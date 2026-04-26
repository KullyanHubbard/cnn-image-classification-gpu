import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MobileNetV2 transfer-learning model for cats vs dogs."
    )
    parser.add_argument("--train-dir", default="training_set")
    parser.add_argument("--val-dir", default="test_set")
    parser.add_argument("--output-model", default="results/best_transfer_model.keras")
    parser.add_argument("--metrics-file", default="results/transfer_learning_metrics.json")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def configure_runtime(seed, no_mixed_precision):
    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if gpus and not no_mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    print("TensorFlow:", tf.__version__)
    print("GPU devices:", gpus if gpus else "not detected")
    print("Mixed precision:", mixed_precision.global_policy())


def load_dataset(directory, image_size, batch_size, seed, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def prepare_datasets(args):
    train_ds = load_dataset(
        args.train_dir,
        args.image_size,
        args.batch_size,
        args.seed,
        shuffle=True,
    )
    val_ds = load_dataset(
        args.val_dir,
        args.image_size,
        args.batch_size,
        args.seed,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(args):
    weights = None if args.weights == "none" else args.weights
    input_shape = (args.image_size, args.image_size, 3)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def callbacks(output_model):
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            output_model,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def fine_tune_base_model(base_model, fine_tune_layers):
    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - fine_tune_layers)

    for index, layer in enumerate(base_model.layers):
        if index < fine_tune_at or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def best_accuracy(history):
    if not history.history.get("val_accuracy"):
        return None, None

    best_index = int(np.argmax(history.history["val_accuracy"]))
    return best_index + 1, float(history.history["val_accuracy"][best_index])


def confusion_matrix(model, val_ds, threshold):
    y_true_batches = []
    y_prob_batches = []

    for images, labels in val_ds:
        y_true_batches.append(labels.numpy().reshape(-1))
        y_prob_batches.append(model.predict(images, verbose=0).reshape(-1))

    y_true = np.concatenate(y_true_batches).astype(int)
    y_prob = np.concatenate(y_prob_batches)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = float(np.mean(y_true == y_pred))
    return {"accuracy": accuracy, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def save_metrics(args, class_names, head_history, fine_history, eval_metrics, matrix):
    head_epoch, head_acc = best_accuracy(head_history)
    fine_epoch, fine_acc = best_accuracy(fine_history) if fine_history else (None, None)
    payload = {
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "class_names": class_names,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "weights": args.weights,
        "head_epochs": args.head_epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_layers": args.fine_tune_layers,
        "head_best_epoch": head_epoch,
        "head_best_val_accuracy": head_acc,
        "fine_tune_best_epoch": fine_epoch,
        "fine_tune_best_val_accuracy": fine_acc,
        "best_model_eval": {
            "loss": float(eval_metrics[0]),
            "accuracy": float(eval_metrics[1]),
            "auc": float(eval_metrics[2]),
        },
        "threshold": args.threshold,
        "confusion_matrix": matrix,
    }

    metrics_path = Path(args.metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    args = parse_args()
    os.makedirs(Path(args.output_model).parent, exist_ok=True)
    configure_runtime(args.seed, args.no_mixed_precision)
    train_ds, val_ds, class_names = prepare_datasets(args)
    print("Class names:", class_names)

    model, base_model = build_model(args)
    compile_model(model, args.head_lr)
    model.summary()

    train_callbacks = callbacks(args.output_model)
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=train_callbacks,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
    )

    fine_history = None
    if args.fine_tune_epochs > 0:
        fine_tune_base_model(base_model, args.fine_tune_layers)
        compile_model(model, args.fine_tune_lr)
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.head_epochs + args.fine_tune_epochs,
            initial_epoch=args.head_epochs,
            callbacks=train_callbacks,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps,
        )

    best_model = tf.keras.models.load_model(args.output_model)
    eval_metrics = best_model.evaluate(val_ds, verbose=1)
    matrix = confusion_matrix(best_model, val_ds, args.threshold)
    payload = save_metrics(args, class_names, head_history, fine_history, eval_metrics, matrix)

    print("\nBest model:", args.output_model)
    print(f"Eval accuracy: {payload['best_model_eval']['accuracy'] * 100:.2f}%")
    print(f"Eval AUC: {payload['best_model_eval']['auc']:.4f}")
    print("Confusion matrix:", matrix)
    print("Metrics file:", args.metrics_file)


if __name__ == "__main__":
    main()
