# CNN Image Classification

Proyek ini menggunakan Convolutional Neural Network (CNN) untuk klasifikasi gambar kucing dan anjing.

## Isi Repository

```text
.
|-- cnn.ipynb
|-- Hasil Data CNN.txt
|-- training_set/
|   |-- cats/
|   `-- dogs/
|-- test_set/
|   |-- cats/
|   `-- dogs/
|-- single_prediction/
|   |-- cat_or_dog_1.jpg
|   `-- cat_or_dog_2.jpg
`-- results/
    `-- best_model.keras
```

## Dataset

Dataset sudah disertakan dalam repository.

- `training_set/cats`: 4.001 gambar
- `training_set/dogs`: 4.001 gambar
- `test_set/cats`: 1.001 gambar
- `test_set/dogs`: 1.001 gambar
- `single_prediction`: 2 gambar untuk uji prediksi individual

## Model

Model hasil training tersedia di:

```text
results/best_model.keras
```

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Pillow

Install dependency:

```bash
pip install tensorflow numpy matplotlib pillow
```

## Penggunaan

Buka dan jalankan `cnn.ipynb` di Jupyter Notebook untuk:

- Melatih model CNN
- Mengevaluasi performa model
- Menyimpan model terbaik ke `results/best_model.keras`
- Melakukan prediksi gambar dari folder `single_prediction`

Ringkasan hasil eksperimen tersimpan di `Hasil Data CNN.txt`.
