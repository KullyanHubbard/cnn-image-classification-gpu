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
pip install -r requirements.txt
```

## Menjalankan di Windows dan Ubuntu

Repository ini bisa dipakai dari Windows dan Ubuntu sekaligus dengan cara clone repo yang sama di masing-masing OS, lalu sinkronisasi lewat GitHub.

### Windows

```powershell
git pull origin main
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Setelah selesai bekerja:

```powershell
git add .
git commit -m "Update from Windows"
git push origin main
```

### Ubuntu

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
git clone https://github.com/KullyanHubbard/cnn-image-classification.git
cd cnn-image-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Setelah selesai bekerja:

```bash
git add .
git commit -m "Update from Ubuntu"
git push origin main
```

Sebelum mulai kerja di OS lain, selalu jalankan:

```bash
git pull origin main
```

## Penggunaan

Buka dan jalankan `cnn.ipynb` di Jupyter Notebook untuk:

- Melatih model CNN
- Mengevaluasi performa model
- Menyimpan model terbaik ke `results/best_model.keras`
- Melakukan prediksi gambar dari folder `single_prediction`

Ringkasan hasil eksperimen tersimpan di `Hasil Data CNN.txt`.
