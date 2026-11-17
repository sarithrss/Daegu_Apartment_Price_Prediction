# Prediksi Harga Jual Apartemen di Daegu

Capstone Project Modul 3: Machine Learning Regression
Bootcamp Data Science & Machine Learning, Purwadhika

**Penulis:** [isi namamu]

---

## Project Introduction

### Latar belakang

Daegu adalah kota di Korea Selatan dengan lahan hunian yang terbatas dan aktivitas bisnis yang padat, sehingga apartemen menjadi salah satu solusi kebutuhan hunian. Unit apartemen ditawarkan oleh perorangan maupun perusahaan, yang menentukan sendiri harga jualnya di sebuah platform. Harga dipengaruhi banyak faktor sekaligus, baik internal (ukuran, umur bangunan, fasilitas) maupun eksternal (jarak ke stasiun subway, fasilitas di sekitar), sehingga penjual sulit menetapkan harga yang sesuai pasar. Harga yang terlalu tinggi membuat unit sulit terjual, sedangkan harga yang terlalu rendah membuat keuntungan tidak maksimal.

### Tujuan

Membangun model regresi yang memprediksi `SalePrice` (harga jual dalam Won) berdasarkan karakteristik unit, sebagai **harga acuan** bagi penjual.

### Dataset

- Sumber: data bootcamp modul Machine Learning, Purwadhika.
- 4.123 baris dan 11 kolom (10 fitur dan 1 target). Setelah 1.422 baris duplikat dibuang, tersisa **2.701 baris**.
- Fitur: tipe lorong (`HallwayType`), waktu tempuh dan stasiun subway terdekat (`TimeToSubway`, `SubwayStation`), jumlah fasilitas dan universitas di sekitar, parkir basement, tahun dibangun, fasilitas di dalam apartemen, dan ukuran (`Size(sqf)`).
- Target: `SalePrice`.

### Pendekatan

1. Exploratory Data Analysis (termasuk laporan otomatis dengan `ydata-profiling`).
2. Preprocessing: duplikat dibuang, outlier dan data janggal dipertahankan dan dicatat sebagai keterbatasan, One-Hot Encoding untuk kolom kategorikal (di dalam pipeline).
3. Pemodelan: split data 80:20, benchmark lima model dengan 5-fold cross-validation (Linear Regression, KNN, Decision Tree, Random Forest, XGBoost), dan baseline yang selalu memprediksi rata-rata.
4. Hyperparameter tuning XGBoost dengan `RandomizedSearchCV` (40 kombinasi).
5. Evaluasi pada test set dan interpretasi (feature importance, analisis error).

Metrik evaluasi: RMSE (utama), MAE, MAPE, dan R² sebagai pelengkap.

### Hasil utama

Model akhir: **XGBoost Regressor setelah tuning** (pipeline lengkap dengan encoding).

| Model (test set, 541 unit) | RMSE | MAE | MAPE | R² |
| --- | --- | --- | --- | --- |
| Baseline (selalu memprediksi rata-rata) | 103.098 | 84.205 | 54,4% | ≈ 0 |
| XGBoost default | 48.005 | 38.963 | 19,9% | 0,783 |
| **XGBoost setelah tuning (model final)** | **47.642** | **38.773** | **19,8%** | **0,786** |

- Error turun sekitar 54% dibandingkan baseline.
- Pada cross-validation, Random Forest, XGBoost, dan Decision Tree hampir setara (selisih jauh di bawah standar deviasi antar fold), sehingga XGBoost dipilih bukan karena selisih skor, melainkan karena model pohon cocok untuk data ini dan XGBoost fleksibel untuk dituning.
- Tuning hanya memberi peningkatan kecil, dan hanya XGBoost yang dituning.

### Temuan utama

- Tipe lorong (`HallwayType`) adalah faktor paling dominan (sekitar 71% dari feature importance), disusul fasilitas sekitar, parkir basement, ukuran, dan fasilitas di dalam apartemen.
- Model cenderung memprediksi unit murah terlalu tinggi dan unit mahal terlalu rendah. Akurasi terbaik pada tipe terraced, terburuk pada tipe mixed.
- Sebagai harga acuan, berikan prediksi dalam bentuk **rentang**. Pada data test, rentang ± 38.773 (MAE) hanya mencakup sekitar 53% unit, ± 62.000 mencakup sekitar 80%, dan ± 74.000 mencakup sekitar 90%.

### Limitasi

- Duplikat dibuang walaupun bisa jadi merupakan unit berbeda dengan karakteristik yang sama.
- Kemungkinan banyak unit berasal dari kompleks yang sama, sehingga performa pada kompleks baru kemungkinan lebih rendah.
- Fitur terbatas (tidak ada lantai, kondisi unit, koordinat lokasi, waktu transaksi), dan data hanya dari satu kota.
- Terdapat data yang tidak konsisten (88 unit tanpa subway terdekat tetapi tercatat 5-10 menit ke subway, dan 355 unit dengan parkir basement 0) yang dipertahankan karena tidak ada dasar untuk mengoreksinya.
- Error sekitar 20% belum cukup presisi untuk dijadikan penentu harga tunggal.

---

## Isi repository

| File | Keterangan |
| --- | --- |
| `README.md` | Pengantar proyek (file ini) |
| `Capstone_Project_Modul_3.ipynb` | Jupyter Notebook: dokumentasi pengerjaan dari business understanding sampai rekomendasi |
| `model_xgboost_daegu.pkl` | Model final (pipeline preprocessing + XGBoost) yang disimpan dengan Pickle |
| `data_daegu_apartment.csv` | Dataset yang dipakai (agar notebook dapat dijalankan) |
| `requirements.txt` | Daftar library beserta versinya |

## Cara memakai model

Model yang disimpan adalah pipeline lengkap, jadi data baru cukup berupa tabel dengan kolom yang sama seperti data latih (tanpa `SalePrice`):

```python
import pickle
import pandas as pd

with open('model_xgboost_daegu.pkl', 'rb') as f:
    model = pickle.load(f)

unit_baru = pd.DataFrame([{
    'HallwayType': 'terraced',
    'TimeToSubway': '0-5min',
    'SubwayStation': 'Kyungbuk_uni_hospital',
    'N_FacilitiesNearBy(ETC)': 0.0,
    'N_FacilitiesNearBy(PublicOffice)': 3.0,
    'N_SchoolNearBy(University)': 2.0,
    'N_Parkinglot(Basement)': 1270.0,
    'YearBuilt': 2007,
    'N_FacilitiesInApt': 10,
    'Size(sqf)': 1643,
}])

print(model.predict(unit_baru))
```

Gunakan versi library yang sama seperti di `requirements.txt` agar model dapat dimuat tanpa masalah.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, seaborn, matplotlib, ydata-profiling.
