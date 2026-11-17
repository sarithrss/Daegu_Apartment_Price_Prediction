# Daegu Apartment Price Prediction: A Machine Learning Regression Approach

## 1. Project Overview

Proyek ini mengembangkan model Machine Learning untuk memprediksi harga jual apartemen di Daegu, Korea Selatan, berdasarkan karakteristik properti, fasilitas apartemen, serta aksesibilitas terhadap fasilitas publik dan transportasi.

Penentuan harga jual apartemen menjadi tantangan bagi penjual karena harga dipengaruhi oleh berbagai faktor, seperti ukuran bangunan, tipe lorong, tahun pembangunan, fasilitas sekitar, dan akses menuju stasiun subway.

Fokus utama proyek ini adalah memanfaatkan pendekatan Machine Learning Regression untuk menghasilkan estimasi harga jual yang lebih objektif dan berbasis data, sehingga dapat membantu penjual dalam menentukan harga yang kompetitif.

🎯 **Key Objectives:**
- Objective 1: Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap harga jual apartemen di Daegu.
- Objective 2: Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik dan pola hubungan antarvariabel.
- Objective 3: Membandingkan performa beberapa algoritma Machine Learning Regression untuk memperoleh model prediksi terbaik.
- Objective 4: Melakukan hyperparameter tuning untuk meningkatkan performa model terpilih.
- Objective 5: Memberikan rekomendasi harga jual apartemen berdasarkan hasil prediksi dan evaluasi model.

## 2. Data Sources

- **Dataset (Daegu Apartment):** Dataset yang disediakan dalam Bootcamp Data Science & Machine Learning Purwadhika, terdiri dari 4.123 baris dan 11 kolom (10 independent variables dan 1 target variable).
- **Data Preprocessing:** Sebanyak 1.422 baris duplikat dihapus, sehingga menghasilkan 2.701 baris data untuk proses analisis dan pemodelan.

**Features:**
- `HallwayType`: Tipe lorong apartemen.
- `TimeToSubway`: Waktu tempuh menuju stasiun subway terdekat.
- `SubwayStation`: Nama stasiun subway terdekat.
- `N_FacilitiesNearBy(ETC)`: Jumlah fasilitas umum lainnya di sekitar apartemen.
- `N_FacilitiesNearBy(PublicOffice)`: Jumlah kantor pemerintahan di sekitar apartemen.
- `N_SchoolNearBy(University)`: Jumlah universitas di sekitar apartemen.
- `N_Parkinglot(Basement)`: Jumlah kapasitas parkir basement.
- `YearBuilt`: Tahun pembangunan apartemen.
- `N_FacilitiesInApt`: Jumlah fasilitas di dalam apartemen.
- `Size(sqf)`: Luas apartemen dalam square feet.

**Target Variable:**
- `SalePrice`: Harga jual apartemen dalam Korean Won (KRW).

## 3. Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Statistical Analysis:** Descriptive Statistics
- **Data Visualization:** Matplotlib, Seaborn
- **Exploratory Data Analysis:** ydata-profiling
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Evaluation:** RMSE, MAE, MAPE, R²
- **Hyperparameter Tuning:** RandomizedSearchCV
- **Environment:** Jupyter Notebook

## 4. Project Structure

- `README.md` — Project overview, business insights, dan hasil pemodelan.
- `Daegu Apartment - ML Regresi.ipynb` — Jupyter Notebook berisi proses EDA, data preprocessing, model development, evaluation, dan business recommendations.
- `data_daegu_apartment.csv` — Dataset yang digunakan untuk analisis dan pemodelan.
- `model_xgboost_daegu.pkl` — Final trained model menggunakan Tuned XGBoost Regressor beserta preprocessing pipeline.
- `PPT` — Summary dari isi project.

## 📊 5. Summary of Findings

### 💡 5.1 Business Insights

Note: Analisis dilakukan menggunakan dataset apartemen di Daegu, Korea Selatan. Model dibangun melalui perbandingan lima algoritma Machine Learning Regression dan dievaluasi menggunakan test set sebanyak 541 observasi.

**Model terbaik: Tuned XGBoost Regressor**

Hasil evaluasi model:

| Model | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| Baseline | 103.098 | 84.205 | 54,4% | ≈ 0 |
| XGBoost Default | 48.005 | 38.963 | 19,9% | 0,783 |
| **XGBoost Tuned** | **47.642** | **38.773** | **19,8%** | **0,786** |

**Key Findings:**

- **Model Performance (R² = 0,786):** Model mampu menjelaskan sekitar 78,6% variasi harga jual apartemen. RMSE turun sekitar 54% dibandingkan baseline, menunjukkan peningkatan kemampuan prediksi yang signifikan.
- **Dominant Feature – HallwayType (±71%):** Tipe lorong apartemen menjadi faktor paling dominan dalam feature importance, disusul fasilitas sekitar, kapasitas parkir basement, ukuran apartemen, dan fasilitas internal.
- **Prediction Error (MAPE = 19,8%):** Model masih memiliki rata-rata persentase error yang cukup besar. Model cenderung overestimate pada apartemen dengan harga rendah dan underestimate pada apartemen dengan harga tinggi.
- **Apartment Type Analysis:** Prediksi relatif lebih akurat pada tipe terraced, sedangkan tipe mixed memiliki tingkat kesalahan prediksi yang lebih tinggi.
- **Price Estimation Range:** Rentang prediksi ±62.000 KRW mencakup sekitar 80% harga aktual pada test set, sedangkan ±74.000 KRW mencakup sekitar 90%.

### 🚀 5.2 Actionable Recommendations

- **Strategi Penetapan Harga** — Gunakan hasil prediksi XGBoost sebagai harga acuan untuk membantu penjual menentukan harga jual yang lebih kompetitif.
- **Property Characteristics** — Perhatikan tipe lorong, ukuran apartemen, fasilitas sekitar, dan kapasitas parkir basement sebagai faktor penting dalam proses penilaian properti.
- **Risk Management in Pricing** — Gunakan rentang estimasi harga, bukan hanya satu nilai prediksi, untuk mengantisipasi potensi kesalahan estimasi.
- **Model Improvement** — Tambahkan variabel seperti lantai apartemen, kondisi bangunan, koordinat lokasi, dan tanggal transaksi untuk meningkatkan akurasi model.
- **Further Evaluation** — Lakukan pengujian pada data dari kompleks apartemen yang belum pernah digunakan dalam training untuk mengukur kemampuan generalisasi model.

### ⚠️ 5.3 Project Limitations

- Penghapusan data duplikat berpotensi menghilangkan unit apartemen berbeda yang memiliki karakteristik identik.
- Beberapa data memiliki ketidakkonsistenan, tetapi tetap dipertahankan karena tidak tersedia informasi yang cukup untuk melakukan koreksi.
- Dataset hanya mencakup apartemen di Daegu sehingga hasil model belum tentu dapat digeneralisasi ke kota lain.
- MAPE sebesar 19,8% menunjukkan bahwa model belum cukup akurat untuk menjadi satu-satunya dasar penentuan harga jual.

## 6. Contact

- **Nama:** Sari Theresia
- **Email:** saritheresia88@gmail.com
