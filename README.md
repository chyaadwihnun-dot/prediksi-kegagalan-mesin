# Prediksi Kegagalan Mesin (Predictive Maintenance)

Aplikasi web sederhana untuk memprediksi apakah sebuah mesin berisiko mengalami kegagalan, berdasarkan data sensor. Dibuat sebagai Tugas Besar Machine Learning (Finalisasi).

**Penyusun:** Cahya Adwi Hanun (202343501846) - Kelas X7L
**Mata kuliah:** Machine Learning - Universitas PGRI Indonesia (UNINDRA)

---

## 1. Latar Belakang

Kegagalan mesin yang tidak terduga menyebabkan produksi terhenti dan biaya perbaikan yang besar. Dengan memantau data sensor (suhu, kecepatan putar, torsi, dan keausan alat), model machine learning dapat memperkirakan risiko kegagalan lebih awal sehingga perawatan dapat dijadwalkan sebelum mesin rusak (*predictive maintenance*).

## 2. Dataset

- **Nama:** AI4I 2020 Predictive Maintenance Dataset
- **Sumber:** UCI Machine Learning Repository (id 601)
- **Target:** `Machine failure` (0 = tidak gagal, 1 = gagal)

| Fitur | Keterangan |
|---|---|
| Type | Kualitas produk: L (Low), M (Medium), H (High) |
| Air temperature (K) | Suhu udara |
| Process temperature (K) | Suhu proses |
| Rotational speed (rpm) | Kecepatan putar |
| Torque (Nm) | Torsi |
| Tool wear (min) | Lama pemakaian / keausan alat |

## 3. Preprocessing

1. Menghapus kolom identitas: `UDI`, `Product ID`.
2. Menghapus kolom jenis kegagalan (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) agar tidak terjadi kebocoran data (*data leakage*), karena kolom tersebut adalah bagian dari label kegagalan.
3. One-hot encoding pada kolom `Type` (`drop_first=True`) sehingga menjadi `Type_L` dan `Type_M`.
4. Mengganti tanda `[ ]` pada nama kolom menjadi `( )`.
5. Membagi data: 80% training dan 20% testing (`random_state=42`, `stratify=y`).

## 4. Model

**Random Forest Classifier** dengan parameter:

| Parameter | Nilai |
|---|---|
| n_estimators | 200 |
| max_depth | 10 |
| min_samples_leaf | 5 |
| class_weight | balanced |
| random_state | 42 |

`class_weight='balanced'` dipakai karena jumlah data kegagalan jauh lebih sedikit dibanding data normal (data tidak seimbang).

Model disimpan sebagai `model_rf.joblib`, dan urutan fitur disimpan di `fitur.joblib` agar input aplikasi sama persis dengan saat training.

## 5. Hasil Evaluasi

> Isi bagian ini dengan hasil evaluasi dari Colab (accuracy, precision, recall, dan F1-score untuk kelas "gagal", serta confusion matrix).

| Metrik | Nilai |
|---|---|
| Accuracy | ... |
| Precision (gagal) | ... |
| Recall (gagal) | ... |
| F1-score (gagal) | ... |

## 6. Struktur Folder

```
prediksi-kegagalan-mesin/
├── app.py              # aplikasi Streamlit
├── requirements.txt    # daftar library
├── latih_model.py      # skrip training (membuat file .joblib)
├── model_rf.joblib     # model Random Forest hasil training
├── fitur.joblib        # urutan kolom fitur
├── jalankan.bat        # pintasan menjalankan di Windows
└── README.md           # dokumentasi
```

## 7. Cara Menjalankan

**Prasyarat:** Python 3.9 atau lebih baru.

1. Unduh atau clone repo ini, pastikan keempat file (`app.py`, `requirements.txt`, `model_rf.joblib`, `fitur.joblib`) berada di satu folder.
2. Buka terminal (Command Prompt) di folder tersebut.
3. Pasang library:
   ```
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```
   python -m streamlit run app.py
   ```
5. Buka `http://localhost:8501` di browser (biasanya terbuka otomatis).

Di Windows, bisa juga cukup klik dua kali `jalankan.bat`.

**Catatan:** versi scikit-learn di komputer sebaiknya sama dengan versi di Google Colab saat model dilatih. Jika muncul error saat memuat model, samakan versinya dengan `pip install scikit-learn==<versi>`.

## 8. Cara Menggunakan Aplikasi

1. Pilih **Tipe produk** (L, M, atau H).
2. Isi nilai sensor: suhu udara, suhu proses, kecepatan putar, torsi, dan keausan alat.
3. Klik **Prediksi**.
4. Aplikasi menampilkan status **Aman** atau **Berisiko Gagal** beserta probabilitas kegagalan.

## 9. Link

- GitHub: https://github.com/chyaadwihnun-dot/prediksi-kegagalan-mesin
- Google Colab (training model): *isi link Colab*
- Video YouTube: *isi link YouTube*

## 10. Teknologi

Python, pandas, scikit-learn, joblib, Streamlit.
