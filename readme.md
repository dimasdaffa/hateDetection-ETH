# Hate Speech Detection Erick Ten Hag Manchester United Twitter
# DIMAS DAFFA ERNANDA
# A11.2022.14079
# STKI A11.4701
# Streamlit https://hatedetection-eth.streamlit.app/

## Deskripsi Proyek
Proyek ini bertujuan untuk mendeteksi ujaran kebencian (hate speech) pada teks tweet Erick Ten Hag di Twitter menggunakan metode Logistic Regression. Dataset telah melalui berbagai tahapan preprocessing, oversampling, dan vektorisasi sebelum digunakan untuk melatih model.

---

## Struktur File

1. **`Crawl_Twitter_ETH.ipynb`**
   - File ini digunakan untuk mengambil data mentah dari Twitter.
   - Data yang dihasilkan disimpan dalam file `TweethateETHMU.csv`.

2. **`Oversampling.ipynb`**
   - File ini melakukan oversampling pada dataset `TweethateETHMU.csv` untuk menangani ketidakseimbangan kelas.
   - Dataset hasil oversampling disimpan dalam file `Oversampled_Tweet_Dataset.csv`.

3. **`AugmentasiDataset.ipynb`**
   - Notebook ini digunakan untuk melakukan augmentasi data dengan fokus pada menambah variasi teks untuk kelas hate speech.
   - Proses augmentasi dilakukan dengan memanfaatkan dataset Profanity in English, sebuah dataset yang berisi daftar kata-kata kasar atau ujaran kebencian.
   - Dataset hasil augmentasi ini bertujuan untuk memperkuat pelatihan model dalam mendeteksi hate speech dengan menambah data pada kelas yang kurang seimbang.
   - Hasil augmentasi dapat digunakan untuk menggantikan atau melengkapi dataset yang telah melalui proses oversampling sebelumnya.

4. **`LogisticRegHateDetection.ipynb`**
   - File ini melatih model Logistic Regression menggunakan dataset hasil oversampling dan augmented dataset.
   - Model yang telah dilatih disimpan dalam file `logistic_regression_model.pkl`.

5. **`TestModel.ipynb`**
   - File ini digunakan untuk menguji model Logistic Regression pada teks baru.
   - Model yang diuji diambil dari file `logistic_regression_model.pkl`.

6. **`streamlit.py`**
   - File ini berisi antarmuka web sederhana menggunakan Streamlit untuk mendeteksi ujaran kebencian berdasarkan model yang telah dilatih.
   - Menggunakan model dan vektorisasi yang disimpan dalam file `logistic_reg_model.pkl` dan `tfidf_vectorizer.pkl`.

---

## Cara Menjalankan Proyek

### Prasyarat
1. Python (versi >= 3.7)
2. Instalasi pustaka yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

### Langkah-langkah
1. **`Crawl_Twitter_ETH.ipynb`**
   - File ini digunakan untuk mengambil data mentah dari Twitter.
   - Data yang dihasilkan disimpan dalam file `TweethateETHMU.csv`.

2. **`Oversampling.ipynb`**
   - File ini melakukan oversampling pada dataset mentah untuk menangani ketidakseimbangan kelas.
   - Dataset hasil oversampling disimpan dalam file `dataset/Oversampled_Tweet_Dataset.csv`.

3. **`AugmentasiDataset.ipynb`**
   - Notebook ini digunakan untuk melakukan augmentasi data pada dataset oversampled guna menambah variasi pada kelas hate speech.
   - Augmentasi menggunakan dataset `dataset/profanity_en.csv`.
   - Dataset hasil augmentasi disimpan dalam file `dataset/augmented_dataset_with_profanities.csv`.

4. **`LogisticRegHateDetection.ipynb`**
   - File ini melatih model Logistic Regression menggunakan dataset hasil augmentasi.
   - Model yang telah dilatih disimpan dalam file `logistic_reg_model.pkl`.
   - Vektorisasi teks disimpan dalam file `tfidf_vectorizer.pkl`.

5. **`TestModel.ipynb`**
   - File ini digunakan untuk menguji model Logistic Regression pada teks baru.

6. **`streamlit.py`**
   - File ini menyediakan antarmuka web berbasis Streamlit untuk mendeteksi hate speech.
   - Menggunakan model dan vektorisasi yang telah disimpan untuk memprediksi input pengguna.

---

## Hasil Evaluasi Model

- **Accuracy**: {nilai_akurasi}
- **Precision**: {nilai_precision}
- **Recall**: {nilai_recall}

---

## Diskusi Hasil
1. **Kekuatan Model**:
   - Logistic Regression memberikan performa yang baik untuk tugas deteksi hate speech, terutama dalam mendeteksi label mayoritas (`hate`).
   - Dengan preprocessing yang baik (pembersihan teks dan TF-IDF vectorization), model dapat menangkap pola-pola ujaran kebencian dari teks secara efektif.

2. **Kelemahan Model**:
   - Meskipun oversampling membantu menyeimbangkan dataset, model terkadang kesulitan mendeteksi no-hate dengan struktur bahasa yang kompleks atau ambigu.
   - Akurasi untuk kelas minoritas (`no-hate`) dapat ditingkatkan lebih lanjut.

3. **Potensi Pengembangan**:
   - Eksplorasi model lain seperti Support Vector Machine (SVM) atau Neural Networks dapat meningkatkan kemampuan model dalam mendeteksi hate speech.
   - Menambahkan lebih banyak data untuk pelatihan, terutama dari sumber yang bervariasi, dapat membantu model belajar lebih baik.

4. **Penggunaan Nyata**:
   - Model ini dapat digunakan sebagai dasar untuk sistem moderasi konten di media sosial untuk memfilter ujaran kebencian secara otomatis.
   - Integrasi dengan platform moderasi dapat membantu dalam menjaga lingkungan online yang lebih aman.

---

## Penyimpanan Model
- **Model**: `logistic_reg_model.pkl`
- **TF-IDF Vectorizer**: `tfidf_vectorizer.pkl`

