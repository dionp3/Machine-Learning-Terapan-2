# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

---

## Project Overview

Proyek ini berfokus pada pengembangan sistem rekomendasi film untuk mengatasi masalah **"information overload"** yang sering dialami pengguna di platform *streaming*. Dengan melimpahnya pilihan film, pengguna kesulitan menemukan konten yang relevan dan menarik bagi mereka. Sistem rekomendasi hadir sebagai solusi untuk mempersonalisasi pengalaman pengguna, meningkatkan *engagement*, dan membantu penemuan film-film baru yang sesuai dengan preferensi individu.

---

## Business Understanding

Proyek ini mengklarifikasi beberapa masalah bisnis inti dan menetapkan tujuan yang jelas.

### Problem Statements

1.  **Kelebihan Pilihan (Information Overload):** Pengguna dihadapkan pada ribuan film di platform *streaming*, sehingga sulit menemukan film yang menarik dan relevan secara efisien.
2.  **Rendahnya Engagement Pengguna:** Tanpa rekomendasi yang personal dan relevan, pengguna mungkin merasa frustrasi dan kurang termotivasi untuk menjelajahi katalog, yang dapat mengurangi *engagement* mereka dengan platform.
3.  **Kesulitan dalam Penemuan Film Baru:** Pengguna cenderung menonton film dari genre atau sutradara yang sudah mereka kenal, sehingga melewatkan banyak film potensial di luar lingkaran preferensi mereka.

### Goals

1.  **Menyediakan Rekomendasi Film Personal:** Membangun sistem yang mampu merekomendasikan film yang sangat mungkin disukai oleh seorang pengguna berdasarkan data *rating* sebelumnya.
2.  **Meningkatkan Penemuan Konten (Content Discovery):** Membantu pengguna menemukan film-film baru yang relevan yang mungkin tidak akan mereka temukan sendiri.
3.  **Meningkatkan Pengalaman Pengguna:** Menyajikan daftar film yang menarik dan personal untuk setiap pengguna, sehingga meningkatkan kepuasan dan loyalitas terhadap platform.

### Solution Approach

Kami mengimplementasikan sistem rekomendasi menggunakan pendekatan **Collaborative Filtering** dengan dua strategi utama:

1.  **User-Based Collaborative Filtering:** Merekomendasikan film berdasarkan kesamaan selera *rating* antar pengguna.
2.  **Matrix Factorization (SVD):** Menguraikan matriks *rating* pengguna-item menjadi faktor-faktor laten tersembunyi untuk memprediksi *rating* yang belum diketahui.

---

## Data Understanding

Dataset yang digunakan adalah **MovieLens 100k Dataset**. Dataset ini berisi 100.000 *rating* (1-5) dari 943 pengguna pada 1.682 film, dengan setiap pengguna memberikan *rating* minimal 20 film.

**Sumber Data:**
Dataset diunduh langsung dari GroupLens Research: [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/).

**Variabel-variabel utama pada dataset meliputi:**
* `user_id`: ID unik pengguna.
* `item_id` / `movie_id`: ID unik film.
* `rating`: Nilai *rating* yang diberikan pengguna untuk film.
* `timestamp`: Waktu *rating* diberikan.
* `movie_title`: Judul film.
* `genre`: Informasi biner mengenai genre film.
* `age`, `gender`, `occupation`, `zip_code`: Informasi demografi pengguna.

**Insight dari EDA:**
Data menunjukkan *sparsity* yang tinggi (sekitar 93.68%), yang umum dalam sistem rekomendasi, menandakan bahwa sebagian besar kombinasi pengguna-film belum di-*rating*. Distribusi *rating* cenderung positif (banyak *rating* 3 dan 4), dengan film populer seperti 'Star Wars (1977)' mendapatkan banyak perhatian.

---

## Data Preparation

Tahap persiapan data melibatkan pembentukan data ke format yang sesuai untuk model *collaborative filtering*. Proses utama yang dilakukan adalah:

1.  **Pemilihan Kolom Relevan:** Hanya kolom `user_id`, `item_id`, dan `rating` yang digunakan.
2.  **Pembuatan Objek Dataset untuk Surprise:** Data diubah ke format yang kompatibel dengan library `surprise`.
3.  **Pembagian Data Latih dan Uji:** Data dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian untuk evaluasi model yang objektif dan mencegah *overfitting*.

Tujuan dari persiapan data ini adalah untuk memastikan kompatibilitas dengan model dan memungkinkan evaluasi kinerja model secara objektif.

---

## Modeling

Proyek ini mengimplementasikan dua model sistem rekomendasi *collaborative filtering* untuk memprediksi *rating* film. Output utama adalah *top-N recommendation* untuk pengguna tertentu.

### Solusi Rekomendasi 1: User-Based Collaborative Filtering (k-NN)
* **Pendekatan**: Mengidentifikasi pengguna serupa dan merekomendasikan film yang disukai oleh pengguna serupa tersebut.
* **Kelebihan**: Intuitif, tidak memerlukan fitur item, dapat menemukan item yang tidak terduga.
* **Kekurangan**: Masalah skalabilitas dan *sparsity* pada dataset besar, serta masalah *cold start* untuk pengguna baru.

### Solusi Rekomendasi 2: Matrix Factorization (SVD)
* **Pendekatan**: Menguraikan matriks *rating* menjadi dua matriks laten (pengguna dan item) yang lebih kecil untuk memprediksi *rating*.
* **Kelebihan**: Akurasi tinggi pada data *sparse*, lebih skalabel, efektif menangani *sparsity*.
* **Kekurangan**: Kurang intuitif, tetap memiliki masalah *cold start* untuk pengguna/item yang benar-benar baru, komputasi bisa intensif untuk dataset sangat besar.

---

## Evaluation

Kinerja model dievaluasi menggunakan metrik standar untuk prediksi *rating*: **Root Mean Squared Error (RMSE)** dan **Mean Absolute Error (MAE)**.

### Metrik Evaluasi

1.  **Root Mean Squared Error (RMSE):** Mengukur akar kuadrat dari rata-rata kuadrat perbedaan antara *rating* prediksi dan aktual. Lebih sensitif terhadap kesalahan besar.
    $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
2.  **Mean Absolute Error (MAE):** Mengukur rata-rata nilai absolut perbedaan antara *rating* prediksi dan aktual. Memberikan bobot yang sama untuk semua kesalahan.
    $$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

### Hasil Evaluasi

* **User-Based Collaborative Filtering (KNN):**
    * **RMSE**: `1.0194`
    * **MAE**: `0.8038`
* **Matrix Factorization (SVD):**
    * **RMSE**: `0.9352`
    * **MAE**: `0.7375`

### Analisis Hasil

Berdasarkan metrik evaluasi, model **Matrix Factorization (SVD)** menunjukkan kinerja yang **lebih unggul** (RMSE dan MAE lebih rendah) dibandingkan dengan User-Based Collaborative Filtering (KNN). SVD lebih akurat dalam memprediksi *rating* karena kemampuannya menangkap **pola tersembunyi (latent factors)** dalam data *rating* yang *sparse* secara lebih efektif. Selain itu, rekomendasi SVD cenderung lebih relevan dan logis, merekomendasikan film-film populer yang lebih mungkin disukai pengguna.

### Kesimpulan

Secara keseluruhan, proyek ini berhasil membangun dan membandingkan dua model sistem rekomendasi **collaborative filtering**. Model **SVD** terbukti lebih unggul dalam hal **akurasi prediksi *rating***, menjadikannya pilihan yang lebih baik untuk sistem rekomendasi film ini.
