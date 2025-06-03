# Laporan Proyek Machine Learning - Sistem Rekomendasi Film

## Project Overview

Proyek ini fokus pada pengembangan **sistem rekomendasi film** untuk mengatasi masalah **"information overload"** yang sering dialami pengguna di platform *streaming* digital. Dengan melimpahnya pilihan film, pengguna seringkali kesulitan menemukan konten yang relevan dan menarik bagi mereka.

Sistem rekomendasi ini penting karena secara signifikan dapat **meningkatkan pengalaman pengguna** dengan menyajikan film-film yang dipersonalisasi berdasarkan preferensi mereka. Dengan menganalisis pola *rating* dan interaksi, sistem rekomendasi membantu pengguna menemukan konten baru yang mungkin mereka sukai, mendorong *engagement* yang lebih tinggi dengan platform, dan pada akhirnya, meningkatkan loyalitas pengguna.

Menurut Ricci et al. (2011) dalam buku *Recommender Systems Handbook*, sistem rekomendasi merupakan alat krusial untuk mengatasi *information overload* dengan menyaring informasi yang relevan bagi pengguna dan memfasilitasi penemuan konten yang efektif.

-----

## Business Understanding

Bagian ini mengklarifikasi masalah bisnis inti yang ingin diselesaikan oleh proyek sistem rekomendasi film, serta menetapkan tujuan yang terukur.

### Problem Statements

1.  **Kelebihan Pilihan (Information Overload):** Pengguna merasa kewalahan dengan banyaknya pilihan film di platform *streaming*, sehingga sulit bagi mereka untuk menemukan film yang menarik dan relevan secara efisien.
2.  **Rendahnya Engagement Pengguna:** Tanpa rekomendasi personal dan relevan, pengguna mungkin merasa frustrasi atau bosan, yang dapat mengurangi interaksi dan waktu mereka di platform.
3.  **Kesulitan dalam Penemuan Film Baru:** Pengguna cenderung terpaku pada genre atau sutradara yang sudah familiar, menghambat penemuan film-film baru yang berpotensi mereka sukai.

### Goals

1.  **Menyediakan Rekomendasi Film Personal:** Membangun sistem yang mampu memprediksi *rating* yang mungkin diberikan pengguna pada film yang belum ditonton, sehingga dapat merekomendasikan film dengan potensi disukai tertinggi.
2.  **Meningkatkan Penemuan Konten (Content Discovery):** Membantu pengguna menemukan film-film baru yang relevan di luar preferensi eksplisit mereka, memperluas cakrawala tontonan mereka.
3.  **Meningkatkan Pengalaman Pengguna:** Menyajikan daftar film yang menarik dan dipersonalisasi, yang akan meningkatkan kepuasan dan loyalitas pengguna terhadap platform.

### Solution Approach

Kami mengimplementasikan sistem rekomendasi menggunakan pendekatan **Collaborative Filtering**. Kami mengeksplorasi dua algoritma dalam kategori ini untuk membandingkan efektivitasnya:

1.  **User-Based Collaborative Filtering (menggunakan k-Nearest Neighbors - k-NN):** Pendekatan ini merekomendasikan film dengan mengidentifikasi pengguna lain yang memiliki pola *rating* serupa dengan pengguna target.
2.  **Matrix Factorization (menggunakan Singular Value Decomposition - SVD):** Pendekatan ini menguraikan matriks *rating* pengguna-item yang *sparse* menjadi faktor-faktor laten tersembunyi untuk memprediksi *rating* yang belum diketahui.

-----

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **MovieLens 100k Dataset**. Dataset ini sangat populer untuk riset sistem rekomendasi, berisi **100.000 *rating*** (skala 1-5) dari **943 pengguna unik** untuk **1.682 film unik**. Setiap pengguna telah memberikan *rating* minimal 20 film.

**Sumber Data:**
Dataset diunduh langsung dari situs GroupLens Research: [http://files.grouplens.org/datasets/movielens/ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip).

**Uraian Variabel (Fitur):**

* `user_id`: ID unik pengguna.
* `item_id` / `movie_id`: ID unik film.
* `rating`: Nilai *rating* yang diberikan pengguna (1-5).
* `timestamp`: Waktu *rating* diberikan (dalam format Unix *timestamp*).
* `movie_title`: Judul lengkap film.
* `release_date`: Tanggal rilis film, menunjukkan kapan film tersebut pertama kali dirilis ke publik.
* `video_release_date`: Tanggal rilis video film. Untuk dataset ini, kolom ini seringkali memiliki nilai yang sama dengan `release_date` atau tidak relevan, dan umumnya diabaikan dalam analisis.
* `imdb_url`: Tautan (URL) langsung ke halaman film tersebut di Internet Movie Database (IMDb.com), menyediakan akses ke informasi film yang lebih detail.
* **Genre**: Sebanyak 19 kolom biner (0 atau 1) yang menunjukkan kategori genre film. Nilai 1 berarti film termasuk dalam genre tersebut, 0 berarti tidak. Daftar lengkap genre adalah:
    * `unknown`
    * `Action`
    * `Adventure`
    * `Animation`
    * `Children's`
    * `Comedy`
    * `Crime`
    * `Documentary`
    * `Drama`
    * `Fantasy`
    * `Film-Noir`
    * `Horror`
    * `Musical`
    * `Mystery`
    * `Romance`
    * `Sci-Fi`
    * `Thriller`
    * `War`
    * `Western`
* `age`: Usia pengguna.
* `gender`: Jenis kelamin pengguna (`M` untuk Pria, `F` untuk Wanita).
* `occupation`: Pekerjaan pengguna.
* `zip_code`: Kode pos tempat tinggal pengguna.

**Insight dari EDA (Exploratory Data Analysis):**
Analisis data awal menunjukkan bahwa dataset memiliki **sparsity yang tinggi** (sekitar 93.68%), yang umum dalam sistem rekomendasi karena tidak semua pengguna me-*rating* semua film. Distribusi *rating* cenderung positif, dengan *rating* 3 dan 4 paling sering diberikan. Film-film populer seperti 'Star Wars (1977)' mendapatkan jumlah *rating* tertinggi.

-----

## Data Preparation

Tahap persiapan data sangat penting untuk mengonversi data mentah ke format yang sesuai untuk *modeling*. Proses yang dilakukan adalah:

1.  **Pemilihan Kolom Relevan:** Hanya kolom `user_id`, `item_id`, dan `rating` yang digunakan untuk pelatihan model rekomendasi.
2.  **Pembuatan Objek Dataset untuk Library Surprise:** Data diubah menjadi objek `Dataset` yang spesifik untuk library `surprise` yang digunakan dalam *modeling*.
3.  **Pembagian Data Latih dan Uji:** Data dibagi menjadi 80% untuk set pelatihan dan 20% untuk set pengujian (`test_size=0.2`). Ini dilakukan untuk mencegah *overfitting* dan memungkinkan evaluasi kinerja model yang objektif pada data yang belum pernah dilihat.

-----

## Modeling and Results

Proyek ini membangun dan melatih dua model *collaborative filtering* untuk memprediksi *rating* film dan menyajikan *top-N recommendation*.

### Solusi Rekomendasi 1: User-Based Collaborative Filtering (k-NN)

Model ini dilatih untuk menemukan pengguna dengan selera serupa dan merekomendasikan film yang disukai oleh "tetangga" tersebut.

  * **Kelebihan**: Intuitif, tidak memerlukan fitur item yang kaya, dapat menemukan item yang tidak terduga.
  * **Kekurangan**: Masalah skalabilitas dan *sparsity* pada dataset besar, serta tantangan *cold start* untuk pengguna baru.

### Solusi Rekomendasi 2: Matrix Factorization (SVD)

Model ini menguraikan matriks *rating* menjadi faktor-faktor laten untuk memprediksi *rating* yang hilang.

  * **Kelebihan**: Umumnya memberikan akurasi yang lebih tinggi pada data *sparse*, lebih skalabel dibandingkan k-NN untuk dataset yang lebih besar, dan efektif dalam menangani *sparsity*.
  * **Kekurangan**: Kurang intuitif karena melibatkan faktor laten abstrak, masih memiliki masalah *cold start* untuk pengguna atau item yang sama sekali baru, dan komputasi bisa intensif untuk dataset yang sangat besar.

### Contoh Top-N Recommendation (untuk `user_id` 100)

**Dari User-Based KNN:**

  * Film: Great Day in Harlem, A (1994) (Estimasi Rating: 5.00)
  * Film: They Made Me a Criminal (1939) (Estimasi Rating: 5.00)
  * Film: Marlene Dietrich: Shadow and Light (1996) (Estimasi Rating: 5.00)
  * Film: Saint of Fort Washington, The (1993) (Estimasi Rating: 5.00)
  * Film: Santa with Muscles (1996) (Estimasi Rating: 5.00)
  * Film: Aiqing wansui (1994) (Estimasi Rating: 5.00)
  * Film: Someone Else's America (1995) (Estimasi Rating: 5.00)
  * Film: Entertaining Angels: The Dorothy Day Story (1996) (Estimasi Rating: 5.00)
  * Film: Prefontaine (1997) (Estimasi Rating: 5.00)
  * Film: Star Kid (1997) (Estimasi Rating: 5.00)

**Dari Matrix Factorization (SVD):**

  * Film: Third Man, The (1949) (Estimasi Rating: 4.22)
  * Film: Schindler's List (1993) (Estimasi Rating: 4.20)
  * Film: Fargo (1996) (Estimasi Rating: 4.19)
  * Film: Shawshank Redemption, The (1994) (Estimasi Rating: 4.10)
  * Film: Raiders of the Lost Ark (1981) (Estimasi Rating: 4.08)
  * Film: Braveheart (1995) (Estimasi Rating: 4.06)
  * Film: Close Shave, A (1995) (Estimasi Rating: 4.03)
  * Film: Arsenic and Old Lace (1944) (Estimasi Rating: 4.01)
  * Film: Usual Suspects, The (1995) (Estimasi Rating: 4.00)
  * Film: Wallace & Gromit: The Best of Aardman Animation (1996) (Estimasi Rating: 3.99)

-----

## Evaluation

Kinerja kedua model dievaluasi menggunakan metrik standar untuk masalah prediksi *rating*.

### Metrik Evaluasi

1.  **Root Mean Squared Error (RMSE):**
    * **Deskripsi**: Mengukur akar kuadrat dari rata-rata kuadrat perbedaan antara *rating* yang diprediksi dan *rating* aktual.
    * **Karakteristik**: Metrik ini memberikan bobot lebih pada kesalahan prediksi yang besar (karena dikuadratkan), sehingga sensitif terhadap *outlier*.
    * **Interpretasi**: Nilai **RMSE yang lebih rendah** menunjukkan kinerja model yang lebih baik.
    * **Rumus:**
        $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

2.  **Mean Absolute Error (MAE):**
    * **Deskripsi**: Mengukur rata-rata dari nilai absolut perbedaan antara *rating* yang diprediksi dan *rating* aktual.
    * **Karakteristik**: Metrik ini memberikan bobot yang sama untuk semua kesalahan dan kurang sensitif terhadap *outlier* dibandingkan RMSE.
    * **Interpretasi**: Nilai **MAE yang lebih rendah** juga menunjukkan kinerja model yang lebih baik.
    * **Rumus:**
        $$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

---

### Hasil Evaluasi

Berdasarkan *output* dari proses *modeling*, metrik evaluasi untuk kedua model adalah sebagai berikut:

#### Untuk User-Based Collaborative Filtering (KNN):
* **RMSE**: `1.0194`
* **MAE**: `0.8038`

#### Untuk Matrix Factorization (SVD):
* **RMSE**: `0.9352`
* **MAE**: `0.7375`

---

### Analisis Hasil

Dari hasil di atas, model **Matrix Factorization (SVD)** menunjukkan kinerja yang **lebih unggul** dibandingkan dengan User-Based Collaborative Filtering (KNN).

* Nilai **RMSE SVD yang lebih rendah** (`0.9352` berbanding `1.0194`) mengindikasikan bahwa prediksi SVD memiliki deviasi kuadrat rata-rata yang lebih kecil dari *rating* aktual. Artinya, model SVD lebih akurat dalam memprediksi *rating*, terutama untuk kesalahan prediksi yang signifikan.
* Nilai **MAE SVD yang lebih rendah** (`0.7375` berbanding `0.8038`) juga menunjukkan bahwa SVD memiliki rata-rata kesalahan absolut yang lebih kecil. Ini berarti, secara umum, perbedaan antara *rating* prediksi dan *rating* aktual pada SVD lebih kecil.

Perbedaan kinerja ini konsisten dengan literatur dan praktik di mana **Matrix Factorization** seringkali unggul dalam akurasi pada dataset **sparse** dibandingkan pendekatan berbasis kesamaan tradisional. Model SVD mampu menangkap **pola tersembunyi (latent factors)** dalam data *rating* yang lebih efektif dalam memprediksi *rating* yang belum diketahui. Selain itu, seperti yang terlihat dari contoh rekomendasi, SVD cenderung memberikan rekomendasi film-film yang lebih populer dan realistis, berbeda dengan KNN yang dalam kasus ini merekomendasikan film-film dengan *rating* estimasi sempurna namun kurang dikenal.

### Dampak terhadap Business Understanding (Tambahan)

Hasil evaluasi ini secara langsung dan signifikan berdampak pada tujuan bisnis yang telah ditetapkan:

* **Menjawab Problem Statements:**
    * **Kelebihan Pilihan:** Kedua model, khususnya SVD yang lebih akurat, efektif dalam memfilter ribuan film dan menyajikan daftar *top-N* yang relevan. Ini secara langsung mengurangi *information overload* dengan menyediakan pilihan terkurasi yang sesuai dengan preferensi pengguna.
    * **Rendahnya Engagement:** Dengan memberikan rekomendasi film yang memiliki estimasi *rating* tinggi dan relevan (terutama SVD yang cenderung merekomendasikan film populer dengan akurasi lebih baik), model ini meningkatkan kemungkinan pengguna menemukan konten yang mereka sukai, sehingga mendorong *engagement* yang lebih tinggi dan waktu yang dihabiskan di platform.
    * **Kesulitan Penemuan Film Baru:** Model SVD, dengan kemampuannya menangkap faktor laten, dapat merekomendasikan film yang mungkin tidak secara langsung terkait dengan genre yang sering ditonton pengguna, namun relevan dengan selera laten mereka. Ini secara efektif membantu pengguna menjelajahi dan menemukan film di luar kebiasaan tontonan mereka.

* **Pencapaian Goals:**
    * **Menyediakan Rekomendasi Film Personal:** Kedua algoritma berhasil memprediksi *rating* dan menghasilkan rekomendasi personal berdasarkan data historis pengguna. Model SVD menunjukkan akurasi yang lebih tinggi, yang berarti prediksinya lebih dapat diandalkan dan personal untuk setiap pengguna.
    * **Meningkatkan Penemuan Konten:** Dengan menyediakan daftar film yang diprediksi akan disukai pengguna, sistem ini secara aktif memfasilitasi penemuan konten baru yang relevan, seperti yang ditunjukkan oleh daftar rekomendasi dari SVD yang berisi film-film berkualitas tinggi yang mungkin belum pernah ditemukan pengguna.
    * **Meningkatkan Pengalaman Pengguna:** Akurasi prediksi yang lebih baik dari SVD berkontribusi pada pengalaman pengguna yang lebih memuaskan, karena rekomendasi yang diberikan lebih sering sesuai dengan preferensi mereka, meningkatkan kepuasan dan loyalitas terhadap platform.

* **Dampak Solusi Statement:**
    * **User-Based Collaborative Filtering (k-NN):** Meskipun berhasil melatih model dan menghasilkan rekomendasi, nilai RMSE dan MAE yang lebih tinggi menunjukkan bahwa prediksinya kurang akurat dibandingkan SVD. Selain itu, rekomendasi *top-N* cenderung menghasilkan film yang kurang populer atau mungkin merupakan artefak dari *sparsity* data, yang kurang optimal untuk tujuan bisnis meningkatkan *engagement* melalui penemuan konten populer.
    * **Matrix Factorization (SVD):** Pendekatan ini terbukti lebih berdampak positif. Akurasi yang lebih tinggi (RMSE dan MAE lebih rendah) menunjukkan kemampuannya yang unggul dalam memprediksi *rating* yang akurat. Rekomendasi yang dihasilkan SVD juga terlihat lebih relevan secara komersial dan cenderung merekomendasikan film-film yang lebih diakui. Ini secara langsung mendukung tujuan bisnis untuk meningkatkan *engagement* dan penemuan konten berkualitas. Oleh karena itu, pendekatan SVD terbukti lebih berhasil dalam menjawab *problem statements* dan mencapai *goals* proyek ini.

---

### Kesimpulan

Secara keseluruhan, proyek ini berhasil membangun dan membandingkan dua model sistem rekomendasi **collaborative filtering**. Model **SVD** terbukti lebih unggul dalam hal **akurasi prediksi *rating***. Kinerja superior SVD ini menjadikannya pilihan yang lebih baik untuk sistem rekomendasi film ini, karena secara efektif **menjawab *problem statements*** terkait *information overload* dan *engagement*, serta **berhasil mencapai *goals*** untuk menyediakan rekomendasi personal yang meningkatkan penemuan konten dan pengalaman pengguna. Pendekatan **Matrix Factorization (SVD)** terbukti lebih berdampak positif dalam konteks bisnis yang direncanakan.

-----


