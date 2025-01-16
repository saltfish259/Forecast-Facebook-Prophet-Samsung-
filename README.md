# Laporan Proyek Machine Learning - Razif Zulvikar Hatuwe

## A. Domain Proyek 
Latar Belakang 

Industri saham merupakan bagian dari sistem keuangan global yang memiliki volatilitas tinggi. Harga saham perusahaan sering kali dipengaruhi oleh faktor eksternal seperti kondisi ekonomi, keputusan kebijakan, hingga kinerja perusahaan itu sendiri. Samsung Electronics, sebagai salah satu perusahaan teknologi terkemuka di dunia, memiliki dampak signifikan terhadap pasar saham global. Prediksi harga saham Samsung Electronics penting bagi investor, analis pasar, dan pemegang saham untuk memahami potensi tren yang akan datang dan mengoptimalkan keputusan investasi mereka.

Mengapa dan Bagaimana masalah ini Di Selesaikan

Menghadapi volatilitas pasar, investor memerlukan informasi yang akurat tentang kemungkinan pergerakan harga saham ke depan. Dengan melakukan forecasting pada harga saham Samsung Electronics, kita bisa memproyeksikan harga saham berdasarkan data historis yang ada, sehingga membantu dalam perencanaan strategis investasi. Proyek ini menggunakan algoritma Facebook Prophet karena kemampuannya dalam menangani data deret waktu yang memiliki tren dan musiman. Selain itu, dengan melakukan tuning parameter, model dapat dioptimalkan untuk meminimalkan kesalahan dalam prediksi.

## B. Business Understanding

Propblem Statements 
1. Bagaimana memprediksi harga penutupan saham Samsung Electronics untuk periode satu bulan mendatang dengan tingkat akurasi tinggi?

2. Bagaimana evaluasi performa model forecasting sehingga hasil prediksi bisa dipertanggungjawabkan?

Goals
1. Menghasilkan prediksi harga saham Samsung Electronics yang akurat dengan menggunakan model Facebook Prophet dengan parameter yang optimal untuk periode 2024-09-25 hingga 2024-10-25.

2. Mengukur performa model dengan metrik MAE (Mean Absolute Error) dan RMSE (Root Mean Square Error) sehingga dapat dievaluasi keakuratan prediksi.

Solution Statements
1. Menggunakan model Facebook Prophet dengan tuning parameter seperti ```changepoint_prior_scale=0.5```, ```seasonality_prior_scale=0.01```, dan ```n_changepoints=30```.

2. Menggnuakan regresi tambahan pada fitur yang berkoleasi tinggi dengan harga penutupan, seperti ```Open```, ```High```, ```Low```, dan ```Close``` untuk meningkatkan akurasi model.

## C. Data Understanding
Informasi data yang di gunakan sebagai berikut : 
1. Jumlah Data : 1430 Baris dan 7 Kolom 
2. Kondisi Data : 
                    - Missing Value : Tidak terdapat missing value pada kolom apapun. 
                    - Duplikat : Tidak terdapat data duplikat.
                    - Outlier : 
                                - Kolom Open, High, low, Close, dan Adj Close : Tidak ada outlier. 
                                - Kolom Volume : Terdapat 66 outlier.
3. Tautan Sumber Data : https://www.kaggle.com/datasets/caesarmario/samsung-electronics-stock-historical-price.
4. Uraian Fitur :
Dataset berisi informasi historis harga saham Samsung Electronics dari 2019-01-02 hingga 2024-10-25. Dataset terdiri dari kolom berikut: 
| Nama Tabel | Info Table                                     |
|------------|------------------------------------------------|
| Date       | Tanggal pencatatan saham.                      |
| Open       | Harga pembukaan saham dari hari tersebut.      |
| High       | Harga tertinggi saham pada hari tersebut.      |
| Low        | Harga terendah saham pada hari tersebut.       |
| Close      | Harga penutupan saham pada hari tersebut.      |
| Adj Close  | Harga penutupan saham yang telah di sesuaikan. |
| Volume     | Jumlah transaksi saham yang terjadi.           |
    
## D. Data Preparation
1. Membagi Data Menjadi Data Latih dan Data Uji: Memisahkan dataset menjadi dua bagian: ```train_data``` dan ```test_data```.
    - ```train_data``` berisi semua baris dari dataset asli (```data```) yang memiliki tanggal sebelum 25 September 2024. Data ini akan digunakan untuk melatih model.
    - ```test_data berisi``` semua baris dengan tanggal 25 September 2024 dan setelahnya. Data ini akan digunakan untuk menguji kinerja model setelah dilatih.

2. Mengubah nama kolom di ```train_data``` dan ```test_data``` agar sesuai dengan format yang dibutuhkan oleh model Facebook Prophet:
    - Kolom ```Date``` diubah menjadi ```ds```, yang merupakan singkatan dari "date stamp". Kolom ini berisi tanggal. 
    - Kolom ```Adj Close``` diubah menjadi ```y```, yang berisi nilai target yang ingin diprediksi (dalam hal ini, harga penutupan saham yang disesuaikan)

3. Feature Engineering: Menambahkan regresor tambahan berdasarkan kolom ```Open```, ```High```, ```Low```, ```Close```, dan ```Adj Close``` yang berkorelasi tinggi dengan target (```Adj Close```).

## E. modeling
Model yang digunakan adalah Facebook Prophet. Model ini dipilih karena kemampuannya untuk menangani tren dan pola musiman dalam data deret waktu. 
1. Parameter Tuning: 
    - Changepoint Prior Scale: 0.5 – Meningkatkan fleksibilitas model untuk menangkap perubahan tren.
    - Seasonality Prior Scale: 0.01 – Mengurangi sensitivitas terhadap pola musiman untuk menghindari overfitting.
    - n_changepoints=30 - Model menjadi fleksibel dalam mengikuti perubahan tren. 

2. Proses modeling 
    - Model dilatih dengan data train dan diujikan pada data test untuk melihat performa prediksi pada periode 2024-09-21 hingga 2024-10-21.
    - Regresi tambahan pada kolom ```Open```, ```High```, ```Low```, ```Close```, dan ```Adj Close``` digunakan untuk memperbaiki akurasi prediksi.

3. Cara kerja model
    - **Memecah Data Deret waktu**, Model memecah data deret waktu menjadi beberapa komponen: Tren, Musiman, dan efek tambahan dari regresor. 
        - ```yearly_seasonality=True``` : Mengaktifkan komponen musiman tahunan, sehingga model dapat menangkap pola yang terjadi setiap tahun.
        - ```weekly_seasonality=True``` : Mengaktifkan komponen musiman mingguan, untuk menangkap pola yang terjadi setiap minggu.
        - ```daily_seasonality=True``` : Mengaktifkan komponen musiman harian, untuk menangkap pola yang terjadi setiap hari.

    - **Mengidentifikasi Tren**, Model mendeteksi titik perubahan untuk mengidentifikasi ketika tren berubah, berdasarkan jumlah titik perubahan yang ditentukan. 
        - ```changepoint_prior_scale=0.5``` : Menentukan seberapa sensitif model terhadap berubahan tren. Nilai yang lebih tinggi membuat model lebih responsif terhadap perubahan tren, tetapi berisiko overfitting jika terlalu sensitif.
        - ```n_changepoints=30``` : Menentukan jumlah maksumum titik perubahan yang dapat dideteksi dalam data. Model akan mencari hingga 30 titik perubahan untuk menyesuaikan tren yang mungkin berubah seiring waktu.
     
    - **Menangkap Pola Musiman**, Model dapat menangkap pola musiman yang berulang dalam data berdasarkan mengaturan musiman yang diaktifkan.
        - **Musiman diaktifkan (yearly, weekly, daily)** : Dengan ketika parameter musiman diaktifkan, model dapat menangkap berbagai pola yang terjadi dalam periode waktu yang berbeda, membantu menignkatkan akurasi prediksi.
        - ```seasonality_prior_scale=0.01``` : Mengontrol seberapa besar pengaruh pola musiman dalam model. Nilai rendah membuat model lebih konservatif terhadap musiman, membantu mencegah overfitting pada noise musiman yang mungkin tidak konsisten.

    - **Menggunakan Regressor**, Efek dari regresor (Open, High, Low, Close) ditambahkan untuk memperbaiki prediksi berdasarkan informasi yang relevan.
        - ```model.add_regressor('Open')```, ```model.add_regressor('High')```, ```model.add_regressor('Low')```, ```model.add_regressor('Close')``` : Menambahkan variabel regresor yang dianggap memiliki hubungan signifikan dengan harga penutupan (```y```). Ini memungkikan model untuk menggunakan informasi tambahan dlam meningkatkan prediksi harga.

    - **Pelatihan Model**, Model dilatih menggunakan data historis, yang memungkikan ia belajar dari pola dan hubungan dalam data untuk membuat prediksi di masa depan. 
        - ```model.fit(train_data[['ds', 'y', 'Open', 'High', 'Low', 'Close']])``` : Melatih model menggunakan dataset yang terdiri dari tanggal (```ds```), harga penutupan yang ingin diprediksi (```y```), dan regresor yang telah di tambahkan. Proses pelatihan ini memungkikan model untuk memahami pola dalam data historis dan mengembangkan kemampuan prediktif.

## F. Evaluasi 
**Hasil Metrik Evaluasi**

1. Mean Absolute Error (MAE): Menyatakan rata-rata selisih absolut antara harga prediksi dan harga sebenarnya, yang pada proyek ini mencapai 141.1164.

2. Root Mean Squared Error (RMSE): Mengukur selisih kuadrat rata-rata, yang menunjukkan performa prediksi secara keseluruhan, dengan nilai 157.8733 .


**Kesimpulan** 

Secara keseluruhan, model yang dievaluasi telah menjawab problem Statement, mencapai goals yang diharapkan, dan memberikan dampak positif sesuai dengan solusi yang direncanakan. hal ini menunjukkan bahwa pendekatan yang digunakan dalam proyek ini efektif dan dapat diandalan untuk prediksi harga saham di masa mendatang.





