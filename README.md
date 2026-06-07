# Smart-Grouping System 

# Deskripsi
_Smart-Grouping System_ merupakan proyek data mining yang bertujuan untuk mengoptimalkan pembentukan kelompok kolaboratif murid menggunakan algoritme K-Means Clustering. Sistem ini memanfaatkan data _Cognitive Load_ dan Resiliensi Belajar untuk mengelompokkan murid berdasarkan kemiripan karakteristik, kemudian menggunakan hasil clustering tersebut sebagai dasar pembentukan kelompok yang heterogen dan seimbang.

# Dataset
Dataset yang digunakan dalam penelitian ini merupakan dataset murid yang terdiri dari 34 data murid dengan dua variabel yaitu _Cognitive Load_ dan Resiliensi Belajar. Data _cognitive load_ diperoleh melalui tes tertulis yang diberikan secara langsung kepada murid untuk mengukur kemampuan kognitif, sedangkan data resiliensi belajar diperoleh melalui angket dengan skala likert yang digunakan untuk mengukur aspek afektif murid.

|Variabel|Deskripsi|
|---------|---------|
|No|Nomor urut data murid|
|Nama Siswa|Identitas murid yang menjadi objek penelitian|
|Cognitive Load|Skor kemampuan kognitif murid|
|Resiliensi Belajar| Skor reiliensi belajar|

# 1. Persiapan Data
# Import Library
Tahap pertama dalam proyek ini adalah mengimport library Python yang diperlukan untuk memproses pengolahan data, clustering, evaluasi model, dan visualisasi model. Library yang digunakan meliputi **Pandas** untuk membaca dan mengelola dataset, **Matplotlib** untuk menampilkan visualisasi data, **StandardScaler** dari Scikit-Learn untuk normalisasi data, **KMeans** untuk proses clustering, dan **Sillhouetter Score** untuk mengevaluasi kualitas cluster yang terbentuk.
```Python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

# Membaca Dataset
Pada tahap ini dilakukan proses membaca dataset menggunakan library **Pandas**. `pd.read_excel()` digunakan untuk membaca file dataset berformat Excel dan mengubahnya ke dalam bentuk **Data Frame** sehingga data dapat diolah dan dianalisis lebih lanjut. Dataset kemudian disimpan ke dalam variabel `df` yang memuat seluruh data murid beserta data yang diperlukan dalam penelitian. Selanjutnya `df` dipanggil untuk menampilkan isi dataset sehingga dapat diketahui struktur data, jumlah kolom, tipe data, serta memastikan bahwa dataset telah berhasil dibaca dan siap digunakan dalam tahap _preprocessing_ dan analisis selanjutnya.

# 2. Pre-Processing Data
# Informasi Dataset
Pada tahap ini digunakan `info()` untuk menampilkan informasi umum mengenai dataset. Informasi yang ditampilkan meliputi jumlah baris dan kolom, nama variabel, tipe data setiap variabel, dan jumlah data yang tidak kosong. Melalui informasi ini, dapat diketahui apakah terdapat _missing values_ pada dataset serta memastikan bahwa tipe data pada setiap variabel sudah sesuai untuk proses analisis dan pembuatan model. 
# Identifikasi Missing Value
Pada tahap ini dilakukan pemeriksaan _missing values_ menggunakan `print(df.isnull().sum())`. Fungsi `isnull()` digunakan untuk mendeteksi nilai yang kosong pada setiap variabel, sedangan  `sum()` digunakan untuk menghitung jumlah nilai nilai kosong. Tujuan dari tahap ini adalah untuk memastikan bahwa dataset tidak memiliki data yang hilang sehingga proses analisis dan _clustering_ dapat dilakukan dengan lebih akurat. Jika terdapat _missing value_ maka diperlukan penangan, seperti menghapus data atau melakukan imputasi data. Berdasarkan hasil pemeriksaan, seluruh variabel memiliki nilai _missing value_ sebesar 0 yang menunjukkan bahwa dataset lengkap dan siap digunakan.
# Seleksi Variabel
Pada tahap ini dilakukan seleksi variabel dengan memilih variabel yang akan digunakan dalam proses _clustering_. Variabel yang dipilih adalah variabel **Cognitive Load** dan **Resiliensi** karena kedua variabel tersebut menjadi dasar dalam pengelompokkan murid. Proses seleksi variabel ini dilakukan dengan mengambil kedua kolom tersebut dari dataset dan menyimpannya ke dalam variabel `x`. Sedangkan variabel **No** dan **Nama Siswa** tidak digunakan dalam proses clustering karena hanya berfungsi sebagai identitas data dan tidak memengaruhi hasil pengelompokkan. Data hasil seleksi variabel ini, selanjutnya digunakan sebagai input pada tahap normalisasi dan pemodelan menggunakan algortma K-Means Clustering.
```python
x = df[['Cognitive Load','Resiliensi']]
```
# Normalisasi Data
Pada tahap ini dilakukan normalisasi data menggunakan **StandardScaler** untuk menyamakan skala nilai pada setiap variabel yang digunakan dalam proses clustering. Normalisasi ini diperlukan karena algoritma K-Means mengelempokkan data berdasarkan perhitungan jarak, sehingga perbedaan rentang nilai antar variabel dapat mempengaruhi hasil pengelompokan. Fungsi `StandardScaler()` untuk membentuk model normalisasi, sedangkan `scaler.fit_transform(x)` digunakan untuk menghitung dan mengubah data ke dalam skala standar dengan rata-rata mendekati 0 dan standar deviasi mendekati 1. Hasil normalisasi ini kemudian disimpan dalam variabel `x_scaled` dan digunakan sebagai input pada proses clustering.
```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```
# Hasil Normalisasi Data
Pada tahap ini ditampilkan hasil normalisasi data yang telah dilakukan menggunakan metode **StandardScaler**. Nilai pada `x_scaled` ini tidak lagi berupa nilai asli, melainkan nilai hasil transformasi yang merepresentasikan posisi setiap data terhadap rata-rata keseluruhan data.
```python
x_scaled
```

# 3. Elbow Method
# Jumlah Cluster Optimal
Pada tahap ini dilakukan penentuan jumlah cluster yang optimal menggunakan Elbow Method. Metode ini dilakukan dengan menjalankan algoritma K-Means pada berbagai jumlah cluster yaitu dari 1 hingga 10 cluster. Untuk setiap jumlah cluster, dihitung nilai **Within Cluster Sum of Squares (WCSS)** atau jumlah kuadrat jarak data terhadap centroid pada setiap cluster yang sama. Nilai WCSS kemudian disimpan pada variabel `wcs`. Semakin kecil nilai   wcss   menunjukkan bahwa data dalam cluster homogen, namun penambahan jumlah cluster yang terlalu banyak juga kurang efisien. Oleh karena itu, nilai WCSS selanjutnya divisualisasikan dalam bentuk grafik untuk menemukan titik siku (elbow point) yang menunjukkan jumlah cluster paling optimal untuk digunakan pada proses clustering.
```python
wcs = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(x_scaled)
  wcs.append(kmeans.inertia_)
```
# Visualisasi Elbow Method
Pada tahap ini dilakukan visualisasi hasil perhitungan WCSS menggunakan grafik Elbow Method. Grafik ini dibuat dengan menampilkan hubungan antara jumlah cluster (Number pf Clusters) pada sumbu x dan nilai WCSS pada sumbu y. Visualisasi ini bertujuan untuk membantu menentukan jumlah cluster yang optimal dengan mengidentifikasi titik siku pada grafik. Titik tersebut menunjukkan kondisi ketika penambahan jumlah cluster tidak lagi memberikan penurunan nilai WCSS yang signifikan. Jumlah cluster pada titik siku ini yang dipilih sebagai jumlah cluster optimal untuk digunakan pada proses K-Means Clustering.
```python
plt.figure(figsize=(10,5))
plt.plot(range(1,11), wcs)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
```
<img width="841" height="470" alt="image" src="https://github.com/user-attachments/assets/41d315b4-d98a-49a0-8c8f-ae6c43bdc374" />

# 4. Pemodelan K-Means Clustering
# Inisialisasi Model K-Means
Pada tahap ini dilakukan pembentukan model K-Means Clustering dengan menggunakan3 cluster sebagai jumlah cluster yang telah ditentukan berdasarkan hasil Elbow Method. Parameter `n_clusters=3` digunakan untuk menentukan jumlah cluster yang akan dibentuk, sedangkan `init='k-means++'` digunakan untuk memilih centroid awal secara lebih optimal sehingga dapat meningkatkan kualitas hasil clustering dan mengurangi kemungkinan terjebak pada solusi lokal. Parameter `random_state=42` digunakan untuk memastikan hasil clustering yang diperoleh tetap konsisten setiap kali program dijalankan. Model K-Means yang telah dibentuk selanjutnya digunakan untuk mengelompokkan murid berdasarkan kemiripan nilai Cognitive Load dan Resiliensi Belajar.
```python
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=42
)
```
# Proses Clustering
Pada tahap ini dilakukan proses clustering menggunakan model K-Means yang telah dibentuk sebelumnya. `fit_predict()` digunakan untuk menjalankan proses pelatihan model _(fitting)_ sekaligus menentukan cluster untuk setiap data murid berdasarkan nilai Cognitive Load dan Resiliensi yang telah dinormalisasi. Hasil pengelompokkan tersebut, disimpan dalam variabel `cluster` yang berisi label cluster untuk masing-masing murid. Label cluster ini menunjukkan kelompok setiap murid berdasarkan tingkat kemiripan karakteristik yang dimiliki dan selanjutnya digunakan untuk analisis hasil clustering serta pembentuk kelompok kolaboratif.
```python
cluster = kmeans.fit_predict(x_scaled)
```
# Menambahkan Hasil Cluster ke Dataset
Pada tahap ini, hasil clustering yang tersimpan dalam variabel `cluster` ditambahkan ke dalam dataset sebagai kolom baru bernama **Cluster**. Setiap nilai pada kolom tersebut menunjukkan label cluster yang dimiliki oleh masing-masing murid berdasarkan hasil pengelompokan menggunakan algoritma K-Means. Penambahan kolom cluster bertujuan untuk memudahkan proses analisis dan identifikasi anggota pada setiap cluster, sehingga hasil pengelompokan dapat digunakan sebagai dasar dalam pembentukan kelompok kolaboratif yang heterogen dan seimbang.
```python
df['Cluster'] = cluster
```

# 5. Evaluasi 
Pada tahap ini dilakukan evaluasi terhadap hasil clustering menggunakan **Silhouette Score**. Metode ini digunakan untuk mengukur kualitas cluster yang terbentuk berdasarkan tingkat kemiripan data dalam satu cluster dan perbedaan data antar cluster. Fungsi `silhouette_score()` menghitung nilai silhouette dengan menggunakan data yang telah dinormalisasi (`x_scaled`) dan label hasil clustering (`cluster`). Nilai Silhouette Score berada pada rentang -1 hingga 1, di mana nilai yang semakin mendekati 1 menunjukkan bahwa data dalam satu cluster memiliki kemiripan yang tinggi dan terpisah dengan baik dari cluster lainnya. Hasil evaluasi ini digunakan untuk menilai apakah proses clustering yang dilakukan telah menghasilkan pengelompokan yang baik dan representatif.
```python
silhouette = silhouette_score(x_scaled, cluster)
```
# Hasil Evaluasi Silhouette Score
Berdasarkan hasil perhitungan menggunakan metode **Silhouette Score**, diperoleh nilai sebesar **0,5587**. Nilai tersebut menunjukkan bahwa hasil clustering yang terbentuk memiliki kualitas yang cukup baik, karena berada di atas 0,5. Hal ini mengindikasikan bahwa siswa dalam cluster yang sama memiliki tingkat kemiripan karakteristik yang relatif tinggi, sementara perbedaan antar cluster juga cukup jelas. Dengan demikian, pengelompokan murid berdasarkan nilai Cognitive Load dan Resiliensi Belajar menggunakan algoritma K-Means dapat dikatakan berhasil menghasilkan cluster yang cukup representatif untuk digunakan dalam pembentukan kelompok kolaboratif yang heterogen dan seimbang.
```python
silhouette
```
# Hasil Clustering
Pada tahap ini ditampilkan hasil pengelompokan siswa menggunakan algoritma **K-Means Clustering**. Hasil clustering ditambahkan ke dalam dataset pada kolom **Cluster**, yang menunjukkan kelompok tempat setiap siswa berada berdasarkan kemiripan nilai **Cognitive Load** dan **Resiliensi Belajar**. Berdasarkan hasil pengelompokan, terbentuk **3 cluster**, yaitu Cluster 0, Cluster 1, dan Cluster 2. Setiap cluster merepresentasikan kelompok siswa dengan karakteristik yang relatif serupa. Hasil clustering ini selanjutnya digunakan sebagai dasar dalam pembentukan kelompok kolaboratif yang heterogen dan seimbang sehingga setiap kelompok dapat terdiri atas siswa dengan profil belajar yang beragam.
```python
print("Hasil Clustering")
print(df)
```

# Hasil Output

| No | Nama Siswa                       | Cognitive Load | Resiliensi | Cluster |
| -- | -------------------------------- | -------------- | ---------- | ------- |
| 1  | Andaru Azkha Aranatama           | 82             | 45         | 1       |
| 2  | Anggita Septi Tree Rahayu        | 76             | 50         | 1       |
| 3  | Aria Huda Galuh                  | 35             | 88         | 0       |
| 4  | Arieffatul Umami                 | 40             | 85         | 0       |
| 5  | Arkana Ezar Altaffiras Isnanto   | 60             | 65         | 2       |
| 6  | Askana Adzra Farrastra           | 58             | 70         | 2       |
| 7  | Calista Puteri Sabina            | 90             | 35         | 1       |
| 8  | Chesa Kenzie Yusuf Asadel        | 88             | 38         | 1       |
| 9  | Cindy Aulia                      | 45             | 82         | 0       |
| 10 | Icha Aulia Rahma                 | 55             | 68         | 2       |
| 11 | Ika Fatmawati                    | 30             | 92         | 0       |
| 12 | Isna Afa Nada Delia              | 72             | 55         | 1       |
| 13 | Jagat Pramudhita M               | 48             | 80         | 0       |
| 14 | Kania Amelia Artanti             | 62             | 63         | 2       |
| 15 | Miftahul Jannah                  | 85             | 42         | 1       |
| 16 | Muhammad Azwangga Reyhan Fahreza | 38             | 89         | 0       |
| 17 | Muhammad Prabu Penggalih         | 52             | 72         | 2       |
| 18 | Muhammad Rafi Maulana            | 78             | 48         | 1       |
| 19 | Muhammad Rafka Alfarezi          | 42             | 84         | 0       |
| 20 | Naufal Adil Fadhila              | 65             | 60         | 2       |
| 21 | Naufal Alif Wijaya               | 33             | 90         | 0       |
| 22 | Naufal Haidar Ahmad              | 80             | 46         | 1       |
| 23 | Naufal Satria Nur Rasyid         | 50             | 75         | 2       |
| 24 | Nayla Fauzia Azzahra Nugraha     | 47             | 78         | 0       |
| 25 | Prince Gabriel Leonard Al Jabbar | 74             | 53         | 1       |
| 26 | Raditiya Praba Perwira           | 36             | 90         | 0       |
| 27 | Radittya Putra Rifa'i            | 68             | 58         | 2       |
| 28 | Sakhi Almer Faidh                | 92             | 30         | 1       |
| 29 | Salim Khoirul Lisan              | 41             | 86         | 0       |
| 30 | Salsabila Syfa Ramadhani         | 57             | 69         | 2       |
| 31 | Sayyid Hasan Al Anshory          | 34             | 93         | 0       |
| 32 | Sayyidina Zafar Alhamzah         | 79             | 44         | 1       |
| 33 | Shofiya Nadaa Khoirunnisa        | 53             | 74         | 2       |
| 34 | Tania Uzlah Wahyu Bachtiar       | 59             | 87         | 0       |

# Visualisasi Hasil Clustering
Pada tahap ini dilakukan visualisasi hasil clustering menggunakan **scatter plot** untuk melihat persebaran siswa berdasarkan nilai Cognitive Load dan Resiliensi Belajar. Sumbu horizontal (`x`) menunjukkan nilai Cognitive Load, sedangkan sumbu vertikal (`y`) menunjukkan nilai Resiliensi. Setiap titik pada grafik merepresentasikan satu siswa, dan warna yang berbeda menunjukkan cluster tempat siswa tersebut berada. Visualisasi ini membantu dalam mengidentifikasi pola pengelompokan data serta melihat tingkat pemisahan antar cluster yang terbentuk. Dengan adanya scatter plot, hasil clustering dapat diinterpretasikan secara lebih mudah sehingga karakteristik masing-masing cluster dapat diamati dengan lebih jelas.
```python
plt.figure(figsize=(10,5))
plt.scatter(
    df['Cognitive Load'],
    df['Resiliensi'],
    c=df['Cluster'],
    cmap='rainbow'
)
plt.xlabel('Cognitive Load')
plt.ylabel('Resiliensi')
plt.title('Hasil Clustering')
plt.show()
```
<img width="841" height="470" alt="image" src="https://github.com/user-attachments/assets/993167bd-17a5-41c1-8507-a65de4dc3e30" />


# 6. Pembagian Anggota Setiap Cluster
Pada tahap ini ditampilkan daftar anggota pada setiap cluster yang terbentuk dari proses K-Means Clustering. Pengelompokan dilakukan berdasarkan kemiripan nilai Cognitive Load dan Resiliensi Belajar, sehingga siswa dalam cluster yang sama memiliki karakteristik yang relatif serupa. Berdasarkan hasil clustering, terbentuk 3 cluster, yaitu Cluster 0 yang beranggotakan 13 siswa, Cluster 1 yang beranggotakan 11 siswa, dan Cluster 2 yang beranggotakan 10 siswa. Informasi anggota setiap cluster digunakan sebagai dasar untuk memahami karakteristik masing-masing kelompok serta menjadi acuan dalam pembentukan kelompok kolaboratif yang heterogen dan seimbang pada tahap selanjutnya.
```python
print("Anggota Setiap Cluster")

for i in sorted(df['Cluster'].unique()):
  print(f"Cluster {i}")
  anggota = df[df['Cluster'] == i]['Nama Siswa']
  for nama in anggota:
    print("-",nama)
```
# Hasil 

#### Cluster 0

| No | Nama Siswa |
|----|------------|
| 1 | Aria Huda Galuh |
| 2 | Arieffatul Umami |
| 3 | Cindy Aulia |
| 4 | Ika Fatmawati |
| 5 | Jagat Pramudhita M |
| 6 | Muhammad Azwangga Reyhan Fahreza |
| 7 | Muhammad Rafka Alfarezi |
| 8 | Naufal Alif Wijaya |
| 9 | Nayla Fauzia Azzahra Nugraha |
| 10 | Raditiya Praba Perwira |
| 11 | Salim Khoirul Lisan |
| 12 | Sayyid Hasan Al Anshory |
| 13 | Tania Uzlah Wahyu Bachtiar |

#### Cluster 1

| No | Nama Siswa |
|----|------------|
| 1 | Andaru Azkha Aranatama |
| 2 | Anggita Septi Tree Rahayu |
| 3 | Calista Puteri Sabina |
| 4 | Chesa Kenzie Yusuf Asadel |
| 5 | Isna Afa Nada Delia |
| 6 | Miftahul Jannah |
| 7 | Muhammad Rafi Maulana |
| 8 | Naufal Haidar Ahmad |
| 9 | Prince Gabriel Leonard Al Jabbar |
| 10 | Sakhi Almer Faidh |
| 11 | Sayyidina Zafar Alhamzah |

#### Cluster 2

| No | Nama Siswa |
|----|------------|
| 1 | Arkana Ezar Altaffiras Isnanto |
| 2 | Askana Adzra Farrastra |
| 3 | Icha Aulia Rahma |
| 4 | Kania Amelia Artanti |
| 5 | Muhammad Prabu Penggalih |
| 6 | Naufal Adil Fadhila |
| 7 | Naufal Satria Nur Rasyid |
| 8 | Radittya Putra Rifa'i |
| 9 | Salsabila Syfa Ramadhani |
| 10 | Shofiya Nadaa Khoirunnisa |
