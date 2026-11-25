# Catatan model 123450112 — Transfer Learning ResNet18 + One-Class SVM

Penulis: 123450112

Status: Implementasi Transfer Learning menggunakan ResNet18 sebagai feature extractor dan One-Class SVM sebagai detektor anomali. Implementasi sebelumnya telah diuji pada data simulasi. Penyesuaian penggunaan data primer masih perlu dilakukan dan diverifikasi.

## Peran dan arsitektur

Model menggunakan pendekatan Transfer Learning dengan ResNet18 pretrained pada ImageNet sebagai ekstraktor fitur untuk mendeteksi anomali suara mesin. Data audio direpresentasikan dalam bentuk spektrogram log-mel sebelum diproses menggunakan jaringan ResNet18.

Arsitektur ResNet18 digunakan tanpa lapisan klasifikasi terakhir (`fc`), yang diganti dengan `Identity` sehingga jaringan menghasilkan representasi fitur dari masukan spektrogram. Bobot pretrained ResNet18 dibekukan (*frozen*) selama proses ekstraksi fitur sehingga tidak dilakukan pelatihan ulang terhadap parameter jaringan.

Fitur hasil ekstraksi ResNet18 kemudian distandardisasi menggunakan `StandardScaler` sebelum digunakan sebagai masukan algoritma One-Class SVM. One-Class SVM mempelajari karakteristik data normal dari subset latih untuk membedakan sampel normal dan anomali.

Skor anomali kontinu diperoleh dari fungsi keputusan One-Class SVM dengan arah skor disesuaikan sehingga nilai yang lebih tinggi menunjukkan tingkat anomali yang lebih besar. Skor tersebut digunakan untuk menghitung AUROC.

## Strategi pelatihan

Model tidak menggunakan optimizer berbasis gradien seperti Adam atau SGD karena bobot ResNet18 dibekukan dan hanya digunakan untuk ekstraksi fitur. Proses deteksi anomali dilakukan menggunakan One-Class SVM yang mempelajari distribusi fitur dari data latih normal.

Pendekatan ini dipilih agar representasi fitur dari model pretrained dapat dimanfaatkan tanpa melakukan pelatihan ulang seluruh jaringan. Selain itu, penggunaan One-Class SVM sesuai dengan karakteristik protokol tim yang menyediakan data latih hanya dari kelas normal.

## Regularisasi

- **Pembekuan bobot ResNet18 (*frozen feature extractor*)** untuk mempertahankan representasi fitur hasil pretraining dan menghindari perubahan bobot jaringan selama eksperimen.
- **StandardScaler** untuk menyamakan skala fitur hasil ekstraksi sebelum digunakan oleh One-Class SVM.
- **One-Class SVM** untuk membentuk batas keputusan berdasarkan karakteristik data normal sehingga sampel yang menyimpang dapat diidentifikasi sebagai anomali.
- **Kernel RBF** untuk memungkinkan pemodelan batas keputusan nonlinier pada ruang fitur.

## Pencarian hyperparameter (tabel)

| Percobaan | Feature Extractor | Detektor | Konfigurasi | val_macro_f1 |
| --- | --- | --- | --- | --- |
| baseline (simulasi) | ResNet18 pretrained | One-Class SVM | Kernel RBF | 0.646775 |
| perubahan parameter nu | ResNet18 pretrained | One-Class SVM | — | — |
| perubahan parameter gamma | ResNet18 pretrained | One-Class SVM | — | — |
| perubahan konfigurasi kernel | ResNet18 pretrained | One-Class SVM | — | — |

## Ablation (>= 2)

1. **Dengan vs tanpa StandardScaler**: menguji pengaruh standardisasi fitur terhadap performa deteksi anomali menggunakan One-Class SVM.
2. **ResNet18 pretrained vs ekstraksi fitur alternatif**: menguji kontribusi penggunaan representasi fitur hasil pretraining ResNet18 terhadap kemampuan membedakan suara normal dan anomali.
3. **Perbandingan konfigurasi One-Class SVM**: menguji pengaruh perubahan parameter `nu`, `gamma`, dan kernel terhadap hasil deteksi anomali.

## Hasil konfigurasi model

Hasil pengujian implementasi Transfer Learning sebelumnya:

- **val_acc**: `0.646825`
- **val_macro_f1**: `0.646775`
- **val_auroc**: `0.767425`

Hasil tersebut merupakan hasil implementasi sebelumnya dan belum boleh dianggap sebagai hasil final pada data primer. Evaluasi perlu dijalankan kembali setelah sumber data pada `run.py` disesuaikan dengan data primer Tim 14.

## Penyesuaian pada Data Primer

Data primer yang digunakan berasal dari sesi `sesi-20261117-181631` dan ditempatkan pada folder `data/sesi_primer/`.

Pembagian data mengikuti protokol resmi Tim 14 melalui `src/protokol/data.py`:

- **Latih**: 90 segmen audio, seluruhnya kelas normal (`0`).
- **Validasi**: 144 segmen audio, terdiri atas 36 normal dan 108 anomali.
- **Uji**: 135 segmen audio, terdiri atas 27 normal dan 108 anomali.

Penyesuaian implementasi dilakukan dengan menggunakan `data.muat_segmen()` sebagai sumber data primer, kemudian menjalankan tahapan ekstraksi spektrogram, ekstraksi fitur ResNet18, standardisasi fitur, dan deteksi anomali menggunakan One-Class SVM.

Data latih hanya digunakan untuk mempelajari karakteristik normal. Data validasi digunakan untuk evaluasi selama pengembangan model, sedangkan data uji disimpan untuk evaluasi akhir.

Data hash resmi Tim 14:

`c4949d6dc75b47c6c569dcdfc5812f3f`

*Catatan: Hasil evaluasi data primer akan dicatat setelah implementasi Transfer Learning selesai disesuaikan dan dijalankan menggunakan pembagian data resmi. Hasil pengujian data simulasi tidak dicampurkan dengan hasil data primer.*