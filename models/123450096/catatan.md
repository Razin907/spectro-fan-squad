# Catatan Model: Transformer Encoder, jalur `trf`

Penulis: Razin Hafid Hamdi (123450096)
Status: protokol data simulasi selesai dan `run.py` sudah melewati `data.muat()` (menunggu rekaman meja nyata.)

## Peran dan arsitektur

Model menerima log-mel 64 pita sepanjang 1 detik. Sumbu waktu dipotong menjadi patch
(10 bingkai per patch), tiap patch diproyeksikan dengan Conv1d ke dimensi 64, ditambah
positional encoding, lalu diproses dua lapis Transformer encoder (4 head, feed-forward 128).
Representasi dirata-ratakan dan diteruskan ke kepala keluaran: jumlah logit = jumlah kelas
untuk klasifikasi, atau satu nilai untuk regresi. Bentuk masukan yang diterima `[N, waktu,
pita]`, `[N, pita, waktu]`, dan `[N, 1, pita, waktu]`; bentuk keluaran mengikuti
`src.protokol.data.TUGAS`.

Protokol tim memisahkan data per sesi dan hanya memasukkan bunyi normal ke data latih.
Transformer tetap memakai target biner 0/1, sehingga skor AUROC saat ini adalah uji jalur,
bukan performa final. Jalur AE tetap acuan utama untuk deteksi anomali; hasil Transformer
dibandingkan memakai split dan metrik yang sama.

## Alasan optimizer

AdamW dipilih karena attention Transformer peka terhadap skala gradien per parameter;
AdamW memberi langkah adaptif tanpa mencampur weight decay ke dalam gradien. Weight decay
dipisahkan (decoupled) sehingga regularisasi tetap terkontrol. Learning rate awal 1e-3
dengan cosine annealing, gradient clipping 1.0, dan early stopping pada macro-F1 validasi
(regresi: val loss). Alasan ini akan diuji ulang lewat pencarian hyperparameter setelah
data asli tersedia.

## Regularisasi

- Dropout 0.1 pada encoder dan sebelum kepala keluaran.
- Weight decay 1e-2 pada AdamW.
- Label smoothing 0.05 untuk klasifikasi.
- Class weighting otomatis untuk kelas tidak seimbang (bisa dimatikan untuk ablation).
- Early stopping dengan patience 8 epoch dan pemulihan bobot validasi terbaik.

## Pencarian hyperparameter (rencana, diisi setelah data asli ada)

| Percobaan | Patch | d_model | nlayer | Dropout | LR | val_macro_f1 (3 seed) |
| --- | --- | --- | --- | --- | --- | --- |
| dasar | 10 | 64 | 2 | 0.1 | 1e-3 | — |
| patch lebih panjang | 20 | 64 | 2 | 0.1 | 1e-3 | — |
| model lebih lebar | 10 | 128 | 2 | 0.1 | 1e-3 | — |
| lebih dangkal | 10 | 64 | 1 | 0.1 | 1e-3 | — |

Semua percobaan memakai split per sesi protokol dan diulang 3 seed (1, 2, 3) pada
konfigurasi akhir.

## Ablation (>= 2)

1. `--tanpa-posisi`: menghapus positional encoding untuk menguji apakah urutan patch
   benar-benar dipakai model.
2. `--tanpa-bobot-kelas`: membandingkan class weighting dengan cross-entropy biasa.
3. `--patch 20`: memperbesar patch untuk melihat trade-off jumlah token dan detail waktu.
4. `--dropout 0.0`: menguji apakah regularisasi dropout membantu pada data tim.

## Hasil 3 seed konfigurasi akhir

Belum ada; data asli belum tersedia. Angka akan ditulis dari `results/123450096/metrics.json`
setelah pelatihan pada data tim.

## Uji jalur pada data simulasi

Perintah `python models/123450096/run.py --seed 1` memuat data langsung melalui
`src.protokol.data.muat()` pada folder sesi simulasi (`sesi/t14-anomali-suara_simulasi/`).
Data yang dipakai: 162 jendela latih (hanya normal), 252 jendela validasi (144 normal /
108 anomali), dan 198 jendela uji (90 normal / 108 anomali). Bentuk masukan `[N, 100, 64]`.
`metrics.json` ditulis lewat `src.common.evaluate`; AUROC validasi dihitung terpisah oleh
`tools/nilai_auroc.py` dan ditulis ke `results/123450096/auroc.json` (0,529 pada uji jalur
ini). Angka tersebut belum hasil ilmiah karena data masih berasal dari sesi simulasi dan
model masih dilatih sebagai klasifikasi normal.
