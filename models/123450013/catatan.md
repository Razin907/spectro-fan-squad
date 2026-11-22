# Catatan model 123450013 — Autoencoder (`ae`)

Penulis: Azzahra Putri Kamilah (123450013)
Status: Protokol data simulasi selesai dan `run.py` terintegrasi dengan `src.protokol.data.muat()` (menunggu rekaman meja nyata).

## Peran dan arsitektur

Mendeteksi anomali suara kipas dengan Autoencoder (AE) yang dilatih hanya pada data normal (`y == 0`). Masukan berupa spektrogram log-mel 64 pita sepanjang 1 detik (`[N, 100, 64]`) dari `src.protokol.data.muat()` yang diratakan menjadi vektor 6.400 fitur. Arsitektur baseline adalah MLP simetris (`6400 -> 256 -> 16 -> 256 -> 6400`, aktivasi ReLU, bottleneck 16). Skor anomali adalah reconstruction error (MSE per jendela). Ambang keputusan biner diambil dari persentil ke-95 skor error data latih normal, bukan dari data uji. AUROC dihitung dari skor kontinu (`pred_val_skor.npy`) sehingga tidak membutuhkan ambang.

## Alasan optimizer

Menggunakan **Adam** (`lr=1e-3`) karena rekonstruksi spektrogram log-mel yang diratakan memiliki skala gradien yang bervariasi antar pita frekuensi. Adam memberi laju belajar adaptif per parameter sehingga MSE pada data normal konvergen lebih cepat dan stabil dalam 30 epoch dibandingkan SGD biasa.

## Regularisasi

- **Bottleneck berdimensi rendah (16)**: memaksa jaringan mempelajari representasi padat dari bunyi normal saja, sehingga pola anomali sulit direkonstruksi.
- **Pelatihan khusus kelas normal**: pola anomali tidak masuk ke ruang laten.
- **Rencana pada data nyata**: menguji `weight_decay = 1e-4` dan `Dropout(0.1)` agar AE tidak terlalu ekspresif (anomali ikut terekonstruksi baik).

## Pencarian hyperparameter (tabel)

| Percobaan | Arsitektur | Bottleneck | LR | Weight Decay | val_macro_f1 (3 seed) | val_auroc |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (simulasi) | MLP (`256-16-256`) | 16 | 1e-3 | 0 | 0.6518 ± 0.0580 | 0.7438 (seed 1) |
| bottleneck sempit | MLP (`256-8-256`) | 8 | 1e-3 | 0 | — (diisi saat data asli) | — |
| bottleneck lebar | MLP (`256-32-256`) | 32 | 1e-3 | 0 | — (diisi saat data asli) | — |
| AE konvolusional | Conv2D AE | 16 | 1e-3 | 1e-4 | — (diisi saat data asli) | — |

## Ablation (>= 2)

1. **Ukuran bottleneck (8 vs 16 vs 32)**: apakah penyempitan ruang laten mempertajam pemisahan skor anomali atau justru merusak rekonstruksi suara normal.
2. **MLP vs AE konvolusional pada bottleneck yang sama**: pengaruh struktur lokal waktu-frekuensi terhadap AUROC.
3. **Regularisasi (tanpa vs `weight_decay 1e-4` + `dropout 0.1`)**: apakah pembatasan kapasitas mencegah anomali ikut terekonstruksi dengan baik.

## Hasil 3 seed konfigurasi akhir

Hasil uji jalur pada data simulasi (`sesi/t14-anomali-suara_simulasi/`):
- Seed 1: `val_acc = 0.5913`, `val_macro_f1 = 0.5850`, `val_auroc = 0.7438`
- Seed 2: `val_acc = 0.7103`, `val_macro_f1 = 0.6822`
- Seed 3: `val_acc = 0.7103`, `val_macro_f1 = 0.6883`
- Rerata ± simpangan baku (simulasi): `val_acc = 0.6706 ± 0.0687`, `val_macro_f1 = 0.6518 ± 0.0580`

Angka dari data simulasi hanya untuk menguji jalur kode, bukan hasil ilmiah. Angka final diperbarui setelah data meja nyata tersedia.