# Catatan model 123450013 â€” Autoencoder (`ae`)

Penulis: Azzahra Putri Kamilah (123450013)
Status: Sprint 1 selesai untuk seed 1 pada data primer meja (`sesi-20261117-181631`). Tiga seed dan penyetelan dikerjakan di Sprint 2.

## Peran dan arsitektur

Mendeteksi anomali suara kipas dengan Autoencoder (AE) yang dilatih hanya pada data normal. Masukan: log-mel 64 pita, 1 detik (`[N, 100, 64]`) dari `src.protokol.data.muat()` (48 kHz diturunkan ke 16 kHz, 1 detik pertama dibuang, jendela 1 detik hop 0,5 detik), diratakan menjadi 6.400 fitur. Arsitektur baseline: MLP simetris `6400 -> 256 -> 16 -> 256 -> 6400` (ReLU). Skor anomali adalah reconstruction error (MSE per jendela). Ambang biner diambil dari persentil ke-95 skor data latih normal, bukan data uji. AUROC dihitung dari skor kontinu (`pred_val_skor.npy`) sehingga tidak bergantung pada ambang.

## Data dan protokol

- Latih: ulangan 1-3, hanya normal (90 jendela). Validasi: ulangan 4 (144 jendela: 36 normal, 108 anomali). Uji: ulangan 5, dievaluasi sekali di sprint terakhir.
- 72 dari 76 kondisi terekam dipakai (no 12, 13, 21, 22 dikeluarkan karena Wi-Fi ESP32 putus).
- `data_hash` pada `metrics.json`: `c4949d6dc75b47c6c569dcdfc5812f3f`, sama dengan `data_hash` resmi di `PROTOKOL.md`.

## Alasan optimizer

**Adam** (`lr=1e-3`, 30 epoch, batch 32) karena rekonstruksi log-mel yang diratakan memiliki skala gradien yang bervariasi antar pita frekuensi, dan laju belajar adaptif per parameter membuat MSE pada data normal konvergen cepat tanpa penalaan jadwal.

## Regularisasi

- Bottleneck sempit (16) membatasi kapasitas representasi.
- Pelatihan hanya pada kelas normal.
- Direncanakan di Sprint 2: `weight_decay = 1e-4` (`--wd`) dan dropout 0.1.

## Hasil data primer meja (seed 1)

| Konfigurasi | Seed | val AUROC | val_acc | val_macro_f1 |
| --- | --- | --- | --- | --- |
| MLP 256-16-256, Adam 1e-3, 30 epoch | 1 | 0.6062 | 0.3056 | 0.2822 |

Catatan pembacaan:
- AUROC 0.606 hanya sedikit di atas tebakan acak (0.5). Ini hasil baseline apa adanya.
- `val_acc` dan `macro_f1` rendah karena ambang persentil ke-95 dari data latih terlalu tinggi: hampir semua jendela validasi diprediksi normal, padahal 75% validasi adalah anomali. AUROC tidak terpengaruh ambang, jadi itu metrik utamanya.
- Pembanding non-DL (Isolation Forest pada MFCC, `results/baseline/`, seed 1): val AUROC 0.6507. AE (0.6062) belum melampaui pembanding pada baseline Sprint 1; selisih kecil dan baru satu seed.
- Dugaan penyebab AUROC rendah (belum diuji): data latih normal hanya 90 jendela untuk model 6.400 fitur, dan anomali lembut (kertas, selang pompa saat pompa mati) mirip normal. Akan diuji di Sprint 2.

## Pencarian hyperparameter (rencana Sprint 2)

| Percobaan | Arsitektur | Bottleneck | LR | Weight Decay | val_auroc (3 seed) |
| --- | --- | --- | --- | --- | --- |
| baseline | MLP 256-16-256 | 16 | 1e-3 | 0 | seed 1: 0.6062 |
| bottleneck sempit | MLP 256-8-256 | 8 | 1e-3 | 0 | diisi Sprint 2 |
| bottleneck lebar | MLP 256-32-256 | 32 | 1e-3 | 0 | diisi Sprint 2 |
| regularisasi | MLP 256-16-256 | 16 | 1e-3 | 1e-4 | diisi Sprint 2 |
| AE konvolusional | Conv2D AE | 16 | 1e-3 | 1e-4 | diisi Sprint 2 |

Perintah contoh: `python models/123450013/run.py --seed 1 --bottleneck 8`, lalu `python tools/nilai_auroc.py --nim 123450013 --model ae --seed 1`.

## Ablation (>= 2, Sprint 2)

1. Ukuran bottleneck (8 vs 16 vs 32).
2. MLP vs AE konvolusional pada bottleneck yang sama.
3. Regularisasi (tanpa vs `weight_decay 1e-4` dan dropout).

Setiap percobaan mengubah satu hal, memakai 3 seed, dan dicatat di sini. Ukuran ONNX (KB) dan latensi (ms) di laptop meja juga dicatat tiap varian.

## Catatan data simulasi (hanya uji jalur, bukan hasil ilmiah)

Pada data simulasi `sesi/t14-anomali-suara_simulasi/`, seed 1 menghasilkan val AUROC 0.7438. Angka ini tidak dipakai untuk kesimpulan.
