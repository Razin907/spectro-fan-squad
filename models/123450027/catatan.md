# Catatan model 123450027 — CNN 2D (`cnn`)

Penulis: 123450027
Status: Sprint 1 selesai — dilatih pada data primer meja (seed 1), `metrics.json` valid (`data_hash` c4949d6dc75b47c6c569dcdfc5812f3f). Tiga seed dan penyetelan dikerjakan di Sprint 2.

## Peran dan arsitektur

Model menerima spektrogram log-mel 64 pita sepanjang 1 detik yang ditata menjadi tensor 2D `[N, 1, 64, 100]` (`[N, kanal, pita, waktu]`). Arsitektur terdiri dari 3 blok konvolusi bertingkat (`1 -> 16 -> 32 -> 64` kanal), di mana setiap blok memuat `Conv2d(kernel_size=3, padding=1)`, `BatchNorm2d`, `ReLU`, dan `MaxPool2d(2)`. Peta fitur diringkas dengan `AdaptiveAvgPool2d(1)`, diratakan (`Flatten`), melewati `Dropout(0.3)`, dan dipetakan ke lapisan `Linear(64, n_keluaran)`. Skor anomali kontinu untuk AUROC dihitung sebagai `1 - P(kelas normal)`.

## Alasan optimizer

Menggunakan **Adam** (`lr=1e-3`) karena kombinasi lapisan konvolusi 2D dan `BatchNorm2d` membutuhkan adaptasi skala gradien per filter agar konvergen stabil dalam anggaran 15 epoch di CPU laptop tanpa perlu penalaan jadwal *learning rate* manual pada tahap awal.

## Regularisasi

- **`BatchNorm2d`** pada setiap blok konvolusi untuk menstabilkan distribusi aktivasi.
- **`Dropout(0.3)`** sebelum kepala klasifikasi linear untuk mengurangi *co-adaptation* antar filter.
- **`AdaptiveAvgPool2d(1)`** (*Global Average Pooling*) untuk menekan jumlah parameter kepala linear sekaligus mencegah *overfitting* spasial.
- **Pembobotan kelas otomatis (`class weighting`)** pada *cross-entropy loss*.

## Pencarian hyperparameter (tabel)

| Percobaan | Kanal Konvolusi | Dropout | LR | Batch | val_auroc (seed 1) |
| --- | --- | --- | --- | --- | --- |
| baseline (data primer meja) | `16-32-64` | 0.3 | 1e-3 | 64 | 0.4715 |
| jaringan lebih ramping | `8-16-32` | 0.3 | 1e-3 | 64 | — (Sprint 2) |
| dropout lebih rendah | `16-32-64` | 0.1 | 1e-3 | 64 | — (Sprint 2) |
| lr lebih kecil | `16-32-64` | 0.3 | 3e-4 | 32 | — (Sprint 2) |

## Ablation (>= 2) — direncanakan, dijalankan di Sprint 2

1. **Dengan vs tanpa `Dropout(0.3)`**: menguji kontribusi regularisasi dropout pada kepala klasifikasi CNN.
2. **Dengan vs tanpa `BatchNorm2d`**: menguji dampak normalisasi batch terhadap kecepatan konvergensi dan stabilitas skor validasi.
3. **Kedalaman blok konvolusi (`2 blok` vs `3 blok`)**: menguji apakah fitur frekuensi-waktu lokal cukup ditangkap oleh reseptif yang lebih dangkal.

## Hasil Sprint 1 (seed 1, data primer meja)

Latih 90 jendela normal; validasi 144 jendela (36 normal, 108 anomali); 15 epoch; 15,7 detik di CPU laptop.

| Model | val_auroc | val_acc | val_macro_f1 |
| --- | --- | --- | --- |
| CNN 2D (seed 1) | 0.4715 | 0.25 | 0.20 |
| Pembanding non-DL: Isolation Forest + MFCC (seed 1) | 0.6507 | 0.2986 | 0.2690 |

Seed 2 dan seed 3: diisi di Sprint 2.

*Catatan:* Sesuai protokol tim, `X_train` hanya berisi kelas normal `0`, sehingga CNN terawasi memprediksi kelas normal secara konstan (val_acc 0.25 = proporsi normal di validasi). Loss latih turun (1.04 → 0.054) karena model makin yakin "normal", bukan karena belajar membedakan anomali. Skor `1 - P(normal)` belum diskriminatif (AUROC di sekitar tingkat acak) dan di bawah pembanding non-DL. Penyesuaian strategi deteksi/skor dibahas di Sprint 2.