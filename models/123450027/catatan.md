# Catatan model 123450027 — CNN 2D (`cnn`)

Penulis: 123450027
Status: Protokol data simulasi selesai dan `run.py` sudah terintegrasi dengan `src.protokol.data.muat()` (menunggu rekaman meja).

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

| Percobaan | Kanal Konvolusi | Dropout | LR | Batch | val_macro_f1 (3 seed) |
| --- | --- | --- | --- | --- | --- |
| baseline (simulasi) | `16-32-64` | 0.3 | 1e-3 | 64 | 0.3636 ± 0.0000 |
| jaringan lebih ramping | `8-16-32` | 0.3 | 1e-3 | 64 | — (diisi saat data asli) |
| dropout lebih rendah | `16-32-64` | 0.1 | 1e-3 | 64 | — (diisi saat data asli) |
| lr lebih kecil | `16-32-64` | 0.3 | 3e-4 | 32 | — (diisi saat data asli) |

## Ablation (>= 2)

1. **Dengan vs tanpa `Dropout(0.3)`**: menguji kontribusi regularisasi dropout pada kepala klasifikasi CNN.
2. **Dengan vs tanpa `BatchNorm2d`**: menguji dampak normalisasi batch terhadap kecepatan konvergensi dan stabilitas skor validasi.
3. **Kedalaman blok konvolusi (`2 blok` vs `3 blok`)**: menguji apakah fitur frekuensi-waktu lokal cukup ditangkap oleh reseptif yang lebih dangkal.

## Hasil 3 seed konfigurasi akhir

Hasil uji jalur pada data simulasi (`sesi/t14-anomali-suara_simulasi/`):

- **Seed 1**: `val_acc = 0.5714`, `val_macro_f1 = 0.3636`, `val_auroc = 0.3612`
- **Seed 2**: `val_acc = 0.5714`, `val_macro_f1 = 0.3636`
- **Seed 3**: `val_acc = 0.5714`, `val_macro_f1 = 0.3636`

*(Catatan: Sesuai protokol tim, `X_train` hanya berisi kelas normal `0`, sehingga CNN terawasi standar memprediksi kelas normal secara konstan pada uji jalur ini. Penyesuaian strategi deteksi/skor dan angka final akan dilengkapi setelah data meja nyata tersedia).*