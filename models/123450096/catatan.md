# Catatan Model: Masked Patch Reconstruction Transformer, jalur `trf`

Penulis: Razin Hafid Hamdi (123450096)
Status: **Sprint 1 selesai pada data primer meja nyata** (`data/sesi_primer/`, `sesi-20261117-181631`, seed 1); penyetelan hyperparameter, ablation, dan 3 seed dikerjakan di Sprint 2.

## Peran dan arsitektur

Model menerima spektrogram log-mel 64 pita sepanjang 1 detik (`[N, 100, 64]`) dari `src.protokol.data.muat()`. Karena data latih protokol tim hanya memuat bunyi normal (`label 0`), arsitektur `PatchTransformer` (`models/123450096/run.py`, **150.400 parameter**) dirancang untuk deteksi anomali *one-class* berbasis **Masked Patch Modeling (MPM)**:

- Sumbu waktu dipotong menjadi `10` patch non-tumpang-tindih (`patch = 10` bingkai = 100 ms, dimensi `640`) dan diproyeksikan linear ke `d_model = 64`.
- Saat pelatihan, `40%` patch (`4` dari `10` patch per sampel) diganti token `[MASK]` terlatih, ditambahkan *positional encoding* terlatih, diproses dua lapis `nn.TransformerEncoder` (4 head, `dim_ff = 128`, GELU, Pre-LN), lalu direkonstruksi kembali ke `[10, 64]` oleh kepala dekoder linear. Loss MSE dihitung hanya pada patch yang di-mask agar model wajib memanfaatkan *self-attention* lintas konteks waktu.
- Saat evaluasi (deterministik), tiap patch dari 10 patch di-mask bergiliran (*leave-one-patch-out*) dan rata-rata galat rekonstruksi MSE per jendela dijadikan skor anomali kontinu (`results/123450096/pred_val_skor.npy`).
- Ambang klasifikasi biner (`pred_val.npy`) diambil dari persentil ke-95 skor galat pada data latih normal saja (tanpa kebocoran data validasi/uji).

## Alasan optimizer

**AdamW** (`lr = 1e-3`, `weight_decay = 1e-2`) dipilih karena mekanisme *self-attention* Transformer peka terhadap perbedaan skala gradien antar lapisan proyeksi Q/K/V dan dekoder rekonstruksi. AdamW memisahkan *weight decay* dari gradien adaptif sehingga regularisasi tetap terkontrol. Pelatihan memakai jadwal *cosine annealing* (`eta_min = 5e-5`) selama 80 epoch pada data normal saja dan *gradient clipping* (`max_norm = 1.0`).

## Regularisasi

- **Masked Patch Modeling (`mask_ratio = 0.40`)**: mencegah pemetaan identitas (*identity shortcut*) sehingga encoder hanya dapat merekonstruksi pola normal dari konteks patch sekitar.
- **Dropout `0.1`** di dalam lapisan encoder dan sebelum dekoder linear.
- **Weight decay `1e-2`** pada AdamW dan *gradient clipping* `1.0`.
- **Pelatihan murni kelas normal** (tidak memakai data validasi untuk *early stopping* atau pemilihan ambang).

## Hasil Sprint 1 pada data primer meja (seed 1)

Data primer `data/sesi_primer/` (`data_hash = c4949d6dc75b47c6c569dcdfc5812f3f`, 72 kondisi valid setelah mengeluarkan `no=12, 13, 21, 22` akibat putus Wi-Fi ESP32; split `90` jendela latih normal / `144` validasi / `135` uji):

| Model | Jalur | Seed | `val_auroc` *(utama)* | `val_acc` | `val_macro_f1` | Durasi (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Pembanding non-DL tim (`Isolation Forest pada MFCC`, `results/baseline/`) | `baseline` | 1 | **0.650720** | 0.298611 | 0.268962 | 8.3 |
| **Transformer (`123450096`, `PatchTransformer` MPM)** | `trf` | 1 | **0.558899** | **0.333333** | **0.317401** | 272.6 |

### Rincian AUROC validasi per jenis gangguan (diagnosis awal)

| Gangguan (vs normal pada `ulangan-4`) | Transformer (`trf`) | Pembanding non-DL (`baseline`) |
| --- | --- | --- |
| `kertas menyentuh baling kipas` | **0.770833** | 0.766204 |
| `benda logam jatuh di meja` | 0.482253 | **0.615741** |
| `ujung selang pompa terangkat dari air` | 0.423611 | **0.570216** |

**Temuan awal untuk Sprint 2:**
- Pada gangguan berkelanjutan yang kuat secara spektral (`kertas menyentuh baling kipas`), Transformer sudah sedikit melampaui pembanding non-DL (`0.770833` vs `0.766204`).
- Pada `benda logam jatuh di meja`, benturan sendok hanya berlangsung < 0,3 detik sehingga rata-rata galat di seluruh 10 patch meredam sinyal impulsif; pada Sprint 2 perlu diuji agregasi skor berbasis `max` / `top-k` patch error dan ukuran patch lebih halus (`patch = 5`).
- Pada `ujung selang pompa terangkat dari air`, separuh kondisi validasi berada pada `pompa = 0` (pompa mati) sehingga sinyal akustiknya sangat dekat dengan normal.

## Pencarian hyperparameter (rencana Sprint 2)

| Percobaan | Patch | `d_model` | `nlayer` | `mask_ratio` | Agregasi skor | `val_auroc` (3 seed) |
| --- | --- | --- | --- | --- | --- | --- |
| dasar (Sprint 1, seed 1) | 10 | 64 | 2 | 0.40 | mean patch MSE | 0.558899 (1 seed) |
| patch lebih halus | 5 | 64 | 2 | 0.40 | mean / max patch MSE | — (Sprint 2) |
| model lebih lebar | 10 | 128 | 2 | 0.40 | mean / max patch MSE | — (Sprint 2) |
| rasio mask lebih rendah | 10 | 64 | 2 | 0.20 | mean / max patch MSE | — (Sprint 2) |

## Ablation (>= 2, rencana Sprint 2)

1. `--tanpa-posisi`: menghapus *positional encoding* untuk menguji apakah urutan waktu antar patch berkontribusi pada rekonstruksi.
2. `--tanpa-mask`: melatih rekonstruksi tanpa masking untuk menguji apakah *Masked Patch Modeling* mencegah *identity mapping*.
3. `--patch 5` vs `--patch 10` vs `--patch 20`: menguji resolusi patch terhadap anomali impulsif (sendok logam) vs berkelanjutan (kertas).
4. `--dropout 0.0` vs `--dropout 0.1`: menguji pengaruh regularisasi dropout pada data latih 90 jendela normal.

## Hasil 3 seed konfigurasi akhir

Akan diisi pada Sprint 2 (seed 1, 2, 3) setelah penyetelan dan ablation selesai. Pada Sprint 1 (seed 1, data primer): `val_auroc = 0.558899`, `val_acc = 0.333333`, `val_macro_f1 = 0.317401`, `data_hash = c4949d6dc75b47c6c569dcdfc5812f3f`.

## Riwayat uji jalur simulasi & pemeriksaan bentuk masukan/keluaran (Tugas #15)

Catatan ini merangkum keluaran run simulasi pada commit [`2ae6598`](https://github.com/Razin907/spectro-fan-squad/commit/2ae6598) (PR #8) beserta pemeriksaan bentuk tensor masukan/keluaran untuk memudahkan verifikasi kontrak `models/123450096/run.py`.

### 1. Pemeriksaan bentuk masukan dan keluaran

1. **Kerangka awal pada data simulasi (`2ae6598`, PR #8 — 108.866 parameter):**
   - Keluaran `src.protokol.data.muat()`: `X_train (162, 100, 64)`, `X_val (252, 100, 64)`, `X_test (198, 100, 64)`.
   - Alur tensor `PatchTransformer`: `[N, 100, 64]` $\rightarrow$ transpose `[N, 64, 100]` $\rightarrow$ `Conv1d(64, 64, kernel_size=10, stride=10)` menjadi `[N, 10, 64]` token patch $\rightarrow$ `TransformerEncoder` $\rightarrow$ `mean(dim=1)` `[N, 64]` $\rightarrow$ `head` logits `[N, 2]`.
   - Berkas keluaran validasi: `results/123450096/pred_val.npy` berdimensi `(252,)` bertipe `int64` rentang `[0, 1]`, dan `results/123450096/pred_val_skor.npy` berdimensi `(252,)` bertipe `float32` rentang `[0.439707, 0.555540]`.
2. **Pembaruan jalur *one-class* MPM pada data primer (`04b9fa2` / `main` — 150.400 parameter):**
   - Keluaran `src.protokol.data.muat()`: `X_train (90, 100, 64)`, `X_val (144, 100, 64)`, `X_test (135, 100, 64)`.
   - Alur tensor `PatchTransformer`: `[N, 100, 64]` $\rightarrow$ `patchify` `[N, 10, 640]` $\rightarrow$ `embed` + `[MASK]` + `pos` `[N, 10, 64]` $\rightarrow$ `TransformerEncoder` $\rightarrow$ `decoder` rekonstruksi `[N, 10, 640]`.
   - Berkas keluaran validasi: `pred_val.npy` `(144,)` `int64` dan `pred_val_skor.npy` `(144,)` `float32`.

### 2. Keluaran run simulasi (`2ae6598`, `--seed 1`)

Perintah uji:
```bash
python models/123450096/run.py --seed 1
python tools/nilai_auroc.py --nim 123450096 --model trf --seed 1
```

Keluaran `results/123450096/metrics.json` dan `results/123450096/auroc.json` pada commit `2ae6598`:
- `data_hash`: `922eddb536c7f5bcd3433be03c7ddbb1`
- `model`: `"trf"`, `seed`: `1`, `n_val`: `252` (`108` anomali / `144` normal), `n_test`: `0`, `durasi_detik`: `26.4` (`cpu laptop`)
- `val_acc`: `0.52381`, `val_macro_f1`: `0.522335`, `val_auroc`: `0.529128`
