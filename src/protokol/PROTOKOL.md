# Protokol evaluasi tim 14 - Deteksi anomali suara mesin

Protokol ini dibekukan setelah disetujui pembimbing. Data simulasi dipakai untuk
menguji pemotong dan jalur evaluasi sebelum data meja tersedia pada 11 November 2026.

| Butir | Isi |
|---|---|
| Kasus | Deteksi anomali suara mesin (kipas dan pompa) |
| Sumber dataset | `sesi/t14-anomali-suara_simulasi/audio/` (SINTETIS, pengganti sementara) |
| Jenis tugas | klasifikasi biner: `0` normal, `1` anomali |
| Target / label | `0` = bunyi normal, `1` = salah satu gangguan (kertas, logam, atau pompa) |
| Ukuran data latih / validasi / uji | 162 / 252 / 198 jendela; 4 / 3 / 2 sesi; 104 berkas WAV (80 slot + 24 harian) |
| Cara split | per `sesi_id`, bukan per berkas: latih `slot-01`..`slot-03` + `harian-2026-11-11`; validasi `slot-04` + `harian-2026-11-12`..`13`; uji `slot-05` + `harian-2026-11-14` |
| Pemeriksaan kebocoran | `python -m src.protokol.data` memastikan irisan sesi latih/val/uji kosong |
| Metrik utama untuk komparasi | AUROC biner pada validasi dan uji |
| Praproses yang sama untuk semua model | WAV 16 kHz mono; buang 1 detik pertama; jendela 1 detik dengan hop 0,5 detik; log-mel 64 pita, 100 bingkai; normalisasi memakai rata-rata dan simpangan baku data latih saja |
| Anggaran komputasi per run | <= 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` saat dibekukan | `922eddb536c7f5bcd3433be03c7ddbb1` (dari `results/baseline/metrics.json`) |

## Catatan split

`split.indeks` dari starter tidak dipakai karena unit pemisahan wajib sesi/blok. Split acak
per jendela akan membuat potongan dari sesi yang sama muncul di latih dan uji. Pemeriksaan
irisan sesi dijalankan oleh `python -m src.protokol.data`.

## Catatan metrik

`src/common/evaluate.py` adalah milik starter dan tidak diubah. Penilai itu menulis
akurasi dan macro-F1. AUROC dihitung terpisah oleh `tools/nilai_auroc.py` dari
`results/<nim>/pred_val_skor.npy` dan ditulis ke `results/<nim>/auroc.json`.

## Catatan data simulasi

- Data simulasi hanya untuk menguji pemotong, split per sesi, normalisasi, dan jalur
  evaluasi. Angka AUROC dari data ini bukan hasil ilmiah final.
- Data latih hanya berisi potongan normal dari slot dan rekaman harian.
- Data validasi dan uji memuat normal dan anomali.
- Saat data meja tiba, ganti isi `audio/` dan `audio/index.csv`, lalu perbarui
  `data/MANIFEST.json`; kode protokol dan model tidak berubah.

Disetujui pembimbing pada: `[ ]` (tautan pull request: `[ ]`)
