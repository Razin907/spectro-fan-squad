# Protokol evaluasi tim 14 - Deteksi anomali suara mesin

Protokol ini memakai data primer rekaman meja uji nyata tim 14 (`sesi-20261117-181631`, folder `data/sesi_primer/`).

| Butir | Isi |
| --- | --- |
| Kasus | Deteksi anomali suara mesin (kipas dan pompa) |
| Sumber dataset | `data/sesi_primer/` (rekaman meja nyata `sesi-20261117-181631`, `simulasi=false`, 2026-11-17 18:16:31; lihat `data/MANIFEST.json` & `data/CATATAN-SESI.md`) |
| Jenis tugas | klasifikasi biner: `0` normal, `1` anomali |
| Target / label | `0` = bunyi `normal`, `1` = salah satu gangguan (`kertas menyentuh baling kipas`, `benda logam jatuh di meja`, atau `ujung selang pompa terangkat dari air`) |
| Ukuran data latih / validasi / uji | 90 / 144 / 135 jendela; 3 / 1 / 1 ulangan; 72 kondisi valid dari 76 terekam (`no` 12, 13, 21, 22 dikeluarkan karena putus Wi-Fi ESP32) |
| Cara split | per `ulangan` (blok replikasi penuh), bukan per jendela: latih `ulangan-1`..`ulangan-3` (hanya normal, 10 kondisi = 90 jendela); validasi `ulangan-4` (4 normal + 12 anomali = 144 jendela); uji `ulangan-5` (3 normal + 12 anomali = 135 jendela) |
| Pemeriksaan kebocoran | `python -m src.protokol.data` memastikan irisan nomor kondisi latih/val/uji kosong |
| Metrik utama untuk komparasi | AUROC biner pada validasi dan uji |
| Pembanding non-DL tim | Isolation Forest pada MFCC (`tools/pembanding_iforest.py`, `results/baseline/`) |
| Praproses yang sama untuk semua model | WAV 48 kHz mono diturunkan ke 16 kHz dengan filter FIR anti-aliasing (rasio 3:1); buang 1 detik pertama setiap kondisi (tersisa 5 detik); jendela 1 detik dengan hop 0,5 detik (9 jendela/kondisi); log-mel 64 pita, 100 bingkai; normalisasi memakai rata-rata dan simpangan baku data latih saja |
| Anggaran komputasi per run | <= 20 menit CPU laptop |
| Evaluasi data uji | sekali, oleh Penjaga Data, pada sprint terakhir (`--final`) |
| `data_hash` data primer | `c4949d6dc75b47c6c569dcdfc5812f3f` (dari `data/MANIFEST.json` + `split:2026`) |

## Catatan split

`split.indeks` dari starter tidak dipakai karena unit pemisahan wajib blok replikasi (`ulangan`). Split acak per jendela akan membuat potongan dari kondisi yang sama muncul di latih dan uji. Pemeriksaan irisan kondisi dijalankan oleh `python -m src.protokol.data`.

## Catatan metrik

`src/common/evaluate.py` adalah milik starter dan tidak diubah. Penilai itu menulis akurasi dan macro-F1 ke `results/<nim>/metrics.json` beserta `data_hash`. AUROC biner dihitung terpisah oleh `tools/nilai_auroc.py` dari `results/<nim>/pred_val_skor.npy` dan ditulis ke `results/<nim>/auroc.json`.

## Catatan data primer meja

- Sesi `sesi-20261117-181631` merupakan lanjutan dari sesi 186 mulai kondisi `no=5` sampai `no=80` (kondisi 1–4 dilewati, total 76 kondisi terekam).
- Empat kondisi (`no=12, 13, 21, 22`) dikeluarkan karena sambungan Wi-Fi ESP32 terputus dan mematikan/mengacaukan aktuator (terverifikasi pada `log.txt`, `aktuator.csv`, dan `daya.csv`; rincian lengkap ada di `data/CATATAN-SESI.md`).
- Audio direkam pada laju asli mikrofon USB 48 kHz (`audio-0001.wav`..`audio-0009.wav`, masing-masing ~60 detik) lalu diselaraskan menggunakan stempel waktu `t_mulai_host` dan `t_selesai_host` di `label.csv` dan `audio_index.csv` sebelum diturunkan ke 16 kHz.
- Setiap anggota mengekstrak arsip `sesi_primer.zip` ke `data/sesi_primer/` (diabaikan oleh `.gitignore`) agar memperoleh `data_hash = c4949d6dc75b47c6c569dcdfc5812f3f` yang identik.
