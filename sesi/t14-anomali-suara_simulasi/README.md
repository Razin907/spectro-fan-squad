# Sesi simulasi - Deteksi anomali suara mesin

- Tim: 14
- Pembimbing: Ardika Satria
- Mode: simulasi
- Dibuat: 2026-11-11
- Faktor: gangguan (4 taraf), kipas (2 taraf), pompa (2 taraf)
- Kombinasi faktor: 16
- Ulangan: 5
- Total kondisi: 80
- Estimasi durasi perekaman: 520.0 detik (8.7 menit), belum termasuk penyiapan.

## Isi folder

- `kondisi.csv` - daftar 80 kondisi untuk dijalankan.
- `sesi.json` - ringkasan dan metadata sesi.
- `dryrun.md` - urutan latihan tanpa merekam.
- `audio/index.csv` - daftar berkas WAV dan labelnya.
- `audio/slot/` - rekaman sintetis per kondisi slot.
- `audio/harian/` - rekaman sintetis program harian.
- `audio/` - tempat rekaman sesi nyata.

## Status

- [x] Daftar 80 kondisi dibuat.
- [x] Folder sesi simulasi dibuat.
- [x] Bahan fisik disiapkan.
- [x] Dryrun dilakukan.
- [x] Folder sesi diterima Penjaga Data.
- [ ] Rekaman audio sesi nyata dilakukan.

## Catatan untuk Penjaga Data

Folder ini berisi kondisi dan audio sintetis, bukan rekaman meja nyata. Setelah bahan siap
dan dryrun selesai, sesi nyata memakai urutan `kondisi.csv` yang sama. Setiap
rekaman mengikuti `durasi_detik`, lalu jeda `jeda_detik` sebelum kondisi berikutnya.
