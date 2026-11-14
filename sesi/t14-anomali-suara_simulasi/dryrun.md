# Dryrun - Deteksi anomali suara mesin

Tanggal simulasi (`meja.py`): 2026-11-11
Status dryrun (tanpa meja & tanpa merekam): selesai pada 2026-11-13

## Tujuan

Melatih urutan gerakan, pergantian kondisi, dan timing tanpa merekam audio.

## Urutan latihan

1. Mempersiapkan kartu atau layar berisi instruksi kondisi.
2. Membaca satu kondisi, melakukan gerakan selama `durasi_detik`, lalu berhenti.
3. Menunggu `jeda_detik` sebelum kondisi berikutnya.
4. Mengulangi untuk satu sesi 16 kondisi sampai lancar.
5. Setelah mahir baru lanjut ke sesi nyata dengan perekaman.

Sesi pertama (16 kondisi):

1. ujung selang pompa terangkat dari air (6 detik, jeda 0.5 detik)
2. normal (6 detik, jeda 0.5 detik)
3. benda logam jatuh di meja (6 detik, jeda 0.5 detik)
4. kertas menyentuh baling kipas (6 detik, jeda 0.5 detik)
5. benda logam jatuh di meja (6 detik, jeda 0.5 detik)
6. ujung selang pompa terangkat dari air (6 detik, jeda 0.5 detik)
7. ujung selang pompa terangkat dari air (6 detik, jeda 0.5 detik)
8. kertas menyentuh baling kipas (6 detik, jeda 0.5 detik)
9. normal (6 detik, jeda 0.5 detik)
10. benda logam jatuh di meja (6 detik, jeda 0.5 detik)
11. benda logam jatuh di meja (6 detik, jeda 0.5 detik)
12. kertas menyentuh baling kipas (6 detik, jeda 0.5 detik)
13. ujung selang pompa terangkat dari air (6 detik, jeda 0.5 detik)
14. kertas menyentuh baling kipas (6 detik, jeda 0.5 detik)
15. normal (6 detik, jeda 0.5 detik)
16. normal (6 detik, jeda 0.5 detik)

## Gerakan yang dilatih

- **Kertas menyentuh baling kipas**: mematikan kipas saat latihan, menyentuh pelan dengan karton, lalu menyalakan kembali untuk sesi nyata.
- **Benda logam jatuh di meja**: menjatuhkan sendok dari tanda 10 cm (tidak dilempar), dan memastikan permukaan stabil.
- **Ujung selang pompa terangkat dari air**: mengangkat perlahan, menjaga wadah air agar tidak tumpah.
- **Normal**: tidak ada gangguan, menjaga posisi serta suara ruangan tetap sama.
