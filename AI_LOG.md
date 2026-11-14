# AI_LOG — Tim 14 (spectro-fan-squad)

Satu baris per pemakaian AI yang bermakna (ChatGPT, Claude, Copilot, Bravais, dan sejenisnya).
Ringkas dan jujur; transkrip penuh tidak diminta. Yang dinilai bukan "pakai AI atau tidak",
melainkan "mengerti atau tidak" — isi kolom terakhir dengan apa yang Anda ubah dari keluarannya.

| Tanggal | Siapa (NIM) | Alat | Untuk apa | Apa yang diubah dari keluarannya |
| --- | --- | --- | --- | --- |
| 2026-11-12 | 123450096 | Antigravity (Swarm) | Audit pemotong jendela (`src/protokol/data.py`), `PROTOKOL.md`, dan `rencana/t14-anomali-suara.yaml` | tolak ubah `sprint/sprint0/123450027.md` & `models/123450013/run.py`; rapikan header `PROTOKOL.md`; kerjakan `git push` & PR sendiri |
| 2026-11-12 | 123450096 | Antigravity (Swarm) | Kerangka Transformer (`models/123450096/run.py` dan `catatan.md`) | tolak fallback sintetis AI (`_buat_data_simulasi`, `--sim-*`, monkeypatch `evaluate`) dan arahkan penghapusannya; pertahankan `PatchTransformer` + `data.muat()` & terima pembaruan `catatan.md` |
| 2026-10-31 | 123450013 | Claude | ide nama tim dan avatar | memilih Spectro Fan Squad dari daftar, memperbaiki ejaan "Squard" menjadi "Squad", memilih avatar 🌀 |
| 2026-10-31 | 123450013 | Claude | draf Kontrak Tim (pembagian jalur, tugas sampingan, standup, rencana jika anggota hilang) | mengisi nama dan NIM anggota, menetapkan standup Selasa 20.00 WIB sesuai kesepakatan tim, lalu memilih komponen Autoencoder di kartu Tim saya |
| 2026-11-01 | 123450013 | Claude | langkah Git (clone, cabang 123450013/kerja, commit, push) dan membaca error | menjalankan sendiri di Git Bash, membuat cabang 123450013/kerja, menyalin models/TEMPLATE menjadi models/123450013, lalu commit dan push |
| 2026-11-01 | 123450013 | Claude | draf catatan.md jalur AE (rencana dan variasi percobaan) | memakai rencana awal (variasi bottleneck, MLP vs konvolusional, regularisasi), menambah bagian ambang keputusan dari persentil ke-95 skor data latih; belum dijalankan |
| 2026-11-14 | 123450013 | Claude | kerangka run.py Autoencoder (MLP, bottleneck 16, skor reconstruction error, ambang persentil ke-95) | mengganti JALUR menjadi "ae", menyimpan pred_val_skor.npy untuk AUROC, menjalankan sendiri dan mendapati run pertama masih memakai versi lama, memperbaikinya, lalu val AUROC 0.744 pada data simulasi |
| 2026-11-14 | 123450013 | Claude | membaca error run_all.py (baseline_mlp tidak ada, .wav tidak ikut repo) dan menyusun catatan kartu #6 | memeriksa riwayat Git dan isi run_all.py sendiri, membangkitkan data simulasi dengan meja.py, menjalankan run_all.py dengan --tanpa-baseline |
