# Catatan Model: Autoencoder (AE), jalur utama
Penulis: Azzahra Putri Kamilah (123450013)
Status: Sprint 0, perencanaan

## Tujuan
Mendeteksi anomali suara kipas dengan Autoencoder yang dilatih hanya pada data normal. Anomali (kertas menyentuh baling, benda logam jatuh, selang pompa terangkat) diharapkan menghasilkan reconstruction error lebih besar daripada data normal.

## Rencana
- Masukan: Log-mel 64 pita, 1 s, dari protokol tim (data.py).
- Latih: hanya data normal (data bersama).
- Skor anomali: reconstruction error (MSE) per sampel.
- Ambang keputusan: ditentukan dari data validasi, bukan data uji. Data uji dievaluasi sekali di sprint terakhir.
- Prapemrosesan: 1 detik pertama tiap kondisi dibuang; normalisasi dari statistik data latih saja.
- Metrik: AUROC (rata-rata ± simpangan dari 3 seed), ukuran ONNX (KB), latensi (ms) di laptop meja.
- Pembanding: Isolation Forest pada MFCC (dipegang Wulan).

## Hal yang akan dicoba (rencana awal, dapat berubah di Sprint 1)
1. Baseline AE: MLP sederhana (Log-mel diratakan, encoder 2 lapis, decoder simetris), bottleneck 16.
2. Ukuran bottleneck: 8, 16, 32. Pertanyaan: apakah bottleneck lebih kecil membuat anomali lebih mudah terdeteksi, atau justru merusak rekonstruksi data normal?
3. Arsitektur: AE konvolusional kecil dibandingkan dengan MLP pada bottleneck yang sama.
4. Regularisasi: tanpa regularisasi, weight decay 1e-4, dan dropout 0.1. Pertanyaan: apakah AE terlalu pandai merekonstruksi sehingga anomali pun ikut terekonstruksi dengan baik (overfit)?
5. Ukuran dan latensi: bandingkan ukuran ONNX (KB) dan latensi (ms) tiap varian, lalu pilih yang seimbang antara AUROC dan batasan laptop meja.

Aturan percobaan: ubah satu hal per percobaan, gunakan 3 seed, catat konfigurasi dan AUROC tiap percobaan di bagian Catatan eksperimen.

## Catatan eksperimen
(Diisi mulai Sprint 1: konfigurasi, seed, AUROC, durasi latih.)
## Ambang keputusan
Ambang dipilih dari data validasi, misalnya persentil ke-95 dari reconstruction error data normal. AUROC sendiri tidak butuh ambang.
## Hambatan
- Baseline dan run.py belum bisa dijalankan karena src/protokol/data.py dan MANIFEST.json masih diisi Penjaga Data & Evaluasi.