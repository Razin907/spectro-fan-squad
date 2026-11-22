# Catatan Sesi & Pemeriksaan Mutu — Tim 14

Dokumen ini mencatat penerimaan data primer meja uji untuk kasus **Deteksi Anomali Suara Mesin** (Tugas #5), hasil pemeriksaan mutu umum dan khusus tim, serta daftar kondisi yang dikeluarkan dari pemrosesan protokol.

---

## 1. Sesi yang Dipakai

| Butir | Isi |
| --- | --- |
| ID Sesi | `sesi-20261117-181631` (tim 14, `versi_perekam: 1.13`, `simulasi=False`) |
| Waktu perekaman | `2026-11-17T18:16:31` s.d. `2026-11-17T18:24:47` (host `sditera-civ`) |
| Status sesi | Lanjutan dari sesi 186 (`sesi-20261117-181527`), dimulai dari kondisi `no=5/80` |
| Lokasi lokal standar | `data/sesi_primer/` (sesuai `data/MANIFEST.json`; diabaikan `.gitignore`, didistribusikan lewat penyimpanan tim / `sesi_primer.zip`) |
| Berkas dalam `data/MANIFEST.json` | 16 berkas: 7 CSV (`aktuator`, `audio_index`, `daya`, `digital`, `imu`, `label`, `lingkungan`) + 9 audio (`audio/audio-0001.wav`..`audio-0009.wav`) |
| `data_hash` resmi tim | `c4949d6dc75b47c6c569dcdfc5812f3f` (dihitung dari `data/MANIFEST.json` + `split:2026`) |

---

## 2. Pemeriksaan Mutu Umum (Panduan Metode Bagian 8)

1. **Integritas berkas terhadap `manifest.json` perekam:**
   - Seluruh 16 berkas di `data/MANIFEST.json` (plus `rencana.yaml`) **cocok 100%** SHA-256 dan ukuran byte-nya terhadap `data/sesi_primer/manifest.json`.
   - `log.txt` berselisih 103 byte (`1.631` byte aktual vs `1.528` byte di `manifest.json`) karena baris penutup `"Selesai. Data di ..."` ditulis perekam setelah hash dibuat; karena itu `log.txt` tidak dimasukkan ke `data/MANIFEST.json`.
2. **Kelengkapan CSV sensor dan nilai kosong:**
   - Seluruh berkas CSV memiliki **0 sel kosong** (`empty_cells = 0`).
   - `label.csv`: **76 baris** (`no=5..80`), 0 duplikat kombinasi, durasi tiap kondisi `6,0001–6,0006` detik, seluruh stempel waktu berada dalam cakupan `audio_index.csv`.
   - `audio_index.csv`: **9 berkas WAV kontinu** (`audio-0001.wav`..`audio-0008.wav` ~60,00 detik + `audio-0009.wav` 15,29 detik; total 495,29 detik; celah antar-potongan `0,0000–0,0005` detik). Mikrofon `NYK NEMESIS: USB Audio` merekam pada laju asli **`48.000 Hz`** mono 16-bit PCM (bukan `16.000 Hz` seperti rencana), sehingga `src/protokol/data.py` melakukan *downsampling* 3:1 ke `16.000 Hz` dengan filter FIR anti-aliasing.
   - `aktuator.csv`: 441 baris event (`kipas`, `pompa`, `lampu`, `buzzer`, `servo`).
   - `daya.csv`: 85.501 baris kanal `kipas` (kondisi sehat: `kipas=120` di `95,9–115,3 mA`, `kipas=220` di `175–211,7 mA`).
   - `imu.csv`: 170.246 baris, cakupan waktu penuh `494,85` detik.
   - `lingkungan.csv`: 434 baris stabil (`suhu 23,0–23,9 °C`, `RH 53,0–58,0%`, `lux 207,5–230,0`).
   - `digital.csv`: 0 baris (tidak ada sensor digital aktif di rencana tim 14).
3. **Kejadian gangguan perangkat keras di `log.txt`:**
   - `18:16:33.997`: `ESP32: antrean perintah penuh` (awal kondisi `no=5`; arus kipas di `daya.csv` tetap normal `105,6 mA` untuk `kipas=120`, sehingga `no=5` dipertahankan).
   - `18:17:20.783 – 18:17:23.544`: **Sambungan Wi-Fi ke ESP32 putus dan disambung ulang** (merusak kondisi `no=12` dan `no=13`).
   - `18:18:20.147 – 18:18:25.561`: **Sambungan Wi-Fi ke ESP32 putus dan disambung ulang** (merusak kondisi `no=21` dan `no=22`).

---

## 3. Pemeriksaan Khusus Tim 14

### A. Jumlah kondisi (80 rencana vs 76 terekam)

- **Terekam:** 76 dari 80 kondisi (`no=5` sampai `80`).
- **Selisih (-4 kondisi):** kondisi `1` sampai `4` terlewat karena sesi lanjutan dari sesi 186 (`log.txt`: *"mulai kondisi 5/80, kondisi 1–4 dilewati"*).
- **Kombinasi yang tidak ada di `label.csv`:**
  - `kertas menyentuh baling kipas`, `kipas=120 / pompa=200`, `ulangan=3`
  - `benda logam jatuh di meja`, `kipas=120 / pompa=200`, `ulangan=2`
  - `benda logam jatuh di meja`, `kipas=220 / pompa=0`, `ulangan=2` dan `ulangan=3`
- Seluruh 20 kondisi `normal` serta seluruh kondisi pada `ulangan=1` (16/16), `ulangan=4` (16/16, validasi), dan `ulangan=5` (16/16, uji) terekam lengkap.

### B. Kejelasan bunyi kertas dan sendok logam pada potongan audio

- **Audio hening:** 0 kondisi (`RMS` seluruh 76 kondisi berada di `0,0348–0,2115`).
- **Kertas menyentuh baling kipas (19 kondisi):**
  - Secara agregat, rata-rata `RMS` (`0,0819`) dan energi frekuensi tinggi `> 2 kHz` (`24,2%`) jelas di atas kelas `normal` (`RMS 0,0559`, `HF > 2 kHz 11,6%`).
  - Gesekan kertas terdengar sangat jelas (lonjakan `RMS` 1,4×–4,5× dan `HF > 2 kHz` hingga `71,5%`) pada 9 kondisi utama (`no=10, 18, 38, 40, 53, 55, 56, 75, 76`).
  - Catatan keterbatasan: pada sebagian kondisi (`no=11, 15, 16, 25, 28, 36, 47, 68, 73`), amplitudo relatif lembut (`RMS 0,037–0,045`), dan pada `no=62` sentuhan kertas dominan muncul di detik `0–1` (`RMS 0,136`) sebelum jendela stabil.
- **Benda logam / sendok jatuh di meja (17 kondisi):**
  - Puncak impulsif tajam (`peak 0,25–1,00`, `crest factor 6,1–15,9`, rasio *spike* 10 ms hingga `20×`) terdeteksi jelas pada 11 kondisi (`no=14, 29, 31, 32, 41, 43, 49, 60, 64, 72, 74`).
  - Catatan keterbatasan: pada `no=14` (`t_peak=0,67 s`) dan `no=64` (`t_peak=0,01 s`) benturan terjadi di 1 detik pertama sehingga terbuang oleh aturan buang 1 detik awal; 6 kondisi lain (`no=7, 8, 20, 48, 52, 57`) memiliki puncak lemah (`0,10–0,17`).
- **Catatan mutu tambahan (dipertahankan, tidak dikeluarkan):**
  - Terdapat *clipping* (`|x| >= 0,999`, 22.739 sampel di 29 kondisi, termasuk 9 kondisi berlabel `normal`).
  - Pada kelas `ujung selang pompa terangkat dari air` (20 kondisi), separuhnya (10 kondisi) terjadi saat `pompa=0` (pompa mati) sehingga secara fisik tidak menghasilkan bunyi pompa menyedot udara. Kondisi ini tetap dipertahankan sesuai desain faktorial untuk analisis kesalahan di Sprint 2.

---

## 4. Kondisi yang Dikeluarkan Beserta Alasannya

Sebanyak **4 kondisi (`no = 12, 13, 21, 22`)** dikeluarkan secara objektif berdasarkan bukti log perangkat keras (`log.txt`, `aktuator.csv`, dan `daya.csv`) karena status aktuator fisik tidak sesuai label rencana akibat putusnya koneksi Wi-Fi ESP32:

| `no` | `ulangan` | Label rencana (`gangguan`, `kipas`, `pompa`) | Bukti objektif & alasan pengeluaran |
| --- | --- | --- | --- |
| **12** | `2` (latih) | `normal`, `kipas=220`, `pompa=200` | Wi-Fi ESP32 putus pukul `18:17:20.783–18:17:23.544` di tengah kondisi; `daya.csv` mencatat arus kipas rata-rata **`-0,1 mA`** (kipas mati total, sampel turun ke 118) dan `aktuator.csv` tertahan di `(220, 0)`. |
| **13** | `2` (latih) | `normal`, `kipas=220`, `pompa=0` | Berlangsung saat pemulihan sambungan ESP32; pada detik `+4,26 s` seluruh aktuator di-reset ke `0` lalu baru disetel ulang `kipas=220` di `+4,39 s`; arus rata-rata kipas anjlok ke **`139,6 mA`** (`min -0,5 mA`). |
| **21** | `1` (latih*) | `ujung selang pompa terangkat dari air`, `kipas=120`, `pompa=0` | Wi-Fi ESP32 putus pukul `18:18:20.147–18:18:25.561`; **`0` sampel `daya.csv`** (100% dropout sensor) dan `aktuator.csv` tertahan di `(120, 200)` (pompa masih menyala padahal label `pompa=0`). |
| **22** | `5` (uji) | `normal`, `kipas=220`, `pompa=0` | ESP32 masih putus hingga detik `+1,8 s` lalu aktuator dimatikan ke `(0, 0)` pada detik `+2,12 s`; `daya.csv` mencatat arus kipas rata-rata **`7,4 mA`** (kipas mati hampir sepanjang kondisi). |

*\*`no=21` berlabel anomali pada `ulangan=1` sehingga tidak masuk latih normal-only, namun tetap didaftarkan di `KONDISI_DIKELUARKAN`.*

---

## 5. Ringkasan Data Akhir Protokol (`src/protokol/data.py`)

- **Total kondisi terekam:** 76 (`no=5..80`) → **dikeluarkan:** 4 (`no=12, 13, 21, 22`) → **dipakai:** **72 kondisi valid**.
- **Split per ulangan:**
  - **Latih (`ulangan-1..3`, normal saja):** 10 kondisi normal × 9 jendela = **90 jendela** (`label 0`).
  - **Validasi (`ulangan-4`, utuh 16/16 kondisi):** 4 normal (36) + 12 anomali (108) = **144 jendela**.
  - **Uji (`ulangan-5`, 15/16 kondisi):** 3 normal (27) + 12 anomali (108) = **135 jendela**.
- **Verifikasi semua anggota:** letakkan folder hasil ekstrak di `data/sesi_primer/`, lalu jalankan `python -m src.protokol.data` dan `python -m src.common.baseline --seed 2026`; pastikan `data_hash = c4949d6dc75b47c6c569dcdfc5812f3f`.
