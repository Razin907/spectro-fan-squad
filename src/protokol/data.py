"""Pemuat data kasus tim 14: deteksi anomali suara mesin (data primer meja).

Protokol yang dipakai:

- folder sesi primer ``data/sesi_primer/`` (sesuai ``data/MANIFEST.json``);
- kondisi dengan gangguan perangkat keras ESP32 Wi-Fi putus (``no`` 12, 13, 21, 22)
  dikeluarkan sesuai ``data/CATATAN-SESI.md`` (tersisa 72 dari 76 kondisi terekam);
- audio WAV mono 16-bit (48 kHz pada perekam meja) diselaraskan ke ``label.csv`` melalui
  ``audio_index.csv``, lalu diturunkan laju cupliknya ke 16 kHz dengan filter FIR anti-aliasing;
- buang 1 detik pertama setiap kondisi (tersisa 5 detik);
- potong jendela 1 detik dengan hop 0,5 detik (9 jendela per kondisi);
- ubah menjadi log-mel 64 pita, 100 bingkai;
- latih hanya potongan normal (ulangan 1, 2, 3);
- validasi (ulangan 4) dan uji (ulangan 5) memuat normal dan semua jenis anomali;
- normalisasi memakai statistik data latih saja;
- pemisahan dilakukan per ``ulangan`` (blok replikasi penuh), bukan per jendela.

Kontrak mengikuti starter:

    TUGAS = "klasifikasi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)
"""
from __future__ import annotations

import csv
import wave

import numpy as np

from src.common import manifest

TUGAS = "klasifikasi"

FOLDER_SESI = manifest.AKAR / "data" / "sesi_primer"

# Kondisi dikeluarkan karena putus sambungan Wi-Fi ESP32 / aktuator mati
# (terverifikasi di log.txt, aktuator.csv, dan daya.csv; dan lihat data/CATATAN-SESI.md).
KONDISI_DIKELUARKAN = frozenset({12, 13, 21, 22})

SR = 16000
N_MELS = 64
N_FFT = 1024
WIN_LENGTH = 400
HOP_LENGTH = 160
N_FRAME = 100
HOP_WINDOW = SR // 2
BUANG_AWAL_DETIK = 1.0
DURASI_BERSIH_DETIK = 5.0

ULANGAN_LATIH = ("1", "2", "3")
ULANGAN_VAL = ("4",)
ULANGAN_UJI = ("5",)


def _filterbank_mel() -> np.ndarray:
    """Bangun filter triangular 64 pita dari 0 sampai Nyquist (8 kHz)."""
    def hz_ke_mel(f: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_ke_hz(m: np.ndarray | float) -> np.ndarray | float:
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    titik_mel = np.linspace(hz_ke_mel(0.0), hz_ke_mel(SR / 2.0), N_MELS + 2)
    titik_hz = np.asarray(mel_ke_hz(titik_mel))
    titik_bin = np.floor((N_FFT + 1) * titik_hz / SR).astype(int)
    titik_bin = np.clip(titik_bin, 0, N_FFT // 2)

    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        kiri, tengah, kanan = titik_bin[m - 1], titik_bin[m], titik_bin[m + 1]
        if tengah <= kiri:
            tengah = min(kiri + 1, N_FFT // 2)
        if kanan <= tengah:
            kanan = min(tengah + 1, N_FFT // 2)
        if kiri < tengah:
            fb[m - 1, kiri:tengah] = np.linspace(
                0.0, 1.0, tengah - kiri, endpoint=False, dtype=np.float32
            )
        if tengah < kanan:
            fb[m - 1, tengah:kanan] = np.linspace(
                1.0, 0.0, kanan - tengah, endpoint=False, dtype=np.float32
            )
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-9)
    return fb


def _buat_lpf(faktor: int, num_taps: int = 63) -> np.ndarray:
    """Filter FIR low-pass windowed-sinc untuk anti-aliasing sebelum desimasi."""
    cutoff = 0.45 / float(faktor)  # 0.9 * Nyquist keluaran (7200 Hz saat 48 -> 16 kHz)
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    h *= np.hanning(num_taps)
    h /= h.sum()
    return h.astype(np.float32)


_FB = _filterbank_mel()
_LPF_CACHE: dict[int, np.ndarray] = {}


def _turunkan_laju(x: np.ndarray, sr_asal: int) -> np.ndarray:
    if sr_asal == SR:
        return x.astype(np.float32, copy=False)
    if sr_asal % SR == 0:
        faktor = sr_asal // SR
        if faktor not in _LPF_CACHE:
            _LPF_CACHE[faktor] = _buat_lpf(faktor)
        tersaring = np.convolve(x, _LPF_CACHE[faktor], mode="same")
        return tersaring[::faktor].astype(np.float32)
    n_baru = int(round(len(x) * SR / sr_asal))
    t_lama = np.linspace(0.0, 1.0, len(x), endpoint=False)
    t_baru = np.linspace(0.0, 1.0, n_baru, endpoint=False)
    return np.interp(t_baru, t_lama, x).astype(np.float32)


def _baca_wav_obj(wf: wave.Wave_read, nama: str) -> tuple[np.ndarray, int]:
    if wf.getnchannels() != 1:
        raise ValueError(f"{nama}: audio harus mono, ditemukan {wf.getnchannels()} kanal")
    if wf.getsampwidth() != 2:
        raise ValueError(f"{nama}: audio harus PCM 16-bit")
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


def _logmel(segmen: np.ndarray) -> np.ndarray:
    """Ubah satu jendela 1 detik (16 kHz) menjadi log-mel [100, 64]."""
    segmen = np.asarray(segmen, dtype=np.float32)
    panjang = (N_FRAME - 1) * HOP_LENGTH + WIN_LENGTH
    if len(segmen) < panjang:
        segmen = np.pad(segmen, (0, panjang - len(segmen)))
    else:
        segmen = segmen[:panjang]

    bingkai = np.lib.stride_tricks.sliding_window_view(segmen, WIN_LENGTH)[::HOP_LENGTH]
    bingkai = bingkai[:N_FRAME] * np.hanning(WIN_LENGTH).astype(np.float32)
    spektrum = np.fft.rfft(bingkai, n=N_FFT, axis=1)
    daya = (spektrum.real ** 2 + spektrum.imag ** 2).astype(np.float32)
    mel = daya @ _FB.T
    return np.log(mel + 1e-6).astype(np.float32)


def _saring_kondisi_valid(labels: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in labels if int(r["no"]) not in KONDISI_DIKELUARKAN]


def _baca_label_saja() -> list[dict[str, str]]:
    jalur_label = FOLDER_SESI / "label.csv"
    if not jalur_label.exists():
        raise FileNotFoundError(
            f"Data primer tidak ditemukan di {FOLDER_SESI}. "
            "Pastikan folder sesi primer sudah diekstrak (lihat data/MANIFEST.json)."
        )
    with jalur_label.open(encoding="utf-8", newline="") as f:
        return _saring_kondisi_valid(list(csv.DictReader(f)))


def _baca_sesi_mentah() -> tuple[
    list[dict[str, str]],
    list[tuple[float, float, int, np.ndarray]],
]:
    jalur_label = FOLDER_SESI / "label.csv"
    jalur_idx = FOLDER_SESI / "audio_index.csv"
    if not jalur_label.exists() or not jalur_idx.exists():
        raise FileNotFoundError(
            f"Data primer tidak ditemukan di {FOLDER_SESI}. "
            "Pastikan folder sesi primer sudah diekstrak (lihat data/MANIFEST.json)."
        )
    with jalur_label.open(encoding="utf-8", newline="") as f:
        labels = _saring_kondisi_valid(list(csv.DictReader(f)))
    with jalur_idx.open(encoding="utf-8", newline="") as f:
        audio_idx = list(csv.DictReader(f))
    potongan: list[tuple[float, float, int, np.ndarray]] = []
    for row in audio_idx:
        jalur_wav = FOLDER_SESI / "audio" / row["berkas"]
        with wave.open(str(jalur_wav), "rb") as wf:
            x, sr = _baca_wav_obj(wf, str(jalur_wav))
        t0 = float(row["t_mulai_host"])
        potongan.append((t0, t0 + len(x) / sr, sr, x))
    return labels, potongan


def _potong_kondisi(
    potongan: list[tuple[float, float, int, np.ndarray]],
    t_mulai: float,
    t_selesai: float,
) -> np.ndarray:
    """Ambil audio kondisi setelah buang 1 detik pertama, lalu turunkan ke 16 kHz."""
    t0 = t_mulai + BUANG_AWAL_DETIK
    t1 = t_selesai
    bagian: list[np.ndarray] = []
    sr_ketemu: int | None = None
    for a0, a1, sr, x in potongan:
        if t1 > a0 and t0 < a1:
            i0 = max(0, int(round((t0 - a0) * sr)))
            i1 = min(len(x), int(round((t1 - a0) * sr)))
            if i1 > i0:
                bagian.append(x[i0:i1])
                sr_ketemu = sr
    if not bagian or sr_ketemu is None:
        raise ValueError(f"Tidak ada audio pada rentang [{t0:.3f}, {t1:.3f}]")
    seg = np.concatenate(bagian)
    seg16 = _turunkan_laju(seg, sr_ketemu)

    target = int(round(DURASI_BERSIH_DETIK * SR))
    if abs(len(seg16) - target) <= int(0.1 * SR):
        if len(seg16) < target:
            seg16 = np.pad(seg16, (0, target - len(seg16)))
        else:
            seg16 = seg16[:target]
    return seg16


def _bagi_ulangan(labels: list[dict[str, str]]) -> tuple[set[str], set[str], set[str]]:
    tersedia = {r["ulangan"].strip() for r in labels}
    latih = set(ULANGAN_LATIH)
    val = set(ULANGAN_VAL)
    uji = set(ULANGAN_UJI)
    kurang = (latih | val | uji) - tersedia
    if kurang:
        raise ValueError(f"Ulangan protokol tidak lengkap di label.csv: kurang {sorted(kurang)}")
    if (latih & val) or (latih & uji) or (val & uji):
        raise ValueError("Ulangan bocor antar split")
    return latih, val, uji


def _kumpulkan_segmen(
    labels: list[dict[str, str]],
    potongan: list[tuple[float, float, int, np.ndarray]],
    ulangan_set: set[str],
    hanya_normal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    W: list[np.ndarray] = []
    y: list[int] = []
    for row in labels:
        if row["ulangan"].strip() not in ulangan_set:
            continue
        label = 0 if row["gangguan"].strip().lower() == "normal" else 1
        if hanya_normal and label != 0:
            continue
        seg16 = _potong_kondisi(
            potongan,
            float(row["t_mulai_host"]),
            float(row["t_selesai_host"]),
        )
        for awal in range(0, len(seg16) - SR + 1, HOP_WINDOW):
            W.append(seg16[awal : awal + SR])
            y.append(label)
    if not W:
        raise ValueError(f"Tidak ada jendela audio untuk ulangan {sorted(ulangan_set)}")
    return np.stack(W).astype(np.float32), np.asarray(y, dtype=np.int64)


def muat_segmen() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Kembalikan jendela audio 1 detik 16 kHz mentah pada split protokol (untuk MFCC non-DL)."""
    labels, potongan = _baca_sesi_mentah()
    u_latih, u_val, u_uji = _bagi_ulangan(labels)
    Wtr, ytr = _kumpulkan_segmen(labels, potongan, u_latih, hanya_normal=True)
    Wva, yva = _kumpulkan_segmen(labels, potongan, u_val, hanya_normal=False)
    Wte, yte = _kumpulkan_segmen(labels, potongan, u_uji, hanya_normal=False)
    return Wtr, ytr, Wva, yva, Wte, yte


def _muat_detail() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    dict[str, list[str]],
]:
    labels, potongan = _baca_sesi_mentah()
    u_latih, u_val, u_uji = _bagi_ulangan(labels)

    Wtr, ytr = _kumpulkan_segmen(labels, potongan, u_latih, hanya_normal=True)
    Wva, yva = _kumpulkan_segmen(labels, potongan, u_val, hanya_normal=False)
    Wte, yte = _kumpulkan_segmen(labels, potongan, u_uji, hanya_normal=False)

    Xtr = np.stack([_logmel(w) for w in Wtr])
    Xva = np.stack([_logmel(w) for w in Wva])
    Xte = np.stack([_logmel(w) for w in Wte])

    if np.unique(ytr).tolist() != [0]:
        raise ValueError("Data latih harus hanya berisi normal (label 0).")
    if len(np.unique(yva)) < 2:
        raise ValueError("Data validasi harus memuat normal dan anomali.")
    if len(np.unique(yte)) < 2:
        raise ValueError("Data uji harus memuat normal dan anomali.")

    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xva = ((Xva - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)

    info = {
        "latih": [f"ulangan-{u}" for u in sorted(u_latih)],
        "val": [f"ulangan-{u}" for u in sorted(u_val)],
        "uji": [f"ulangan-{u}" for u in sorted(u_uji)],
    }
    return Xtr, ytr, Xva, yva, Xte, yte, info


def muat():
    Xtr, ytr, Xva, yva, Xte, yte, _ = _muat_detail()
    return Xtr, ytr, Xva, yva, Xte, yte


def periksa_split() -> dict[str, object]:
    """Periksa bahwa tidak ada ulangan maupun nomor kondisi yang muncul di dua split."""
    labels = _baca_label_saja()
    latih, val, uji = _bagi_ulangan(labels)
    no_latih = {r["no"] for r in labels if r["ulangan"].strip() in latih}
    no_val = {r["no"] for r in labels if r["ulangan"].strip() in val}
    no_uji = {r["no"] for r in labels if r["ulangan"].strip() in uji}
    return {
        "kondisi_dipakai": len(labels),
        "kondisi_dikeluarkan": sorted(KONDISI_DIKELUARKAN),
        "latih": [f"ulangan-{u}" for u in sorted(latih)],
        "val": [f"ulangan-{u}" for u in sorted(val)],
        "uji": [f"ulangan-{u}" for u in sorted(uji)],
        "irisan_latih_val": sorted(no_latih & no_val),
        "irisan_latih_uji": sorted(no_latih & no_uji),
        "irisan_val_uji": sorted(no_val & no_uji),
    }


if __name__ == "__main__":
    info = periksa_split()
    print(f"kondisi dipakai     : {info['kondisi_dipakai']} (dikeluarkan: {info['kondisi_dikeluarkan']})")
    print(f"latih : {len(info['latih'])} ulangan -> {info['latih']}")
    print(f"val   : {len(info['val'])} ulangan -> {info['val']}")
    print(f"uji   : {len(info['uji'])} ulangan -> {info['uji']}")
    print()
    print(f"irisan kondisi latih-val: {info['irisan_latih_val'] or 'kosong'}")
    print(f"irisan kondisi latih-uji: {info['irisan_latih_uji'] or 'kosong'}")
    print(f"irisan kondisi val-uji  : {info['irisan_val_uji'] or 'kosong'}")
