"""Pemuat data kasus tim 14: deteksi anomali suara mesin.

Protokol yang dipakai:

- audio WAV 16 kHz dari folder sesi simulasi;
- buang 1 detik pertama setiap kondisi;
- potong jendela 1 detik dengan hop 0,5 detik;
- ubah menjadi log-mel 64 pita, 100 bingkai;
- latih hanya potongan normal;
- validasi dan uji memuat normal dan anomali;
- normalisasi memakai statistik data latih saja;
- pemisahan dilakukan per ``sesi_id``, bukan per berkas.

Kontrak mengikuti starter:

    TUGAS = "klasifikasi"
    muat() -> (X_train, y_train, X_val, y_val, X_test, y_test)
"""
from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np

from src.common import manifest

TUGAS = "klasifikasi"

SESI = manifest.AKAR / "sesi" / "t14-anomali-suara_simulasi"
AUDIO = SESI / "audio"

SR = 16000
N_MELS = 64
N_FFT = 1024
WIN_LENGTH = 400
HOP_LENGTH = 160
N_FRAME = 100
HOP_WINDOW = SR // 2


def _filterbank_mel() -> np.ndarray:
    """Bangun filter triangular 64 pita dari 0 sampai Nyquist."""
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


_FB = _filterbank_mel()


def _baca_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1:
            raise ValueError(f"{path}: audio harus mono, ditemukan {f.getnchannels()} kanal")
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: audio harus PCM 16-bit")
        sr = f.getframerate()
        data = f.readframes(f.getnframes())
    x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


def _logmel(segmen: np.ndarray) -> np.ndarray:
    """Ubah satu jendela 1 detik menjadi log-mel [100, 64]."""
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


def _baca_index() -> list[dict[str, str]]:
    path = AUDIO / "index.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} belum ada. Jalankan: python meja.py --simulasi jalankan ../rencana/t14-anomali-suara.yaml"
        )
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _bagi_sesi(rows: list[dict[str, str]]) -> tuple[set[str], set[str], set[str]]:
    """Bagi sesi secara lepas: 3 slot + 1 harian latih, sisanya val/uji."""
    slot = sorted({r["sesi_id"] for r in rows if r.get("jenis") == "slot"})
    harian = sorted({r["sesi_id"] for r in rows if r.get("jenis") == "harian"})
    if len(slot) < 5:
        raise ValueError(f"Butuh minimal 5 sesi slot, ditemukan {len(slot)}")
    if len(harian) < 4:
        raise ValueError(f"Butuh minimal 4 sesi harian, ditemukan {len(harian)}")

    latih = set(slot[:3]) | set(harian[:1])
    val = set(slot[3:4]) | set(harian[1:3])
    uji = set(slot[4:5]) | set(harian[3:4])

    for nama_a, a in (("latih", latih), ("val", val), ("uji", uji)):
        for nama_b, b in (("latih", latih), ("val", val), ("uji", uji)):
            if nama_a < nama_b and a & b:
                raise ValueError(f"Sesi bocor antara {nama_a} dan {nama_b}: {sorted(a & b)}")
    return latih, val, uji


def _muat_split(
    rows: list[dict[str, str]],
    sesi: set[str],
    hanya_normal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[int] = []
    for row in rows:
        if row["sesi_id"] not in sesi:
            continue
        label = int(row["label"])
        if hanya_normal and label != 0:
            continue
        x, sr = _baca_wav(AUDIO / row["berkas"])
        if sr != SR:
            raise ValueError(f"{row['berkas']}: sample rate {sr}, seharusnya {SR}")
        x = x[SR:]  # buang 1 detik pertama
        for awal in range(0, len(x) - SR + 1, HOP_WINDOW):
            X.append(_logmel(x[awal:awal + SR]))
            y.append(label)
    if not X:
        raise ValueError(f"Tidak ada jendela audio untuk sesi {sorted(sesi)}")
    return np.stack(X), np.asarray(y, dtype=np.int64)


def _muat_detail() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    dict[str, list[str]],
]:
    rows = _baca_index()
    sesi_latih, sesi_val, sesi_uji = _bagi_sesi(rows)

    Xtr, ytr = _muat_split(rows, sesi_latih, hanya_normal=True)
    Xva, yva = _muat_split(rows, sesi_val, hanya_normal=False)
    Xte, yte = _muat_split(rows, sesi_uji, hanya_normal=False)

    if np.unique(ytr).tolist() != [0]:
        raise ValueError("Data latih harus hanya berisi normal (label 0).")
    if len(np.unique(yva)) < 2:
        raise ValueError("Data validasi harus memuat normal dan anomali.")
    if len(np.unique(yte)) < 2:
        raise ValueError("Data uji harus memuat normal dan anomali.")

    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xte = (Xte - mu) / sd

    info = {
        "latih": sorted(sesi_latih),
        "val": sorted(sesi_val),
        "uji": sorted(sesi_uji),
    }
    return Xtr, ytr, Xva, yva, Xte, yte, info


def muat():
    Xtr, ytr, Xva, yva, Xte, yte, _ = _muat_detail()
    return Xtr, ytr, Xva, yva, Xte, yte


def periksa_split() -> dict[str, object]:
    """Periksa bahwa tidak ada sesi yang muncul di dua split."""
    latih, val, uji = _bagi_sesi(_baca_index())
    return {
        "latih": sorted(latih),
        "val": sorted(val),
        "uji": sorted(uji),
        "irisan_latih_val": sorted(latih & val),
        "irisan_latih_uji": sorted(latih & uji),
        "irisan_val_uji": sorted(val & uji),
    }


if __name__ == "__main__":
    info = periksa_split()
    print(f"latih : {len(info['latih'])} sesi -> {info['latih']}")
    print(f"val   : {len(info['val'])} sesi -> {info['val']}")
    print(f"uji   : {len(info['uji'])} sesi -> {info['uji']}")
    print(f"irisan latih-val: {info['irisan_latih_val'] or 'kosong'}")
    print(f"irisan latih-uji: {info['irisan_latih_uji'] or 'kosong'}")
    print(f"irisan val-uji  : {info['irisan_val_uji'] or 'kosong'}")
