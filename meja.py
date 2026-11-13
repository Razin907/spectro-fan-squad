"""Jalankan rencana eksperimen "meja" dari berkas YAML.

Contoh:

    python meja.py --simulasi jalankan rencana/t14-anomali-suara.yaml

Mode simulasi membuat daftar kondisi dan folder sesi.
Mode ini tidak merekam audio dan tidak menggerakkan perangkat keras.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import wave
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - pesan jelas bila PyYAML tidak ada
    yaml = None

AKAR = Path(__file__).resolve().parent
FOLDER_SESI = AKAR / "sesi"


def _cari_rencana(nama: str) -> Path:
    """Cari berkas rencana di jalur langsung atau folder rencana/."""
    jalur = Path(nama)
    kandidat = [jalur] if jalur.is_absolute() else [
        Path.cwd() / jalur,
        AKAR / jalur,
        AKAR / "rencana" / jalur.name,
    ]
    for k in kandidat:
        if k.is_file():
            return k.resolve()
    raise SystemExit(f"Rencana tidak ditemukan: {nama}")


def _muat_rencana(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit("PyYAML belum terpasang. Jalankan: pip install PyYAML")
    isi = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(isi, dict):
        raise SystemExit(f"Rencana harus berupa objek YAML: {path}")
    return isi


def _faktor_rencana(plan: dict[str, Any]) -> tuple[list[str], list[list[Any]], dict[str, str]]:
    faktor = plan.get("faktor")
    if not isinstance(faktor, dict) or not faktor:
        raise SystemExit("Rencana tidak punya bagian 'faktor'.")

    nama: list[str] = []
    taraf: list[list[Any]] = []
    template: dict[str, str] = {}
    for kunci, nilai in faktor.items():
        if isinstance(nilai, dict):
            daftar = nilai.get("taraf")
            if not isinstance(daftar, list):
                continue
            if nilai.get("instruksi"):
                template[kunci] = str(nilai["instruksi"])
        elif isinstance(nilai, list):
            daftar = nilai
        else:
            continue
        if not daftar:
            raise SystemExit(f"Faktor '{kunci}' tidak punya taraf.")
        nama.append(kunci)
        taraf.append(daftar)

    if not nama:
        raise SystemExit("Tidak ada faktor dengan daftar taraf pada rencana.")
    return nama, taraf, template


def _buat_kondisi(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[list[Any]]]:
    nama, taraf, template = _faktor_rencana(plan)
    ulangan = int(plan.get("ulangan", 1))
    durasi = float(plan.get("durasi_kondisi_detik", 6))
    jeda = float(plan.get("jeda_detik", 0.5))
    seed = int(plan.get("seed", 0))
    if ulangan < 1:
        raise SystemExit("'ulangan' minimal 1.")
    if durasi <= 0:
        raise SystemExit("'durasi_kondisi_detik' harus lebih dari 0.")
    if jeda < 0:
        raise SystemExit("'jeda_detik' tidak boleh negatif.")

    kombinasi = list(itertools.product(*taraf))
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    nomor = 0
    for blok in range(1, ulangan + 1):
        urutan = kombinasi.copy()
        rng.shuffle(urutan)
        for urut, nilai in enumerate(urutan, start=1):
            nomor += 1
            row: dict[str, Any] = {
                "kondisi_ke": nomor,
                "ulangan": blok,
                "urut_ulangan": urut,
            }
            row.update(dict(zip(nama, nilai)))
            row["instruksi"] = " | ".join(t.replace("{taraf}", str(row[k])) for k, t in template.items())
            row["durasi_detik"] = durasi
            row["jeda_detik"] = jeda
            row["mulai_relatif_detik"] = round((nomor - 1) * (durasi + jeda), 3)
            rows.append(row)
    return rows, nama, taraf


def _kolom_csv(rows: list[dict[str, Any]], nama_faktor: list[str]) -> list[str]:
    return ["kondisi_ke", "ulangan", "urut_ulangan", *nama_faktor, "instruksi",
            "durasi_detik", "jeda_detik", "mulai_relatif_detik"]


def _tulis_csv(path: Path, rows: list[dict[str, Any]], kolom: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _tulis_wav(path: Path, data: np.ndarray, sr: int) -> None:
    """Tulis satu kanal PCM 16-bit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def _audio_sintetis(
    seed: int,
    durasi: float,
    sr: int,
    kipas: Any,
    pompa: Any,
    gangguan: str,
) -> np.ndarray:
    """Bangkitkan campuran suara mesin dengan ciri gangguan yang berbeda."""
    rng = np.random.default_rng(seed)
    n = int(round(durasi * sr))
    t = np.arange(n, dtype=np.float32) / float(sr)
    x = 0.018 * rng.normal(0.0, 1.0, n).astype(np.float32)

    f_kipas = 42.0 + 0.38 * float(kipas)
    fase = float(rng.uniform(0.0, 2.0 * np.pi))
    x += (0.10 + 0.04 * float(kipas) / 220.0) * np.sin(2.0 * np.pi * f_kipas * t + fase)
    x += 0.04 * np.sin(2.0 * np.pi * 2.0 * f_kipas * t + 0.5 * fase)
    x += 0.012 * rng.normal(0.0, 1.0, n).astype(np.float32)

    if float(pompa) > 0:
        f_pompa = 30.0 + 0.03 * float(pompa)
        x += 0.08 * np.sin(2.0 * np.pi * f_pompa * t + fase / 2.0)
        x += 0.025 * np.sin(2.0 * np.pi * 2.0 * f_pompa * t + fase)
        x += 0.01 * rng.normal(0.0, 1.0, n).astype(np.float32)

    x *= 1.0 + 0.04 * np.sin(2.0 * np.pi * 0.6 * t + fase)

    if "kertas" in gangguan:
        for _ in range(6):
            awal = int(rng.integers(0, max(1, n - sr // 3)))
            panjang = int(rng.integers(sr // 50, sr // 10))
            if awal + panjang > n:
                panjang = n - awal
            if panjang <= 1:
                continue
            env = np.hanning(panjang).astype(np.float32)
            noise = rng.normal(0.0, 1.0, panjang).astype(np.float32)
            noise = np.diff(noise, prepend=np.float32(0.0))
            x[awal:awal + panjang] += 0.35 * noise * env

    elif "logam" in gangguan:
        for _ in range(2):
            awal = int(rng.integers(0, max(1, n - sr // 2)))
            panjang = min(sr // 2, n - awal)
            if panjang <= 1:
                continue
            tt = np.arange(panjang, dtype=np.float32) / float(sr)
            frekuensi = float(rng.uniform(1800.0, 5000.0))
            ring = 0.45 * np.exp(-tt * 10.0) * np.sin(2.0 * np.pi * frekuensi * tt + fase)
            x[awal:awal + panjang] += ring
            klik = rng.normal(0.0, 0.6, min(240, panjang)).astype(np.float32)
            x[awal:awal + len(klik)] += klik * np.hanning(len(klik)).astype(np.float32)

    elif "pompa" in gangguan:
        x += 0.14 * np.sin(2.0 * np.pi * 9.0 * t + fase) * (
            0.5 + 0.5 * np.sin(2.0 * np.pi * 0.8 * t)
        )
        derau = rng.normal(0.0, 1.0, n).astype(np.float32)
        kernel = np.ones(200, dtype=np.float32) / 200.0
        derau = np.convolve(derau, kernel, mode="same").astype(np.float32)
        x += 0.10 * derau

    return np.clip(x, -0.99, 0.99).astype(np.float32)


def _bangkitkan_audio(
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    folder: Path,
    seed: int,
) -> list[dict[str, Any]]:
    """Tulis WAV slot dan harian, lalu kembalikan isi audio/index.csv."""
    audio = folder / "audio"
    slot_dir = audio / "slot"
    harian_dir = audio / "harian"
    slot_dir.mkdir(parents=True, exist_ok=True)
    harian_dir.mkdir(parents=True, exist_ok=True)
    for lama in list(slot_dir.glob("*.wav")) + list(harian_dir.glob("*.wav")):
        lama.unlink()

    rekam = plan.get("rekam") or {}
    audio_cfg = rekam.get("audio") or {}
    sr = int(audio_cfg.get("sample_rate", 16000))
    durasi = float(plan.get("durasi_kondisi_detik", 6))
    faktor = plan.get("faktor") or {}
    kipas_levels = list(faktor.get("kipas") or [120, 220])
    pompa_levels = list(faktor.get("pompa") or [0, 200])
    index: list[dict[str, Any]] = []

    for row in rows:
        ulangan = int(row["ulangan"])
        nomor = int(row["kondisi_ke"])
        sesi_id = f"slot-{ulangan:02d}"
        nama = f"sesi-{ulangan:02d}_kondisi-{nomor:02d}.wav"
        gangguan = str(row["gangguan"])
        label = 0 if gangguan.strip().lower() == "normal" else 1
        data = _audio_sintetis(
            seed + nomor * 1009,
            durasi,
            sr,
            row["kipas"],
            row["pompa"],
            gangguan,
        )
        _tulis_wav(slot_dir / nama, data, sr)
        index.append({
            "berkas": f"slot/{nama}",
            "sesi_id": sesi_id,
            "jenis": "slot",
            "label": label,
            "kondisi_ke": nomor,
            "ulangan": ulangan,
            "gangguan": gangguan,
            "kipas": row["kipas"],
            "pompa": row["pompa"],
            "instruksi": row["instruksi"],
            "durasi_detik": durasi,
        })

    harian = plan.get("harian") or {}
    mulai = date.fromisoformat(str(harian.get("mulai", "2026-11-11")))
    jumlah_hari = int(harian.get("jumlah_hari", 4))
    file_per_hari = int(harian.get("file_per_hari", 6))
    for h in range(jumlah_hari):
        tanggal = mulai + timedelta(days=h)
        sesi_id = f"harian-{tanggal.isoformat()}"
        for f in range(file_per_hari):
            nama = f"{tanggal.isoformat()}_{f + 1:02d}.wav"
            kipas = kipas_levels[(h + f) % len(kipas_levels)]
            pompa = pompa_levels[(h * 2 + f) % len(pompa_levels)]
            data = _audio_sintetis(
                seed + 100000 + h * 100 + f,
                durasi,
                sr,
                kipas,
                pompa,
                "normal",
            )
            _tulis_wav(harian_dir / nama, data, sr)
            index.append({
                "berkas": f"harian/{nama}",
                "sesi_id": sesi_id,
                "jenis": "harian",
                "label": 0,
                "kondisi_ke": "",
                "ulangan": "",
                "gangguan": "normal",
                "kipas": kipas,
                "pompa": pompa,
                "instruksi": "Rekaman harian normal",
                "durasi_detik": durasi,
            })

    kolom = [
        "berkas", "sesi_id", "jenis", "label", "kondisi_ke", "ulangan",
        "gangguan", "kipas", "pompa", "instruksi", "durasi_detik",
    ]
    _tulis_csv(audio / "index.csv", index, kolom)
    return index


def _perbarui_manifest_simulasi(folder: Path, index: list[dict[str, Any]]) -> None:
    """Arahkan data/MANIFEST.json ke audio simulasi yang baru ditulis."""
    audio = folder / "audio"
    berkas = [audio / "index.csv", *sorted(audio.rglob("*.wav"))]
    try:
        relatif = [p.resolve().relative_to(AKAR.resolve()).as_posix() for p in berkas]
    except ValueError:
        print("[peringatan] folder sesi di luar repo; data/MANIFEST.json tidak diubah.")
        return
    isi = {
        "sumber": "simulasi meja.py --simulasi",
        "berkas": relatif,
        "catatan": (
            f"SINTETIS: {len(index)} rekaman, bukan data meja nyata. "
            "Ganti manifest ini saat data 11 Nov tersedia."
        ),
        "sintetis": True,
    }
    (AKAR / "data" / "MANIFEST.json").write_text(
        json.dumps(isi, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def jalankan(
    nama_rencana: str,
    simulasi: bool = True,
    out: str | None = None,
    tanpa_audio: bool = False,
) -> int:
    if not simulasi:
        raise SystemExit("meja.py hanya mendukung mode simulasi; sertakan --simulasi.")

    plan_path = _cari_rencana(nama_rencana)
    plan = _muat_rencana(plan_path)
    rows, nama_faktor, taraf = _buat_kondisi(plan)
    kolom = _kolom_csv(rows, nama_faktor)
    ulangan = int(plan.get("ulangan", 1))
    kombinasi = len(rows) // ulangan
    durasi = float(plan.get("durasi_kondisi_detik", 6))
    jeda = float(plan.get("jeda_detik", 0.5))
    audio_aktif = not tanpa_audio

    nama_folder = plan_path.stem + "_simulasi"
    folder = Path(out) if out else FOLDER_SESI / nama_folder
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audio").mkdir(exist_ok=True)

    jumlah_audio = 0
    if audio_aktif:
        index_audio = _bangkitkan_audio(plan, rows, folder, int(plan.get("seed", 0)))
        jumlah_audio = len(index_audio)
        _perbarui_manifest_simulasi(folder, index_audio)

    _tulis_csv(folder / "kondisi.csv", rows, kolom)
    (folder / "audio" / ".gitkeep").write_text("", encoding="utf-8")

    print(f"Tim {plan.get('tim', '?')} - {plan.get('judul', 'eksperimen')}")
    print(f"Rencana: {plan_path}")
    print("Faktor: " + ", ".join(f"{n} ({len(t)} taraf)" for n, t in zip(nama_faktor, taraf)))
    print(f"Kombinasi faktor: {kombinasi}")
    print(f"Ulangan: {ulangan}")
    print(f"Total: {len(rows)} kondisi")
    print()
    print(f"{'no':>3} | ul | " + " | ".join(f"{n:>10}" for n in nama_faktor) + " | instruksi")
    print("-" * 100)
    for row in rows:
        nilai = " | ".join(f"{str(row[n]):>10}" for n in nama_faktor)
        print(f"{row['kondisi_ke']:3d} | {row['ulangan']:2d} | {nilai} | {row['instruksi']}")

    print()
    print(f"Selesai: {len(rows)} kondisi siap dijalankan (mode simulasi).")
    if audio_aktif:
        print(f"Audio simulasi: {jumlah_audio} berkas WAV di {folder / 'audio'}")
        print("Manifest: data/MANIFEST.json diperbarui untuk audio simulasi.")
    print(f"Folder sesi: {folder}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--simulasi", action="store_true", help="wajib: jalankan mode simulasi")
    ap.add_argument("--out", default=None, help="folder sesi alternatif")
    ap.add_argument("--tanpa-audio", action="store_true", help="hanya tulis daftar kondisi")
    sub = ap.add_subparsers(dest="perintah", required=True)
    p = sub.add_parser("jalankan", help="buat daftar kondisi dan folder sesi simulasi")
    p.add_argument("rencana", help="berkas rencana YAML")
    p.add_argument("--simulasi", action="store_true", dest="simulasi_sub", help="wajib: jalankan mode simulasi")
    p.add_argument("--out", default=None, dest="out_sub", help="folder sesi alternatif")
    p.add_argument("--tanpa-audio", action="store_true", dest="tanpa_audio_sub", help="hanya tulis daftar kondisi")
    a = ap.parse_args()
    if a.perintah == "jalankan":
        simulasi = a.simulasi or a.simulasi_sub
        if not simulasi:
            ap.error("meja.py hanya mendukung mode simulasi; sertakan --simulasi.")
        out = a.out if a.out is not None else a.out_sub
        tanpa_audio = a.tanpa_audio or a.tanpa_audio_sub
        return jalankan(a.rencana, simulasi, out, tanpa_audio)
    return 1


if __name__ == "__main__":
    sys.exit(main())
