"""Satu perintah reproduksi tim.

    python run_all.py [--seeds 1 2 3] [--final] [--hanya <nim> ...]

Menjalankan pembanding non-DL tim dan ``models/<nim>/run.py`` setiap anggota untuk tiap seed,
memperbarui ``results/*/auroc.json``, lalu merakit ``reports/hasil_komparasi.md`` dari
``results/*/metrics.json`` dan ``results/*/auroc.json`` (hanya seed terakhir yang tersimpan di
berkas JSON; sebaran antar-seed dicatat dari keluaran tiap run).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent


def jalankan(perintah: list[str]) -> None:
    print("$", " ".join(perintah), flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(perintah, check=True, cwd=AKAR, env=env)


def _baca_auroc(path: Path) -> str:
    if not path.exists():
        return "—"
    try:
        d = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return "—"
    val = d["metrik"].get("val_auroc") if isinstance(d.get("metrik"), dict) else d.get("val_auroc")
    return f"{float(val):.4f}" if isinstance(val, (int, float)) else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--hanya", nargs="*", default=None, help="NIM tertentu saja")
    ap.add_argument("--tanpa-baseline", action="store_true")
    a = ap.parse_args()

    if not a.tanpa_baseline:
        perintah_bl = [
            sys.executable,
            str(AKAR / "tools" / "pembanding_iforest.py"),
            "--seed",
            str(a.seeds[0]),
        ]
        if a.final:
            perintah_bl.append("--final")
        jalankan(perintah_bl)

    anggota = sorted(
        p for p in (AKAR / "models").iterdir()
        if p.is_dir() and (p / "run.py").exists() and p.name != "TEMPLATE"
        and (a.hanya is None or p.name in a.hanya)
    )
    ringkas: dict[str, list[tuple[dict, str]]] = {}
    for m in anggota:
        for s in a.seeds:
            is_final = a.final and s == a.seeds[-1]
            perintah = [sys.executable, str(m / "run.py"), "--seed", str(s)]
            if is_final:
                perintah.append("--final")
            jalankan(perintah)

            res_dir = AKAR / "results" / m.name
            hasil = json.loads((res_dir / "metrics.json").read_text("utf-8"))
            if (res_dir / "pred_val_skor.npy").exists():
                cmd_auroc = [
                    sys.executable,
                    str(AKAR / "tools" / "nilai_auroc.py"),
                    "--nim",
                    m.name,
                    "--model",
                    str(hasil.get("model", m.name)),
                    "--seed",
                    str(s),
                ]
                if is_final and (res_dir / "pred_test_skor.npy").exists():
                    cmd_auroc.append("--final")
                jalankan(cmd_auroc)
            auroc_str = _baca_auroc(res_dir / "auroc.json")
            ringkas.setdefault(m.name, []).append((hasil, auroc_str))

    baris = [
        "| NIM | Model | seed | val_auroc | val_acc | val_macro_f1 | test_acc | durasi (s) | perangkat |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    data_hash = "—"
    bl = AKAR / "results" / "baseline" / "metrics.json"
    if bl.exists():
        h = json.loads(bl.read_text("utf-8"))
        mt = h["metrik"]
        data_hash = h.get("data_hash", data_hash)
        auroc_bl = _baca_auroc(AKAR / "results" / "baseline" / "auroc.json")
        baris.append(
            f"| baseline | iforest mfcc | {h['seed']} | {auroc_bl} | {mt['val_acc']:.4f} | {mt['val_macro_f1']:.4f} | "
            f"— | {h.get('durasi_detik', 0):.0f} | {h.get('perangkat', '')} |"
        )
    for nim, daftar in ringkas.items():
        for h, auroc_str in daftar:
            mt = h["metrik"]
            if data_hash == "—" and h.get("data_hash"):
                data_hash = h["data_hash"]
            baris.append(
                f"| {nim} | {h.get('model', '')} | {h['seed']} | {auroc_str} | {mt['val_acc']:.4f} | {mt['val_macro_f1']:.4f} | "
                f"{mt.get('test_acc', float('nan')):.4f} | {h.get('durasi_detik', 0):.0f} | {h.get('perangkat', '')} |"
            )

    (AKAR / "reports").mkdir(exist_ok=True)
    out = AKAR / "reports" / "hasil_komparasi.md"
    out.write_text(
        "# Hasil komparasi (dibangkitkan run_all.py)\n\n"
        f"`data_hash`: `{data_hash}`\n\n" + "\n".join(baris) + "\n",
        encoding="utf-8",
    )
    print("tabel komparasi →", out.relative_to(AKAR))


if __name__ == "__main__":
    main()
