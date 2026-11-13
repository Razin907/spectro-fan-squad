"""Hitung AUROC dari skor prediksi tanpa mengubah penilai starter.

Contoh:

    python tools/nilai_auroc.py --nim 123450096
    python tools/nilai_auroc.py --nim 123450096 --final

Label diambil dari ``src.protokol.data.muat()``. Skor validasi diambil dari
``results/<nim>/pred_val_skor.npy`` yang ditulis oleh ``run.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AKAR))


def auroc(y: np.ndarray, skor: np.ndarray) -> float:
    """AUROC biner dengan rata-rata peringkat untuk nilai seri."""
    y = (np.asarray(y).reshape(-1) != 0).astype(np.int64)
    skor = np.asarray(skor, dtype=np.float64).reshape(-1)
    if len(y) != len(skor):
        raise ValueError(f"Panjang label {len(y)} != skor {len(skor)}")
    positif = y == 1
    negatif = y == 0
    n_pos = int(positif.sum())
    n_neg = int(negatif.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC butuh minimal satu normal dan satu anomali")

    urut = np.argsort(skor, kind="mergesort")
    nilai = skor[urut]
    peringkat = np.empty(len(skor), dtype=np.float64)
    i = 0
    while i < len(nilai):
        j = i + 1
        while j < len(nilai) and nilai[j] == nilai[i]:
            j += 1
        peringkat[urut[i:j]] = (i + 1 + j) / 2.0
        i = j

    jumlah_u = float(peringkat[positif].sum()) - n_pos * (n_pos + 1) / 2.0
    return jumlah_u / (n_pos * n_neg)


def _muat_skor(path: Path, n: int) -> np.ndarray:
    if not path.exists():
        raise SystemExit(
            f"{path} belum ada. Jalankan model lebih dulu: "
            f"python models/<nim>/run.py --seed 1"
        )
    skor = np.load(path).reshape(-1)
    if len(skor) != n:
        raise SystemExit(f"{path}: panjang {len(skor)}, seharusnya {n}")
    return skor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nim", required=True)
    ap.add_argument("--model", default="trf")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--final", action="store_true", help="hitung juga AUROC data uji")
    a = ap.parse_args()

    from src.protokol import data

    _, _, _, y_val, _, y_test = data.muat()
    out = AKAR / "results" / a.nim
    skor_val = _muat_skor(out / "pred_val_skor.npy", len(y_val))
    hasil: dict[str, object] = {
        "model": a.model,
        "seed": a.seed,
        "metrik": {"val_auroc": round(float(auroc(y_val, skor_val)), 6)},
        "n_val": int(len(y_val)),
        "n_pos_val": int((np.asarray(y_val).reshape(-1) != 0).sum()),
        "n_neg_val": int((np.asarray(y_val).reshape(-1) == 0).sum()),
        "sumber_skor": "pred_val_skor.npy",
    }
    if a.final:
        if y_test is None:
            raise SystemExit("Protokol tidak menyediakan y_test")
        skor_test = _muat_skor(out / "pred_test_skor.npy", len(y_test))
        hasil["metrik"]["test_auroc"] = round(float(auroc(y_test, skor_test)), 6)  # type: ignore[index]
        hasil["n_test"] = int(len(y_test))
        hasil["sumber_skor_test"] = "pred_test_skor.npy"

    if hasil["seed"] is None:
        metrics = out / "metrics.json"
        if metrics.exists():
            try:
                hasil["seed"] = json.loads(metrics.read_text(encoding="utf-8")).get("seed")
            except (OSError, json.JSONDecodeError):
                pass

    out.mkdir(parents=True, exist_ok=True)
    (out / "auroc.json").write_text(
        json.dumps(hasil, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(hasil["metrik"]), "->", out / "auroc.json")


if __name__ == "__main__":
    main()
