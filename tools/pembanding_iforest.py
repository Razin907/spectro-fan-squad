"""Pembanding non-DL tim 14: Isolation Forest pada fitur MFCC.

Contoh:

    python tools/pembanding_iforest.py --seed 1
    python tools/pembanding_iforest.py --seed 1 --final

Menggunakan jendela audio 1 detik 16 kHz dan pemisahan ulangan yang persis sama
dengan protokol DL tim (``src.protokol.data.muat_segmen``), lalu mengekstrak
MFCC (DCT-II atas log-mel 64 pita) beserta delta orde pertama dan ringkasan
rata-rata/simpangan baku per jendela. Evaluasi dilakukan lewat ``src.common.evaluate``
dan ``tools/nilai_auroc.py`` ke ``results/baseline/``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

AKAR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402


def _matriks_dct(n_mels: int, n_mfcc: int) -> np.ndarray:
    """Matriks DCT-II ortonormal untuk mengubah log-mel menjadi MFCC."""
    n = np.arange(n_mels, dtype=np.float32)
    k = np.arange(n_mfcc, dtype=np.float32)[:, None]
    basis = np.cos(np.pi / n_mels * (n[None, :] + 0.5) * k)
    basis[0] *= np.sqrt(1.0 / n_mels)
    basis[1:] *= np.sqrt(2.0 / n_mels)
    return basis.astype(np.float32)


def _ekstrak_fitur_mfcc(segmen: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """Ekstrak ringkasan MFCC + delta (mean dan std per jendela -> 4 * n_mfcc fitur)."""
    dct = _matriks_dct(data.N_MELS, n_mfcc)
    fitur: list[np.ndarray] = []
    for w in segmen:
        lm = data._logmel(w)               # [100, 64]
        mfcc = lm @ dct.T                  # [100, n_mfcc]
        delta = np.diff(mfcc, axis=0, prepend=mfcc[:1])
        vektor = np.concatenate(
            [
                mfcc.mean(axis=0),
                mfcc.std(axis=0),
                delta.mean(axis=0),
                delta.std(axis=0),
            ],
            axis=0,
        )
        fitur.append(vektor)
    return np.stack(fitur).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nim", default="baseline")
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--n-mfcc", type=int, default=13)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    Wtr, ytr, Wva, _, Wte, _ = data.muat_segmen()
    normal = np.asarray(ytr) == 0
    Ftr = _ekstrak_fitur_mfcc(Wtr[normal], n_mfcc=a.n_mfcc)
    Fva = _ekstrak_fitur_mfcc(Wva, n_mfcc=a.n_mfcc)

    scaler = StandardScaler()
    Ftr_s = scaler.fit_transform(Ftr)
    Fva_s = scaler.transform(Fva)

    detektor = IsolationForest(
        n_estimators=a.n_estimators,
        contamination=a.contamination,
        random_state=a.seed,
    )
    detektor.fit(Ftr_s)

    pred_val = np.where(detektor.predict(Fva_s) == 1, 0, 1).astype(np.int64)
    skor_val = (-detektor.decision_function(Fva_s)).astype(np.float32)

    pred_test = None
    skor_test = None
    if a.final and Wte is not None:
        Fte_s = scaler.transform(_ekstrak_fitur_mfcc(Wte, n_mfcc=a.n_mfcc))
        pred_test = np.where(detektor.predict(Fte_s) == 1, 0, 1).astype(np.int64)
        skor_test = (-detektor.decision_function(Fte_s)).astype(np.float32)

    durasi = time.time() - t0
    out = AKAR / "results" / a.nim
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", pred_val)
    np.save(out / "pred_val_skor.npy", skor_val)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    cmd_eval = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", a.nim,
        "--model", a.model,
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat,
    ]
    cmd_auroc = [
        sys.executable, str(AKAR / "tools" / "nilai_auroc.py"),
        "--nim", a.nim,
        "--model", a.model,
        "--seed", str(a.seed),
    ]
    if pred_test is not None and skor_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        np.save(out / "pred_test_skor.npy", skor_test)
        cmd_eval += ["--pred-test", str(out / "pred_test.npy")]
        cmd_auroc.append("--final")

    subprocess.run(cmd_eval, check=True, cwd=AKAR, env=env)
    subprocess.run(cmd_auroc, check=True, cwd=AKAR, env=env)


if __name__ == "__main__":
    main()
