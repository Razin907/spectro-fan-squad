"""Jalur AE (Autoencoder): dilatih hanya pada data normal; skor anomali = reconstruction error.

    python models/123450013/run.py --seed 1 [--final]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "ae"
BOTTLENECK = 16
EPOCH = 30


def _skor(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(1).numpy()    # reconstruction error per jendela


def latih_dan_prediksi(seed: int, final: bool):
    """Kembalikan (pred_val, pred_test|None, skor_val, skor_test|None)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    def ke_tensor(X):
        return torch.tensor(np.asarray(X, dtype=np.float32)).reshape(len(X), -1)

    Xtr_t, Xva_t = ke_tensor(Xtr), ke_tensor(Xva)
    # Latih hanya pada data normal (protokol: data latih sudah normal; difilter lagi demi aman)
    normal = torch.tensor(np.asarray(ytr) == 0)
    Xn = Xtr_t[normal]

    d = Xn.shape[1]
    model = nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, BOTTLENECK),      # encoder
        nn.ReLU(), nn.Linear(BOTTLENECK, 256), nn.ReLU(), nn.Linear(256, d),  # decoder
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(EPOCH):
        model.train()
        for idx in torch.randperm(len(Xn)).split(32):
            opt.zero_grad()
            loss = ((model(Xn[idx]) - Xn[idx]) ** 2).mean()
            loss.backward()
            opt.step()

    s_tr = _skor(model, Xn)
    s_va = _skor(model, Xva_t)
    ambang = float(np.percentile(s_tr, 95))      # ambang dari data latih, bukan data uji
    pred_val = (s_va > ambang).astype(np.int64)  # 0 normal, 1 anomali
    pred_test = s_te = None
    if final and Xte is not None:
        s_te = _skor(model, ke_tensor(Xte))
        pred_test = (s_te > ambang).astype(np.int64)
    return pred_val, pred_test, s_va, s_te


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test, s_va, s_te = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    np.save(out / "pred_val_skor.npy", np.asarray(s_va))     # dibaca tools/nilai_auroc.py
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        np.save(out / "pred_test_skor.npy", np.asarray(s_te))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()