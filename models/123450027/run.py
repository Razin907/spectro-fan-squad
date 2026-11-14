"""Jalur CNN: jaringan konvolusi 2D pada log-mel (64 pita, 1 detik).

    python models/123450027/run.py --seed 1 [--final]
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
JALUR = "cnn"                                   # cnn | tl | rnn | trf | ae
N_MELS = 64                                     # masukan tim: log-mel 64 pita


def _rapikan(X, n_mels: int = N_MELS) -> np.ndarray:
    """Kembalikan array [N, 1, pita, waktu] untuk CNN 2D."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4:
        if X.shape[1] == 1:                     # [N, 1, pita, waktu]
            X = X[:, 0]
        elif X.shape[-1] == 1:                  # [N, pita, waktu, 1]
            X = X[..., 0]
        else:
            raise ValueError(f"bentuk X 4D tidak dikenali: {X.shape}")
    if X.ndim == 2:                             # hasil ratakan [N, pita*waktu]
        if X.shape[1] % n_mels != 0:
            raise ValueError(f"{X.shape[1]} fitur tidak bisa dipecah menjadi {n_mels} pita")
        X = X.reshape(len(X), n_mels, -1)
    if X.ndim != 3:
        raise ValueError(f"bentuk X harus 2D/3D/4D, diterima: {X.shape}")
    if X.shape[1] != n_mels and X.shape[2] == n_mels:   # [N, waktu, pita] -> [N, pita, waktu]
        X = X.transpose(0, 2, 1)
    return np.ascontiguousarray(X[:, None, :, :])


def latih_dan_prediksi(a):
    """Kembalikan (pred_val, pred_test|None, skor_val, skor_test|None)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    tugas = getattr(data, "TUGAS", "klasifikasi")
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    Xtr, Xva = _rapikan(Xtr), _rapikan(Xva)
    Xte = _rapikan(Xte) if Xte is not None else None

    # Standarisasi memakai mean/std data latih saja
    mu, sd = float(Xtr.mean()), float(Xtr.std()) + 1e-8
    ke_t = lambda X: torch.from_numpy((X - mu) / sd).float()
    Xtr_t, Xva_t = ke_t(Xtr), ke_t(Xva)

    if tugas == "regresi":
        n_keluaran = 1
        ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.float32).reshape(-1))
        bobot = None
    else:
        ytr = np.asarray(ytr, dtype=np.int64).reshape(-1)
        yva = np.asarray(yva, dtype=np.int64).reshape(-1)
        n_keluaran = max(2, int(max(ytr.max(initial=0), yva.max(initial=0))) + 1)
        ytr_t = torch.from_numpy(ytr)
        jumlah = np.bincount(ytr, minlength=n_keluaran).astype(np.float32)
        bobot = torch.tensor(np.clip(jumlah.sum() / (len(jumlah) * np.maximum(jumlah, 1.0)), 0.0, 20.0))
    print(f"[info] tugas={tugas}  X_latih={tuple(Xtr.shape)}  X_val={tuple(Xva.shape)}  "
          f"keluaran={n_keluaran}", flush=True)
    if tugas != "regresi":
        print(f"[info] jumlah per kelas (latih)={np.bincount(ytr).tolist()}", flush=True)

    def blok(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

    model = nn.Sequential(
        blok(1, 16), blok(16, 32), blok(32, 64),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(64, n_keluaran),
    )
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)    # tulis alasan pilihan optimizer di catatan.md

    for ep in range(1, a.epoch + 1):
        model.train()
        urutan = torch.randperm(len(Xtr_t))
        total = 0.0
        for i in range(0, len(urutan), a.batch):
            idx = urutan[i:i + a.batch]
            opt.zero_grad()
            keluar = model(Xtr_t[idx])
            if tugas == "regresi":
                loss = F.mse_loss(keluar.squeeze(1), ytr_t[idx])
            else:
                loss = F.cross_entropy(keluar, ytr_t[idx], weight=bobot)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        print(f"  epoch {ep:3d}/{a.epoch}  loss_latih={total / len(urutan):.4f}", flush=True)

    def keluaran(X_t):
        model.eval()
        hasil = []
        with torch.no_grad():
            for i in range(0, len(X_t), 256):
                hasil.append(model(X_t[i:i + 256]))
        return torch.cat(hasil)

    def prediksi_dan_skor(X_t):
        k = keluaran(X_t)
        if tugas == "regresi":
            nilai = k.squeeze(1).numpy().astype(np.float32)
            return nilai, nilai
        prob = torch.softmax(k, dim=1)
        pred = k.argmax(1).numpy().astype(np.int64)
        skor = (1.0 - prob[:, 0]).numpy().astype(np.float32)    # 1 - P(normal), makin tinggi makin anomali
        return pred, skor

    pred_val, skor_val = prediksi_dan_skor(Xva_t)
    pred_test = skor_test = None
    if a.final and Xte is not None:
        pred_test, skor_test = prediksi_dan_skor(ke_t(Xte))
    return pred_val, pred_test, skor_val, skor_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")
    ap.add_argument("--epoch", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    t0 = time.time()
    pred_val, pred_test, skor_val, skor_test = latih_dan_prediksi(a)
    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    np.save(out / "pred_val_skor.npy", np.asarray(skor_val))     # dibaca tools/nilai_auroc.py
    perintah = [sys.executable, "-m", "src.common.evaluate", "--nim", NIM, "--model", JALUR, "--seed", str(a.seed),
                "--pred-val", str(out / "pred_val.npy"), "--durasi", f"{durasi:.1f}", "--perangkat", a.perangkat]
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        np.save(out / "pred_test_skor.npy", np.asarray(skor_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]
    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()