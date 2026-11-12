"""Jalur Transformer: encoder kecil atas potongan (patch) waktu log-mel.

Kontrak tim tetap dipakai:

    python models/123450096/run.py --seed 1            # latih + prediksi validasi
    python models/123450096/run.py --seed 1 --final    # + prediksi data uji (sekali)

Masukan model mengikuti protokol tim: log-mel 64 pita, 1 detik. Setiap contoh
dipotong menjadi beberapa patch waktu, diproyeksikan ke dimensi model, lalu
diproses oleh Transformer encoder. Keluaran menyesuaikan ``src.protokol.data.TUGAS``:
klasifikasi -> satu logit per kelas, regresi -> satu nilai.

Data latih, validasi, dan uji dimuat langsung dari ``src.protokol.data.muat()``
(memakai sesi aktif protokol tim tanpa data sintetis cadangan).
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))
from src.protokol import data  # noqa: E402  (protokol tim yang dibekukan)

NIM = Path(__file__).resolve().parent.name     # nama folder = NIM
JALUR = "trf"                                   # cnn | tl | rnn | trf | ae
N_MELS = 64                                     # masukan tim: log-mel 64 pita


# ---------------------------------------------------------------------------
# Utilitas data
# ---------------------------------------------------------------------------
def _rapikan_input(X: np.ndarray, n_mels: int = N_MELS) -> np.ndarray:
    """Kembalikan X berbentuk [N, waktu, pita] dari beberapa bentuk yang lazim."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4:
        if X.shape[1] == 1:                 # [N, 1, pita, waktu]
            X = X[:, 0]
        elif X.shape[-1] == 1:              # [N, pita, waktu, 1]
            X = X[..., 0]
        else:
            raise ValueError(f"bentuk X 4D tidak dikenali: {X.shape}")
    if X.ndim == 2:
        if X.shape[1] == n_mels:            # satu bingkai per contoh
            X = X[:, None, :]
        elif X.shape[1] % n_mels == 0:      # hasil ratakan [N, pita*waktu]
            X = X.reshape(len(X), n_mels, -1).transpose(0, 2, 1)
        else:
            raise ValueError(
                f"X 2D berukuran {X.shape[1]} fitur; tidak bisa dipecah menjadi {n_mels} pita"
            )
    if X.ndim != 3:
        raise ValueError(f"bentuk X harus 2D/3D/4D, diterima {X.ndim}D: {X.shape}")

    # [N, pita, waktu] -> [N, waktu, pita]; [N, waktu, pita] dibiarkan.
    if X.shape[2] == n_mels:
        pass
    elif X.shape[1] == n_mels:
        X = X.transpose(0, 2, 1)
    elif X.shape[2] < X.shape[1]:
        pass
    elif X.shape[1] < X.shape[2]:
        X = X.transpose(0, 2, 1)
    else:
        raise ValueError(f"tidak bisa menentukan sumbu pita/waktu pada X {X.shape}")
    return np.ascontiguousarray(X)


def _muat_data() -> tuple[np.ndarray, ...]:
    """Muat data protokol langsung dari src.protokol.data.muat()."""
    hasil = data.muat()
    if not isinstance(hasil, (tuple, list)) or len(hasil) != 6:
        raise ValueError("data.muat() harus mengembalikan 6 array")
    if hasil[0] is None or hasil[2] is None:
        raise ValueError("X_train dan X_val tidak boleh None")
    return tuple(hasil)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PatchTransformer(nn.Module):
    """Transformer encoder kecil di atas patch waktu log-mel."""

    def __init__(
        self,
        n_mels: int,
        n_keluaran: int,
        n_frame: int,
        d_model: int = 64,
        nhead: int = 4,
        nlayer: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        patch: int = 10,
        pakai_posisi: bool = True,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) harus habis dibagi nhead ({nhead})")
        if patch < 1:
            raise ValueError("ukuran patch minimal 1")

        self.patch = int(patch)
        self.pakai_posisi = pakai_posisi
        self.embed = nn.Conv1d(n_mels, d_model, kernel_size=self.patch, stride=self.patch)
        n_token = max(1, (int(n_frame) + self.patch - 1) // self.patch)
        self.pos = nn.Parameter(torch.zeros(1, n_token, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        lapisan = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            lapisan, num_layers=nlayer, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_keluaran)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, waktu, pita] -> [N, pita, waktu] untuk Conv1d patch.
        x = x.transpose(1, 2)
        sisa = x.size(-1) % self.patch
        if sisa:
            x = F.pad(x, (0, self.patch - sisa))
        x = self.embed(x).transpose(1, 2)          # [N, token, d_model]

        n_token = x.size(1)
        if n_token > self.pos.size(1):
            pos = F.interpolate(
                self.pos.transpose(1, 2), size=n_token, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos[:, :n_token]
        if self.pakai_posisi:
            x = x + pos

        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(self.dropout(x))


# ---------------------------------------------------------------------------
# Latih dan prediksi
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prediksi_logits(model: nn.Module, X: torch.Tensor, batch: int) -> torch.Tensor:
    model.eval()
    keluaran = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            keluaran.append(model(X[i:i + batch]))
    return torch.cat(keluaran, dim=0)


def _latih(
    model: nn.Module,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    tugas: str,
    n_kelas: int,
    a: argparse.Namespace,
) -> tuple[nn.Module, list[float]]:
    from src.common.evaluate import metrik_klasifikasi

    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr)).float()
    Xva_t = torch.from_numpy(np.ascontiguousarray(Xva)).float()
    if tugas == "regresi":
        ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.float32).reshape(-1))
        yva_t = torch.from_numpy(np.asarray(yva, dtype=np.float32).reshape(-1))
    else:
        ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.int64).reshape(-1))
        yva_t = torch.from_numpy(np.asarray(yva, dtype=np.int64).reshape(-1))

    bobot = None
    if tugas != "regresi" and not a.tanpa_bobot_kelas:
        jumlah = np.bincount(ytr_t.numpy(), minlength=n_kelas).astype(np.float32)
        bobot_np = jumlah.sum() / (len(jumlah) * np.maximum(jumlah, 1.0))
        bobot = torch.tensor(np.clip(bobot_np, 0.0, 20.0), dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    jadwal = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.epoch), eta_min=a.lr * 0.05
    )
    rng = np.random.default_rng(a.seed)
    riwayat: list[float] = []
    terbaik = -float("inf")
    patience = 0
    state_terbaik = copy.deepcopy(model.state_dict())

    for epoch in range(1, a.epoch + 1):
        model.train()
        urut = rng.permutation(len(Xtr_t))
        total_loss = 0.0
        for i in range(0, len(urut), a.batch):
            idx = torch.from_numpy(urut[i:i + a.batch])
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            if tugas == "regresi":
                loss = F.mse_loss(logits.squeeze(1), ytr_t[idx])
            else:
                loss = F.cross_entropy(
                    logits,
                    ytr_t[idx],
                    weight=bobot,
                    label_smoothing=a.label_smoothing,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(idx)

        model.eval()
        with torch.no_grad():
            logits_va = _prediksi_logits(model, Xva_t, a.batch)
            if tugas == "regresi":
                loss_va = float(F.mse_loss(logits_va.squeeze(1), yva_t).item())
                skor = -loss_va
            else:
                loss_va = float(
                    F.cross_entropy(logits_va, yva_t, weight=bobot).item()
                )
                skor = metrik_klasifikasi(yva_t.numpy(), logits_va.argmax(1).numpy())["macro_f1"]
        jadwal.step()
        riwayat.append(loss_va)

        if epoch == 1 or epoch % a.cetak == 0:
            print(
                f"  epoch {epoch:3d}/{a.epoch}  loss_latih={total_loss / len(urut):.4f}  "
                f"loss_val={loss_va:.4f}  skor_val={skor:.4f}",
                flush=True,
            )
        if skor > terbaik + 1e-5:
            terbaik = skor
            patience = 0
            state_terbaik = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= a.patience:
                print(f"  berhenti awal pada epoch {epoch} (patience {a.patience})", flush=True)
                break

    model.load_state_dict(state_terbaik)
    return model, riwayat


def _prediksi(model: nn.Module, X: np.ndarray, tugas: str, batch: int) -> np.ndarray:
    X_t = torch.from_numpy(np.ascontiguousarray(X)).float()
    logits = _prediksi_logits(model, X_t, batch)
    if tugas == "regresi":
        return logits.squeeze(1).numpy().astype(np.float32)
    return logits.argmax(1).numpy().astype(np.int64)


def _prediksi_skor(model: nn.Module, X: np.ndarray, tugas: str, batch: int) -> np.ndarray:
    """Simpan skor kontinu untuk AUROC: 1 - P(kelas normal)."""
    X_t = torch.from_numpy(np.ascontiguousarray(X)).float()
    logits = _prediksi_logits(model, X_t, batch)
    if tugas == "regresi" or logits.shape[1] == 1:
        return logits.squeeze(1).numpy().astype(np.float32)
    prob = torch.softmax(logits, dim=1)
    return (1.0 - prob[:, 0]).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Penilaian (selalu lewat src.common.evaluate)
# ---------------------------------------------------------------------------
def _jalankan_evaluasi(
    out: Path,
    pred_test: np.ndarray | None,
    seed: int,
    durasi: float,
    perangkat: str,
) -> None:
    argumen = [
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", perangkat,
    ]
    if pred_test is not None:
        argumen += ["--pred-test", str(out / "pred_test.npy")]

    lingkungan = os.environ.copy()
    # Penilai mencetak panah "→"; konsol Windows lama memakai cp1252.
    lingkungan["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(
        [sys.executable, "-m", "src.common.evaluate", *argumen],
        check=True,
        cwd=AKAR,
        env=lingkungan,
    )


# ---------------------------------------------------------------------------
# Program utama
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true", help="sprint terakhir: evaluasi data uji sekali")
    ap.add_argument("--perangkat", default="cpu laptop")

    # Arsitektur dan pelatihan (bisa diubah untuk pencarian hyperparameter/ablation).
    ap.add_argument("--patch", type=int, default=10, help="jumlah bingkai per patch waktu")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayer", type=int, default=2)
    ap.add_argument("--dim-ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cetak", type=int, default=1, help="cetak log tiap N epoch")
    ap.add_argument("--tanpa-posisi", action="store_true", help="ablation: tanpa positional encoding")
    ap.add_argument("--tanpa-bobot-kelas", action="store_true", help="ablation: tanpa class weighting")
    a = ap.parse_args()

    _set_seed(a.seed)
    t0 = time.time()
    tugas = getattr(data, "TUGAS", "klasifikasi")
    Xtr, ytr, Xva, yva, Xte, yte = _muat_data()

    Xtr = _rapikan_input(Xtr, n_mels=N_MELS)
    Xva = _rapikan_input(Xva, n_mels=N_MELS)
    if Xte is not None:
        Xte = _rapikan_input(Xte, n_mels=N_MELS)

    n_mels = Xtr.shape[2]
    if n_mels != N_MELS:
        print(f"[ingat] jumlah pita {n_mels}, protokol tim menyebut {N_MELS}.", flush=True)

    if tugas == "regresi":
        ytr = np.asarray(ytr, dtype=np.float32).reshape(-1)
        yva = np.asarray(yva, dtype=np.float32).reshape(-1)
        n_keluaran = 1
    else:
        ytr = np.asarray(ytr, dtype=np.int64).reshape(-1)
        yva = np.asarray(yva, dtype=np.int64).reshape(-1)
        n_keluaran = max(2, int(max(ytr.max(initial=0), yva.max(initial=0))) + 1)

    model = PatchTransformer(
        n_mels=n_mels,
        n_keluaran=n_keluaran,
        n_frame=Xtr.shape[1],
        d_model=a.d_model,
        nhead=a.nhead,
        nlayer=a.nlayer,
        dim_ff=a.dim_ff,
        dropout=a.dropout,
        patch=a.patch,
        pakai_posisi=not a.tanpa_posisi,
    )
    jumlah_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[nyata] tugas={tugas}  X_latih={Xtr.shape}  X_val={Xva.shape}  "
        f"keluaran={n_keluaran}  parameter={jumlah_parameter:,}",
        flush=True,
    )

    model, _ = _latih(model, Xtr, ytr, Xva, yva, tugas, n_keluaran, a)
    pred_val = _prediksi(model, Xva, tugas, a.batch)
    skor_val = _prediksi_skor(model, Xva, tugas, a.batch)
    pred_test = None
    skor_test = None
    if a.final and Xte is not None and yte is not None:
        pred_test = _prediksi(model, Xte, tugas, a.batch)
        skor_test = _prediksi_skor(model, Xte, tugas, a.batch)

    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    np.save(out / "pred_val_skor.npy", np.asarray(skor_val))
    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        np.save(out / "pred_test_skor.npy", np.asarray(skor_test))
    _jalankan_evaluasi(out, pred_test, a.seed, durasi, a.perangkat)


if __name__ == "__main__":
    main()
