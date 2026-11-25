"""Jalur Transformer: Masked Patch Reconstruction Transformer atas log-mel 64 pita.

Kontrak tim tetap dipakai:

    python models/123450096/run.py --seed 1            # latih + prediksi validasi
    python models/123450096/run.py --seed 1 --final    # + prediksi data uji (sekali)

Masukan model mengikuti protokol tim: log-mel 64 pita, 1 detik ([N, 100, 64]).
Karena data latih protokol hanya berisi bunyi normal (label 0), model dilatih
secara one-class / self-supervised dengan Masked Patch Modeling:
- sumbu waktu dipotong menjadi patch (10 bingkai = 100 ms per patch);
- saat pelatihan, sebagian patch diganti token [MASK] terlatih lalu Transformer
  encoder merekonstruksi patch yang di-mask dari konteks patch lainnya;
- saat evaluasi (deterministik), tiap patch di-mask bergiliran (leave-one-patch-out)
  dan rata-rata galat rekonstruksi MSE menjadi skor anomali kontinu (pred_val_skor.npy);
- ambang klasifikasi biner (pred_val.npy) ditetapkan dari persentil skor data latih
  normal saja (tanpa kebocoran data validasi/uji).
"""
from __future__ import annotations

import argparse
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
# Model: Masked Patch Reconstruction Transformer
# ---------------------------------------------------------------------------
class PatchTransformer(nn.Module):
    """Transformer encoder untuk rekonstruksi patch waktu log-mel yang di-mask."""

    def __init__(
        self,
        n_mels: int,
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

        self.n_mels = int(n_mels)
        self.patch = int(patch)
        self.dim_patch = self.patch * self.n_mels
        self.pakai_posisi = pakai_posisi

        self.embed = nn.Linear(self.dim_patch, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

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
        self.decoder = nn.Linear(d_model, self.dim_patch)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Ubah [N, waktu, pita] -> [N, n_token, patch * pita]."""
        n, t, f = x.shape
        sisa = t % self.patch
        if sisa:
            x = F.pad(x, (0, 0, 0, self.patch - sisa))
            t = x.size(1)
        n_token = t // self.patch
        return x.reshape(n, n_token, self.patch * f)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Kembalikan (pred_patch, target_patch) masing-masing [N, n_token, patch * pita]."""
        target = self.patchify(x)                  # [N, T, patch * pita]
        h = self.embed(target)                     # [N, T, d_model]
        if mask is not None:
            mask_f = mask.unsqueeze(-1).to(h.dtype)
            h = h * (1.0 - mask_f) + self.mask_token * mask_f

        n_token = h.size(1)
        if n_token > self.pos.size(1):
            pos = F.interpolate(
                self.pos.transpose(1, 2), size=n_token, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            pos = self.pos[:, :n_token]
        if self.pakai_posisi:
            h = h + pos

        h = self.encoder(h)
        h = self.norm(h)
        pred = self.decoder(self.dropout(h))
        return pred, target


# ---------------------------------------------------------------------------
# Latih dan skor anomali
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _buat_mask_acak(
    batch_size: int,
    n_token: int,
    mask_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Buat mask biner [B, T] dengan tepat k patch di-mask per sampel."""
    k = max(1, min(n_token - 1, int(round(n_token * mask_ratio)))) if n_token > 1 else 1
    skor = torch.rand((batch_size, n_token), generator=generator)
    idx = skor.topk(k, dim=1).indices
    mask = torch.zeros((batch_size, n_token), dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _latih_one_class(
    model: PatchTransformer,
    Xtr: np.ndarray,
    a: argparse.Namespace,
) -> PatchTransformer:
    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr)).float()
    n_token = model.patchify(Xtr_t[:1]).size(1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    jadwal = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.epoch), eta_min=a.lr * 0.05
    )
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    for epoch in range(1, a.epoch + 1):
        model.train()
        urut = rng.permutation(len(Xtr_t))
        total_loss = 0.0
        for i in range(0, len(urut), a.batch):
            idx = torch.from_numpy(urut[i : i + a.batch])
            xb = Xtr_t[idx]
            if a.tanpa_mask:
                mask = None
            else:
                mask = _buat_mask_acak(len(xb), n_token, a.mask_ratio, gen)

            opt.zero_grad()
            pred, target = model(xb, mask=mask)
            if mask is not None:
                loss = F.mse_loss(pred[mask], target[mask])
            else:
                loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * len(idx)

        jadwal.step()
        if epoch == 1 or epoch == a.epoch or epoch % a.cetak == 0:
            print(
                f"  epoch {epoch:3d}/{a.epoch}  mse_latih={total_loss / len(urut):.4f}",
                flush=True,
            )

    return model


def _hitung_skor_anomali(
    model: PatchTransformer,
    X: np.ndarray,
    batch: int,
    tanpa_mask: bool = False,
) -> np.ndarray:
    """Skor anomali deterministik: rata-rata MSE rekonstruksi leave-one-patch-out."""
    model.eval()
    X_t = torch.from_numpy(np.ascontiguousarray(X)).float()
    skor_semua: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            xb = X_t[i : i + batch]
            if tanpa_mask:
                pred, target = model(xb, mask=None)
                err = ((pred - target) ** 2).mean(dim=(1, 2))
            else:
                n_token = model.patchify(xb[:1]).size(1)
                err_akum = torch.zeros(len(xb), dtype=torch.float32)
                for p in range(n_token):
                    mask = torch.zeros((len(xb), n_token), dtype=torch.bool)
                    mask[:, p] = True
                    pred, target = model(xb, mask=mask)
                    err_akum += ((pred[:, p, :] - target[:, p, :]) ** 2).mean(dim=1)
                err = err_akum / float(n_token)
            skor_semua.append(err.numpy().astype(np.float32))
    return np.concatenate(skor_semua, axis=0)


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
    # Penilai mencetak panah "->"; konsol Windows lama memakai cp1252.
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
    ap.add_argument("--epoch", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--mask-ratio", type=float, default=0.4, help="rasio patch yang di-mask saat latih")
    ap.add_argument("--persentil-ambang", type=float, default=95.0, help="persentil ambang dari skor latih normal")
    ap.add_argument("--cetak", type=int, default=10, help="cetak log tiap N epoch")
    ap.add_argument("--tanpa-posisi", action="store_true", help="ablation: tanpa positional encoding")
    ap.add_argument("--tanpa-mask", action="store_true", help="ablation: rekonstruksi tanpa masking")
    a = ap.parse_args()

    _set_seed(a.seed)
    t0 = time.time()
    Xtr, ytr, Xva, _, Xte, yte = _muat_data()

    # Pastikan hanya melatih pada data normal (sesuai protokol deteksi anomali).
    normal = np.asarray(ytr).reshape(-1) == 0
    Xtr = _rapikan_input(Xtr[normal], n_mels=N_MELS)
    Xva = _rapikan_input(Xva, n_mels=N_MELS)
    if Xte is not None:
        Xte = _rapikan_input(Xte, n_mels=N_MELS)

    n_mels = Xtr.shape[2]
    if n_mels != N_MELS:
        print(f"[ingat] jumlah pita {n_mels}, protokol tim menyebut {N_MELS}.", flush=True)

    model = PatchTransformer(
        n_mels=n_mels,
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
        f"[trf-mpm] X_latih_normal={Xtr.shape}  X_val={Xva.shape}  "
        f"mask_ratio={0.0 if a.tanpa_mask else a.mask_ratio:.2f}  parameter={jumlah_parameter:,}",
        flush=True,
    )

    model = _latih_one_class(model, Xtr, a)

    skor_tr = _hitung_skor_anomali(model, Xtr, a.batch, tanpa_mask=a.tanpa_mask)
    skor_va = _hitung_skor_anomali(model, Xva, a.batch, tanpa_mask=a.tanpa_mask)
    ambang = float(np.percentile(skor_tr, a.persentil_ambang))
    pred_val = (skor_va > ambang).astype(np.int64)

    pred_test = None
    skor_test = None
    if a.final and Xte is not None and yte is not None:
        skor_test = _hitung_skor_anomali(model, Xte, a.batch, tanpa_mask=a.tanpa_mask)
        pred_test = (skor_test > ambang).astype(np.int64)

    durasi = time.time() - t0
    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_val.npy", np.asarray(pred_val))
    np.save(out / "pred_val_skor.npy", np.asarray(skor_va))
    if pred_test is not None and skor_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        np.save(out / "pred_test_skor.npy", np.asarray(skor_test))
    _jalankan_evaluasi(out, pred_test, a.seed, durasi, a.perangkat)


if __name__ == "__main__":
    main()
