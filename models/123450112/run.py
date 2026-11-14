"""Transfer Learning: ResNet18 pretrained + One-Class SVM."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AKAR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AKAR))

from src.protokol import data

NIM = Path(__file__).resolve().parent.name
JALUR = "tl"


def siapkan_input(X):
    """Mengubah log-mel menjadi input 3-channel berukuran 224x224."""
    import torch
    import torch.nn.functional as F

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(
            f"Input harus berbentuk (N, waktu, 64), tetapi ditemukan {X.shape}"
        )

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalisasi setiap sampel ke rentang 0-1.
    minimum = X.min(axis=(1, 2), keepdims=True)
    maksimum = X.max(axis=(1, 2), keepdims=True)
    X = (X - minimum) / np.maximum(maksimum - minimum, 1e-8)

    tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    tensor = tensor.repeat(1, 3, 1, 1)
    tensor = F.interpolate(
        tensor, size=(224, 224), mode="bilinear", align_corners=False
    )

    # Normalisasi sesuai bobot pralatih ImageNet.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    return (tensor - mean) / std


def buat_model():
    """Memuat ResNet18 pralatih sebagai ekstraktor fitur."""
    import torch
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def ekstrak_fitur(model, X, batch_size=16):
    """Mengambil fitur dari setiap sampel log-mel."""
    import torch

    X = siapkan_input(X)
    hasil = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            fitur = model(batch)
            hasil.append(fitur.cpu().numpy())

    return np.concatenate(hasil, axis=0)


def latih_dan_prediksi(seed: int, final: bool):
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr, Xva, _, Xte, _ = data.muat()

    # Pastikan data training hanya berisi kelas normal.
    kelas_train = np.unique(ytr)
    if not np.all(kelas_train == 0):
        raise ValueError(
            f"Training seharusnya hanya berisi data normal (0), "
            f"tetapi ditemukan kelas {kelas_train}"
        )

    print("Memuat ResNet18 pretrained...")
    model = buat_model()

    print("Mengekstrak fitur data training...")
    fitur_train = ekstrak_fitur(model, Xtr)

    print("Mengekstrak fitur data validasi...")
    fitur_val = ekstrak_fitur(model, Xva)

    # Standardisasi fitur berdasarkan data training.
    scaler = StandardScaler()
    fitur_train = scaler.fit_transform(fitur_train)
    fitur_val = scaler.transform(fitur_val)

    # One-Class SVM dilatih menggunakan data normal saja.
    detektor = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )
    detektor.fit(fitur_train)

    # One-Class SVM: +1 = normal, -1 = anomali.
    # Prediksi kelas validasi
    pred_val = np.where(detektor.predict(fitur_val) == 1, 0, 1)

    # Skor kontinu untuk AUROC
    # Semakin tinggi skor, semakin kuat indikasi anomali
    skor_val = -detektor.decision_function(fitur_val)
    skor_val = skor_val.astype(np.float32)

    pred_test = None
    if final and Xte is not None:
        print("Mengekstrak fitur data testing...")
        fitur_test = ekstrak_fitur(model, Xte)
        fitur_test = scaler.transform(fitur_test)
        pred_test = np.where(detektor.predict(fitur_test) == 1, 0, 1)

    print("Prediksi validasi:", dict(zip(*np.unique(pred_val, return_counts=True))))

    return pred_val, pred_test, skor_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()
    pred_val, pred_test, skor_val = latih_dan_prediksi(a.seed, a.final)
    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "pred_val.npy", np.asarray(pred_val))
    np.save(out / "pred_val_skor.npy", np.asarray(skor_val))

    perintah = [
        sys.executable, "-m", "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat
    ]

    if pred_test is not None:
        np.save(out / "pred_test.npy", np.asarray(pred_test))
        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)


if __name__ == "__main__":
    main()
