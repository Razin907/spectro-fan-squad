
"""Transfer Learning: ResNet18 pretrained + One-Class SVM."""

from __future__ import annotations

import argparse
import json
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

    if X.ndim != 3 or X.shape[-1] != 64:
        raise ValueError(
            f"Input harus berbentuk (N, waktu, 64), ditemukan {X.shape}"
        )

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Rentang normalisasi ditentukan dari data training.
    minimum = float(X.min())
    maksimum = float(X.max())
    X = (X - minimum) / max(maksimum - minimum, 1e-8)

    tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    tensor = tensor.repeat(1, 3, 1, 1)

    tensor = F.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    return (tensor - mean) / std


def buat_model():
    """Memuat ResNet18 pretrained sebagai ekstraktor fitur."""
    import torch
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def ekstrak_fitur(model, X, minimum, maksimum, batch_size=16):
    """Mengambil fitur ResNet18 dari setiap sampel log-mel."""
    import torch
    import torch.nn.functional as F

    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalisasi menggunakan statistik training yang sama.
    X = (X - minimum) / max(maksimum - minimum, 1e-8)

    tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    tensor = tensor.repeat(1, 3, 1, 1)

    tensor = F.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    hasil = []

    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i + batch_size]
            fitur = model(batch)
            hasil.append(fitur.cpu().numpy())

    return np.concatenate(hasil, axis=0)


def latih_dan_prediksi(seed: int, final: bool):
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Memuat data primer sesuai protokol resmi Tim 14.
    Xtr, ytr, Xva, yva, Xte, yte = data.muat()

    print("Bentuk data:")
    print("Train:", Xtr.shape, "Label:", np.unique(ytr, return_counts=True))
    print("Validation:", Xva.shape, "Label:", np.unique(yva, return_counts=True))
    print("Test:", Xte.shape, "Label:", np.unique(yte, return_counts=True))

    if not np.all(np.unique(ytr) == 0):
        raise ValueError("Data training harus hanya berisi kelas normal (0).")

    if len(np.unique(yva)) != 2:
        raise ValueError("Data validation harus memiliki kelas normal dan anomali.")

    # Statistik normalisasi dihitung dari training saja.
    minimum = float(np.min(Xtr))
    maksimum = float(np.max(Xtr))

    print("\nMemuat ResNet18 pretrained...")
    model = buat_model()

    print("Mengekstrak fitur training...")
    fitur_train = ekstrak_fitur(model, Xtr, minimum, maksimum)

    print("Mengekstrak fitur validation...")
    fitur_val = ekstrak_fitur(model, Xva, minimum, maksimum)

    scaler = StandardScaler()
    fitur_train = scaler.fit_transform(fitur_train)
    fitur_val = scaler.transform(fitur_val)

    print("Melatih One-Class SVM...")
    detektor = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )
    detektor.fit(fitur_train)

    # Prediksi kelas: 0 = normal, 1 = anomali.
    pred_val = np.where(detektor.predict(fitur_val) == 1, 0, 1)

    # Skor lebih tinggi menunjukkan indikasi anomali lebih kuat.
    skor_val = -detektor.decision_function(fitur_val)
    skor_val = skor_val.astype(np.float32)

    if len(np.unique(yva)) == 2:
        auroc_val = float(roc_auc_score(yva, skor_val))
    else:
        auroc_val = None

    pred_test = None
    skor_test = None
    auroc_test = None

    if final:
        print("Mengekstrak fitur testing...")
        fitur_test = ekstrak_fitur(model, Xte, minimum, maksimum)
        fitur_test = scaler.transform(fitur_test)

        pred_test = np.where(detektor.predict(fitur_test) == 1, 0, 1)
        skor_test = -detektor.decision_function(fitur_test)
        skor_test = skor_test.astype(np.float32)

        if len(np.unique(yte)) == 2:
            auroc_test = float(roc_auc_score(yte, skor_test))

    print("\nHasil prediksi validation:")
    print(dict(zip(*np.unique(pred_val, return_counts=True))))
    print("Validation AUROC:", auroc_val)

    if auroc_test is not None:
        print("Test AUROC:", auroc_test)

    return pred_val, pred_test, skor_val, skor_test, auroc_val, auroc_test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--perangkat", default="cpu laptop")
    a = ap.parse_args()

    t0 = time.time()

    (
        pred_val,
        pred_test,
        skor_val,
        skor_test,
        auroc_val,
        auroc_test
    ) = latih_dan_prediksi(a.seed, a.final)

    durasi = time.time() - t0

    out = AKAR / "results" / NIM
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "pred_val.npy", pred_val)
    np.save(out / "pred_val_skor.npy", skor_val)

    perintah = [
        sys.executable,
        "-m",
        "src.common.evaluate",
        "--nim", NIM,
        "--model", JALUR,
        "--seed", str(a.seed),
        "--pred-val", str(out / "pred_val.npy"),
        "--durasi", f"{durasi:.1f}",
        "--perangkat", a.perangkat
    ]

    if pred_test is not None:
        np.save(out / "pred_test.npy", pred_test)
        np.save(out / "pred_test_skor.npy", skor_test)

        perintah += ["--pred-test", str(out / "pred_test.npy")]

    subprocess.run(perintah, check=True, cwd=AKAR)

    # Menyimpan AUROC secara terpisah karena evaluator starter
    # hanya menghitung Accuracy dan Macro-F1.
    hasil_auroc = {
        "val_auroc": auroc_val,
        "test_auroc": auroc_test,
        "seed": a.seed
    }

    (out / "auroc.json").write_text(
        json.dumps(hasil_auroc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )

    print("\nDurasi:", round(durasi, 2), "detik")
    print("Hasil AUROC disimpan di:", out / "auroc.json")


if __name__ == "__main__":
    main()
