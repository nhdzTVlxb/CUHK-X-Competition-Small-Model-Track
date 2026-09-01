from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_skel_imu_v2 import IMU_DIM, SequenceBranch  # noqa: E402

PAMAP2_CLASSES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    24: "rope_jumping",
}

# heart rate + acc/gyro from hand, chest, and ankle = 19 channels.
SELECTED_COLUMNS = [
    2,
    7,
    8,
    9,
    10,
    11,
    12,
    24,
    25,
    26,
    27,
    28,
    29,
    41,
    42,
    43,
    44,
    45,
    46,
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_extracted(data_dir: Path, zip_path: Path) -> Path:
    protocol = data_dir / "PAMAP2_Dataset" / "Protocol"
    if protocol.exists() and list(protocol.glob("subject*.dat")):
        return protocol
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing PAMAP2 zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    nested = data_dir / "PAMAP2_Dataset.zip"
    if nested.exists() and not protocol.exists():
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(data_dir)
    if not protocol.exists():
        raise FileNotFoundError(f"Could not find extracted Protocol directory under {data_dir}")
    return protocol


def read_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep=r"\s+", header=None, na_values="NaN", engine="python")
    activity = frame.iloc[:, 1].to_numpy(dtype=np.int64)
    sensor = frame.iloc[:, SELECTED_COLUMNS].astype(np.float32)
    sensor = sensor.interpolate(limit_direction="both").fillna(0.0)
    return sensor.to_numpy(dtype=np.float32), activity


def make_windows(
    protocol_dir: Path,
    subjects: set[int],
    window: int,
    stride: int,
    min_purity: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(sorted(PAMAP2_CLASSES))}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for path in sorted(protocol_dir.glob("subject*.dat")):
        subject = int(path.stem.replace("subject", ""))
        if subject not in subjects:
            continue
        sensor, activity = read_subject(path)
        for start in range(0, max(0, len(activity) - window + 1), stride):
            labels = activity[start : start + window]
            labels = labels[np.isin(labels, list(PAMAP2_CLASSES))]
            if len(labels) < int(window * min_purity):
                continue
            values, counts = np.unique(labels, return_counts=True)
            best = int(values[counts.argmax()])
            if counts.max() / window < min_purity:
                continue
            xs.append(sensor[start : start + window])
            ys.append(label_to_index[best])
    if not xs:
        raise ValueError("No PAMAP2 windows were produced")
    return xs, np.asarray(ys, dtype=np.int64)


class Pamap2Dataset(Dataset):
    def __init__(self, x: list[np.ndarray], y: np.ndarray, mean: np.ndarray, std: np.ndarray, train: bool):
        self.x = x
        self.y = y
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.train = train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = self.x[index].astype(np.float32, copy=True)
        if self.train:
            if random.random() < 0.35:
                x += np.random.normal(0.0, 0.025, x.shape).astype(np.float32)
            if random.random() < 0.20:
                x = x[::-1].copy()
            if random.random() < 0.20:
                col = random.randrange(IMU_DIM)
                x[:, col] = 0.0
            if random.random() < 0.15:
                start = random.randrange(max(1, x.shape[0] - 8))
                x[start : start + 8] = 0.0
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(int(self.y[index]), dtype=torch.long)


class Pamap2Model(nn.Module):
    def __init__(self, width: int = 160, dropout: float = 0.20, classes: int = len(PAMAP2_CLASSES)):
        super().__init__()
        self.imu = SequenceBranch(IMU_DIM, width, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.imu(x))


def run_epoch(model, loader, criterion, optimizer, device, scaler):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = total = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
        total_loss += float(loss.detach()) * len(y)
        correct += int((logits.argmax(1) == y).sum())
        total += len(y)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "external_data" / "pamap2")
    parser.add_argument("--zip-path", type=Path, default=ROOT / "external_data" / "pamap2" / "pamap2_physical_activity_monitoring.zip")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--min-purity", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pamap2_imu_pretrain.pt")
    args = parser.parse_args()

    seed_all(args.seed)
    protocol_dir = ensure_extracted(args.data_dir, args.zip_path)
    train_x, train_y = make_windows(protocol_dir, set(range(101, 108)), args.window, args.stride, args.min_purity)
    val_x, val_y = make_windows(protocol_dir, {108, 109}, args.window, args.stride, args.min_purity)
    mean = np.concatenate(train_x, axis=0).mean(axis=0)
    std = np.concatenate(train_x, axis=0).std(axis=0).clip(1e-4)

    train_loader = DataLoader(
        Pamap2Dataset(train_x, train_y, mean, std, True),
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        Pamap2Dataset(val_x, val_y, mean, std, False),
        args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pamap2Model(args.width).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.03)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history: list[dict[str, float]] = []
    print(f"pamap2_windows train={len(train_x)} val={len(val_x)} classes={len(PAMAP2_CLASSES)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(f"epoch={epoch:02d} train={train_loss:.4f}/{train_acc:.4f} val={val_loss:.4f}/{val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save(
                {
                    "imu": model.imu.state_dict(),
                    "params": {"width": args.width, "dropout": 0.20, "window": args.window},
                    "mean": mean,
                    "std": std,
                    "best_val_acc": best,
                    "source": "UCI PAMAP2 Physical Activity Monitoring, Protocol dat files",
                    "channels": "heart_rate + hand/chest/ankle accelerometer(6g) and gyroscope",
                    "class_names": PAMAP2_CLASSES,
                },
                args.output,
            )

    args.output.with_name(args.output.stem + "_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"device={device} best_val={best:.6f} checkpoint={args.output}")


if __name__ == "__main__":
    main()
