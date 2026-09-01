from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_skel_imu_v2 import IMU_DIM, SequenceBranch  # noqa: E402


UCI_CHANNELS = 9
UCI_CLASSES = 6


class UciHarDataset(Dataset):
    def __init__(self, root: Path, split: str, mean: np.ndarray, std: np.ndarray, train: bool):
        self.root = root
        self.split = split
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.train = train
        base = root / "UCI HAR Dataset" / split
        signal_dir = base / "Inertial Signals"
        names = [
            "body_acc_x_", "body_acc_y_", "body_acc_z_",
            "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
            "total_acc_x_", "total_acc_y_", "total_acc_z_",
        ]
        channels = []
        for name in names:
            path = signal_dir / f"{name}{split}.txt"
            channels.append(np.loadtxt(path, dtype=np.float32))
        self.x = np.stack(channels, axis=-1)
        labels = np.loadtxt(base / f"y_{split}.txt", dtype=np.int64) - 1
        self.y = labels

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        x = self.x[index].copy()
        if self.train:
            if random.random() < 0.35:
                x = x + np.random.normal(0.0, 0.015, x.shape).astype(np.float32)
            if random.random() < 0.20:
                x = x[::-1].copy()
            if random.random() < 0.20:
                drop = random.randrange(UCI_CHANNELS)
                x[:, drop] = 0.0
        x = (x - self.mean) / self.std
        padded = np.zeros((x.shape[0], IMU_DIM), dtype=np.float32)
        padded[:, :UCI_CHANNELS] = x
        return torch.from_numpy(padded), torch.tensor(int(self.y[index]), dtype=torch.long)


class UciModel(nn.Module):
    def __init__(self, width: int = 160, dropout: float = 0.20):
        super().__init__()
        self.imu = SequenceBranch(IMU_DIM, width, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, UCI_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.imu(x))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_raw(root: Path, split: str) -> np.ndarray:
    signal_dir = root / "UCI HAR Dataset" / split / "Inertial Signals"
    names = [
        "body_acc_x_", "body_acc_y_", "body_acc_z_",
        "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
        "total_acc_x_", "total_acc_y_", "total_acc_z_",
    ]
    return np.stack(
        [np.loadtxt(signal_dir / f"{name}{split}.txt", dtype=np.float32) for name in names],
        axis=-1,
    )


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "external_data" / "uci_har")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "uci_har_imu_pretrain.pt")
    args = ap.parse_args()

    seed_all(args.seed)
    train_raw = load_raw(args.data_dir, "train")
    mean = train_raw.reshape(-1, UCI_CHANNELS).mean(axis=0)
    std = train_raw.reshape(-1, UCI_CHANNELS).std(axis=0).clip(1e-4)
    train_ds = UciHarDataset(args.data_dir, "train", mean, std, True)
    val_ds = UciHarDataset(args.data_dir, "test", mean, std, False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, args.batch_size * 2, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UciModel(args.width).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.03)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()
        print(f"epoch={epoch:02d} train={train_loss:.4f}/{train_acc:.4f} val={val_loss:.4f}/{val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        if val_acc > best:
            best = val_acc
            torch.save(
                {
                    "imu": model.imu.state_dict(),
                    "params": {"width": args.width, "dropout": 0.20},
                    "mean": mean,
                    "std": std,
                    "best_val_acc": best,
                    "source": "UCI HAR Dataset v1.0, inertial signals, 9 channels",
                },
                args.output,
            )
    history_path = args.output.with_name(args.output.stem + "_history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"device={device} best_val={best:.6f} checkpoint={args.output}")


if __name__ == "__main__":
    main()
