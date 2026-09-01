from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import Paths, compute_train_stats
from cuhkx_baseline.model import SkelImuNet, count_parameters


DEFAULT_VAL_USERS = {"user8", "user9", "user23", "user24"}


class FeatureDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: Path,
        stats: dict[str, np.ndarray],
        train: bool,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.stats = stats
        self.train = train

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        skeleton = (item["skeleton"] - self.stats["skeleton_mean"]) / self.stats["skeleton_std"]
        imu = (item["imu"] - self.stats["imu_mean"]) / self.stats["imu_std"]
        if self.train and random.random() < 0.15:
            imu = np.zeros_like(imu)
        if self.train and random.random() < 0.10:
            skeleton = np.zeros_like(skeleton)
        label = int(row["label"])
        return (
            torch.from_numpy(skeleton.astype(np.float32)),
            torch.from_numpy(imu.astype(np.float32)),
            torch.tensor(label, dtype=torch.long),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    losses = []
    correct = 0
    total = 0
    for skeleton, imu, y in tqdm(loader, leave=False):
        skeleton = skeleton.to(device, non_blocking=True)
        imu = imu.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            amp_enabled = scaler is not None and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(skeleton, imu)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        losses.append(float(loss.detach().cpu()))
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().detach().cpu())
        total += int(y.numel())
    return float(np.mean(losses)), float(correct / max(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    paths = Paths(args.root)
    manifest_path = paths.cache_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run scripts/prepare_skel_imu_cache.py first: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    train_all = manifest[manifest["split"] == "train"].copy()
    val_users = DEFAULT_VAL_USERS
    train_users = set(train_all["user"].unique()) - val_users
    train_df = train_all[train_all["user"].isin(train_users)].copy()
    val_df = train_all[train_all["user"].isin(val_users)].copy()
    if args.limit_train:
        train_df = train_df.sample(min(args.limit_train, len(train_df)), random_state=args.seed)
        val_df = val_df.sample(min(max(args.limit_train // 4, 40), len(val_df)), random_state=args.seed)

    stats = compute_train_stats(manifest, paths.cache_dir, train_users)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    np.savez(paths.outputs_dir / "skel_imu_stats.npz", **stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkelImuNet(width=args.width).to(device)
    params = count_parameters(model)
    ckpt_size_mb = params * 4 / (1024 * 1024)
    print(f"device={device} params={params:,} fp32_size~{ckpt_size_mb:.2f}MB")
    print(f"train={len(train_df)} val={len(val_df)} val_users={sorted(val_users)}")

    train_loader = DataLoader(
        FeatureDataset(train_df, paths.cache_dir, stats, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FeatureDataset(val_df, paths.cache_dir, stats, train=False),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None

    best_acc = -1.0
    best_path = paths.outputs_dir / "skel_imu_best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "params": vars(args),
                    "stats_file": "skel_imu_stats.npz",
                    "val_acc": best_acc,
                    "val_users": sorted(val_users),
                },
                best_path,
            )

    with (paths.outputs_dir / "skel_imu_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"best_val_acc={best_acc:.4f}")
    print(f"checkpoint={best_path}")


if __name__ == "__main__":
    main()
