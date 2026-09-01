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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import Paths, enhance_skeleton_sequence
from cuhkx_baseline.model import FusionNetV2, count_parameters

DEFAULT_VAL_USERS = {"user8", "user9", "user23", "user24"}


class FusionV2Dataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir: Path, stats: dict[str, np.ndarray], train: bool) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.stats = stats
        self.train = train

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        skeleton = enhance_skeleton_sequence(item["skeleton"])
        imu = item["imu"].astype(np.float32)
        depth = item["depth"].astype(np.float32) / 127.5 - 1.0
        ir = item["ir"].astype(np.float32) / 127.5 - 1.0

        skeleton = (skeleton - self.stats["skeleton_mean"]) / self.stats["skeleton_std"]
        imu = (imu - self.stats["imu_mean"]) / self.stats["imu_std"]

        if self.train:
            if random.random() < 0.25:
                skeleton = skeleton + np.random.normal(0.0, 0.02, skeleton.shape).astype(np.float32)
            if random.random() < 0.20:
                imu = imu + np.random.normal(0.0, 0.03, imu.shape).astype(np.float32)
            if random.random() < 0.10:
                skeleton = np.zeros_like(skeleton)
            if random.random() < 0.12:
                imu = np.zeros_like(imu)
            if random.random() < 0.08:
                depth = np.zeros_like(depth)
            if random.random() < 0.08:
                ir = np.zeros_like(ir)
            if random.random() < 0.50:
                depth = depth[:, :, :, ::-1].copy()
                ir = ir[:, :, :, ::-1].copy()
            if random.random() < 0.20:
                depth = depth * np.float32(0.9 + 0.2 * random.random())
                ir = ir * np.float32(0.9 + 0.2 * random.random())

        return (
            torch.from_numpy(np.ascontiguousarray(skeleton.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(imu.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(depth.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(ir.astype(np.float32))),
            torch.tensor(int(float(row["label"])), dtype=torch.long),
            row["clip_id"],
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == y).sum().detach().cpu())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    correct = 0
    total = 0
    for skeleton, imu, depth, ir, y, _clip_id in tqdm(loader, leave=False):
        skeleton = skeleton.to(device, non_blocking=True)
        imu = imu.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        ir = ir.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        amp_enabled = scaler is not None and device.type == "cuda"
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(skeleton, imu, depth, ir)
                loss = criterion(logits, y)
            if is_train:
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
        losses.append(float(loss.detach().cpu()))
        correct += accuracy(logits, y)
        total += int(y.numel())
    return float(np.mean(losses)), float(correct / max(total, 1))


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    ids: list[str] = []
    with torch.no_grad():
        for skeleton, imu, depth, ir, y, clip_id in tqdm(loader, desc="logits", leave=False):
            out = model(
                skeleton.to(device, non_blocking=True),
                imu.to(device, non_blocking=True),
                depth.to(device, non_blocking=True),
                ir.to(device, non_blocking=True),
            )
            logits_parts.append(out.float().cpu().numpy())
            y_parts.append(y.numpy())
            ids.extend(list(clip_id))
    return np.concatenate(logits_parts), np.concatenate(y_parts), np.asarray(ids)


def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = df["label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=40)
    weights = np.array([1.0 / max(counts[y], 1) for y in labels], dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_stats(manifest: pd.DataFrame, cache_dir: Path, train_users: set[str]) -> dict[str, np.ndarray]:
    rows = manifest[(manifest["split"] == "train") & (manifest["user"].isin(train_users))]
    sk_sum = np.zeros(238, dtype=np.float64)
    sk_sq = np.zeros(238, dtype=np.float64)
    imu_sum = np.zeros(19, dtype=np.float64)
    imu_sq = np.zeros(19, dtype=np.float64)
    sk_n = 0
    imu_n = 0

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="stats"):
        item = np.load(cache_dir / row["feature_file"])
        sk = enhance_skeleton_sequence(item["skeleton"]).astype(np.float64)
        imu = item["imu"].astype(np.float64)
        sk_sum += sk.sum(axis=0)
        sk_sq += np.square(sk).sum(axis=0)
        imu_sum += imu.sum(axis=0)
        imu_sq += np.square(imu).sum(axis=0)
        sk_n += sk.shape[0]
        imu_n += imu.shape[0]

    sk_mean = sk_sum / max(sk_n, 1)
    imu_mean = imu_sum / max(imu_n, 1)
    return {
        "skeleton_mean": sk_mean.astype(np.float32),
        "skeleton_std": np.sqrt(np.maximum(sk_sq / max(sk_n, 1) - np.square(sk_mean), 1e-5)).astype(np.float32),
        "imu_mean": imu_mean.astype(np.float32),
        "imu_std": np.sqrt(np.maximum(imu_sq / max(imu_n, 1) - np.square(imu_mean), 1e-5)).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--image-base", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="fusion_v2")
    args = parser.parse_args()

    seed_everything(args.seed)
    paths = Paths(args.root)
    cache_dir = paths.fusion_cache_dir
    manifest_path = cache_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run scripts/prepare_fusion_cache.py first: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    train_all = manifest[manifest["split"] == "train"].copy()
    train_users = set(train_all["user"].unique()) - DEFAULT_VAL_USERS
    train_df = train_all[train_all["user"].isin(train_users)].copy()
    val_df = train_all[train_all["user"].isin(DEFAULT_VAL_USERS)].copy()

    stats = compute_stats(manifest, cache_dir, train_users)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    np.savez(paths.outputs_dir / f"{args.output_prefix}_stats.npz", **stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNetV2(width=args.width, image_base=args.image_base).to(device)
    params = count_parameters(model)
    print(f"device={device} params={params:,} fp32_size~{params * 4 / 1024**2:.2f}MB")
    print(f"train={len(train_df)} val={len(val_df)} val_users={sorted(DEFAULT_VAL_USERS)}")

    sampler = make_sampler(train_df) if args.balanced else None
    train_loader = DataLoader(
        FusionV2Dataset(train_df, cache_dir, stats, train=True),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FusionV2Dataset(val_df, cache_dir, stats, train=False),
        batch_size=max(args.batch_size, 1) * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None

    best_acc = -1.0
    best_path = paths.outputs_dir / f"{args.output_prefix}_best.pt"
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
                    "stats_file": f"{args.output_prefix}_stats.npz",
                    "val_acc": best_acc,
                    "val_users": sorted(DEFAULT_VAL_USERS),
                },
                best_path,
            )

    history_path = paths.outputs_dir / f"{args.output_prefix}_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_logits, val_y, val_ids = collect_logits(model, val_loader, device)
    np.savez_compressed(
        paths.outputs_dir / f"{args.output_prefix}_val_logits.npz",
        logits=val_logits,
        y=val_y,
        clip_id=val_ids,
    )

    print(f"best_val_acc={best_acc:.4f}")
    print(f"checkpoint={best_path}")
    print(f"history={history_path}")
    print(f"val_logits={paths.outputs_dir / f'{args.output_prefix}_val_logits.npz'}")


if __name__ == "__main__":
    main()
