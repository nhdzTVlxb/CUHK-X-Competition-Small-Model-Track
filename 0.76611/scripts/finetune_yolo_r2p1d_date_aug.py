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
sys.path.insert(0, str(ROOT / "scripts"))

from cuhkx_baseline.yolo_r2p1d import CHANNELS, N_CLASSES, R2Plus1D34, dequantize_state, load_checkpoint  # noqa: E402
from finetune_yolo_r2p1d import MEAN, STD, infer, make_model, run_epoch, seed_everything, set_trainable  # noqa: E402
from sequence_postprocess import collect_train_metadata  # noqa: E402

VAL_USERS = {"user8", "user9", "user23", "user24"}
DEFAULT_VAL_DATES = {"2025-06-12", "2025-06-13"}


def build_metadata(root: Path) -> pd.DataFrame:
    meta = collect_train_metadata(root)[["clip_id", "user", "start"]].copy()
    meta["date"] = meta["start"].dt.date.astype(str)
    return meta[["clip_id", "user", "date"]]


def build_date_weight_map(root: Path) -> dict[str, float]:
    report_path = root / "outputs" / "validation_gap_train_acc_by_group.csv"
    if report_path.exists():
        report = pd.read_csv(report_path)
        date_rows = report[(report["file"] == "yolo_r2p1d_all_train_logits.npz") & (report["group_type"] == "date")]
        if not date_rows.empty:
            median_acc = float(date_rows["acc"].median())
            weights: dict[str, float] = {}
            for row in date_rows.itertuples(index=False):
                acc = float(row.acc)
                weight = (median_acc / max(acc, 1e-4)) ** 1.5
                weights[str(row.group)] = float(np.clip(weight, 0.8, 1.8))
            return weights
    return {
        "2025-05-07": 0.85,
        "2025-05-08": 0.85,
        "2025-05-30": 0.90,
        "2025-05-31": 1.20,
        "2025-06-01": 1.15,
        "2025-06-02": 1.30,
        "2025-06-09": 1.00,
        "2025-06-10": 0.95,
        "2025-06-11": 0.95,
        "2025-06-12": 1.10,
        "2025-06-13": 1.10,
    }


def temporal_shift(clip: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return clip
    if shift > 0:
        tail = clip[-1:].repeat(shift, 1, 1, 1)
        return torch.cat([clip[shift:], tail], dim=0)
    head = clip[:1].repeat(-shift, 1, 1, 1)
    return torch.cat([head, clip[: clip.shape[0] + shift]], dim=0)


def spatial_shift(clip: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    if dx == 0 and dy == 0:
        return clip
    clip = torch.roll(clip, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        clip[..., :dy, :] = 0
    elif dy < 0:
        clip[..., dy:, :] = 0
    if dx > 0:
        clip[..., :, :dx] = 0
    elif dx < 0:
        clip[..., :, dx:] = 0
    return clip


def channel_jitter(clip: torch.Tensor) -> torch.Tensor:
    rgb_scale = random.uniform(0.88, 1.12)
    ir_scale = random.uniform(0.86, 1.14)
    clip[:, :3] = torch.clamp(clip[:, :3] * rgb_scale, 0.0, 1.0)
    clip[:, 3:] = torch.clamp(clip[:, 3:] * ir_scale, 0.0, 1.0)
    if random.random() < 0.12:
        clip[:, 3:] = 0.0
    if random.random() < 0.05:
        clip[:, :3] = 0.0
    if random.random() < 0.30:
        clip = torch.clamp(clip + torch.randn_like(clip) * 0.012, 0.0, 1.0)
    return clip


def frame_dropout(clip: torch.Tensor, max_drop: int = 1) -> torch.Tensor:
    if random.random() < 0.30:
        count = random.randint(1, max_drop)
        idx = random.sample(range(clip.shape[0]), count)
        clip[idx] = 0.0
    return clip


class AugmentedCachedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, train: bool, strong_aug: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train
        self.strong_aug = strong_aug

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        clip = torch.from_numpy(item["tensor"].astype(np.float32)).div_(255.0)
        if self.train:
            if random.random() < 0.5:
                clip = torch.flip(clip, dims=(-1,))
            if self.strong_aug:
                clip = temporal_shift(clip, max_shift=2)
                clip = spatial_shift(clip, max_shift=4)
                clip = channel_jitter(clip)
                clip = frame_dropout(clip, max_drop=1)
        clip = (clip - MEAN) / STD
        return clip, int(row["label"]), str(row["clip_id"])


def make_sampler(frame: pd.DataFrame, meta: pd.DataFrame, root: Path) -> WeightedRandomSampler:
    labels = frame["label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=N_CLASSES)
    class_weights = np.asarray([1.0 / max(np.sqrt(counts[label]), 1.0) for label in labels], dtype=np.float64)
    class_weights /= np.mean(class_weights)

    date_weights = build_date_weight_map(root)
    lookup = meta.set_index("clip_id")
    weights = []
    for idx, clip_id in enumerate(frame["clip_id"].astype(str)):
        date = str(lookup.loc[clip_id, "date"]) if clip_id in lookup.index else "unknown"
        weights.append(class_weights[idx] * date_weights.get(date, 1.0))
    weights = np.asarray(weights, dtype=np.float64)
    weights /= np.mean(weights)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--mode", choices=["head", "last"], default="last")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--strong-aug", action="store_true", default=True)
    parser.add_argument("--split-mode", choices=["user", "date", "all"], default="date")
    parser.add_argument("--val-dates", type=str, default="2025-06-12,2025-06-13")
    parser.add_argument("--output-prefix", type=str, default="finetune_yolo_date_aug")
    args = parser.parse_args()

    seed_everything(args.seed)
    manifest = pd.read_csv(args.cache_dir / "train_manifest.csv")
    if "user" not in manifest:
        manifest["user"] = manifest["path"].astype(str).str.split("/").str[-2]
    if "clip_id" not in manifest:
        manifest["clip_id"] = manifest["path"].astype(str).str.rstrip("/").str.split("/").str[-1]
    meta = build_metadata(args.root)
    manifest = manifest.merge(meta, on="clip_id", how="left")
    manifest["date"] = manifest["date"].fillna("unknown")
    if "user_x" in manifest.columns and "user_y" in manifest.columns:
        manifest["user"] = manifest["user_x"].fillna(manifest["user_y"])
        manifest = manifest.drop(columns=["user_x", "user_y"])
    elif "user_x" in manifest.columns:
        manifest["user"] = manifest["user_x"]
        manifest = manifest.drop(columns=["user_x"])
    elif "user_y" in manifest.columns:
        manifest["user"] = manifest["user_y"]
        manifest = manifest.drop(columns=["user_y"])

    if args.split_mode == "all":
        train = manifest.copy()
        val = manifest.copy()
    elif args.split_mode == "user":
        train = manifest[~manifest["user"].isin(VAL_USERS)].copy()
        val = manifest[manifest["user"].isin(VAL_USERS)].copy()
    else:
        val_dates = {date.strip() for date in args.val_dates.split(",") if date.strip()}
        train = manifest[~manifest["date"].isin(val_dates)].copy()
        val = manifest[manifest["date"].isin(val_dates)].copy()

    if train.empty or val.empty:
        raise ValueError(f"Empty split for mode={args.split_mode}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.checkpoint, args.model_index, args.resume_checkpoint).to(device)
    set_trainable(model, args.mode)
    trainable = [p for p in model.parameters() if p.requires_grad]

    sampler = make_sampler(train, meta, args.root) if args.balanced else None
    train_loader = DataLoader(
        AugmentedCachedDataset(train, args.cache_dir, True, args.strong_aug),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        AugmentedCachedDataset(val, args.cache_dir, False, False),
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None

    output_dir = args.root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.output_prefix}_best.pt"
    history: list[dict[str, float]] = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            criterion,
            args.accumulation,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            device,
            None,
            None,
            criterion,
            1,
        )
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(f"epoch={epoch:02d} train={train_loss:.4f}/{train_acc:.4f} val={val_loss:.4f}/{val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best, "args": vars(args)}, checkpoint_path)

    (output_dir / f"{args.output_prefix}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    val_logits, val_y, val_ids = infer(model, val_loader, device)
    test_manifest = pd.read_csv(args.cache_dir / "test_manifest.csv")
    test_loader = DataLoader(
        AugmentedCachedDataset(test_manifest.assign(label=-1), args.cache_dir, False, False),
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_logits, _test_y, test_ids = infer(model, test_loader, device)
    np.savez_compressed(
        output_dir / f"{args.output_prefix}_logits.npz",
        val_logits=val_logits,
        val_y=val_y,
        val_clip_id=np.asarray(val_ids, dtype=str),
        test_logits=test_logits,
        test_clip_id=np.asarray(test_ids, dtype=str),
        test_path=test_manifest["path"].astype(str).to_numpy(),
    )
    print(
        f"device={device} split={args.split_mode} mode={args.mode} train={len(train)} val={len(val)} "
        f"best_val={best:.6f} checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
