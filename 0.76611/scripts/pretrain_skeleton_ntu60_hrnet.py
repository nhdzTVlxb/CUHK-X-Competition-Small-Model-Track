from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_skel_imu_v2 import SKEL_ENHANCED, SequenceBranch, skeleton_features  # noqa: E402


NTU_CLASSES = 60
LEFT_RIGHT_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sample_indices(n: int, target: int, train: bool) -> np.ndarray:
    if n <= 0:
        return np.zeros(target, dtype=np.int64)
    if n == target:
        return np.arange(n, dtype=np.int64)
    if train and n > target and random.random() < 0.65:
        window = random.randint(max(target, int(n * 0.70)), n)
        start = random.randint(0, n - window)
        return np.linspace(start, start + window - 1, target).round().astype(np.int64)
    return np.linspace(0, n - 1, target).round().astype(np.int64)


def pick_person(keypoint: np.ndarray, score: np.ndarray) -> int:
    if keypoint.shape[0] == 1:
        return 0
    mean_score = np.nanmean(score, axis=(1, 2))
    if not np.isfinite(mean_score).any():
        return 0
    return int(np.nanargmax(mean_score))


def annotation_to_raw(annotation: dict[str, object], frames: int, train: bool) -> np.ndarray:
    keypoint = np.asarray(annotation["keypoint"], dtype=np.float32)
    score = np.asarray(annotation["keypoint_score"], dtype=np.float32)
    person = pick_person(keypoint, score)
    keypoint = keypoint[person]
    score = score[person]
    idx = sample_indices(keypoint.shape[0], frames, train)
    keypoint = keypoint[idx]
    score = score[idx]

    height, width = annotation.get("img_shape", annotation.get("original_shape", (1080, 1920)))
    scale = np.asarray([max(float(width), 1.0), max(float(height), 1.0)], dtype=np.float32)
    xy = keypoint[:, :, :2] / scale
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clip(0.0, 1.0)

    if train and random.random() < 0.35:
        xy += np.random.normal(0.0, 0.006, xy.shape).astype(np.float32)
    if train and random.random() < 0.30:
        xy[:, :, 0] = 1.0 - xy[:, :, 0]
        for left, right in LEFT_RIGHT_PAIRS:
            xy[:, [left, right]] = xy[:, [right, left]]
            score[:, [left, right]] = score[:, [right, left]]

    raw = np.zeros((frames, 17, 4), dtype=np.float32)
    raw[:, :, :2] = xy
    raw[:, :, 3] = score
    return raw.reshape(frames, 17 * 4)


def load_payload(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "annotations" not in payload or "split" not in payload:
        raise ValueError(f"Unexpected NTU HRNet payload format: {path}")
    return payload


def split_annotations(payload: dict[str, object], split_prefix: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    annotations = payload["annotations"]
    lookup = {row["frame_dir"]: row for row in annotations}
    split = payload["split"]
    train_names = split[f"{split_prefix}_train"]
    val_names = split[f"{split_prefix}_val"]
    return [lookup[name] for name in train_names], [lookup[name] for name in val_names]


def compute_stats(records: list[dict[str, object]], frames: int, max_items: int) -> tuple[np.ndarray, np.ndarray]:
    if max_items > 0 and len(records) > max_items:
        records = random.sample(records, max_items)
    total = np.zeros(SKEL_ENHANCED, dtype=np.float64)
    total_sq = np.zeros(SKEL_ENHANCED, dtype=np.float64)
    count = 0
    for row in tqdm(records, desc="ntu stats"):
        feat = skeleton_features(annotation_to_raw(row, frames, train=False)).astype(np.float64)
        total += feat.sum(axis=0)
        total_sq += np.square(feat).sum(axis=0)
        count += feat.shape[0]
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(total_sq / max(count, 1) - mean**2, 1e-5))
    return mean.astype(np.float32), std.astype(np.float32)


class NtuHrnetDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, object]],
        frames: int,
        mean: np.ndarray,
        std: np.ndarray,
        train: bool,
    ) -> None:
        self.records = records
        self.frames = frames
        self.mean = mean
        self.std = std
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        feat = skeleton_features(annotation_to_raw(row, self.frames, self.train))
        if self.train and random.random() < 0.15:
            start = random.randrange(max(1, self.frames - 4))
            feat[start : start + 4] = 0.0
        feat = (feat - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(feat)), torch.tensor(int(row["label"]), dtype=torch.long)


class NtuSkeletonModel(nn.Module):
    def __init__(self, width: int = 160, dropout: float = 0.20) -> None:
        super().__init__()
        self.sk = SequenceBranch(SKEL_ENHANCED, width, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, NTU_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.sk(x))


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
    ap.add_argument("--data-file", type=Path, default=ROOT / "external_data" / "ntu60_hrnet" / "ntu60_hrnet.pkl")
    ap.add_argument("--split-prefix", choices=("xsub", "xview"), default="xsub")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--stats-items", type=int, default=8000)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "ntu60_hrnet_skeleton_pretrain.pt")
    args = ap.parse_args()

    seed_all(args.seed)
    payload = load_payload(args.data_file)
    train_records, val_records = split_annotations(payload, args.split_prefix)
    mean, std = compute_stats(train_records, args.frames, args.stats_items)

    train_loader = DataLoader(
        NtuHrnetDataset(train_records, args.frames, mean, std, True),
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        NtuHrnetDataset(val_records, args.frames, mean, std, False),
        args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NtuSkeletonModel(args.width).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.03)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()
        print(f"epoch={epoch:02d} train={tr_loss:.4f}/{tr_acc:.4f} val={va_loss:.4f}/{va_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best:
            best = va_acc
            torch.save(
                {
                    "sk": model.sk.state_dict(),
                    "params": {"width": args.width, "dropout": 0.20, "frames": args.frames},
                    "mean": mean,
                    "std": std,
                    "best_val_acc": best,
                    "source": "NTU60 HRNet skeleton Kaggle dataset hungkhoi/skeleton-data-of-ntu-rgbd-60-dataset",
                    "license": "unknown on Kaggle mirror; disclose source and official NTU request path if used",
                },
                args.output,
            )
    history_path = args.output.with_name(args.output.stem + "_history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"device={device} best_val={best:.6f} checkpoint={args.output}")


if __name__ == "__main__":
    main()
