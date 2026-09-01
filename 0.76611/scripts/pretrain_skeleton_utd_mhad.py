from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_skel_imu_v2 import SKEL_ENHANCED, SequenceBranch, skeleton_features  # noqa: E402


UTD_CLASSES = 27
SUBJECT_RE = re.compile(r"a(?P<label>\d+)_s(?P<subject>\d+)_t(?P<trial>\d+)_skeleton\.mat$")

# UTD-MHAD uses the Kinect-20 layout. Map its upper/lower body joints into
# COCO-17 slots used by the competition skeleton branch.
KINECT20_TO_COCO17 = {
    0: 3,   # nose/head
    1: 3,
    2: 3,
    3: 3,
    4: 3,
    5: 4,   # left shoulder
    6: 8,   # right shoulder
    7: 5,   # left elbow
    8: 9,   # right elbow
    9: 6,   # left wrist
    10: 10, # right wrist
    11: 12, # left hip
    12: 16, # right hip
    13: 13, # left knee
    14: 17, # right knee
    15: 14, # left ankle
    16: 18, # right ankle
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sample_indices(n: int, target: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(target, dtype=np.int64)
    if n == target:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, target).round().astype(np.int64)


def load_utd_raw(path: Path, frames: int) -> np.ndarray:
    mat = loadmat(path)
    arr = np.asarray(mat["d_skel"], dtype=np.float32)  # joints, xyz, frames
    if arr.ndim != 3 or arr.shape[0] < 20 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected UTD skeleton shape in {path}: {arr.shape}")
    arr = arr[:, :3, sample_indices(arr.shape[2], frames)]
    arr = np.moveaxis(arr, 2, 0)  # frames, joints, xyz

    raw = np.zeros((frames, 17, 4), dtype=np.float32)
    for coco_idx, kinect_idx in KINECT20_TO_COCO17.items():
        raw[:, coco_idx, :3] = arr[:, kinect_idx, :3]
        raw[:, coco_idx, 3] = 1.0
    return raw.reshape(frames, 17 * 4)


def parse_record(path: Path) -> tuple[int, int, int]:
    match = SUBJECT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Cannot parse UTD filename: {path}")
    label = int(match.group("label")) - 1
    subject = int(match.group("subject"))
    trial = int(match.group("trial"))
    return label, subject, trial


def build_records(data_dir: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(data_dir.rglob("*_skeleton.mat")):
        label, subject, trial = parse_record(path)
        records.append({"path": path, "label": label, "subject": subject, "trial": trial})
    if not records:
        raise FileNotFoundError(f"No UTD-MHAD skeleton files found under {data_dir}")
    return records


def compute_stats(records: list[dict[str, object]], frames: int) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(SKEL_ENHANCED, dtype=np.float64)
    total_sq = np.zeros(SKEL_ENHANCED, dtype=np.float64)
    count = 0
    for row in tqdm(records, desc="utd stats"):
        feat = skeleton_features(load_utd_raw(Path(row["path"]), frames)).astype(np.float64)
        total += feat.sum(axis=0)
        total_sq += np.square(feat).sum(axis=0)
        count += feat.shape[0]
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(total_sq / max(count, 1) - mean**2, 1e-5))
    return mean.astype(np.float32), std.astype(np.float32)


class UtdSkeletonDataset(Dataset):
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
        raw = load_utd_raw(Path(row["path"]), self.frames)
        feat = skeleton_features(raw)
        if self.train:
            if random.random() < 0.35:
                feat = feat + np.random.normal(0.0, 0.025, feat.shape).astype(np.float32)
            if random.random() < 0.20:
                scale = np.random.uniform(0.92, 1.08)
                feat[:, :51] *= np.float32(scale)
            if random.random() < 0.15:
                start = random.randrange(max(1, self.frames - 4))
                feat[start : start + 4] = 0.0
        feat = (feat - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(feat)), torch.tensor(int(row["label"]), dtype=torch.long)


class UtdSkeletonModel(nn.Module):
    def __init__(self, width: int = 160, dropout: float = 0.20) -> None:
        super().__init__()
        self.sk = SequenceBranch(SKEL_ENHANCED, width, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, UTD_CLASSES),
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
    ap.add_argument("--data-dir", type=Path, default=ROOT / "external_data" / "utd_mhad")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-subjects", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "utd_mhad_skeleton_pretrain.pt")
    args = ap.parse_args()

    seed_all(args.seed)
    records = build_records(args.data_dir)
    val_subjects = set(args.val_subjects)
    train_records = [row for row in records if int(row["subject"]) not in val_subjects]
    val_records = [row for row in records if int(row["subject"]) in val_subjects]
    mean, std = compute_stats(train_records, args.frames)

    train_loader = DataLoader(
        UtdSkeletonDataset(train_records, args.frames, mean, std, True),
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        UtdSkeletonDataset(val_records, args.frames, mean, std, False),
        args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UtdSkeletonModel(args.width).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
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
                    "source": "UTD-MHAD skeleton files from Kaggle dasmehdixtr/human-action-recognition-dataset",
                    "license": "DbCL-1.0",
                },
                args.output,
            )
    history_path = args.output.with_name(args.output.stem + "_history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"device={device} best_val={best:.6f} checkpoint={args.output}")


if __name__ == "__main__":
    main()
