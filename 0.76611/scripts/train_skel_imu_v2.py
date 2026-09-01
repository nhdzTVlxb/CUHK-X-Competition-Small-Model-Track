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

from cuhkx_baseline.features import Paths

N_CLASSES = 40
SKEL_JOINTS = 17
SKEL_RAW = 68
SKEL_ENHANCED = 238
IMU_DIM = 19
VAL_USERS = {"user8", "user9", "user23", "user24"}

# COCO-17 parent graph. Relative vectors remove most subject-specific pose offsets.
PARENTS = np.asarray(
    [-1, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14],
    dtype=np.int64,
)


def skeleton_features(raw: np.ndarray) -> np.ndarray:
    """Build subject-robust per-frame pose and motion features."""
    seq = raw.astype(np.float32, copy=False).reshape(-1, SKEL_JOINTS, 4)
    xyz = seq[:, :, :3]
    score = seq[:, :, 3:4]

    hip = (xyz[:, 11:12] + xyz[:, 12:13]) * 0.5
    centered = xyz - hip
    shoulder_span = np.linalg.norm(xyz[:, 5] - xyz[:, 6], axis=1, keepdims=True)
    hip_span = np.linalg.norm(xyz[:, 11] - xyz[:, 12], axis=1, keepdims=True)
    body_span = np.maximum(np.maximum(shoulder_span, hip_span), 0.08)
    centered = centered / body_span[:, None, :]

    bones = np.zeros_like(centered)
    valid = PARENTS >= 0
    bones[:, valid] = centered[:, valid] - centered[:, PARENTS[valid]]
    velocity = np.diff(centered, axis=0, prepend=centered[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])

    # 51 centered pose + 51 score-weighted pose + 51 bones + 34 velocity +
    # 34 acceleration + 17 confidence channels.
    out = np.concatenate(
        [
            centered.reshape(len(seq), -1),
            (centered * score).reshape(len(seq), -1),
            bones.reshape(len(seq), -1),
            velocity[:, :, :2].reshape(len(seq), -1),
            acceleration[:, :, :2].reshape(len(seq), -1),
            score.reshape(len(seq), -1),
        ],
        axis=1,
    )
    # Keep the declared size stable if a future source changes keypoint count.
    if out.shape[1] < SKEL_ENHANCED:
        out = np.pad(out, ((0, 0), (0, SKEL_ENHANCED - out.shape[1])))
    return out[:, :SKEL_ENHANCED].astype(np.float32, copy=False)


class CachedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, stats: dict[str, np.ndarray], train: bool):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.stats = stats
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        sk = skeleton_features(item["skeleton"])
        imu = item["imu"].astype(np.float32, copy=False)
        sk = (sk - self.stats["sk_mean"]) / self.stats["sk_std"]
        imu = (imu - self.stats["imu_mean"]) / self.stats["imu_std"]

        if self.train:
            if random.random() < 0.25:
                sk = sk + np.random.normal(0.0, 0.025, sk.shape).astype(np.float32)
            if random.random() < 0.15:
                imu = imu + np.random.normal(0.0, 0.03, imu.shape).astype(np.float32)
            if random.random() < 0.10:
                sk = np.zeros_like(sk)
            if random.random() < 0.12:
                imu = np.zeros_like(imu)

        return (
            torch.from_numpy(np.ascontiguousarray(sk)),
            torch.from_numpy(np.ascontiguousarray(imu)),
            torch.tensor(int(float(row["label"])), dtype=torch.long),
            row["clip_id"],
        )


class ResidualTCN(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, groups=1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(width),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SequenceBranch(nn.Module):
    def __init__(self, in_dim: int, width: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(in_dim, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            ResidualTCN(width, 1, dropout),
            ResidualTCN(width, 2, dropout),
            ResidualTCN(width, 4, dropout),
        )
        self.gru = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.out = nn.Sequential(nn.LayerNorm(width * 3), nn.Linear(width * 3, width), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x.transpose(1, 2))
        x = self.tcn(x).transpose(1, 2)
        x, _ = self.gru(x)
        weights = torch.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        weighted = (x * weights).sum(dim=1)
        pooled = torch.cat([weighted, x.mean(dim=1), x.amax(dim=1)], dim=1)
        return self.out(pooled)


class SkelImuV2(nn.Module):
    def __init__(self, width: int = 160, dropout: float = 0.20):
        super().__init__()
        self.sk = SequenceBranch(SKEL_ENHANCED, width, dropout)
        self.imu = SequenceBranch(IMU_DIM, width, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, N_CLASSES),
        )

    def forward(self, skeleton: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.sk(skeleton), self.imu(imu)], dim=1))


def stats_for(frame: pd.DataFrame, cache_dir: Path, train_users: set[str]) -> dict[str, np.ndarray]:
    rows = frame[(frame["split"] == "train") & frame["user"].isin(train_users)]
    sk_sum = np.zeros(SKEL_ENHANCED, np.float64)
    sk_sq = np.zeros(SKEL_ENHANCED, np.float64)
    imu_sum = np.zeros(IMU_DIM, np.float64)
    imu_sq = np.zeros(IMU_DIM, np.float64)
    n_sk = n_imu = 0
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="stats"):
        item = np.load(cache_dir / row["feature_file"])
        sk = skeleton_features(item["skeleton"]).astype(np.float64)
        imu = item["imu"].astype(np.float64)
        sk_sum += sk.sum(0)
        sk_sq += np.square(sk).sum(0)
        imu_sum += imu.sum(0)
        imu_sq += np.square(imu).sum(0)
        n_sk += sk.shape[0]
        n_imu += imu.shape[0]
    sk_mean = sk_sum / max(n_sk, 1)
    imu_mean = imu_sum / max(n_imu, 1)
    return {
        "sk_mean": sk_mean.astype(np.float32),
        "sk_std": np.sqrt(np.maximum(sk_sq / max(n_sk, 1) - sk_mean**2, 1e-5)).astype(np.float32),
        "imu_mean": imu_mean.astype(np.float32),
        "imu_std": np.sqrt(np.maximum(imu_sq / max(n_imu, 1) - imu_mean**2, 1e-5)).astype(np.float32),
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def run_epoch(model, loader, criterion, optimizer, device, scaler):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = total = 0
    for sk, imu, y, _ids in tqdm(loader, leave=False):
        sk, imu, y = sk.to(device), imu.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(sk, imu)
                loss = criterion(logits, y)
            if train:
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


def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_y, all_ids = [], [], []
    with torch.no_grad():
        for sk, imu, y, ids in tqdm(loader, desc="logits", leave=False):
            out = model(sk.to(device), imu.to(device))
            all_logits.append(out.float().cpu().numpy())
            all_y.append(y.numpy())
            all_ids.extend(ids)
    return np.concatenate(all_logits), np.concatenate(all_y), np.asarray(all_ids)


def load_transfer_imu(
    model: nn.Module,
    checkpoint: Path,
    device: torch.device,
    full: bool,
    shared_channels: int,
) -> None:
    """Transfer compatible UCI temporal weights and optionally shared sensor channels."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    source = payload.get("imu", payload.get("model", payload))
    target = model.imu.state_dict()
    transferable = {
        key: value
        for key, value in source.items()
        if (full or not key.startswith("input."))
        and key in target
        and tuple(value.shape) == tuple(target[key].shape)
    }
    if shared_channels > 0:
        key = "input.0.weight"
        if key in source and key in target and source[key].ndim == target[key].ndim:
            count = min(shared_channels, source[key].shape[1], target[key].shape[1])
            target[key][:, :count] = source[key][:, :count]
            transferable.pop(key, None)
            input_note = f"shared_input_channels={count}"
        else:
            input_note = "shared_input_channels=0"
    else:
        input_note = "shared_input_channels=0"
    target.update(transferable)
    model.imu.load_state_dict(target)
    print(f"loaded_external_imu={checkpoint} transferred={len(transferable)} full={full} {input_note}")


def load_transfer_skeleton(model: nn.Module, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    source = payload.get("sk", payload.get("model", payload))
    target = model.sk.state_dict()
    transferable = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    target.update(transferable)
    model.sk.load_state_dict(target)
    print(f"loaded_external_skeleton={checkpoint} transferred={len(transferable)}")


def set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(trainable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--init-imu-checkpoint", type=Path, default=None)
    ap.add_argument("--transfer-full-imu", action="store_true")
    ap.add_argument("--transfer-shared-channels", type=int, default=0)
    ap.add_argument("--init-skel-checkpoint", type=Path, default=None)
    ap.add_argument("--skel-lr-mult", type=float, default=1.0)
    ap.add_argument("--imu-lr-mult", type=float, default=1.0)
    ap.add_argument("--head-lr-mult", type=float, default=1.0)
    ap.add_argument("--freeze-skel-epochs", type=int, default=0)
    ap.add_argument("--freeze-imu-epochs", type=int, default=0)
    ap.add_argument("--output-prefix", type=str, default="skel_imu_v2")
    args = ap.parse_args()

    seed_all(args.seed)
    paths = Paths(args.root)
    cache = paths.cache_dir
    manifest = pd.read_csv(cache / "manifest.csv")
    train_all = manifest[manifest.split.eq("train")].copy()
    train_users = set(train_all.user.unique()) - VAL_USERS
    train_df = train_all[train_all.user.isin(train_users)].copy()
    val_df = train_all[train_all.user.isin(VAL_USERS)].copy()
    stats = stats_for(manifest, cache, train_users)
    np.savez(paths.outputs_dir / "skel_imu_v2_stats.npz", **stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkelImuV2(width=args.width).to(device)
    if args.init_skel_checkpoint is not None:
        load_transfer_skeleton(model, args.init_skel_checkpoint, device)
    if args.init_imu_checkpoint is not None:
        load_transfer_imu(
            model,
            args.init_imu_checkpoint,
            device,
            full=args.transfer_full_imu,
            shared_channels=args.transfer_shared_channels,
        )
    train_loader = DataLoader(CachedDataset(train_df, cache, stats, True), args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(CachedDataset(val_df, cache, stats, False), args.batch_size * 2, shuffle=False, num_workers=args.workers)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    if args.freeze_skel_epochs > 0:
        set_trainable(model.sk, False)
    if args.freeze_imu_epochs > 0:
        set_trainable(model.imu, False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.sk.parameters(), "lr": args.lr * args.skel_lr_mult, "name": "skel"},
            {"params": model.imu.parameters(), "lr": args.lr * args.imu_lr_mult, "name": "imu"},
            {"params": model.head.parameters(), "lr": args.lr * args.head_lr_mult, "name": "head"},
        ],
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    best = -1.0
    history = []
    ckpt_path = paths.outputs_dir / f"{args.output_prefix}_best.pt"
    for epoch in range(1, args.epochs + 1):
        if args.freeze_skel_epochs > 0 and epoch == args.freeze_skel_epochs + 1:
            set_trainable(model.sk, True)
            print(f"unfroze_skeleton_at_epoch={epoch}")
        if args.freeze_imu_epochs > 0 and epoch == args.freeze_imu_epochs + 1:
            set_trainable(model.imu, True)
            print(f"unfroze_imu_at_epoch={epoch}")
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()
        print(f"epoch={epoch:02d} train={tr_loss:.4f}/{tr_acc:.4f} val={va_loss:.4f}/{va_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best:
            best = va_acc
            torch.save({"model": model.state_dict(), "params": vars(args), "val_acc": best}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_logits, val_y, val_ids = collect_logits(model, val_loader, device)
    np.savez_compressed(paths.outputs_dir / f"{args.output_prefix}_val_logits.npz", logits=val_logits, y=val_y, clip_id=val_ids)
    with (paths.outputs_dir / f"{args.output_prefix}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"device={device} best_val={best:.4f} checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
