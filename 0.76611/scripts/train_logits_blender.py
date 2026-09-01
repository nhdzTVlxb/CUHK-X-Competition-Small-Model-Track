from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import Paths


@dataclass(frozen=True)
class SplitLogits:
    logits: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    paths: np.ndarray | None = None


def _as_str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(str)


def load_train_split(yolo_path: Path, skel_path: Path) -> SplitLogits:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    yolo_ids = _as_str(yolo["clip_id"])
    skel_ids = _as_str(skel["clip_id"])
    skel_lookup = {cid: i for i, cid in enumerate(skel_ids)}
    idx = np.array([skel_lookup[cid] for cid in yolo_ids], dtype=np.int64)
    logits = np.concatenate([yolo["logits"], skel["logits"][idx]], axis=1).astype(np.float32)
    return SplitLogits(logits=logits, y=yolo["y"].astype(np.int64), ids=yolo_ids)


def load_train_split_three(yolo_path: Path, skel_path: Path, thermal_path: Path) -> SplitLogits:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    thermal = np.load(thermal_path, allow_pickle=True)
    yolo_ids = _as_str(yolo["clip_id"])
    skel_lookup = {cid: i for i, cid in enumerate(_as_str(skel["clip_id"]))}
    thermal_lookup = {cid: i for i, cid in enumerate(_as_str(thermal["clip_id"]))}
    skel_idx = np.array([skel_lookup[cid] for cid in yolo_ids], dtype=np.int64)
    thermal_idx = np.array([thermal_lookup[cid] for cid in yolo_ids], dtype=np.int64)
    logits = np.concatenate(
        [yolo["logits"], skel["logits"][skel_idx], thermal["logits"][thermal_idx]],
        axis=1,
    ).astype(np.float32)
    return SplitLogits(logits=logits, y=yolo["y"].astype(np.int64), ids=yolo_ids)


def load_val_split(yolo_path: Path, skel_path: Path) -> SplitLogits:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    yolo_ids = _as_str(yolo["clip_id"])
    skel_ids = _as_str(skel["clip_id"])
    skel_lookup = {cid: i for i, cid in enumerate(skel_ids)}
    idx = np.array([skel_lookup[cid] for cid in yolo_ids], dtype=np.int64)
    logits = np.concatenate([yolo["logits"], skel["logits"][idx]], axis=1).astype(np.float32)
    return SplitLogits(logits=logits, y=yolo["y"].astype(np.int64), ids=yolo_ids)


def load_val_split_three(yolo_path: Path, skel_path: Path, thermal_path: Path) -> SplitLogits:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    thermal = np.load(thermal_path, allow_pickle=True)
    yolo_ids = _as_str(yolo["clip_id"])
    skel_lookup = {cid: i for i, cid in enumerate(_as_str(skel["clip_id"]))}
    thermal_lookup = {cid: i for i, cid in enumerate(_as_str(thermal["clip_id"]))}
    skel_idx = np.array([skel_lookup[cid] for cid in yolo_ids], dtype=np.int64)
    thermal_idx = np.array([thermal_lookup[cid] for cid in yolo_ids], dtype=np.int64)
    logits = np.concatenate(
        [yolo["logits"], skel["logits"][skel_idx], thermal["logits"][thermal_idx]],
        axis=1,
    ).astype(np.float32)
    return SplitLogits(logits=logits, y=yolo["y"].astype(np.int64), ids=yolo_ids)


def load_test_split(yolo_path: Path, skel_path: Path) -> tuple[np.ndarray, np.ndarray]:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    yolo_paths = _as_str(yolo["path"])
    skel_paths = _as_str(skel["path"])
    skel_lookup = {path: i for i, path in enumerate(skel_paths)}
    idx = np.array([skel_lookup[path] for path in yolo_paths], dtype=np.int64)
    logits = np.concatenate([yolo["logits"], skel["logits"][idx]], axis=1).astype(np.float32)
    return logits, yolo_paths


def load_test_split_three(yolo_path: Path, skel_path: Path, thermal_path: Path) -> tuple[np.ndarray, np.ndarray]:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    thermal = np.load(thermal_path, allow_pickle=True)
    yolo_paths = _as_str(yolo["path"])
    skel_lookup = {path: i for i, path in enumerate(_as_str(skel["path"]))}
    thermal_lookup = {path: i for i, path in enumerate(_as_str(thermal["path"]))}
    skel_idx = np.array([skel_lookup[path] for path in yolo_paths], dtype=np.int64)
    thermal_idx = np.array([thermal_lookup[path] for path in yolo_paths], dtype=np.int64)
    logits = np.concatenate(
        [yolo["logits"], skel["logits"][skel_idx], thermal["logits"][thermal_idx]],
        axis=1,
    ).astype(np.float32)
    return logits, yolo_paths


class LogitBlender(nn.Module):
    def __init__(self, in_dim: int = 80, hidden: int = 64, num_classes: int = 40, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().detach().cpu())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    losses = []
    correct = 0.0
    total = 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        correct += float((logits.argmax(dim=1) == y).sum().detach().cpu())
        total += int(y.numel())
    return float(np.mean(losses)), correct / max(total, 1)


def infer(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    parts = []
    loader = DataLoader(torch.from_numpy(x.astype(np.float32)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            parts.append(logits.float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--output-prefix", type=str, default="yolo_skel_blender")
    args = parser.parse_args()

    paths = Paths(args.root)
    yolo_train = paths.outputs_dir / "yolo_r2p1d_all_train_logits.npz"
    yolo_val = paths.outputs_dir / "yolo_r2p1d_val_logits.npz"
    yolo_test = paths.outputs_dir / "yolo_r2p1d_logits.npz"
    skel_train = paths.outputs_dir / "skel_imu_v2_train_logits.npz"
    skel_val = paths.outputs_dir / "skel_imu_v2_val_logits.npz"
    skel_test = paths.outputs_dir / "skel_imu_v2_test_logits.npz"

    thermal_train = paths.outputs_dir / "thermal_r2p1d_train_logits.npz"
    thermal_val = paths.outputs_dir / "thermal_r2p1d_train_logits.npz"
    thermal_test = paths.outputs_dir / "thermal_r2p1d_test_logits.npz"

    if thermal_train.exists() and thermal_test.exists():
        train = load_train_split_three(yolo_train, skel_train, thermal_train)
        val = load_val_split_three(yolo_val, skel_val, thermal_val)
        test_x, test_paths = load_test_split_three(yolo_test, skel_test, thermal_test)
    else:
        train = load_train_split(yolo_train, skel_train)
        val = load_val_split(yolo_val, skel_val)
        test_x, test_paths = load_test_split(yolo_test, skel_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogitBlender(in_dim=train.logits.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    print(f"device={device} train={len(train.y)} val={len(val.y)} params={sum(p.numel() for p in model.parameters()):,}")

    train_ds = TensorDataset(torch.from_numpy(train.logits), torch.from_numpy(train.y))
    val_ds = TensorDataset(torch.from_numpy(val.logits), torch.from_numpy(val.y))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_acc = -1.0
    best_path = paths.outputs_dir / f"{args.output_prefix}_best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "params": vars(args),
                    "val_acc": best_acc,
                },
                best_path,
            )

    history_path = paths.outputs_dir / f"{args.output_prefix}_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_logits = infer(model, val.logits, device)
    test_logits = infer(model, test_x, device)
    np.savez_compressed(paths.outputs_dir / f"{args.output_prefix}_val_logits.npz", logits=val_logits, y=val.y, clip_id=val.ids)
    np.savez_compressed(paths.outputs_dir / f"{args.output_prefix}_test_logits.npz", logits=test_logits, path=test_paths)

    submission = pd.DataFrame({"path": test_paths, "prediction": test_logits.argmax(axis=1).astype(int)})
    out_csv = paths.outputs_dir / f"submission_{args.output_prefix}.csv"
    submission.to_csv(out_csv, index=False)
    print(f"best_val_acc={best_acc:.4f}")
    print(f"checkpoint={best_path}")
    print(f"submission={out_csv}")


if __name__ == "__main__":
    main()
