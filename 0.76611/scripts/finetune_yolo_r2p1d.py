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

from cuhkx_baseline.yolo_r2p1d import (  # noqa: E402
    CHANNELS,
    N_CLASSES,
    R2Plus1D34,
    dequantize_state,
    load_checkpoint,
)

VAL_USERS = {"user8", "user9", "user23", "user24"}
MEAN = torch.tensor(
    (0.43216, 0.394666, 0.37645, (0.43216 + 0.394666 + 0.37645) / 3.0),
    dtype=torch.float32,
).view(1, CHANNELS, 1, 1)
STD = torch.tensor(
    (0.22803, 0.22145, 0.216989, (0.22803 + 0.22145 + 0.216989) / 3.0),
    dtype=torch.float32,
).view(1, CHANNELS, 1, 1)


class CachedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        clip = torch.from_numpy(item["tensor"].astype(np.float32)).div_(255.0)
        if self.train and random.random() < 0.5:
            clip = torch.flip(clip, dims=(-1,))
        clip = (clip - MEAN) / STD
        return clip, int(row["label"]), str(row["clip_id"])


def make_model(checkpoint: Path, model_index: int, resume_checkpoint: Path | None) -> nn.Module:
    model = R2Plus1D34()
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        return model
    packed = load_checkpoint(checkpoint)["models_packed"][model_index]
    model.load_state_dict(dequantize_state(packed))
    return model


def set_trainable(model: nn.Module, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = mode != "head"
    if mode == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad = True
    elif mode == "last":
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        for parameter in model.encoder.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.head.parameters():
            parameter.requires_grad = True


def make_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    labels = frame["label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=N_CLASSES)
    weights = np.asarray([1.0 / max(counts[label], 1) for label in labels], dtype=np.float64)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    criterion: nn.Module,
    accumulation: int,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total = correct = 0
    losses: list[float] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (inputs, labels, _ids) in enumerate(tqdm(loader, leave=False)):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(inputs)
                loss = criterion(logits, labels)
                scaled_loss = loss / accumulation if training else loss
            if training:
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(1) == labels).sum().detach().cpu())
        total += int(labels.numel())
    return float(np.mean(losses)), correct / max(total, 1)


def infer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    ids: list[str] = []
    with torch.inference_mode():
        for inputs, labels, batch_ids in tqdm(loader, desc="infer", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(inputs) + model(torch.flip(inputs, dims=(-1,)))
            logits_parts.append(output.float().cpu().numpy())
            labels_parts.append(labels.numpy())
            ids.extend(batch_ids)
    return np.concatenate(logits_parts), np.concatenate(labels_parts), ids


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--mode", choices=["head", "last"], default="head")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="finetune_yolo_head")
    args = parser.parse_args()

    seed_everything(args.seed)
    manifest = pd.read_csv(args.cache_dir / "train_manifest.csv")
    if "user" not in manifest:
        manifest["user"] = manifest["path"].astype(str).str.split("/").str[-2]
    train = manifest[~manifest["user"].isin(VAL_USERS)].copy()
    val = manifest[manifest["user"].isin(VAL_USERS)].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.checkpoint, args.model_index, args.resume_checkpoint).to(device)
    set_trainable(model, args.mode)
    trainable = [p for p in model.parameters() if p.requires_grad]
    sampler = make_sampler(train) if args.balanced else None
    train_loader = DataLoader(
        CachedDataset(train, args.cache_dir, True),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        CachedDataset(val, args.cache_dir, False),
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if device.type == "cuda"
        else None
    )

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
        print(
            f"epoch={epoch:02d} train={train_loss:.4f}/{train_acc:.4f} "
            f"val={val_loss:.4f}/{val_acc:.4f}"
        )
        if val_acc > best:
            best = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_acc": best,
                    "args": vars(args),
                },
                checkpoint_path,
            )
    (output_dir / f"{args.output_prefix}_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    best_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    val_logits, val_y, val_ids = infer(model, val_loader, device)
    test_manifest = pd.read_csv(args.cache_dir / "test_manifest.csv")
    test_loader = DataLoader(
        CachedDataset(
            test_manifest.assign(label=-1),
            args.cache_dir,
            False,
        ),
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
        f"device={device} mode={args.mode} train={len(train)} val={len(val)} "
        f"best_val={best:.6f} checkpoint={checkpoint_path}"
    )


if __name__ == "__main__":
    main()
