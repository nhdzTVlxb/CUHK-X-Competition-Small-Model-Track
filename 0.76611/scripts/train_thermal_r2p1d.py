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

from cuhkx_baseline.yolo_r2p1d import N_CLASSES, R2Plus1D34, dequantize_state, load_checkpoint, predict_batch_logits

VAL_USERS = {"user8", "user9", "user23", "user24"}
MEAN = 0.45
STD = 0.25


class ThermalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        tensor = item["tensor"].astype(np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                tensor = tensor[:, :, :, ::-1].copy()
            if random.random() < 0.25:
                tensor = np.clip(tensor * (0.85 + 0.3 * random.random()), 0.0, 1.0)
        tensor = (tensor - MEAN) / STD
        raw_label = row["label"]
        label = int(raw_label) if pd.notna(raw_label) and str(raw_label) != "" else -1
        return torch.from_numpy(tensor), label, str(row["path"]), str(row["clip_id"])


def make_model(checkpoint: Path, model_index: int = 0) -> nn.Module:
    model = R2Plus1D34()
    packed = load_checkpoint(checkpoint)["models_packed"][model_index]
    state = dequantize_state(packed)
    source_stem = state["encoder.stem.0.weight"]
    state["encoder.stem.0.weight"] = source_stem.mean(dim=1, keepdim=True)
    old = model.encoder.stem[0]
    replacement = type(old)(
        1,
        old.out_channels,
        old.kernel_size,
        old.stride,
        old.padding,
        bias=old.bias is not None,
    )
    model.encoder.stem[0] = replacement
    filtered = {key: value for key, value in state.items() if key != "encoder.stem.0.weight"}
    model.load_state_dict(filtered, strict=False)
    with torch.no_grad():
        model.encoder.stem[0].weight.copy_(state["encoder.stem.0.weight"])
    return model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def run_epoch(model, loader, criterion, optimizer, device, scaler):
    training = optimizer is not None
    model.train(training)
    total = correct = 0
    losses = []
    for tensor, y, _paths, _ids in tqdm(loader, leave=False):
        tensor = tensor.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        amp = scaler is not None and device.type == "cuda"
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(tensor)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(1) == y).sum().detach().cpu())
        total += int(y.numel())
    return float(np.mean(losses)), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "thermal_r2p1d_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model-index", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    manifest = pd.read_csv(args.cache_dir / "train_manifest.csv")
    train = manifest[~manifest["user"].isin(VAL_USERS)].copy()
    val = manifest[manifest["user"].isin(VAL_USERS)].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.checkpoint, args.model_index).to(device)
    train_loader = DataLoader(ThermalDataset(train, args.cache_dir, True), args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(ThermalDataset(val, args.cache_dir, False), max(1, args.batch_size * 2), shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None
    best = -1.0
    history = []
    output = args.root / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "thermal_r2p1d_best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, None)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(f"epoch={epoch:02d} train={train_loss:.4f}/{train_acc:.4f} val={val_loss:.4f}/{val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best, "args": vars(args)}, best_path)
    (output / "thermal_r2p1d_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"device={device} train={len(train)} val={len(val)} best_val={best:.6f} checkpoint={best_path}")


if __name__ == "__main__":
    main()
