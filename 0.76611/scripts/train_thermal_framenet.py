from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import Paths, sample_indices  # noqa: E402

N_CLASSES = 40
N_FRAMES = 8
IMAGE_SIZE = 112
BATCH_SIZE = 32
EPOCHS = 8
MEAN = 0.5
STD = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_files(path: Path) -> list[Path]:
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", p.name)],
    ) if path.exists() else []


def thermal_dir(paths: Paths, row: pd.Series, split: str) -> Path:
    if split == "train":
        action, user, trial = str(row["path"]).split("/")
        return paths.train_root / "Thermal" / action / user / trial
    clip_id = str(row["clip_id"])
    return paths.test_root / clip_id / "Thermal"


def load_clip(path: Path) -> np.ndarray:
    files = image_files(path)
    output = np.zeros((N_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if not files:
        return output
    for dst, source_index in enumerate(sample_indices(len(files), N_FRAMES)):
        try:
            with Image.open(files[int(source_index)]) as image:
                image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                output[dst] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
        except Exception:
            continue
    return output


def prepare_cache(root: Path, manifest: pd.DataFrame, cache_dir: Path, force: bool) -> pd.DataFrame:
    paths = Paths(root)
    records: list[dict[str, object]] = []
    for split, frame in (("train", manifest), ("test", pd.DataFrame())):
        if split == "test":
            frame = pd.read_csv(paths.test_csv)
            frame["clip_id"] = frame["path"].astype(str).str.rstrip("/").str.split("/").str[-1]
            frame["label"] = -1
            frame["user"] = ""
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"cache {split}"):
            row_dict = row._asdict()
            clip_id = str(row_dict["clip_id"])
            relative = Path(split) / f"{clip_id}.npy"
            output = cache_dir / relative
            if force or not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, load_clip(thermal_dir(paths, pd.Series(row_dict), split)))
            records.append(
                {
                    "split": split,
                    "clip_id": clip_id,
                    "path": str(row_dict["path"]),
                    "feature_file": str(relative).replace("\\", "/"),
                    "label": int(row_dict["label"]),
                    "user": str(row_dict.get("user", "")),
                }
            )
    result = pd.DataFrame(records)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_dir / "manifest.csv", index=False)
    result[result["split"].eq("train")].to_csv(cache_dir / "train_manifest.csv", index=False)
    result[result["split"].eq("test")].to_csv(cache_dir / "test_manifest.csv", index=False)
    return result


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.main(x) + self.skip(x))


class FrameNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            Block(32, 32),
            Block(32, 64, 2),
            Block(64, 128, 2),
            Block(128, 256, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = x.shape
        features = self.encoder(x.reshape(batch * frames, channels, height, width)).flatten(1)
        return self.head(features).reshape(batch, frames, -1).mean(1)


class ThermalClips(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        clip = np.load(self.cache_dir / row["feature_file"]).astype(np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                clip = clip[:, :, :, ::-1].copy()
            if random.random() < 0.25:
                clip = np.clip(clip * random.uniform(0.85, 1.15), 0.0, 1.0)
        clip = torch.from_numpy((clip - MEAN) / STD)
        return clip, int(row["label"]), str(row["clip_id"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    ids: list[str] = []
    for x, labels, clip_ids in loader:
        logits_parts.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
        labels_parts.append(labels.numpy())
        ids.extend(clip_ids)
    logits = np.concatenate(logits_parts)
    labels = np.concatenate(labels_parts)
    return float((logits.argmax(1) == labels).mean()), logits, labels, ids


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    losses: list[float] = []
    correct = total = 0
    for x, labels, _ids in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, labels)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(1) == labels).sum().detach().cpu())
        total += int(labels.numel())
    return float(np.mean(losses)), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "thermal_framenet_v1")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="thermal_framenet_f0_v1")
    args = parser.parse_args()

    seed_everything(args.seed + args.fold)
    source = pd.read_csv(args.root / "cache" / "thermal_r2p1d_v1" / "train_manifest.csv")
    source["user"] = source["user"].astype(str)
    if args.force_cache or not (args.cache_dir / "train_manifest.csv").exists():
        manifest = prepare_cache(args.root, source, args.cache_dir, args.force_cache)
    else:
        manifest = pd.read_csv(args.cache_dir / "manifest.csv")

    train = manifest[manifest["split"].eq("train")].copy()
    test = manifest[manifest["split"].eq("test")].copy()
    splitter = GroupKFold(5)
    if args.fold < 0:
        train_frame = train.copy()
        val_frame = train.copy()
    else:
        splits = list(splitter.split(train, train["label"], train["user"]))
        train_index, val_index = splits[args.fold]
        train_frame = train.iloc[train_index].copy()
        val_frame = train.iloc[val_index].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameNet().to(device)
    train_loader = DataLoader(
        ThermalClips(train_frame, args.cache_dir, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ThermalClips(val_frame, args.cache_dir, False),
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ThermalClips(test, args.cache_dir, False),
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None

    output_dir = args.root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{args.output_prefix}_best.pt"
    history: list[dict[str, float]] = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_acc, _val_logits, _val_y, _val_ids = evaluate(model, val_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"epoch={epoch:02d} train={train_loss:.4f}/{train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "val_acc": best, "args": vars(args)}, checkpoint)

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    val_acc, val_logits, val_y, val_ids = evaluate(model, val_loader, device)
    test_logits_parts: list[np.ndarray] = []
    test_ids: list[str] = []
    model.eval()
    with torch.no_grad():
        for x, _labels, ids in tqdm(test_loader, desc="test", leave=False):
            test_logits_parts.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
            test_ids.extend(ids)
    test_logits = np.concatenate(test_logits_parts)
    np.savez_compressed(
        output_dir / f"{args.output_prefix}_logits.npz",
        val_logits=val_logits,
        val_y=val_y,
        val_clip_id=np.asarray(val_ids, dtype=str),
        test_logits=test_logits,
        test_clip_id=np.asarray(test_ids, dtype=str),
        test_path=test["path"].astype(str).to_numpy(),
    )
    (output_dir / f"{args.output_prefix}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"device={device} fold={args.fold} train={len(train_frame)} val={len(val_frame)} best_val={best:.6f}")


if __name__ == "__main__":
    main()
