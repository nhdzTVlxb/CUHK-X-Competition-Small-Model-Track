from __future__ import annotations

import argparse
import random
import re
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

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural(path: Path) -> list[int | str]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def discover(root: Path) -> pd.DataFrame:
    rows = []
    thermal_root = root / "Thermal"
    for action in sorted(p for p in thermal_root.iterdir() if p.is_dir()):
        m = re.match(r"^(\d+)", action.name)
        if not m:
            continue
        for user in sorted(p for p in action.glob("user*") if p.is_dir()):
            um = re.match(r"user(\d+)", user.name, re.I)
            if not um:
                continue
            for trial in sorted(p for p in user.iterdir() if p.is_dir()):
                frames = sorted(
                    [p for p in trial.rglob("*") if p.suffix.lower() in IMAGE_EXT],
                    key=natural,
                )
                if frames:
                    rows.append(
                        dict(
                            label=int(m.group(1)),
                            user=int(um.group(1)),
                            frames=frames,
                            clip=str(trial),
                        )
                    )
    return pd.DataFrame(rows)


def sample_indices(n: int, k: int, train: bool) -> list[int]:
    edges = np.linspace(0, n, k + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo = min(int(a), n - 1)
        hi = min(max(int(np.ceil(b)) - 1, int(a)), n - 1)
        out.append(random.randint(lo, hi) if train else (lo + hi) // 2)
    return out


class ThermalClips(Dataset):
    def __init__(self, table: pd.DataFrame, train: bool = False, num_frames: int = 8, image_size: int = 112):
        self.table = table.reset_index(drop=True)
        self.train = train
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        frames = row.frames
        imgs = []
        flip = self.train and random.random() < 0.5
        for j in sample_indices(len(frames), self.num_frames, self.train):
            im = Image.open(frames[j]).convert("RGB").resize((self.image_size, self.image_size))
            if flip:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1) / 255.0
            imgs.append((x - 0.5) / 0.25)
        return torch.stack(imgs), int(row.label), idx


class Block(nn.Module):
    def __init__(self, a: int, b: int, s: int = 1):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(a, b, 3, s, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, 1, 1, bias=False),
            nn.BatchNorm2d(b),
        )
        self.d = (
            nn.Sequential(nn.Conv2d(a, b, 1, s, bias=False), nn.BatchNorm2d(b))
            if (a != b or s != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c(x) + self.d(x))


class FrameNet(nn.Module):
    def __init__(self, num_classes: int = 40):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            Block(32, 32),
            Block(32, 64, 2),
            Block(64, 128, 2),
            Block(128, 256, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        z = self.f(x.reshape(b * t, c, h, w)).flatten(1)
        return self.fc(z).reshape(b, t, -1).mean(1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    pred = []
    truth = []
    with torch.no_grad():
        for x, y, _ in loader:
            pred.append(model(x.to(device)).argmax(1).cpu())
            truth.append(y)
    pred_t = torch.cat(pred)
    truth_t = torch.cat(truth)
    return float((pred_t == truth_t).float().mean().item()), pred_t, truth_t


def train_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    device: torch.device,
    epochs: int,
    batch_size: int,
    num_workers: int,
    num_frames: int,
    image_size: int,
) -> tuple[FrameNet, DataLoader]:
    train_loader = DataLoader(
        ThermalClips(train_df, True, num_frames=num_frames, image_size=image_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        ThermalClips(valid_df, False, num_frames=num_frames, image_size=image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = FrameNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            total += float(loss.item())
        acc, _, _ = evaluate(model, valid_loader, device)
        print(f"epoch={epoch:02d} loss={total / max(len(train_loader), 1):.3f} val_acc={acc:.3f}")
    return model, valid_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "Training" / "data" / "HAR" / "data")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    thermal_root = args.root / "Thermal"
    df = discover(args.root)
    if len(df) == 0:
        raise FileNotFoundError(f"No thermal clips found under {thermal_root}")

    gkf = GroupKFold(5)
    df = df.copy()
    df["fold"] = -1
    for fold, (_, va) in enumerate(gkf.split(df, df.label, df.user)):
        df.loc[va, "fold"] = fold

    train_df = df[df.fold != args.fold].copy()
    valid_df = df[df.fold == args.fold].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} clips={len(df)} train={len(train_df)} val={len(valid_df)}")
    print(f"val_users={sorted(valid_df.user.unique().tolist())}")
    model, valid_loader = train_fold(
        train_df,
        valid_df,
        device,
        args.epochs,
        args.batch_size,
        args.num_workers,
        args.num_frames,
        args.image_size,
    )
    acc, pred, truth = evaluate(model, valid_loader, device)
    print(f"held-out_subject_accuracy={acc:.4f}")
    print(f"prediction_classes={len(set(pred.tolist()))}")


if __name__ == "__main__":
    main()
