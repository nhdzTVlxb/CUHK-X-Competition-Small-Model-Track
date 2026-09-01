from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cuhkx_baseline.yolo_r2p1d import N_CLASSES, R2Plus1D34, dequantize_state, load_checkpoint  # noqa: E402
from finetune_yolo_r2p1d import MEAN, STD, make_model, seed_everything, set_trainable  # noqa: E402
from sequence_postprocess import collect_train_metadata  # noqa: E402


def build_metadata(root: Path) -> pd.DataFrame:
    meta = collect_train_metadata(root)[["clip_id", "user", "start"]].copy()
    meta["date"] = meta["start"].dt.date.astype(str)
    return meta[["clip_id", "user", "date"]]


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


def augment(clip: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        clip = torch.flip(clip, dims=(-1,))
    clip = temporal_shift(clip)
    clip = spatial_shift(clip)
    rgb_scale = random.uniform(0.90, 1.10)
    ir_scale = random.uniform(0.88, 1.12)
    clip[:, :3] = torch.clamp(clip[:, :3] * rgb_scale, 0.0, 1.0)
    clip[:, 3:] = torch.clamp(clip[:, 3:] * ir_scale, 0.0, 1.0)
    if random.random() < 0.10:
        clip[:, 3:] = 0.0
    if random.random() < 0.04:
        clip[:, :3] = 0.0
    if random.random() < 0.25:
        clip = torch.clamp(clip + torch.randn_like(clip) * 0.010, 0.0, 1.0)
    if random.random() < 0.25:
        clip[random.randrange(clip.shape[0])] = 0.0
    return clip


class DistillDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        clean = torch.from_numpy(item["tensor"].astype(np.float32)).div_(255.0)
        student = augment(clean.clone()) if self.train else clean.clone()
        return (student - MEAN) / STD, (clean - MEAN) / STD, int(row["label"]), str(row["clip_id"])


def date_class_sampler(frame: pd.DataFrame, meta: pd.DataFrame) -> WeightedRandomSampler:
    labels = frame["label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=N_CLASSES)
    class_weight = np.asarray([1.0 / max(np.sqrt(counts[label]), 1.0) for label in labels], dtype=np.float64)
    class_weight /= class_weight.mean()
    date_weight = {
        "2025-05-31": 1.35,
        "2025-06-01": 1.30,
        "2025-06-02": 2.00,
        "2025-06-12": 1.15,
        "2025-06-13": 1.20,
    }
    lookup = meta.set_index("clip_id")["date"].to_dict()
    weights = np.asarray(
        [class_weight[i] * date_weight.get(str(lookup.get(str(cid), "")), 1.0)
         for i, cid in enumerate(frame["clip_id"].astype(str))],
        dtype=np.float64,
    )
    weights /= weights.mean()
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def infer_teacher(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        clean = model(inputs)
        flipped = model(torch.flip(inputs, dims=(-1,)))
    return (clean + flipped) * 0.5


def train_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    ce_weight: float,
    kd_weight: float,
    temperature: float,
) -> tuple[float, float]:
    student.train()
    teacher.eval()
    total = correct = 0
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for student_x, teacher_x, labels, _ids in tqdm(loader, leave=False):
        student_x = student_x.to(device, non_blocking=True)
        teacher_x = teacher_x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            with torch.no_grad():
                teacher_logits = infer_teacher(teacher, teacher_x)
            student_logits = student(student_x)
            ce = F.cross_entropy(student_logits, labels, label_smoothing=0.03)
            kd = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)
            loss = ce_weight * ce + kd_weight * kd
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        correct += int((student_logits.argmax(1) == labels).sum().detach().cpu())
        total += int(labels.numel())
    return float(np.mean(losses)), correct / max(total, 1)


@torch.no_grad()
def evaluate(student: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray, list[str]]:
    student.eval()
    losses: list[float] = []
    logits_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    ids: list[str] = []
    for student_x, _teacher_x, labels, batch_ids in tqdm(loader, desc="infer", leave=False):
        student_x = student_x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = student(student_x) + student(torch.flip(student_x, dims=(-1,)))
            loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
        losses.append(float(loss.cpu()))
        logits_parts.append(logits.float().cpu().numpy())
        labels_parts.append(labels.cpu().numpy())
        ids.extend(batch_ids)
    logits = np.concatenate(logits_parts)
    labels = np.concatenate(labels_parts)
    return float(np.mean(losses)), float(np.mean(logits.argmax(1) == labels)), logits, labels, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--kd-weight", type=float, default=0.70)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-prefix", type=str, default="finetune_yolo_distill_m0_v1")
    args = parser.parse_args()

    seed_everything(args.seed)
    manifest = pd.read_csv(args.cache_dir / "train_manifest.csv")
    meta = build_metadata(args.root)
    manifest = manifest.merge(meta, on="clip_id", how="left")
    manifest["date"] = manifest["date"].fillna("unknown")
    train = manifest.copy()
    val = manifest.copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = make_model(args.checkpoint, args.model_index, None).to(device)
    teacher = make_model(args.checkpoint, args.model_index, None).to(device)
    set_trainable(student, "head")
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    teacher.eval()

    sampler = date_class_sampler(train, meta)
    train_loader = DataLoader(
        DistillDataset(train, args.cache_dir, True),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        DistillDataset(val, args.cache_dir, False),
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None

    output_dir = args.root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.output_prefix}_best.pt"
    history: list[dict[str, float]] = []
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            student, teacher, train_loader, device, optimizer, scaler,
            ce_weight=1.0, kd_weight=args.kd_weight, temperature=args.temperature,
        )
        val_loss, val_acc, _val_logits, _val_y, _val_ids = evaluate(student, val_loader, device)
        scheduler.step()
        row = {"epoch": float(epoch), "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)
        print(f"epoch={epoch:02d} train={train_loss:.4f}/{train_acc:.4f} val={val_loss:.4f}/{val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save({"model": student.state_dict(), "val_acc": best, "args": vars(args)}, checkpoint_path)

    (output_dir / f"{args.output_prefix}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student.load_state_dict(state["model"])
    _val_loss, _val_acc, val_logits, val_y, val_ids = evaluate(student, val_loader, device)
    test_manifest = pd.read_csv(args.cache_dir / "test_manifest.csv").assign(label=-1)
    test_loader = DataLoader(
        DistillDataset(test_manifest, args.cache_dir, False),
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_logits_parts: list[np.ndarray] = []
    test_ids: list[str] = []
    student.eval()
    with torch.no_grad():
        for student_x, _teacher_x, _labels, batch_ids in tqdm(test_loader, desc="test", leave=False):
            student_x = student_x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = student(student_x) + student(torch.flip(student_x, dims=(-1,)))
            test_logits_parts.append(logits.float().cpu().numpy())
            test_ids.extend(batch_ids)
    test_logits = np.concatenate(test_logits_parts)
    np.savez_compressed(
        output_dir / f"{args.output_prefix}_logits.npz",
        val_logits=val_logits,
        val_y=val_y,
        val_clip_id=np.asarray(val_ids, dtype=str),
        test_logits=test_logits,
        test_clip_id=np.asarray(test_ids, dtype=str),
        test_path=test_manifest["path"].astype(str).to_numpy(),
    )
    print(f"device={device} train={len(train)} best_val={best:.6f} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
