from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from predict_yolo_r2p1d import CachedVideoDataset
from cuhkx_baseline.yolo_r2p1d import (
    CHANNELS,
    N_CLASSES,
    R2Plus1D34,
    dequantize_state,
    load_checkpoint,
)

VAL_USERS = {"user8", "user9", "user23", "user24"}


class ValCachedVideoDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir: Path) -> None:
        self.inner = CachedVideoDataset(manifest, cache_dir)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        clip, _path, clip_id = self.inner[idx]
        row = self.inner.manifest.iloc[idx]
        return clip, int(float(row["label"])), clip_id


class TestCachedVideoDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir: Path) -> None:
        self.inner = CachedVideoDataset(manifest, cache_dir)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        clip, _path, clip_id = self.inner[idx]
        return clip, -1, clip_id


def normalize_clip(item: np.ndarray) -> torch.Tensor:
    mean = torch.tensor(
        (0.43216, 0.394666, 0.37645, (0.43216 + 0.394666 + 0.37645) / 3.0),
        dtype=torch.float32,
    ).view(1, CHANNELS, 1, 1)
    std = torch.tensor(
        (0.22803, 0.22145, 0.216989, (0.22803 + 0.22145 + 0.216989) / 3.0),
        dtype=torch.float32,
    ).view(1, CHANNELS, 1, 1)
    clip = torch.from_numpy(item.astype(np.float32)).div_(255.0)
    return clip.sub_(mean).div_(std)


def spatial_resize(clip: torch.Tensor, size: int) -> torch.Tensor:
    batch, frames, channels, height, width = clip.shape
    flat = clip.reshape(batch * frames, channels, height, width)
    resized = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False)
    return resized.reshape(batch, frames, channels, size, size)


def spatial_view(clip: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "base":
        return clip
    if mode == "zoom_in":
        enlarged = spatial_resize(clip, 144)
        return enlarged[..., 8:136, 8:136]
    if mode == "zoom_out":
        smaller = spatial_resize(clip, 112)
        batch, frames, channels, height, width = smaller.shape
        flat = smaller.reshape(batch * frames, channels, height, width)
        padded = F.pad(flat, (8, 8, 8, 8), mode="replicate")
        return padded.reshape(batch, frames, channels, 128, 128)
    if mode in {"shift_left", "shift_right"}:
        pixels = 8
        flat = clip.reshape(-1, clip.shape[2], clip.shape[3], clip.shape[4])
        if mode == "shift_left":
            shifted = F.pad(flat[..., pixels:], (0, pixels, 0, 0), mode="replicate")
        else:
            shifted = F.pad(flat[..., :-pixels], (pixels, 0, 0, 0), mode="replicate")
        return shifted.reshape_as(clip)
    raise ValueError(f"Unsupported view: {mode}")


def temporal_view(clip: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "base":
        return clip
    if mode == "temporal_prev":
        return torch.roll(clip, shifts=1, dims=1)
    if mode == "temporal_next":
        return torch.roll(clip, shifts=-1, dims=1)
    raise ValueError(f"Unsupported temporal view: {mode}")


def predict_pair(model: torch.nn.Module, clip: torch.Tensor) -> torch.Tensor:
    return model(clip) + model(torch.flip(clip, dims=(-1,)))


def infer(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    view: str,
    temporal: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    clip_ids: list[str] = []
    with torch.inference_mode():
        for inputs, y, ids in tqdm(loader, desc=f"{view}/{temporal}"):
            inputs = inputs.to(device, non_blocking=True)
            inputs = spatial_view(inputs, view)
            inputs = temporal_view(inputs, temporal)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = predict_pair(model, inputs)
            parts.append(logits.float().cpu().numpy())
            labels.append(y.numpy())
            clip_ids.extend(str(value) for value in ids)
    return (
        np.concatenate(parts, axis=0),
        np.concatenate(labels, axis=0),
        np.asarray(clip_ids, dtype=str),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v1")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "external_models" / "ensemble_packed.pt",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "yolo_view_tta_logits.npz")
    parser.add_argument(
        "--views",
        nargs="+",
        default=["base", "zoom_in", "zoom_out", "shift_left", "shift_right"],
    )
    parser.add_argument(
        "--temporals",
        nargs="+",
        default=["base", "temporal_prev", "temporal_next"],
    )
    args = parser.parse_args()

    train_manifest = pd.read_csv(args.cache_dir / "train_manifest.csv")
    if "user" not in train_manifest:
        train_manifest["user"] = train_manifest["path"].astype(str).str.split("/").str[-2]
    val_manifest = train_manifest[train_manifest["user"].isin(VAL_USERS)].copy()
    test_manifest = pd.read_csv(args.cache_dir / "test_manifest.csv")

    val_loader = DataLoader(
        ValCachedVideoDataset(val_manifest, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        TestCachedVideoDataset(test_manifest, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    all_val: dict[str, np.ndarray] = {}
    all_test: dict[str, np.ndarray] = {}
    val_y: np.ndarray | None = None
    val_ids: np.ndarray | None = None
    test_ids = test_manifest["clip_id"].astype(str).to_numpy()
    test_paths = test_manifest["path"].astype(str).to_numpy()

    for model_index, packed in enumerate(checkpoint["models_packed"]):
        model = R2Plus1D34().to(device)
        model.load_state_dict(dequantize_state(packed))
        model.eval()
        for view in args.views:
            for temporal in args.temporals:
                key = f"model{model_index}_{view}_{temporal}"
                val_logits, current_y, current_val_ids = infer(
                    model, val_loader, device, view, temporal
                )
                test_logits, _unused_y, _unused_ids = infer(
                    model, test_loader, device, view, temporal
                )
                all_val[key] = val_logits
                all_test[key] = test_logits
                if val_y is None:
                    val_y = current_y
                    val_ids = current_val_ids
                elif not np.array_equal(val_y, current_y):
                    raise RuntimeError("Validation labels changed between views")
        del model
        torch.cuda.empty_cache()

    if val_y is None or val_ids is None:
        raise RuntimeError("No views were inferred")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        val_y=val_y,
        val_clip_id=val_ids,
        test_clip_id=test_ids,
        test_path=test_paths,
        **{f"val_{key}": value for key, value in all_val.items()},
        **{f"test_{key}": value for key, value in all_test.items()},
    )
    print(f"device={device}")
    print(f"val_rows={len(val_y)} test_rows={len(test_ids)}")
    print(f"views={args.views} temporals={args.temporals}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
