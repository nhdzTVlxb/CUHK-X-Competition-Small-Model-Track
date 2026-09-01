from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from cuhkx_baseline.features import Paths, iter_test_clips, iter_train_clips

Image.MAX_IMAGE_PIXELS = None

N_CLASSES = 40
N_FRAMES = 16
DETECTION_FRAMES = 8
CHANNELS = 4
IMAGE_SIZE = 128
PERSON_CONFIDENCE = 0.25
CROP_MARGIN = 1.4
MIN_SIDE_FRACTION = 0.35
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)
NUMBER_PATTERN = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    path: str
    depth_dir: str
    ir_dir: str
    label: int | None = None


def natural_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in NUMBER_PATTERN.split(value.casefold())
        if part
    )


def pick_indices(length: int, count: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    return tuple(np.linspace(0, length - 1, count).round().astype(int).tolist())


def list_image_files(clip_dir: Path) -> list[Path]:
    if not clip_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(
        (p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: natural_key(p.name),
    )


def read_image(path: Path, mode: str) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            return image.convert(mode)
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError):
        return None


def window_from_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    values = np.asarray(boxes, dtype=np.float64)
    x0, y0 = values[:, :2].min(axis=0)
    x1, y1 = values[:, 2:].max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    width, height = 640.0, 480.0
    side = max((x1 - x0) * width, (y1 - y0) * height) * CROP_MARGIN
    side = max(side, MIN_SIDE_FRACTION * max(width, height))
    hx, hy = side / width / 2.0, side / height / 2.0
    return (
        max(cx - hx, 0.0),
        max(cy - hy, 0.0),
        min(cx + hx, 1.0),
        min(cy + hy, 1.0),
    )


def build_clip_records(paths: Paths, split: str) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    if split == "test":
        for row in iter_test_clips(paths.test_root, paths.test_csv):
            records.append(
                ClipRecord(
                    clip_id=str(row["clip_id"]),
                    path=str(row["path"]),
                    depth_dir=str(row["depth_dir"]),
                    ir_dir=str(row["ir_dir"]),
                    label=None,
                )
            )
    elif split == "train":
        for row in iter_train_clips(paths.train_root):
            records.append(
                ClipRecord(
                    clip_id=str(row["clip_id"]),
                    path=f"{row['action']}/{row['user']}/{row['trial']}",
                    depth_dir=str(row["depth_dir"]),
                    ir_dir=str(row["ir_dir"]),
                    label=int(float(row["label"])),
                )
            )
    else:
        raise ValueError(f"Unsupported split: {split}")
    return records


def detect_windows(
    clips: list[ClipRecord],
    yolo_path: Path,
    cache_path: Path | None = None,
    batch_size: int = 16,
    device: str | int = "cpu",
) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if set(cached["clip_id"].astype(str)) >= {clip.clip_id for clip in clips}:
            return cached

    from ultralytics import YOLO

    model = YOLO(str(yolo_path))
    boxes_by_clip: list[list[tuple[float, float, float, float]]] = [[] for _ in clips]
    frame_batch: list[np.ndarray] = []
    owner_batch: list[int] = []
    readable = 0

    def flush() -> None:
        nonlocal readable
        if not frame_batch:
            return
        results = model.predict(
            frame_batch,
            classes=[0],
            conf=PERSON_CONFIDENCE,
            verbose=False,
            device=device,
            batch=batch_size,
        )
        for owner, result in zip(owner_batch, results, strict=True):
            readable += 1
            if len(result.boxes):
                box = result.boxes.xyxy[result.boxes.conf.argmax()].tolist()
                height, width = result.orig_shape
                boxes_by_clip[owner].append(
                    (
                        float(box[0]) / width,
                        float(box[1]) / height,
                        float(box[2]) / width,
                        float(box[3]) / height,
                    )
                )
        frame_batch.clear()
        owner_batch.clear()

    for owner, clip in enumerate(clips):
        files = list_image_files(Path(clip.ir_dir))
        for index in sorted(set(pick_indices(len(files), DETECTION_FRAMES))):
            image = read_image(files[index], "RGB")
            if image is None:
                continue
            array = np.asarray(image)
            if not array.any():
                continue
            frame_batch.append(array)
            owner_batch.append(owner)
            if len(frame_batch) >= batch_size:
                flush()
    flush()

    rows = []
    for clip, boxes in zip(clips, boxes_by_clip, strict=True):
        window = window_from_boxes(boxes)
        rows.append(
            {
                "clip_id": clip.clip_id,
                "path": clip.path,
                "x0": window[0] if window else np.nan,
                "y0": window[1] if window else np.nan,
                "x1": window[2] if window else np.nan,
                "y1": window[3] if window else np.nan,
                "has_crop": bool(window),
            }
        )

    df = pd.DataFrame(rows)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    del model
    torch.cuda.empty_cache()
    return df


def crop_and_resize(image: Image.Image, window: tuple[float, float, float, float] | None, size: int, mode: str) -> np.ndarray:
    if window is not None:
        iw, ih = image.size
        image = image.crop(
            (
                round(window[0] * iw),
                round(window[1] * ih),
                round(window[2] * iw),
                round(window[3] * ih),
            )
        )
    resample = getattr(Image, "Resampling", Image).BILINEAR
    arr = np.asarray(image.resize((size, size), resample=resample), dtype=np.uint8)
    if mode == "L":
        return arr[None]
    return np.moveaxis(arr, -1, 0)


def load_clip_tensor(
    depth_dir: Path,
    ir_dir: Path,
    window: tuple[float, float, float, float] | None,
    frames: int = N_FRAMES,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    depth_files = list_image_files(depth_dir)
    ir_files = list_image_files(ir_dir)
    output = np.zeros((frames, CHANNELS, image_size, image_size), dtype=np.uint8)
    for dst_i, src_i in enumerate(pick_indices(len(depth_files), frames)):
        image = read_image(depth_files[src_i], "RGB")
        if image is not None:
            try:
                output[dst_i, :3] = crop_and_resize(image, window, image_size, "RGB")
            except Exception:
                pass
    for dst_i, src_i in enumerate(pick_indices(len(ir_files), frames)):
        image = read_image(ir_files[src_i], "L")
        if image is not None:
            try:
                output[dst_i, 3:] = crop_and_resize(image, window, image_size, "L")
            except Exception:
                pass
    return output


class ClipTensorDataset(Dataset):
    def __init__(self, clips: list[ClipRecord], windows: pd.DataFrame) -> None:
        self.clips = clips
        self.windows = windows.set_index("clip_id")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        row = self.windows.loc[clip.clip_id]
        window = None
        if bool(row["has_crop"]) and not pd.isna(row["x0"]):
            window = (float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"]))
        tensor = load_clip_tensor(Path(clip.depth_dir), Path(clip.ir_dir), window)
        if clip.label is None:
            return torch.from_numpy(tensor.astype(np.float32)), clip.path, clip.clip_id
        return torch.from_numpy(tensor.astype(np.float32)), int(clip.label), clip.path, clip.clip_id


class R2Plus1D34(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from .vendor_ig65m_models import r2plus1d_34_32_kinetics

        encoder = r2plus1d_34_32_kinetics(num_classes=400, pretrained=False)
        conv = encoder.stem[0]
        replacement = type(conv)(
            4,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            replacement.weight[:, :3] = conv.weight
            replacement.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True)
            if conv.bias is not None:
                replacement.bias.copy_(conv.bias)
        encoder.stem[0] = replacement
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(encoder.fc.in_features, N_CLASSES))
        self.encoder.fc = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(inputs.permute(0, 2, 1, 3, 4)))


def unpack_signed(packed: torch.Tensor, shape: tuple[int, ...], bits: int) -> torch.Tensor:
    count = math.prod(shape)
    starts = torch.arange(count, dtype=torch.int64) * bits
    codes = torch.zeros(count, dtype=torch.int16)
    source = packed.to(torch.int16)
    for bit in range(bits):
        positions = starts + bit
        byte_indices = positions >> 3
        shifts = positions & 7
        values = torch.bitwise_and(source[byte_indices] >> shifts, 1)
        codes |= values << bit
    sign, modulus = 1 << (bits - 1), 1 << bits
    signed = torch.where(codes >= sign, codes - modulus, codes)
    return signed.to(torch.int8).reshape(shape)


def dequantize_state(state: Mapping[str, object]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if isinstance(value, Mapping):
            shape = tuple(int(item) for item in value["shape"])
            quantized = unpack_signed(value["packed"], shape, int(value["bits"]))
            output[key] = quantized.float() * value["scale"].float()
        else:
            output[key] = value.float() if hasattr(value, "is_floating_point") and value.is_floating_point() else value
    return output


def load_checkpoint(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt.get("schema_version") != "kuno-yolo-r2p1d-packed-ensemble/v1":
        raise ValueError(f"Unexpected checkpoint schema: {ckpt.get('schema_version')}")
    return ckpt


def load_ensemble_models(checkpoint_path: Path, device: torch.device) -> tuple[list[R2Plus1D34], list[float]]:
    ckpt = load_checkpoint(checkpoint_path)
    weights = [float(w) for w in ckpt["weights"]]
    models: list[R2Plus1D34] = []
    for packed in ckpt["models_packed"]:
        model = R2Plus1D34().to(device)
        state = dequantize_state(packed)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    return models, weights


def predict_batch_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=inputs.device.type, dtype=torch.float16, enabled=inputs.device.type == "cuda"):
        return model(inputs) + model(torch.flip(inputs, dims=(-1,)))


def infer_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "infer",
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    model.eval()
    logits: list[np.ndarray] = []
    paths: list[str] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, y, path_batch, _clip_ids = batch
            else:
                inputs, path_batch, _clip_ids = batch
                y = None
            inputs = inputs.to(device, non_blocking=True)
            out = predict_batch_logits(model, inputs)
            logits.append(out.float().cpu().numpy())
            paths.extend(list(path_batch))
            if y is not None:
                labels.extend(torch.as_tensor(y).cpu().tolist())
    stacked = np.concatenate(logits, axis=0) if logits else np.zeros((0, N_CLASSES), dtype=np.float32)
    label_arr = np.asarray(labels, dtype=np.int64) if labels else None
    return stacked, paths, label_arr


def build_loader(clips: list[ClipRecord], windows: pd.DataFrame, batch_size: int, num_workers: int, shuffle: bool = False) -> DataLoader:
    dataset = ClipTensorDataset(clips, windows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
