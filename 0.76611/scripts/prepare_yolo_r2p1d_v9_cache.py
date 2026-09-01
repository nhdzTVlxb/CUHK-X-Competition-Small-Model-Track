from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.yolo_r2p1d import (
    DETECTION_FRAMES,
    IMAGE_SIZE,
    N_FRAMES,
    build_clip_records,
    list_image_files,
    natural_key,
    pick_indices,
    read_image,
)
from cuhkx_baseline.features import Paths

WIDTH, HEIGHT = 640.0, 480.0
CONF = 0.25
MARGIN = 1.40
MIN_SIDE = 0.35


def window_from_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    values = np.asarray(boxes, dtype=np.float64)
    centers_x = (values[:, 0] + values[:, 2]) / 2.0
    centers_y = (values[:, 1] + values[:, 3]) / 2.0
    cx, cy = float(np.median(centers_x)), float(np.median(centers_y))
    widths = values[:, 2] - values[:, 0]
    heights = values[:, 3] - values[:, 1]
    side = max(float(widths.max()), float(heights.max()))
    # Keep the model's training scale while using a robust center.
    side = max(side * max(WIDTH, HEIGHT) * MARGIN, MIN_SIDE * max(WIDTH, HEIGHT))
    hx, hy = side / WIDTH / 2.0, side / HEIGHT / 2.0
    return (max(cx - hx, 0.0), max(cy - hy, 0.0), min(cx + hx, 1.0), min(cy + hy, 1.0))


def detect_v9(clips: list[dict], yolo_path: Path, device: str, batch_size: int) -> pd.DataFrame:
    from ultralytics import YOLO

    model = YOLO(str(yolo_path))
    boxes_by_clip: list[list[tuple[float, float, float, float]]] = [[] for _ in clips]
    frame_batch: list[np.ndarray] = []
    owner_batch: list[int] = []

    def flush() -> None:
        if not frame_batch:
            return
        results = model.predict(
            frame_batch, classes=[0], conf=CONF, verbose=False, device=device, batch=batch_size
        )
        for owner, result in zip(owner_batch, results, strict=True):
            h, w = result.orig_shape
            for box in result.boxes.xyxy.tolist():
                boxes_by_clip[owner].append(
                    (float(box[0]) / w, float(box[1]) / h, float(box[2]) / w, float(box[3]) / h)
                )
        frame_batch.clear()
        owner_batch.clear()

    def queue_probe(owner: int, files: list[Path]) -> None:
        for index in sorted(set(pick_indices(len(files), DETECTION_FRAMES))):
            image = read_image(files[index], "RGB")
            if image is None:
                continue
            array = np.asarray(image)
            if array.any():
                frame_batch.append(array)
                owner_batch.append(owner)
                if len(frame_batch) >= batch_size:
                    flush()

    for owner, clip in enumerate(tqdm(clips, desc="IR probes")):
        queue_probe(owner, clip["ir_files"])
    flush()

    missed = [i for i, boxes in enumerate(boxes_by_clip) if not boxes]
    for owner in tqdm(missed, desc="Depth fallback probes"):
        queue_probe(owner, clips[owner]["depth_files"])
    flush()

    rows = []
    for clip, boxes in zip(clips, boxes_by_clip, strict=True):
        window = window_from_boxes(boxes)
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "path": clip["path"],
                "x0": window[0] if window else np.nan,
                "y0": window[1] if window else np.nan,
                "x1": window[2] if window else np.nan,
                "y1": window[3] if window else np.nan,
                "has_crop": bool(window),
                "n_boxes": len(boxes),
            }
        )
    del model
    return pd.DataFrame(rows)


def crop_resize(image: Image.Image, window, mode: str) -> np.ndarray:
    if window is not None:
        iw, ih = image.size
        image = image.crop(
            (round(window[0] * iw), round(window[1] * ih), round(window[2] * iw), round(window[3] * ih))
        )
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.uint8)
    return array[None] if mode == "L" else np.moveaxis(array, -1, 0)


def load_tensor(depth_files: list[Path], ir_files: list[Path], window) -> np.ndarray:
    out = np.zeros((N_FRAMES, 4, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for files, mode, begin in ((depth_files, "RGB", 0), (ir_files, "L", 3)):
        for dst, src in enumerate(pick_indices(len(files), N_FRAMES)):
            image = read_image(files[int(src)], mode)
            if image is None:
                continue
            try:
                out[dst, begin : begin + (3 if mode == "RGB" else 1)] = crop_resize(image, window, mode)
            except Exception:
                continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v9")
    ap.add_argument("--yolo-path", type=Path, default=ROOT / "external_models" / "yolo11n.pt")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = Paths(args.root)
    clips = []
    for record in build_clip_records(paths, args.split):
        base_depth = Path(record.depth_dir)
        base_ir = Path(record.ir_dir)
        clips.append(
            {
                "clip_id": record.clip_id,
                "path": record.path,
                "depth_files": list_image_files(base_depth),
                "ir_files": list_image_files(base_ir),
                "label": record.label if record.label is not None else "",
            }
        )
    windows_path = args.cache_dir / f"{args.split}_windows.csv"
    windows = detect_v9(clips, args.yolo_path, args.device, args.batch_size)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(windows_path, index=False)

    records = []
    for clip in tqdm(clips, desc="cache v9 clips"):
        row = windows.loc[windows["clip_id"] == clip["clip_id"]].iloc[0]
        window = None if not bool(row["has_crop"]) else (float(row.x0), float(row.y0), float(row.x1), float(row.y1))
        rel = Path(args.split) / f"{clip['clip_id']}.npz"
        out_path = args.cache_dir / rel
        if args.force or not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "tensor": load_tensor(clip["depth_files"], clip["ir_files"], window),
                "path": clip["path"],
                "clip_id": clip["clip_id"],
            }
            if args.split == "train":
                payload["label"] = int(clip["label"])
            np.savez_compressed(out_path, **payload)
        records.append(
            {
                "clip_id": clip["clip_id"],
                "path": clip["path"],
                "feature_file": str(rel).replace("\\", "/"),
                "label": clip["label"],
            }
        )
    pd.DataFrame(records).to_csv(args.cache_dir / f"{args.split}_manifest.csv", index=False)
    print(f"windows={windows_path}")
    print(f"manifest={args.cache_dir / f'{args.split}_manifest.csv'}")


if __name__ == "__main__":
    main()
