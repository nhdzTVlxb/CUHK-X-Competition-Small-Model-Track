from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


MODALITIES = ("Depth_Color", "IR", "Thermal", "IMU", "Radar", "Skeleton")
NUM_CLASSES = 40
SKELETON_KEYPOINTS = 17
SKELETON_DIMS = 4
SKELETON_FEAT_DIM = SKELETON_KEYPOINTS * SKELETON_DIMS
SKELETON_ENHANCED_FEAT_DIM = 238
IMU_FEAT_DIM = 19
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# COCO-17 parent graph used to derive bone vectors.
COCO17_PARENTS = np.asarray(
    [-1, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def train_root(self) -> Path:
        return self.data_dir / "Training" / "data" / "HAR" / "data"

    @property
    def test_root(self) -> Path:
        return self.data_dir / "Testing" / "data" / "small_model_track_test"

    @property
    def test_csv(self) -> Path:
        candidates = [
            self.data_dir / "Testing" / "test.csv",
            self.data_dir / "Testing" / "test_file" / "test.csv",
            self.root / "test.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Could not find test.csv")

    @property
    def class_mapping_csv(self) -> Path:
        candidates = [
            self.data_dir / "class_mapping.csv",
            self.data_dir / "Training" / "class_mapping.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Could not find class_mapping.csv")

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache" / "skel_imu_v1"

    @property
    def fusion_cache_dir(self) -> Path:
        return self.root / "cache" / "fusion_depth_ir_skel_imu_v1"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"


def parse_action_id(action_name: str) -> int:
    return int(action_name.split("_", 1)[0])


def load_class_mapping(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {int(row["action_id"]): row["action_name"] for row in csv.DictReader(f)}


def sample_indices(n: int, target: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(target, dtype=np.int64)
    if n == target:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, target).round().astype(np.int64)


def _load_keypoint_frame(path: Path) -> np.ndarray:
    out = np.zeros((SKELETON_KEYPOINTS, SKELETON_DIMS), dtype=np.float32)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return out.reshape(-1)

    if not data:
        return out.reshape(-1)
    person = data[0] if isinstance(data, list) else data
    keypoints = np.asarray(person.get("keypoints", []), dtype=np.float32)
    scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float32)

    if keypoints.ndim == 2 and keypoints.shape[0] > 0:
        rows = min(SKELETON_KEYPOINTS, keypoints.shape[0])
        cols = min(3, keypoints.shape[1])
        out[:rows, :cols] = keypoints[:rows, :cols]
    if scores.ndim == 1 and scores.shape[0] > 0:
        rows = min(SKELETON_KEYPOINTS, scores.shape[0])
        out[:rows, 3] = scores[:rows]
    return out.reshape(-1)


def load_skeleton_sequence(clip_dir: Path, frames: int) -> np.ndarray:
    pred_dir = clip_dir / "predictions"
    if not pred_dir.exists():
        pred_dir = clip_dir
    files = sorted(pred_dir.glob("*.json"))
    seq = np.zeros((frames, SKELETON_FEAT_DIM), dtype=np.float32)
    if not files:
        return seq
    for dst_i, src_i in enumerate(sample_indices(len(files), frames)):
        seq[dst_i] = _load_keypoint_frame(files[int(src_i)])
    return seq


def enhance_skeleton_sequence(raw: np.ndarray) -> np.ndarray:
    """Build subject-robust pose, bone, and motion features."""
    seq = raw.astype(np.float32, copy=False).reshape(-1, SKELETON_KEYPOINTS, SKELETON_DIMS)
    xyz = seq[:, :, :3]
    score = seq[:, :, 3:4]

    hip = (xyz[:, 11:12] + xyz[:, 12:13]) * 0.5
    centered = xyz - hip
    shoulder_span = np.linalg.norm(xyz[:, 5] - xyz[:, 6], axis=1, keepdims=True)
    hip_span = np.linalg.norm(xyz[:, 11] - xyz[:, 12], axis=1, keepdims=True)
    body_span = np.maximum(np.maximum(shoulder_span, hip_span), 0.08)
    centered = centered / body_span[:, None, :]

    bones = np.zeros_like(centered)
    valid = COCO17_PARENTS >= 0
    bones[:, valid] = centered[:, valid] - centered[:, COCO17_PARENTS[valid]]
    velocity = np.diff(centered, axis=0, prepend=centered[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])

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
    if out.shape[1] < SKELETON_ENHANCED_FEAT_DIM:
        out = np.pad(out, ((0, 0), (0, SKELETON_ENHANCED_FEAT_DIM - out.shape[1])))
    return out[:, :SKELETON_ENHANCED_FEAT_DIM].astype(np.float32, copy=False)


def _numeric_imu_frame(csv_path: Path) -> np.ndarray:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")
    except Exception:
        return np.zeros((0, IMU_FEAT_DIM), dtype=np.float32)

    drop_tokens = ("时间", "设备", "版本")
    numeric_cols = []
    for col in df.columns:
        if any(token in str(col) for token in drop_tokens):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(series)

    if not numeric_cols:
        return np.zeros((0, IMU_FEAT_DIM), dtype=np.float32)
    arr = pd.concat(numeric_cols, axis=1).fillna(0.0).to_numpy(dtype=np.float32)
    if arr.shape[1] < IMU_FEAT_DIM:
        pad = np.zeros((arr.shape[0], IMU_FEAT_DIM - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > IMU_FEAT_DIM:
        arr = arr[:, :IMU_FEAT_DIM]
    return arr


def load_imu_sequence(clip_dir: Path, steps: int) -> np.ndarray:
    parts = []
    for csv_path in sorted(clip_dir.glob("*.csv")):
        arr = _numeric_imu_frame(csv_path)
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.zeros((steps, IMU_FEAT_DIM), dtype=np.float32)
    arr = np.concatenate(parts, axis=0)
    arr = arr[sample_indices(arr.shape[0], steps)]
    return arr.astype(np.float32, copy=False)


def _image_files(clip_dir: Path) -> list[Path]:
    if not clip_dir.exists():
        return []
    return sorted(p for p in clip_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_image_sequence(
    clip_dir: Path,
    frames: int,
    image_size: int,
    mode: str,
) -> np.ndarray:
    channels = 3 if mode == "RGB" else 1
    seq = np.zeros((frames, channels, image_size, image_size), dtype=np.uint8)
    files = _image_files(clip_dir)
    if not files:
        return seq
    resample = getattr(Image, "Resampling", Image).BILINEAR
    for dst_i, src_i in enumerate(sample_indices(len(files), frames)):
        try:
            with Image.open(files[int(src_i)]) as img:
                arr = np.asarray(img.convert(mode).resize((image_size, image_size), resample=resample), dtype=np.uint8)
        except Exception:
            continue
        if channels == 1:
            seq[dst_i, 0] = arr
        else:
            seq[dst_i] = np.moveaxis(arr, -1, 0)
    return seq


def iter_train_clips(train_root: Path) -> Iterable[dict[str, str]]:
    # Skeleton exists for all usable clips and gives the canonical clip list.
    skel_root = train_root / "Skeleton"
    for action_dir in sorted(p for p in skel_root.iterdir() if p.is_dir()):
        label = parse_action_id(action_dir.name)
        for user_dir in sorted(p for p in action_dir.iterdir() if p.is_dir()):
            for trial_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
                rel = Path(action_dir.name) / user_dir.name / trial_dir.name
                yield {
                    "split": "train",
                    "clip_id": f"{action_dir.name}__{user_dir.name}__{trial_dir.name}",
                    "action": action_dir.name,
                    "label": str(label),
                    "user": user_dir.name,
                    "trial": trial_dir.name,
                    "skeleton_dir": str(train_root / "Skeleton" / rel),
                    "imu_dir": str(train_root / "IMU" / rel),
                    "depth_dir": str(train_root / "Depth_Color" / rel),
                    "ir_dir": str(train_root / "IR" / rel),
                }


def iter_test_clips(test_root: Path, test_csv: Path) -> Iterable[dict[str, str]]:
    df = pd.read_csv(test_csv)
    for _, row in df.iterrows():
        rel = str(row["path"]).strip().strip("/\\")
        clip_id = Path(rel).name
        clip_dir = test_root / clip_id
        yield {
            "split": "test",
            "clip_id": clip_id,
            "path": str(row["path"]),
            "skeleton_dir": str(clip_dir / "Skeleton"),
            "imu_dir": str(clip_dir / "IMU"),
            "depth_dir": str(clip_dir / "Depth_Color"),
            "ir_dir": str(clip_dir / "IR"),
        }


def extract_feature_file(row: dict[str, str], out_path: Path, skel_frames: int, imu_steps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton = load_skeleton_sequence(Path(row["skeleton_dir"]), skel_frames)
    imu = load_imu_sequence(Path(row["imu_dir"]), imu_steps)
    np.savez_compressed(out_path, skeleton=skeleton, imu=imu)


def extract_fusion_feature_file(
    row: dict[str, str],
    out_path: Path,
    skel_frames: int,
    imu_steps: int,
    image_frames: int,
    image_size: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skeleton = load_skeleton_sequence(Path(row["skeleton_dir"]), skel_frames)
    imu = load_imu_sequence(Path(row["imu_dir"]), imu_steps)
    depth = load_image_sequence(Path(row["depth_dir"]), image_frames, image_size, "RGB")
    ir = load_image_sequence(Path(row["ir_dir"]), image_frames, image_size, "L")
    np.savez(out_path, skeleton=skeleton, imu=imu, depth=depth, ir=ir)


def compute_train_stats(manifest: pd.DataFrame, cache_dir: Path, train_users: set[str]) -> dict[str, np.ndarray]:
    sk_sum = np.zeros(SKELETON_FEAT_DIM, dtype=np.float64)
    sk_sq = np.zeros(SKELETON_FEAT_DIM, dtype=np.float64)
    imu_sum = np.zeros(IMU_FEAT_DIM, dtype=np.float64)
    imu_sq = np.zeros(IMU_FEAT_DIM, dtype=np.float64)
    sk_n = 0
    imu_n = 0

    rows = manifest[(manifest["split"] == "train") & (manifest["user"].isin(train_users))]
    for _, row in rows.iterrows():
        item = np.load(cache_dir / row["feature_file"])
        sk = item["skeleton"].astype(np.float64)
        imu = item["imu"].astype(np.float64)
        sk_sum += sk.sum(axis=0)
        sk_sq += np.square(sk).sum(axis=0)
        imu_sum += imu.sum(axis=0)
        imu_sq += np.square(imu).sum(axis=0)
        sk_n += sk.shape[0]
        imu_n += imu.shape[0]

    sk_mean = sk_sum / max(sk_n, 1)
    imu_mean = imu_sum / max(imu_n, 1)
    sk_std = np.sqrt(np.maximum(sk_sq / max(sk_n, 1) - np.square(sk_mean), 1e-6))
    imu_std = np.sqrt(np.maximum(imu_sq / max(imu_n, 1) - np.square(imu_mean), 1e-6))
    return {
        "skeleton_mean": sk_mean.astype(np.float32),
        "skeleton_std": sk_std.astype(np.float32),
        "imu_mean": imu_mean.astype(np.float32),
        "imu_std": imu_std.astype(np.float32),
    }
