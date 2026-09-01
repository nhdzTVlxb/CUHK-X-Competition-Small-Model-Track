from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import (
    Paths,
    extract_fusion_feature_file,
    iter_test_clips,
    iter_train_clips,
    load_image_sequence,
)


def _old_feature_lookup(paths: Paths) -> dict[str, Path]:
    manifest_path = paths.cache_dir / "manifest.csv"
    if not manifest_path.exists():
        return {}
    manifest = pd.read_csv(manifest_path)
    return {
        str(row["clip_id"]): paths.cache_dir / str(row["feature_file"])
        for _, row in manifest.iterrows()
        if isinstance(row.get("clip_id"), str)
    }


def _write_feature(
    row: dict[str, str],
    out_path: Path,
    old_lookup: dict[str, Path],
    skel_frames: int,
    imu_steps: int,
    image_frames: int,
    image_size: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    old_path = old_lookup.get(row["clip_id"])
    if old_path and old_path.exists():
        old = np.load(old_path)
        skeleton = old["skeleton"]
        imu = old["imu"]
    else:
        extract_fusion_feature_file(row, out_path, skel_frames, imu_steps, image_frames, image_size)
        return
    depth = load_image_sequence(Path(row["depth_dir"]), image_frames, image_size, "RGB")
    ir = load_image_sequence(Path(row["ir_dir"]), image_frames, image_size, "L")
    np.savez(out_path, skeleton=skeleton, imu=imu, depth=depth, ir=ir)


def build_manifest(paths: Paths, skel_frames: int, imu_steps: int, image_frames: int, image_size: int, force: bool) -> pd.DataFrame:
    cache_dir = paths.fusion_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    old_lookup = _old_feature_lookup(paths)
    records = []

    train_rows = list(iter_train_clips(paths.train_root))
    for i, row in enumerate(tqdm(train_rows, desc="fusion train features")):
        feature_file = Path("train") / f"{i:05d}.npz"
        row["feature_file"] = str(feature_file).replace("\\", "/")
        out_path = cache_dir / feature_file
        if force or not out_path.exists():
            _write_feature(row, out_path, old_lookup, skel_frames, imu_steps, image_frames, image_size)
        records.append(row)

    test_rows = list(iter_test_clips(paths.test_root, paths.test_csv))
    for i, row in enumerate(tqdm(test_rows, desc="fusion test features")):
        feature_file = Path("test") / f"{i:05d}.npz"
        row["feature_file"] = str(feature_file).replace("\\", "/")
        out_path = cache_dir / feature_file
        if force or not out_path.exists():
            _write_feature(row, out_path, old_lookup, skel_frames, imu_steps, image_frames, image_size)
        records.append(row)

    manifest = pd.DataFrame(records)
    manifest["image_frames"] = image_frames
    manifest["image_size"] = image_size
    manifest_path = cache_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skel-frames", type=int, default=32)
    parser.add_argument("--imu-steps", type=int, default=128)
    parser.add_argument("--image-frames", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = Paths(args.root)
    manifest = build_manifest(paths, args.skel_frames, args.imu_steps, args.image_frames, args.image_size, args.force)
    print(f"manifest={paths.fusion_cache_dir / 'manifest.csv'}")
    print(f"train={(manifest['split'] == 'train').sum()} test={(manifest['split'] == 'test').sum()}")


if __name__ == "__main__":
    main()

