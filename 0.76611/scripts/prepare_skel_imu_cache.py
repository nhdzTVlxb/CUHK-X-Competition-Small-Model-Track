from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import Paths, extract_feature_file, iter_test_clips, iter_train_clips


def build_manifest(paths: Paths, skel_frames: int, imu_steps: int, force: bool) -> pd.DataFrame:
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    records = []

    train_rows = list(iter_train_clips(paths.train_root))
    for i, row in enumerate(tqdm(train_rows, desc="train features")):
        feature_file = Path("train") / f"{i:05d}.npz"
        row["feature_file"] = str(feature_file).replace("\\", "/")
        out_path = paths.cache_dir / feature_file
        if force or not out_path.exists():
            extract_feature_file(row, out_path, skel_frames, imu_steps)
        records.append(row)

    test_rows = list(iter_test_clips(paths.test_root, paths.test_csv))
    for i, row in enumerate(tqdm(test_rows, desc="test features")):
        feature_file = Path("test") / f"{i:05d}.npz"
        row["feature_file"] = str(feature_file).replace("\\", "/")
        out_path = paths.cache_dir / feature_file
        if force or not out_path.exists():
            extract_feature_file(row, out_path, skel_frames, imu_steps)
        records.append(row)

    manifest = pd.DataFrame(records)
    manifest_path = paths.cache_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skel-frames", type=int, default=32)
    parser.add_argument("--imu-steps", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = Paths(args.root)
    manifest = build_manifest(paths, args.skel_frames, args.imu_steps, args.force)
    train_count = int((manifest["split"] == "train").sum())
    test_count = int((manifest["split"] == "test").sum())
    print(f"manifest={paths.cache_dir / 'manifest.csv'}")
    print(f"train={train_count} test={test_count}")


if __name__ == "__main__":
    main()

