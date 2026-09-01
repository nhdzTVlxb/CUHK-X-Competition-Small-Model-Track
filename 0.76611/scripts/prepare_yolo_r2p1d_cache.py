from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import Paths
from cuhkx_baseline.yolo_r2p1d import build_clip_records, detect_windows, load_clip_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--yolo-path", type=Path, default=ROOT / "external_models" / "yolo11n.pt")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = Paths(args.root)
    clips = build_clip_records(paths, args.split)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows_csv = cache_dir / f"{args.split}_windows.csv"
    windows = detect_windows(
        clips,
        yolo_path=args.yolo_path,
        cache_path=windows_csv,
        batch_size=args.batch_size,
        device=args.device,
    )
    records = []
    for clip in tqdm(clips, desc=f"cache {args.split} clips"):
        row = windows.loc[windows["clip_id"] == clip.clip_id].iloc[0]
        window = None
        if bool(row["has_crop"]) and not pd.isna(row["x0"]):
            window = (float(row["x0"]), float(row["y0"]), float(row["x1"]), float(row["y1"]))
        feature_file = Path(args.split) / f"{clip.clip_id}.npz"
        out_path = cache_dir / feature_file
        if args.force or not out_path.exists():
            tensor = load_clip_tensor(Path(clip.depth_dir), Path(clip.ir_dir), window)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            npz = {
                "tensor": tensor.astype("uint8"),
                "path": clip.path,
                "clip_id": clip.clip_id,
            }
            if clip.label is not None:
                npz["label"] = clip.label
            np.savez_compressed(out_path, **npz)
        records.append(
            {
                "clip_id": clip.clip_id,
                "path": clip.path,
                "feature_file": str(feature_file).replace("\\", "/"),
                "label": clip.label if clip.label is not None else "",
            }
        )
    pd.DataFrame(records).to_csv(cache_dir / f"{args.split}_manifest.csv", index=False)
    print(f"windows={windows_csv}")
    print(f"manifest={cache_dir / f'{args.split}_manifest.csv'}")


if __name__ == "__main__":
    main()
