from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import Paths, iter_test_clips, iter_train_clips, sample_indices

FRAMES = 16
IMAGE_SIZE = 128


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or "0"),
    )


def load_thermal(path: Path) -> np.ndarray:
    files = image_files(path)
    output = np.zeros((FRAMES, 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if not files:
        return output
    resample = getattr(Image, "Resampling", Image).BILINEAR
    for dst, src in enumerate(sample_indices(len(files), FRAMES)):
        try:
            with Image.open(files[int(src)]) as image:
                array = np.asarray(
                    image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), resample=resample),
                    dtype=np.uint8,
                )
            output[dst, 0] = array
        except Exception:
            continue
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "thermal_r2p1d_v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = Paths(args.root)
    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []

    train_rows = list(iter_train_clips(paths.train_root))
    for row in tqdm(train_rows, desc="thermal train"):
        clip_id = str(row["clip_id"])
        relative = Path("train") / f"{clip_id}.npz"
        target = cache / relative
        depth_dir = Path(row["depth_dir"])
        train_data_root = depth_dir.parents[3]
        thermal_dir = train_data_root / "Thermal" / depth_dir.relative_to(train_data_root / "Depth_Color")
        if args.force or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                tensor=load_thermal(thermal_dir),
                clip_id=clip_id,
                path=f"{row['action']}/{row['user']}/{row['trial']}",
                label=int(row["label"]),
                user=str(row["user"]),
            )
        records.append(
            {
                "split": "train",
                "clip_id": clip_id,
                "path": f"{row['action']}/{row['user']}/{row['trial']}",
                "feature_file": str(relative).replace("\\", "/"),
                "label": int(row["label"]),
                "user": str(row["user"]),
            }
        )

    test_rows = list(iter_test_clips(paths.test_root, paths.test_csv))
    for row in tqdm(test_rows, desc="thermal test"):
        clip_id = str(row["clip_id"])
        relative = Path("test") / f"{clip_id}.npz"
        target = cache / relative
        if args.force or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                tensor=load_thermal(Path(row["depth_dir"]).parent / "Thermal"),
                clip_id=clip_id,
                path=str(row["path"]),
            )
        records.append(
            {
                "split": "test",
                "clip_id": clip_id,
                "path": str(row["path"]),
                "feature_file": str(relative).replace("\\", "/"),
                "label": "",
                "user": "",
            }
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(cache / "manifest.csv", index=False)
    manifest[manifest["split"].eq("train")].to_csv(cache / "train_manifest.csv", index=False)
    manifest[manifest["split"].eq("test")].to_csv(cache / "test_manifest.csv", index=False)
    print(f"cache={cache}")
    print(f"train={(manifest['split'] == 'train').sum()} test={(manifest['split'] == 'test').sum()}")
    print(f"thermal_present={(manifest['split'] == 'train').sum() - sum(np.all(np.load(cache / p)['tensor'] == 0) for p in manifest.loc[manifest['split'].eq('train'), 'feature_file'])}")


if __name__ == "__main__":
    main()
