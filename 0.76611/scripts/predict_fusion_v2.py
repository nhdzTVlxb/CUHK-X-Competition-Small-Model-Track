from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx_baseline.features import Paths, enhance_skeleton_sequence
from cuhkx_baseline.model import FusionNetV2


class TestFusionV2Dataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir: Path, stats: dict[str, np.ndarray]) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.stats = stats

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        skeleton = enhance_skeleton_sequence(item["skeleton"])
        imu = item["imu"].astype(np.float32)
        depth = item["depth"].astype(np.float32) / 127.5 - 1.0
        ir = item["ir"].astype(np.float32) / 127.5 - 1.0
        skeleton = (skeleton - self.stats["skeleton_mean"]) / self.stats["skeleton_std"]
        imu = (imu - self.stats["imu_mean"]) / self.stats["imu_std"]
        return (
            torch.from_numpy(np.ascontiguousarray(skeleton.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(imu.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(depth.astype(np.float32))),
            torch.from_numpy(np.ascontiguousarray(ir.astype(np.float32))),
            row["path"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "fusion_v2_best.pt")
    parser.add_argument("--stats", type=Path, default=ROOT / "outputs" / "fusion_v2_stats.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_fusion_v2.csv")
    args = parser.parse_args()

    paths = Paths(args.root)
    cache_dir = paths.fusion_cache_dir
    manifest = pd.read_csv(cache_dir / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].copy()
    stats_npz = np.load(args.stats)
    stats = {key: stats_npz[key] for key in stats_npz.files}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    params = ckpt.get("params", {})
    model = FusionNetV2(width=int(params.get("width", 160)), image_base=int(params.get("image_base", 32))).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(TestFusionV2Dataset(test_df, cache_dir, stats), batch_size=args.batch_size)
    paths_out = []
    preds = []
    with torch.no_grad():
        for skeleton, imu, depth, ir, paths_batch in tqdm(loader, desc="predict fusion v2"):
            logits = model(
                skeleton.to(device),
                imu.to(device),
                depth.to(device),
                ir.to(device),
            )
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            paths_out.extend(list(paths_batch))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": paths_out, "prediction": preds}).to_csv(args.output, index=False)
    print(f"wrote={args.output}")
    print(f"rows={len(preds)}")


if __name__ == "__main__":
    main()
