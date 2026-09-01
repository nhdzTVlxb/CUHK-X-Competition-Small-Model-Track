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

from cuhkx_baseline.features import Paths
from cuhkx_baseline.model import SkelImuNet


class TestFeatureDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir: Path, stats: dict[str, np.ndarray]) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.stats = stats

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        skeleton = (item["skeleton"] - self.stats["skeleton_mean"]) / self.stats["skeleton_std"]
        imu = (item["imu"] - self.stats["imu_mean"]) / self.stats["imu_std"]
        return (
            torch.from_numpy(skeleton.astype(np.float32)),
            torch.from_numpy(imu.astype(np.float32)),
            row["path"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "skel_imu_best.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_skel_imu.csv")
    args = parser.parse_args()

    paths = Paths(args.root)
    manifest = pd.read_csv(paths.cache_dir / "manifest.csv")
    test_df = manifest[manifest["split"] == "test"].copy()
    stats_npz = np.load(paths.outputs_dir / "skel_imu_stats.npz")
    stats = {key: stats_npz[key] for key in stats_npz.files}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    width = int(ckpt.get("params", {}).get("width", 128))
    model = SkelImuNet(width=width).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(TestFeatureDataset(test_df, paths.cache_dir, stats), batch_size=args.batch_size)
    paths_out = []
    preds = []
    with torch.no_grad():
        for skeleton, imu, paths_batch in tqdm(loader, desc="predict"):
            skeleton = skeleton.to(device)
            imu = imu.to(device)
            logits = model(skeleton, imu)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            paths_out.extend(list(paths_batch))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": paths_out, "prediction": preds}).to_csv(args.output, index=False)
    print(f"wrote={args.output}")
    print(f"rows={len(preds)}")


if __name__ == "__main__":
    main()

