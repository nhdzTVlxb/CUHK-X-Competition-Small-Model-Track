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
sys.path.insert(0, str(ROOT / "scripts"))

from cuhkx_baseline.features import Paths
from train_skel_imu_v2 import SkelImuV2, skeleton_features


class TestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, stats: dict[str, np.ndarray]):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.stats = stats

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        sk = skeleton_features(item["skeleton"])
        imu = item["imu"].astype(np.float32, copy=False)
        sk = (sk - self.stats["sk_mean"]) / self.stats["sk_std"]
        imu = (imu - self.stats["imu_mean"]) / self.stats["imu_std"]
        return (
            torch.from_numpy(np.ascontiguousarray(sk)),
            torch.from_numpy(np.ascontiguousarray(imu)),
            row["path"],
            row["clip_id"],
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "skel_imu_v2_best.pt")
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--split", choices=("train", "test"), default="test")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_skel_imu_v2.csv")
    ap.add_argument("--logits-output", type=Path, default=ROOT / "outputs" / "skel_imu_v2_test_logits.npz")
    args = ap.parse_args()

    paths = Paths(args.root)
    manifest = pd.read_csv(paths.cache_dir / "manifest.csv")
    test_df = manifest[manifest.split.eq(args.split)].copy()
    stats_npz = np.load(paths.outputs_dir / "skel_imu_v2_stats.npz")
    stats = {k: stats_npz[k] for k in stats_npz.files}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    width = int(ckpt.get("params", {}).get("width", 160))
    model = SkelImuV2(width=width).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(TestDataset(test_df, paths.cache_dir, stats), batch_size=args.batch_size, shuffle=False)
    logits, out_paths, clip_ids = [], [], []
    with torch.no_grad():
        for sk, imu, paths_batch, ids_batch in tqdm(loader, desc="predict skel-imu-v2"):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(sk.to(device), imu.to(device))
            logits.append(out.float().cpu().numpy())
            out_paths.extend(list(paths_batch))
            clip_ids.extend(list(ids_batch))

    logits_arr = np.concatenate(logits, axis=0)
    preds = logits_arr.argmax(axis=1).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": out_paths, "prediction": preds}).to_csv(args.output, index=False)
    np.savez_compressed(args.logits_output, logits=logits_arr, path=np.asarray(out_paths), clip_id=np.asarray(clip_ids))
    print(f"wrote={args.output}")
    print(f"logits={args.logits_output}")
    print(f"rows={len(preds)} classes={len(set(preds.tolist()))}")


if __name__ == "__main__":
    main()
