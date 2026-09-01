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

from cuhkx_baseline.yolo_r2p1d import (
    CHANNELS,
    IMAGE_SIZE,
    KINETICS_MEAN,
    KINETICS_STD,
    N_CLASSES,
    N_FRAMES,
    R2Plus1D34,
    dequantize_state,
    load_checkpoint,
    predict_batch_logits,
)


class CachedVideoDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir: Path) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.cache_dir = cache_dir
        mean = (*KINETICS_MEAN, sum(KINETICS_MEAN) / 3.0)
        std = (*KINETICS_STD, sum(KINETICS_STD) / 3.0)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, CHANNELS, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, CHANNELS, 1, 1)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        clip = torch.from_numpy(item["tensor"].astype(np.float32)).div_(255.0)
        clip = clip.sub_(self.mean).div_(self.std)
        return clip, row["path"], row["clip_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_yolo_r2p1d.csv")
    parser.add_argument("--logits-output", type=Path, default=ROOT / "outputs" / "yolo_r2p1d_logits.npz")
    args = parser.parse_args()

    manifest_path = args.cache_dir / "test_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run scripts/prepare_yolo_r2p1d_cache.py first: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 405:
        raise ValueError(f"Expected 405 test clips, got {len(manifest)}")

    loader = DataLoader(
        CachedVideoDataset(manifest, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    ensemble_logits = np.zeros((len(manifest), N_CLASSES), dtype=np.float32)
    for weight, fold, packed in zip(checkpoint["weights"], checkpoint["folds"], checkpoint["models_packed"], strict=True):
        model = R2Plus1D34().to(device)
        model.load_state_dict(dequantize_state(packed))
        model.eval()
        offset = 0
        for inputs, _paths, _clip_ids in tqdm(loader, desc=f"fold {fold}"):
            inputs = inputs.to(device, non_blocking=True)
            with torch.no_grad():
                logits = predict_batch_logits(model, inputs)
            batch = int(inputs.shape[0])
            ensemble_logits[offset : offset + batch] += float(weight) * logits.float().cpu().numpy()
            offset += batch
        del model
        torch.cuda.empty_cache()

    preds = ensemble_logits.argmax(axis=1).astype(int)
    output = pd.DataFrame({"path": manifest["path"].astype(str), "prediction": preds})
    if output["prediction"].nunique() < 35:
        raise RuntimeError("Prediction collapsed to too few classes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    np.savez_compressed(args.logits_output, logits=ensemble_logits, path=manifest["path"].astype(str).to_numpy())
    print(f"device={device} shape={(N_FRAMES, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)}")
    print(f"wrote={args.output}")
    print(f"logits={args.logits_output}")
    print(f"rows={len(output)} classes={output['prediction'].nunique()}")


if __name__ == "__main__":
    main()
