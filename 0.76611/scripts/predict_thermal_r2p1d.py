from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_thermal_r2p1d import MEAN, STD, ThermalDataset, make_model
from cuhkx_baseline.yolo_r2p1d import N_CLASSES, predict_batch_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "thermal_r2p1d_v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "thermal_r2p1d_best.pt")
    parser.add_argument("--source-checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model-index", type=int, default=0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.cache_dir / f"{args.split}_manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.source_checkpoint, args.model_index)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    loader = DataLoader(ThermalDataset(manifest, args.cache_dir, False), args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    parts = []
    labels = []
    ids = []
    with torch.no_grad():
        for tensor, y, _paths, clip_ids in tqdm(loader, desc=f"thermal {args.split}"):
            parts.append(predict_batch_logits(model, tensor.to(device)).float().cpu().numpy())
            labels.extend(y.numpy().tolist())
            ids.extend(clip_ids)
    logits = np.concatenate(parts, axis=0)
    output = args.output or args.root / "outputs" / f"thermal_r2p1d_{args.split}_logits.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, logits=logits, y=np.asarray(labels), clip_id=np.asarray(ids), path=manifest["path"].astype(str).to_numpy())
    print(f"device={device} rows={len(logits)} classes={np.unique(logits.argmax(1)).size} output={output}")
    if args.split == "train":
        valid = np.asarray(labels) >= 0
        print(f"accuracy={(logits.argmax(1)[valid] == np.asarray(labels)[valid]).mean():.6f}")


if __name__ == "__main__":
    main()
