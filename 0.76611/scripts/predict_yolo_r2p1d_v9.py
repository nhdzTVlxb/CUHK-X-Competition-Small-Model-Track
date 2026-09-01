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
sys.path.insert(0, str(ROOT / "scripts"))

from predict_yolo_r2p1d import CachedVideoDataset
from cuhkx_baseline.yolo_r2p1d import N_CLASSES, R2Plus1D34, dequantize_state, load_checkpoint


def predict_v9(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    def jitter(x: torch.Tensor, shift: int) -> torch.Tensor:
        return torch.roll(x, shifts=shift, dims=1)

    return (
        model(inputs)
        + model(torch.flip(inputs, dims=(-1,)))
        + model(jitter(inputs, 1))
        + model(jitter(inputs, -1))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v9")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--logits-output", type=Path, default=None)
    args = ap.parse_args()

    manifest = pd.read_csv(args.cache_dir / f"{args.split}_manifest.csv")
    if args.split == "test" and len(manifest) != 405:
        raise ValueError(f"Expected 405 test clips, got {len(manifest)}")
    loader = DataLoader(
        CachedVideoDataset(manifest, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    ensemble = np.zeros((len(manifest), N_CLASSES), dtype=np.float32)
    per_model = []
    for weight, fold, packed in zip(checkpoint["weights"], checkpoint["folds"], checkpoint["models_packed"], strict=True):
        model = R2Plus1D34().to(device)
        model.load_state_dict(dequantize_state(packed))
        model.eval()
        parts = []
        with torch.no_grad():
            for inputs, _paths, _ids in tqdm(loader, desc=f"v9 fold {fold}"):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    out = predict_v9(model, inputs.to(device))
                parts.append(out.float().cpu().numpy())
        logits = np.concatenate(parts, axis=0)
        per_model.append(logits)
        ensemble += float(weight) * logits
        del model
        torch.cuda.empty_cache()

    preds = ensemble.argmax(1).astype(int)
    output = args.output or args.root / "outputs" / f"submission_yolo_r2p1d_v9_{args.split}.csv"
    logits_output = args.logits_output or args.root / "outputs" / f"yolo_r2p1d_v9_{args.split}_logits.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    logits_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": manifest["path"].astype(str), "prediction": preds}).to_csv(output, index=False)
    np.savez_compressed(
        logits_output,
        logits=ensemble,
        per_model_logits=np.stack(per_model, axis=0),
        path=manifest["path"].astype(str).to_numpy(),
        clip_id=manifest["clip_id"].astype(str).to_numpy(),
    )
    print(f"wrote={output}")
    print(f"logits={logits_output}")
    print(f"rows={len(preds)} classes={np.unique(preds).size}")


if __name__ == "__main__":
    main()
