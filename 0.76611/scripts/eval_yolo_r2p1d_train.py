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
    N_CLASSES,
    R2Plus1D34,
    dequantize_state,
    load_checkpoint,
    predict_batch_logits,
)

VAL_USERS = {"user8", "user9", "user23", "user24"}


class CachedTrainDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        mean = (0.43216, 0.394666, 0.37645, (0.43216 + 0.394666 + 0.37645) / 3.0)
        std = (0.22803, 0.22145, 0.216989, (0.22803 + 0.22145 + 0.216989) / 3.0)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, CHANNELS, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, CHANNELS, 1, 1)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        item = np.load(self.cache_dir / row["feature_file"])
        clip = torch.from_numpy(item["tensor"].astype(np.float32)).div_(255.0)
        clip = clip.sub_(self.mean).div_(self.std)
        return clip, int(float(row["label"])), row["clip_id"], row["user"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v1")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--all-train", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    manifest = pd.read_csv(args.cache_dir / "train_manifest.csv")
    if "user" not in manifest:
        manifest["user"] = manifest["path"].astype(str).str.split("/").str[-2]
    if not args.all_train:
        manifest = manifest[manifest["user"].isin(VAL_USERS)].copy()
    loader = DataLoader(
        CachedTrainDataset(manifest, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint)
    ensemble = np.zeros((len(manifest), N_CLASSES), dtype=np.float32)
    per_model = []
    for weight, fold, packed in zip(
        checkpoint["weights"], checkpoint["folds"], checkpoint["models_packed"], strict=True
    ):
        model = R2Plus1D34().to(device)
        model.load_state_dict(dequantize_state(packed))
        model.eval()
        outputs = []
        with torch.no_grad():
            for inputs, _y, _ids, _users in tqdm(loader, desc=f"fold {fold}"):
                outputs.append(predict_batch_logits(model, inputs.to(device)).float().cpu().numpy())
        logits = np.concatenate(outputs, axis=0)
        per_model.append(logits)
        ensemble += float(weight) * logits
        del model
        torch.cuda.empty_cache()

    y = manifest["label"].astype(int).to_numpy()
    pred = ensemble.argmax(1)
    print(f"device={device} rows={len(y)} accuracy={float((pred == y).mean()):.6f}")
    for i, logits in enumerate(per_model):
        print(f"model{i} accuracy={float((logits.argmax(1) == y).mean()):.6f}")
    args.root.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    output = args.output or args.root / "outputs" / "yolo_r2p1d_val_logits.npz"
    np.savez_compressed(
        output,
        logits=ensemble,
        per_model_logits=np.stack(per_model, axis=0),
        y=y,
        clip_id=manifest["clip_id"].astype(str).to_numpy(),
        user=manifest["user"].astype(str).to_numpy(),
    )
    pd.DataFrame(
        {
            "clip_id": manifest["clip_id"].astype(str),
            "user": manifest["user"].astype(str),
            "label": y,
            "prediction": pred,
        }
    ).to_csv(output.with_name(output.stem + "_predictions.csv"), index=False)
    print(f"output={output}")


if __name__ == "__main__":
    main()
