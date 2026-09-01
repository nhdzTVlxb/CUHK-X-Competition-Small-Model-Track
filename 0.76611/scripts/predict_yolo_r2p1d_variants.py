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

from predict_yolo_r2p1d import CachedVideoDataset
from cuhkx_baseline.yolo_r2p1d import N_CLASSES, R2Plus1D34, dequantize_state, load_checkpoint, predict_batch_logits


def write_submission(manifest: pd.DataFrame, preds: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": manifest["path"].astype(str), "prediction": preds.astype(int)}).to_csv(output, index=False)


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == y).mean())


def aligned_skel_val(root: Path, clip_ids: np.ndarray) -> np.ndarray | None:
    path = root / "outputs" / "skel_imu_v2_val_logits.npz"
    if not path.exists():
        return None
    sk = np.load(path, allow_pickle=True)
    lookup = {str(cid): i for i, cid in enumerate(sk["clip_id"])}
    if not all(str(cid) in lookup for cid in clip_ids):
        return None
    return sk["logits"][[lookup[str(cid)] for cid in clip_ids]]


def best_yolo_weights(per_model: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    best = (-1.0, 0.5, 0.5)
    for w0 in np.linspace(0.0, 1.0, 101):
        weights = np.asarray([w0, 1.0 - w0], dtype=np.float32)
        logits = weights @ per_model.reshape(2, -1)
        logits = logits.reshape(len(y), N_CLASSES)
        acc = accuracy(logits, y)
        if acc > best[0]:
            best = (acc, float(weights[0]), float(weights[1]))
    return best[1], best[2], best[0]


def best_yolo_skel(per_model: np.ndarray, skel: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    best = (-1.0, 1.0, 0.0, 0.0)
    # Normalize by global logit std so the grid is less sensitive to source scale.
    pm = per_model / np.maximum(per_model.std(axis=(1, 2), keepdims=True), 1e-6)
    sk = skel / max(float(skel.std()), 1e-6)
    flat = pm.reshape(2, -1)
    for w0 in np.linspace(0.0, 1.0, 51):
        yolo = (np.asarray([w0, 1.0 - w0], dtype=np.float32) @ flat).reshape(len(y), N_CLASSES)
        for alpha in np.linspace(0.0, 0.5, 51):
            logits = yolo + alpha * sk
            acc = accuracy(logits, y)
            if acc > best[0]:
                best = (acc, float(w0), float(1.0 - w0), float(alpha))
    return best[1], best[2], best[3], best[0]


def build_test_logits(root: Path, cache_dir: Path, checkpoint_path: Path, batch_size: int, workers: int, force: bool) -> tuple[np.ndarray, pd.DataFrame]:
    out_path = root / "outputs" / "yolo_r2p1d_test_per_model_logits.npz"
    manifest = pd.read_csv(cache_dir / "test_manifest.csv")
    if out_path.exists() and not force:
        arr = np.load(out_path, allow_pickle=True)
        return arr["per_model_logits"], manifest

    loader = DataLoader(
        CachedVideoDataset(manifest, cache_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path)
    all_logits = []
    for fold, packed in zip(checkpoint["folds"], checkpoint["models_packed"], strict=True):
        model = R2Plus1D34().to(device)
        model.load_state_dict(dequantize_state(packed))
        model.eval()
        parts = []
        with torch.no_grad():
            for inputs, _paths, _clip_ids in tqdm(loader, desc=f"test fold {fold}"):
                parts.append(predict_batch_logits(model, inputs.to(device)).float().cpu().numpy())
        all_logits.append(np.concatenate(parts, axis=0))
        del model
        torch.cuda.empty_cache()
    per_model = np.stack(all_logits, axis=0)
    np.savez_compressed(
        out_path,
        per_model_logits=per_model,
        path=manifest["path"].astype(str).to_numpy(),
        clip_id=manifest["clip_id"].astype(str).to_numpy(),
    )
    return per_model, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "yolo_r2p1d_v1")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "external_models" / "ensemble_packed.pt")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = args.root
    per_test, manifest = build_test_logits(root, args.cache_dir, args.checkpoint, args.batch_size, args.workers, args.force)
    val = np.load(root / "outputs" / "yolo_r2p1d_val_logits.npz", allow_pickle=True)
    per_val = val["per_model_logits"]
    y = val["y"]
    w0, w1, val_acc = best_yolo_weights(per_val, y)
    skel_val = aligned_skel_val(root, val["clip_id"])
    skel_test_path = root / "outputs" / "skel_imu_v2_test_logits.npz"

    variants = []
    fixed = {
        "fold0": np.asarray([1.0, 0.0], dtype=np.float32),
        "fold1": np.asarray([0.0, 1.0], dtype=np.float32),
        "equal": np.asarray([0.5, 0.5], dtype=np.float32),
        "valbest": np.asarray([w0, w1], dtype=np.float32),
        "fold0_90": np.asarray([0.9, 0.1], dtype=np.float32),
        "fold0_80": np.asarray([0.8, 0.2], dtype=np.float32),
    }
    for name, weights in fixed.items():
        val_logits = (weights @ per_val.reshape(2, -1)).reshape(len(y), N_CLASSES)
        test_logits = (weights @ per_test.reshape(2, -1)).reshape(len(manifest), N_CLASSES)
        output = root / "outputs" / f"submission_yolo_{name}.csv"
        write_submission(manifest, test_logits.argmax(1), output)
        variants.append(
            {
                "candidate": name,
                "val_acc": accuracy(val_logits, y),
                "w0": float(weights[0]),
                "w1": float(weights[1]),
                "skel_alpha": 0.0,
                "classes": int(np.unique(test_logits.argmax(1)).size),
                "diff_equal": int((test_logits.argmax(1) != fixed["equal"] @ per_test.reshape(2, -1).reshape(2, -1).reshape(2, -1)).sum())
                if False
                else 0,
                "output": str(output),
            }
        )

    if skel_val is not None and skel_test_path.exists():
        sw0, sw1, alpha, sk_acc = best_yolo_skel(per_val, skel_val, y)
        pm_val = per_val / np.maximum(per_val.std(axis=(1, 2), keepdims=True), 1e-6)
        pm_test = per_test / np.maximum(per_test.std(axis=(1, 2), keepdims=True), 1e-6)
        skel_test = np.load(skel_test_path, allow_pickle=True)["logits"]
        skel_test = skel_test / max(float(skel_test.std()), 1e-6)
        val_logits = (np.asarray([sw0, sw1]) @ pm_val.reshape(2, -1)).reshape(len(y), N_CLASSES) + alpha * (
            skel_val / max(float(skel_val.std()), 1e-6)
        )
        test_logits = (np.asarray([sw0, sw1]) @ pm_test.reshape(2, -1)).reshape(len(manifest), N_CLASSES) + alpha * skel_test
        output = root / "outputs" / "submission_yolo_skel_v2_valbest_logits.csv"
        write_submission(manifest, test_logits.argmax(1), output)
        variants.append(
            {
                "candidate": "yolo_skel_v2_valbest_logits",
                "val_acc": accuracy(val_logits, y),
                "w0": sw0,
                "w1": sw1,
                "skel_alpha": alpha,
                "classes": int(np.unique(test_logits.argmax(1)).size),
                "diff_equal": 0,
                "output": str(output),
            }
        )

    equal_preds = pd.read_csv(root / "outputs" / "submission_yolo_equal.csv")["prediction"].astype(int).to_numpy()
    for row in variants:
        preds = pd.read_csv(row["output"])["prediction"].astype(int).to_numpy()
        row["diff_equal"] = int((preds != equal_preds).sum())
    summary = pd.DataFrame(variants).sort_values("val_acc", ascending=False)
    summary.to_csv(root / "outputs" / "yolo_variant_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
