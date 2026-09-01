from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def _as_str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(str)


def load_split(yolo_path: Path, skel_path: Path, thermal_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    thermal = np.load(thermal_path, allow_pickle=True)

    yolo_ids = _as_str(yolo["clip_id"])
    skel_lookup = {cid: i for i, cid in enumerate(_as_str(skel["clip_id"]))}
    thermal_lookup = {cid: i for i, cid in enumerate(_as_str(thermal["clip_id"]))}
    skel_idx = np.array([skel_lookup[cid] for cid in yolo_ids], dtype=np.int64)
    thermal_idx = np.array([thermal_lookup[cid] for cid in yolo_ids], dtype=np.int64)
    x = np.concatenate([yolo["logits"], skel["logits"][skel_idx], thermal["logits"][thermal_idx]], axis=1).astype(np.float32)
    return x, yolo["y"].astype(np.int64), yolo_ids


def load_test(yolo_path: Path, skel_path: Path, thermal_path: Path) -> tuple[np.ndarray, np.ndarray]:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    thermal = np.load(thermal_path, allow_pickle=True)

    yolo_paths = _as_str(yolo["path"])
    skel_lookup = {path: i for i, path in enumerate(_as_str(skel["path"]))}
    thermal_lookup = {path: i for i, path in enumerate(_as_str(thermal["path"]))}
    skel_idx = np.array([skel_lookup[path] for path in yolo_paths], dtype=np.int64)
    thermal_idx = np.array([thermal_lookup[path] for path in yolo_paths], dtype=np.int64)
    x = np.concatenate([yolo["logits"], skel["logits"][skel_idx], thermal["logits"][thermal_idx]], axis=1).astype(np.float32)
    return x, yolo_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-prefix", type=str, default="tri_expert_logreg")
    args = parser.parse_args()

    paths = ROOT / "outputs"
    yolo_train = paths / "yolo_r2p1d_all_train_logits.npz"
    yolo_val = paths / "yolo_r2p1d_val_logits.npz"
    yolo_test = paths / "yolo_r2p1d_logits.npz"
    skel_train = paths / "skel_imu_v2_train_logits.npz"
    skel_val = paths / "skel_imu_v2_val_logits.npz"
    skel_test = paths / "skel_imu_v2_test_logits.npz"
    thermal_train = paths / "thermal_r2p1d_train_logits.npz"
    thermal_test = paths / "thermal_r2p1d_test_logits.npz"

    x_train, y_train, train_ids = load_split(yolo_train, skel_train, thermal_train)
    x_val, y_val, val_ids = load_split(yolo_val, skel_val, thermal_train)
    x_test, test_paths = load_test(yolo_test, skel_test, thermal_test)

    candidates = [0.01, 0.03, 0.1, 0.3, 1.0]
    best = None
    best_acc = -1.0
    best_model = None
    for C in candidates:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=3000,
                solver="lbfgs",
                multi_class="multinomial",
            ),
        )
        model.fit(x_train, y_train)
        acc = float(model.score(x_val, y_val))
        print(f"C={C} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best = C
            best_model = model

    assert best_model is not None
    val_pred = best_model.predict(x_val)
    test_pred = best_model.predict(x_test)
    val_logits = best_model.predict_proba(x_val)
    test_logits = best_model.predict_proba(x_test)

    out_dir = paths
    np.savez_compressed(
        out_dir / f"{args.output_prefix}_val_logits.npz",
        logits=val_logits,
        y=y_val,
        clip_id=val_ids,
    )
    np.savez_compressed(
        out_dir / f"{args.output_prefix}_test_logits.npz",
        logits=test_logits,
        path=test_paths,
    )
    pd.DataFrame({"path": test_paths, "prediction": test_pred.astype(int)}).to_csv(
        out_dir / f"submission_{args.output_prefix}.csv",
        index=False,
    )
    joblib.dump(best_model, out_dir / f"{args.output_prefix}_model.joblib")
    print(f"best_C={best} val_acc={best_acc:.4f}")
    print(f"submission={out_dir / f'submission_{args.output_prefix}.csv'}")


if __name__ == "__main__":
    main()
