from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def _as_str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(str)


def load_features(yolo_path: Path, skel_path: Path, thermal_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    thermal = np.load(thermal_path, allow_pickle=True)

    ids = _as_str(yolo["clip_id"])
    groups = _as_str(yolo["user"]) if "user" in yolo.files else ids
    skel_lookup = {cid: i for i, cid in enumerate(_as_str(skel["clip_id"]))}
    thermal_lookup = {cid: i for i, cid in enumerate(_as_str(thermal["clip_id"]))}
    skel_idx = np.array([skel_lookup[cid] for cid in ids], dtype=np.int64)
    thermal_idx = np.array([thermal_lookup[cid] for cid in ids], dtype=np.int64)
    x = np.concatenate([yolo["logits"], skel["logits"][skel_idx], thermal["logits"][thermal_idx]], axis=1).astype(np.float32)
    y = yolo["y"].astype(np.int64)
    return x, y, ids, groups


def load_test_features(yolo_path: Path, skel_path: Path, thermal_path: Path) -> tuple[np.ndarray, np.ndarray]:
    yolo = np.load(yolo_path, allow_pickle=True)
    skel = np.load(skel_path, allow_pickle=True)
    thermal = np.load(thermal_path, allow_pickle=True)

    paths = _as_str(yolo["path"])
    skel_lookup = {path: i for i, path in enumerate(_as_str(skel["path"]))}
    thermal_lookup = {path: i for i, path in enumerate(_as_str(thermal["path"]))}
    skel_idx = np.array([skel_lookup[path] for path in paths], dtype=np.int64)
    thermal_idx = np.array([thermal_lookup[path] for path in paths], dtype=np.int64)
    x = np.concatenate([yolo["logits"], skel["logits"][skel_idx], thermal["logits"][thermal_idx]], axis=1).astype(np.float32)
    return x, paths


def make_model(C: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            max_iter=3000,
            solver="lbfgs",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-prefix", type=str, default="tri_expert_cvstack")
    args = parser.parse_args()

    outputs = args.root / "outputs"
    yolo_train = outputs / "yolo_r2p1d_all_train_logits.npz"
    yolo_test = outputs / "yolo_r2p1d_logits.npz"
    skel_train = outputs / "skel_imu_v2_train_logits.npz"
    skel_test = outputs / "skel_imu_v2_test_logits.npz"
    thermal_train = outputs / "thermal_r2p1d_train_logits.npz"
    thermal_test = outputs / "thermal_r2p1d_test_logits.npz"

    x, y, ids, groups = load_features(yolo_train, skel_train, thermal_train)
    x_test, test_paths = load_test_features(yolo_test, skel_test, thermal_test)

    candidates = [0.003, 0.01, 0.03, 0.1, 0.3]
    splitter = GroupKFold(n_splits=5)
    best_c = None
    best_score = -1.0
    for C in candidates:
        fold_scores = []
        for tr_idx, va_idx in splitter.split(x, y, groups=groups):
            model = make_model(C)
            model.fit(x[tr_idx], y[tr_idx])
            fold_scores.append(float(model.score(x[va_idx], y[va_idx])))
        mean_score = float(np.mean(fold_scores))
        print(f"C={C} mean_cv_acc={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_c = C

    assert best_c is not None
    oof_logits = np.zeros((len(x), 40), dtype=np.float32)
    test_logits = np.zeros((len(x_test), 40), dtype=np.float32)
    fold_models = []
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
        model = make_model(best_c)
        model.fit(x[tr_idx], y[tr_idx])
        fold_models.append(model)
        oof_probs = model.predict_proba(x[va_idx]).astype(np.float32)
        oof_logits[va_idx] = oof_probs
        score = float((oof_probs.argmax(axis=1) == y[va_idx]).mean())
        fold_scores.append(score)
        test_logits += model.predict_proba(x_test).astype(np.float32) / 5.0
        print(f"fold={fold} acc={score:.4f}")

    oof_acc = float((oof_logits.argmax(axis=1) == y).mean())
    print(f"best_C={best_c}")
    print(f"oof_acc={oof_acc:.4f}")
    print(f"mean_fold_acc={float(np.mean(fold_scores)):.4f}")

    np.savez_compressed(
        outputs / f"{args.output_prefix}_val_logits.npz",
        logits=oof_logits,
        y=y,
        clip_id=ids,
        user=groups,
    )
    np.savez_compressed(
        outputs / f"{args.output_prefix}_test_logits.npz",
        logits=test_logits,
        path=test_paths,
    )
    pd.DataFrame({"path": test_paths, "prediction": test_logits.argmax(axis=1).astype(int)}).to_csv(
        outputs / f"submission_{args.output_prefix}.csv",
        index=False,
    )
    joblib.dump(
        {"best_c": best_c, "fold_models": fold_models},
        outputs / f"{args.output_prefix}_model.joblib",
    )
    print(f"submission={outputs / f'submission_{args.output_prefix}.csv'}")


if __name__ == "__main__":
    main()
