from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 40


EXPERTS = [
    ("yolo", ROOT / "outputs" / "yolo_r2p1d_all_train_logits.npz", ROOT / "outputs" / "yolo_r2p1d_logits.npz"),
    ("yolo_v9", ROOT / "outputs" / "yolo_r2p1d_v9_train_logits.npz", ROOT / "outputs" / "yolo_r2p1d_v9_test_logits.npz"),
    ("skel", ROOT / "outputs" / "skel_imu_v2_train_logits.npz", ROOT / "outputs" / "skel_imu_v2_test_logits.npz"),
    ("thermal", ROOT / "outputs" / "thermal_r2p1d_train_logits.npz", ROOT / "outputs" / "thermal_r2p1d_test_logits.npz"),
    ("thermal_m1", ROOT / "outputs" / "thermal_r2p1d_model1_train_logits.npz", ROOT / "outputs" / "thermal_r2p1d_model1_test_logits.npz"),
]


def _as_str(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(str)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)


def _summary_features(logits: np.ndarray) -> np.ndarray:
    probs = _softmax(logits)
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    top2.sort(axis=1)
    top1 = top2[:, 1]
    second = top2[:, 0]
    entropy = -(probs * np.log(np.maximum(probs, 1e-8))).sum(axis=1) / np.log(probs.shape[1])
    return np.stack(
        [
            logits.max(axis=1),
            logits.mean(axis=1),
            logits.std(axis=1),
            top1,
            second,
            top1 - second,
            entropy,
        ],
        axis=1,
    ).astype(np.float32)


def _expert_block(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.mean(axis=1, keepdims=True)
    scale = logits.std(axis=1, keepdims=True)
    scaled = centered / np.maximum(scale, 1e-6)
    return np.concatenate([centered.astype(np.float32), scaled.astype(np.float32), _summary_features(logits)], axis=1)


def _infer_user(clip_ids: np.ndarray) -> np.ndarray:
    users = []
    for cid in clip_ids:
        parts = str(cid).split("__")
        users.append(parts[1] if len(parts) >= 3 else str(cid))
    return np.asarray(users)


def load_train_pair(path: Path, label_lookup: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    out["clip_id"] = _as_str(out["clip_id"])
    if "y" in out:
        out["y"] = np.asarray(out["y"]).astype(np.int64)
    elif label_lookup is not None:
        out["y"] = np.asarray([label_lookup[cid] for cid in out["clip_id"]], dtype=np.int64)
    else:
        raise KeyError(f"Missing y in {path} and no label lookup provided")
    out["user"] = _as_str(out["user"]) if "user" in out else _infer_user(out["clip_id"])
    return out


def load_test_pair(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    out["path"] = _as_str(out["path"])
    if "clip_id" in out:
        out["clip_id"] = _as_str(out["clip_id"])
    return out


def build_train_features(experts: list[tuple[str, dict[str, np.ndarray]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = experts[0][1]
    clip_ids = base["clip_id"]
    y = base["y"]
    groups = base.get("user", _infer_user(clip_ids))

    feats = []
    vote_preds = []
    for name, data in experts:
        lookup = {cid: i for i, cid in enumerate(data["clip_id"])}
        idx = np.array([lookup[cid] for cid in clip_ids], dtype=np.int64)
        logits = np.asarray(data["logits"][idx], dtype=np.float32)
        feats.append(_expert_block(logits))
        vote_preds.append(logits.argmax(axis=1))

    votes = np.stack(vote_preds, axis=1)
    vote_counts = np.zeros((len(clip_ids), N_CLASSES), dtype=np.float32)
    rows = np.arange(len(clip_ids))
    for expert_idx in range(votes.shape[1]):
        vote_counts[rows, votes[:, expert_idx]] += 1.0
    vote_top2 = np.partition(vote_counts, -2, axis=1)[:, -2:]
    vote_top2.sort(axis=1)
    vote_summary = np.stack(
        [
            vote_counts.max(axis=1),
            vote_top2[:, 0],
            vote_top2[:, 1],
            vote_top2[:, 1] - vote_top2[:, 0],
            (vote_counts.max(axis=1) == votes.shape[1]).astype(np.float32),
            (vote_counts.max(axis=1) >= 3).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    x = np.concatenate(feats + [vote_summary], axis=1)
    return x, y, clip_ids, groups


def build_test_features(experts: list[tuple[str, dict[str, np.ndarray]]]) -> tuple[np.ndarray, np.ndarray]:
    base = experts[0][1]
    paths = base["path"]
    feats = []
    vote_preds = []
    for name, data in experts:
        lookup_key = "path" if "path" in data else "clip_id"
        lookup = {str(key): i for i, key in enumerate(data[lookup_key])}
        idx = np.array([lookup[str(key)] for key in paths], dtype=np.int64)
        logits = np.asarray(data["logits"][idx], dtype=np.float32)
        feats.append(_expert_block(logits))
        vote_preds.append(logits.argmax(axis=1))

    votes = np.stack(vote_preds, axis=1)
    vote_counts = np.zeros((len(paths), N_CLASSES), dtype=np.float32)
    rows = np.arange(len(paths))
    for expert_idx in range(votes.shape[1]):
        vote_counts[rows, votes[:, expert_idx]] += 1.0
    vote_top2 = np.partition(vote_counts, -2, axis=1)[:, -2:]
    vote_top2.sort(axis=1)
    vote_summary = np.stack(
        [
            vote_counts.max(axis=1),
            vote_top2[:, 0],
            vote_top2[:, 1],
            vote_top2[:, 1] - vote_top2[:, 0],
            (vote_counts.max(axis=1) == votes.shape[1]).astype(np.float32),
            (vote_counts.max(axis=1) >= 3).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    x = np.concatenate(feats + [vote_summary], axis=1)
    return x, paths


def candidate_models() -> list[tuple[str, object]]:
    return [
        ("lr_c001", make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000, solver="lbfgs", multi_class="multinomial"))),
        ("lr_c003", make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=5000, solver="lbfgs", multi_class="multinomial"))),
        ("lr_c01", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, solver="lbfgs", multi_class="multinomial"))),
        ("lr_bal_c003", make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=5000, solver="lbfgs", multi_class="multinomial", class_weight="balanced"))),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-prefix", type=str, default="meta_stack_v2")
    args = parser.parse_args()

    expert_pairs = []
    base_train = None
    for name, train_path, test_path in EXPERTS:
        if train_path.exists() and test_path.exists():
            if base_train is None:
                base_train = load_train_pair(train_path)
                label_lookup = {cid: int(y) for cid, y in zip(base_train["clip_id"], base_train["y"], strict=True)}
                expert_pairs.append((name, base_train, load_test_pair(test_path)))
            else:
                expert_pairs.append((name, load_train_pair(train_path, label_lookup), load_test_pair(test_path)))
        else:
            print(f"skip={name} train={train_path.exists()} test={test_path.exists()}")

    if len(expert_pairs) < 3:
        raise FileNotFoundError("Need at least 3 expert pairs to train the stack")

    train_inputs = [(name, train) for name, train, _ in expert_pairs]
    test_inputs = [(name, test) for name, _, test in expert_pairs]
    x, y, clip_ids, groups = build_train_features(train_inputs)
    x_test, test_paths = build_test_features(test_inputs)

    splitter = GroupKFold(n_splits=5)
    rows = []
    best_name = None
    best_oof = -1.0
    best_model = None
    best_oof_logits = None
    for name, model in candidate_models():
        oof = np.zeros((len(x), N_CLASSES), dtype=np.float32)
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
            fold_model = clone(model)
            fold_model.fit(x[tr_idx], y[tr_idx])
            proba = fold_model.predict_proba(x[va_idx]).astype(np.float32)
            oof[va_idx] = proba
            acc = float((proba.argmax(axis=1) == y[va_idx]).mean())
            fold_scores.append(acc)
        oof_acc = float((oof.argmax(axis=1) == y).mean())
        row = {
            "candidate": name,
            "mean_fold_acc": float(np.mean(fold_scores)),
            "oof_acc": oof_acc,
            "fold_min": float(np.min(fold_scores)),
            "fold_max": float(np.max(fold_scores)),
        }
        rows.append(row)
        print(f"{name} mean_fold={row['mean_fold_acc']:.4f} oof={oof_acc:.4f} folds={fold_scores}")
        if oof_acc > best_oof:
            best_oof = oof_acc
            best_name = name
            best_model = model
            best_oof_logits = oof.copy()

    assert best_name is not None and best_model is not None
    best_model.fit(x, y)
    test_proba = best_model.predict_proba(x_test).astype(np.float32)
    preds = test_proba.argmax(axis=1).astype(int)

    out_dir = args.root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{args.output_prefix}_test_logits.npz",
        logits=test_proba,
        path=test_paths,
    )
    np.savez_compressed(
        out_dir / f"{args.output_prefix}_train_oof_logits.npz",
        logits=best_oof_logits if best_oof_logits is not None else np.zeros((len(y), N_CLASSES), dtype=np.float32),
        y=y,
        clip_id=clip_ids,
        user=groups,
    )
    submission = pd.DataFrame({"path": test_paths, "prediction": preds})
    submission.to_csv(out_dir / f"submission_{args.output_prefix}.csv", index=False)
    joblib.dump(best_model, out_dir / f"{args.output_prefix}_model.joblib")
    pd.DataFrame(rows).sort_values("oof_acc", ascending=False).to_csv(out_dir / f"{args.output_prefix}_report.csv", index=False)
    print(f"best={best_name} oof_acc={best_oof:.4f}")
    print(f"submission={out_dir / f'submission_{args.output_prefix}.csv'}")


if __name__ == "__main__":
    main()
