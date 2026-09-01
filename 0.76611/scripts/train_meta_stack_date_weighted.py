from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sequence_postprocess import collect_train_metadata

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
    scale = np.maximum(logits.std(axis=1, keepdims=True), 1e-6)
    scaled = centered / scale
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


def build_train_features(
    experts: list[tuple[str, dict[str, np.ndarray]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = experts[0][1]
    clip_ids = base["clip_id"]
    y = base["y"]
    groups = base.get("user", _infer_user(clip_ids))

    feats = []
    vote_preds = []
    for _name, data in experts:
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
    for _name, data in experts:
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


def date_weight_map(root: Path) -> dict[str, float]:
    report_path = root / "outputs" / "validation_gap_train_acc_by_group.csv"
    if report_path.exists():
        report = pd.read_csv(report_path)
        date_rows = report[(report["file"] == "tri_expert_cvstack_val_logits.npz") & (report["group_type"] == "date")]
        if not date_rows.empty:
            median_acc = float(date_rows["acc"].median())
            weights = {}
            for row in date_rows.itertuples(index=False):
                acc = float(row.acc)
                weight = (median_acc / max(acc, 1e-4)) ** 1.5
                weights[str(row.group)] = float(np.clip(weight, 0.8, 1.8))
            return weights
    return {
        "2025-05-07": 0.85,
        "2025-05-08": 0.85,
        "2025-05-30": 0.90,
        "2025-05-31": 1.20,
        "2025-06-01": 1.15,
        "2025-06-02": 1.30,
        "2025-06-09": 1.00,
        "2025-06-10": 0.95,
        "2025-06-11": 0.95,
        "2025-06-12": 1.10,
        "2025-06-13": 1.10,
    }


def build_metadata(root: Path, clip_ids: np.ndarray) -> pd.DataFrame:
    meta = collect_train_metadata(root)[["clip_id", "user", "start"]].copy()
    meta["date"] = meta["start"].dt.date.astype(str)
    lookup = meta.set_index("clip_id")
    rows = []
    for cid in clip_ids:
        if cid in lookup.index:
            row = lookup.loc[cid]
            rows.append({"clip_id": cid, "user": str(row["user"]), "date": str(row["date"])})
        else:
            rows.append({"clip_id": cid, "user": str(_infer_user(np.asarray([cid]))[0]), "date": "unknown"})
    return pd.DataFrame(rows)


def build_groups(meta: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "user":
        return meta["user"].astype(str).to_numpy()
    if mode == "date":
        return meta["date"].astype(str).to_numpy()
    if mode == "user_date":
        return (meta["user"].astype(str) + "__" + meta["date"].astype(str)).to_numpy()
    raise ValueError(f"Unknown group mode: {mode}")


def build_sample_weights(root: Path, y: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    class_counts = np.bincount(y, minlength=N_CLASSES).astype(np.float32)
    class_weights = np.asarray([1.0 / np.sqrt(max(class_counts[label], 1.0)) for label in y], dtype=np.float32)
    class_weights /= np.mean(class_weights)

    dweights = date_weight_map(root)
    date_weights = np.asarray([dweights.get(str(date), 1.0) for date in meta["date"].astype(str)], dtype=np.float32)
    weights = class_weights * date_weights
    weights /= np.mean(weights)
    return weights.astype(np.float32)


class WeightedMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.20, num_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FoldResult:
    candidate: str
    mean_fold_acc: float
    weighted_oof_acc: float
    oof_acc: float
    hard_date_acc: float
    late_date_acc: float
    fold_min: float
    fold_max: float


def _make_lr(C: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=5000, solver="lbfgs", multi_class="multinomial"),
    )


def _weighted_accuracy(pred: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(pred == y, weights=weights))


def _train_weighted_mlp(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    *,
    hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model = WeightedMLP(x.shape[1], hidden=hidden, dropout=dropout).to(device)
    train_ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
        torch.from_numpy(w.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(x_val.astype(np.float32)).to(device)
    val_y = torch.from_numpy(y_val.astype(np.int64)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_state = None
    best_score = -1.0
    best_metrics = {}
    patience = 8
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by, bw in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            bw = bw.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            loss = (loss * bw).sum() / torch.maximum(bw.sum(), torch.tensor(1e-6, device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_pred = val_logits.argmax(1)
            val_acc = float((val_pred == val_y).float().mean().cpu())
            weighted_val_acc = float(np.average((val_pred.cpu().numpy() == y_val), weights=w_val))
        if weighted_val_acc > best_score:
            best_score = weighted_val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {"val_acc": val_acc, "weighted_val_acc": weighted_val_acc, "epoch": float(epoch)}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x).float().cpu().numpy()
        test_logits = None
    return val_logits, best_state, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-prefix", type=str, default="meta_stack_date_weighted")
    parser.add_argument("--group-mode", type=str, default="date", choices=["user", "date", "user_date"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--mlp-only", action="store_true")
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
    x, y, clip_ids, base_groups = build_train_features(train_inputs)
    x_test, test_paths = build_test_features(test_inputs)

    meta = build_metadata(args.root, clip_ids)
    groups = build_groups(meta, args.group_mode)
    sample_weights = build_sample_weights(args.root, y, meta)

    hard_dates = {"2025-05-31", "2025-06-01", "2025-06-02", "2025-06-12", "2025-06-13"}
    late_dates = {"2025-06-12", "2025-06-13"}
    hard_mask = meta["date"].isin(hard_dates).to_numpy()
    late_mask = meta["date"].isin(late_dates).to_numpy()

    splitter = GroupKFold(n_splits=5)
    rows: list[dict[str, object]] = []
    best_name = None
    best_score = -1.0
    best_oof_logits = None
    best_test_logits = None
    best_model = None

    candidates: list[tuple[str, str, object]] = []
    if not args.mlp_only:
        candidates.extend(
            [
                ("lr_c001", "lr", _make_lr(0.01)),
                ("lr_c003", "lr", _make_lr(0.03)),
                ("lr_c01", "lr", _make_lr(0.10)),
            ]
        )
    candidates.extend(
        [
            ("mlp_h96", "mlp", {"hidden": 96, "dropout": args.dropout}),
            ("mlp_h128", "mlp", {"hidden": 128, "dropout": args.dropout}),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} candidates={len(candidates)} group_mode={args.group_mode}")

    for name, kind, model_spec in candidates:
        oof_logits = np.zeros((len(x), N_CLASSES), dtype=np.float32)
        fold_scores = []
        hard_scores = []
        late_scores = []
        weighted_scores = []
        test_logits = np.zeros((len(x_test), N_CLASSES), dtype=np.float32)

        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x, y, groups=groups), start=1):
            if kind == "lr":
                model = clone(model_spec)
                model.fit(x[tr_idx], y[tr_idx], logisticregression__sample_weight=sample_weights[tr_idx])
                proba = model.predict_proba(x[va_idx]).astype(np.float32)
                test_logits += model.predict_proba(x_test).astype(np.float32) / splitter.get_n_splits()
                fold_model = model
            else:
                train_x = x[tr_idx]
                train_y = y[tr_idx]
                train_w = sample_weights[tr_idx]
                val_logits, state, _metrics = _train_weighted_mlp(
                    train_x,
                    train_y,
                    train_w,
                    x[va_idx],
                    y[va_idx],
                    sample_weights[va_idx],
                    hidden=int(model_spec["hidden"]),
                    dropout=float(model_spec["dropout"]),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                )
                proba = _softmax(val_logits).astype(np.float32)
                fold_model = WeightedMLP(x.shape[1], hidden=int(model_spec["hidden"]), dropout=float(model_spec["dropout"])).to(device)
                # load the best available state from training
                # note: best state was captured during training; re-train once on full fold for test logits
                fold_model = fold_model.to(device)
                fold_model.load_state_dict({k: v.to(device) for k, v in state.items()})
                fold_model.eval()
                with torch.no_grad():
                    test_part = fold_model(torch.from_numpy(x_test.astype(np.float32)).to(device)).float().cpu().numpy()
                test_logits += _softmax(test_part).astype(np.float32) / splitter.get_n_splits()

            oof_logits[va_idx] = proba
            pred = proba.argmax(axis=1)
            acc = float((pred == y[va_idx]).mean())
            weighted_acc = _weighted_accuracy(pred, y[va_idx], sample_weights[va_idx])
            hard_acc = float((pred[hard_mask[va_idx]] == y[va_idx][hard_mask[va_idx]]).mean()) if hard_mask[va_idx].any() else float("nan")
            late_acc = float((pred[late_mask[va_idx]] == y[va_idx][late_mask[va_idx]]).mean()) if late_mask[va_idx].any() else float("nan")
            fold_scores.append(acc)
            weighted_scores.append(weighted_acc)
            hard_scores.append(hard_acc)
            late_scores.append(late_acc)
            print(
                f"{name} fold={fold} acc={acc:.4f} weighted={weighted_acc:.4f} "
                f"hard={hard_acc:.4f} late={late_acc:.4f}"
            )

        oof_acc = float((oof_logits.argmax(axis=1) == y).mean())
        weighted_oof_acc = _weighted_accuracy(oof_logits.argmax(axis=1), y, sample_weights)
        hard_oof_acc = float((oof_logits.argmax(axis=1)[hard_mask] == y[hard_mask]).mean()) if hard_mask.any() else float("nan")
        late_oof_acc = float((oof_logits.argmax(axis=1)[late_mask] == y[late_mask]).mean()) if late_mask.any() else float("nan")
        row = {
            "candidate": name,
            "mean_fold_acc": float(np.mean(fold_scores)),
            "mean_weighted_fold_acc": float(np.mean(weighted_scores)),
            "oof_acc": oof_acc,
            "weighted_oof_acc": weighted_oof_acc,
            "hard_oof_acc": hard_oof_acc,
            "late_oof_acc": late_oof_acc,
            "fold_min": float(np.min(fold_scores)),
            "fold_max": float(np.max(fold_scores)),
        }
        rows.append(row)
        print(
            f"{name} mean_fold={row['mean_fold_acc']:.4f} weighted_fold={row['mean_weighted_fold_acc']:.4f} "
            f"oof={oof_acc:.4f} weighted_oof={weighted_oof_acc:.4f} hard={hard_oof_acc:.4f} late={late_oof_acc:.4f}"
        )
        score = weighted_oof_acc
        if score > best_score:
            best_score = score
            best_name = name
            best_oof_logits = oof_logits.copy()
            best_test_logits = test_logits.copy()
            best_model = kind

    assert best_name is not None and best_oof_logits is not None and best_test_logits is not None

    out_dir = args.root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{args.output_prefix}_train_oof_logits.npz",
        logits=best_oof_logits,
        y=y,
        clip_id=clip_ids,
        user=base_groups,
        date=meta["date"].astype(str).to_numpy(),
    )
    np.savez_compressed(
        out_dir / f"{args.output_prefix}_test_logits.npz",
        logits=best_test_logits,
        path=test_paths,
    )
    pd.DataFrame({"path": test_paths, "prediction": best_test_logits.argmax(axis=1).astype(int)}).to_csv(
        out_dir / f"submission_{args.output_prefix}.csv",
        index=False,
    )
    pd.DataFrame(rows).sort_values("weighted_oof_acc", ascending=False).to_csv(
        out_dir / f"{args.output_prefix}_report.csv", index=False
    )
    joblib.dump(
        {"best_candidate": best_name, "best_kind": best_model, "group_mode": args.group_mode},
        out_dir / f"{args.output_prefix}_model.joblib",
    )
    print(f"best_candidate={best_name} best_score={best_score:.6f}")
    print(f"submission={out_dir / f'submission_{args.output_prefix}.csv'}")


if __name__ == "__main__":
    main()
