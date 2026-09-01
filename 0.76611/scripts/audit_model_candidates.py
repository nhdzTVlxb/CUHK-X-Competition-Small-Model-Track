from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_postprocess import collect_test_metadata, collect_train_metadata_complete


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_N = 201
N_CLASSES = 40
HARD_DATES = {"2025-05-31", "2025-06-01", "2025-06-02"}
PUBLIC_LIKE_DATES = HARD_DATES | {"2025-06-12", "2025-06-13"}
LATE_SEEN_DATES = {"2025-06-12", "2025-06-13"}

KNOWN_LB = {
    "submission_recommended_next.csv": 0.74129,
    "submission_yolo_dual_consensus_v1.csv": 0.72139,
    "submission_anchor_consensus_4of4.csv": 0.72139,
    "submission_tri_expert_cvstack_sequence.csv": 0.72636,
    "submission_blend090_tri_meta_sequence.csv": 0.73134,
    "submission_tri_sequence_template_v2.csv": 0.73631,
    "submission_sparse_template.csv": 0.71641,
    "submission_yolo_r2p1d_v9.csv": 0.67164,
    "submission_yolo_skel_v2_valbest_logits.csv": 0.70646,
    "submission_majority_public3.csv": 0.70149,
    "submission_bestanchor_consensus_4source_bestanchor_pubm0_v1.csv": 0.75124,
    "submission_public.csv": 0.71641,
    "submission_fusion.csv": 0.67164,
    "submission_skel_imu.csv": 0.35323,
}


def norm_path(value: object) -> str:
    return str(value).strip().replace("\\", "/").strip("/")


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def top_margin(logits: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(logits), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def load_train_logits(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    values = np.load(path, allow_pickle=True)
    keys = set(values.files)
    if {"val_logits", "val_y", "val_clip_id"}.issubset(keys):
        y = values["val_y"].astype(int)
        if np.any(y < 0):
            return None
        return values["val_logits"], y, values["val_clip_id"].astype(str)
    if {"logits", "y", "clip_id"}.issubset(keys):
        y = values["y"].astype(int)
        if np.any(y < 0):
            return None
        return values["logits"], y, values["clip_id"].astype(str)
    return None


def load_test_logits(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    values = np.load(path, allow_pickle=True)
    keys = set(values.files)
    if {"test_logits", "test_path"}.issubset(keys):
        return values["test_logits"], values["test_path"].astype(str)
    if {"logits", "path"}.issubset(keys):
        return values["logits"], values["path"].astype(str)
    return None


def metric_row(
    name: str,
    logits: np.ndarray,
    y: np.ndarray,
    clip_ids: np.ndarray,
    train_meta: pd.DataFrame,
) -> dict[str, object]:
    meta = train_meta.reindex(clip_ids)
    dates = meta["date"].fillna("unknown").astype(str).to_numpy()
    users = meta["user"].fillna("unknown").astype(str).to_numpy()
    pred = np.asarray(logits).argmax(axis=1)

    def acc(mask: np.ndarray) -> float:
        if not mask.any():
            return float("nan")
        return float((pred[mask] == y[mask]).mean())

    by_date = []
    for date in sorted(set(dates)):
        mask = dates == date
        by_date.append((date, int(mask.sum()), acc(mask)))
    worst_date = min(by_date, key=lambda item: item[2] if not np.isnan(item[2]) else 2.0)

    by_user = []
    for user in sorted(set(users)):
        mask = users == user
        by_user.append((user, int(mask.sum()), acc(mask)))
    worst_user = min(by_user, key=lambda item: item[2] if not np.isnan(item[2]) else 2.0)

    hard = np.isin(dates, list(HARD_DATES))
    public_like = np.isin(dates, list(PUBLIC_LIKE_DATES))
    late = np.isin(dates, list(LATE_SEEN_DATES))
    margins = top_margin(logits)
    correct = pred == y
    return {
        "candidate": name,
        "kind": "logits",
        "train_rows": int(len(y)),
        "all_acc": acc(np.ones(len(y), dtype=bool)),
        "hard_acc": acc(hard),
        "public_like_acc": acc(public_like),
        "late_seen_acc": acc(late),
        "non_public_like_acc": acc(~public_like),
        "worst_date": worst_date[0],
        "worst_date_n": worst_date[1],
        "worst_date_acc": worst_date[2],
        "worst_user": worst_user[0],
        "worst_user_n": worst_user[1],
        "worst_user_acc": worst_user[2],
        "mean_margin_correct": float(margins[correct].mean()) if correct.any() else float("nan"),
        "mean_margin_wrong": float(margins[~correct].mean()) if (~correct).any() else float("nan"),
    }


def test_diff_row(
    name: str,
    paths: np.ndarray,
    pred: np.ndarray,
    anchor: pd.DataFrame,
    test_meta: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    official_paths = anchor["path_norm"].to_numpy()
    lookup = {norm_path(path): int(label) for path, label in zip(paths, pred, strict=True)}
    aligned = np.asarray([lookup[norm_path(path)] for path in official_paths], dtype=int)
    base = anchor["prediction"].to_numpy(dtype=int)
    diff = aligned != base
    public_mask = np.arange(len(base)) < PUBLIC_N
    hidden_mask = ~public_mask
    date_values = test_meta.reindex(official_paths)["date"].fillna("unknown").astype(str).to_numpy()

    date_counts = {
        date: int((diff & (date_values == date)).sum())
        for date in sorted(set(date_values))
        if int((diff & (date_values == date)).sum()) > 0
    }
    public_date_counts = {
        date: int((diff & public_mask & (date_values == date)).sum())
        for date in sorted(set(date_values))
        if int((diff & public_mask & (date_values == date)).sum()) > 0
    }
    changed = pd.DataFrame(
        {
            "candidate": name,
            "row": np.flatnonzero(diff) + 1,
            "is_public201": public_mask[diff],
            "date": date_values[diff],
            "path": official_paths[diff],
            "anchor_prediction": base[diff],
            "candidate_prediction": aligned[diff],
        }
    )
    row = {
        "candidate": name,
        "kind": "submission",
        "known_lb": KNOWN_LB.get(name),
        "test_rows": int(len(base)),
        "diff_vs_anchor_all": int(diff.sum()),
        "diff_vs_anchor_public201": int((diff & public_mask).sum()),
        "diff_vs_anchor_hidden204": int((diff & hidden_mask).sum()),
        "diff_hard_public201": int((diff & public_mask & np.isin(date_values, list(HARD_DATES))).sum()),
        "diff_late_public201": int((diff & public_mask & np.isin(date_values, list(LATE_SEEN_DATES))).sum()),
        "diff_by_date": json.dumps(date_counts, sort_keys=True),
        "public_diff_by_date": json.dumps(public_date_counts, sort_keys=True),
    }
    return row, changed


def submission_from_logits(path: Path, official_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    loaded = load_test_logits(path)
    if loaded is None:
        return None
    logits, paths = loaded
    pred = np.asarray(logits).argmax(axis=1).astype(int)
    return paths, pred


def collect_default_candidates(outputs: Path) -> list[Path]:
    names = [
        "tri_best_logits.npz",
        "tri_expert_cvstack_test_logits.npz",
        "tri_expert_cvstack_sequence_logits.npz",
        "meta_stack_v2_test_logits.npz",
        "meta_stack_date_weighted_v1_test_logits.npz",
        "blend090_tri_meta_sequence_logits.npz",
        "yolo_r2p1d_logits.npz",
        "yolo_r2p1d_v9_test_logits.npz",
        "skel_imu_v2_test_logits.npz",
        "thermal_r2p1d_test_logits.npz",
        "thermal_r2p1d_model1_test_logits.npz",
        "finetune_yolo_date_aug_last_v1_logits.npz",
        "finetune_yolo_date_aug_last_v1_m1_logits.npz",
    ]
    candidates = [outputs / name for name in names if (outputs / name).exists()]
    candidates.extend(sorted(outputs.glob("submission_*.csv")))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--anchor", type=Path, default=ROOT / "outputs" / "submission_recommended_next.csv")
    parser.add_argument("--output-prefix", type=str, default="candidate_harddate_audit")
    parser.add_argument("candidates", nargs="*", type=Path)
    args = parser.parse_args()

    outputs = args.root / "outputs"
    anchor = pd.read_csv(args.anchor)
    anchor["path_norm"] = anchor["path"].map(norm_path)

    # Include clips whose RGB filenames omit timestamps; their timing can be
    # recovered from the paired skeleton stream.
    train_meta = collect_train_metadata_complete(args.root)
    train_meta["date"] = train_meta["start"].dt.date.astype(str)
    train_meta = train_meta.set_index("clip_id")
    test_meta = collect_test_metadata(args.root)
    test_meta["date"] = test_meta["start"].dt.date.astype(str)
    test_meta["path_norm"] = test_meta["path"].map(norm_path)
    test_meta = test_meta.set_index("path_norm")

    candidates = args.candidates or collect_default_candidates(outputs)
    metric_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    changes: list[pd.DataFrame] = []

    official_paths = anchor["path_norm"].to_numpy()
    for raw_path in candidates:
        path = raw_path if raw_path.is_absolute() else args.root / raw_path
        if not path.exists():
            print(f"skip_missing={path}")
            continue
        name = path.name
        try:
            if path.suffix.lower() == ".npz":
                train_loaded = load_train_logits(path)
                if train_loaded is not None:
                    logits, y, clip_ids = train_loaded
                    metric_rows.append(metric_row(name, logits, y, clip_ids, train_meta))
                test_loaded = submission_from_logits(path, official_paths)
                if test_loaded is not None:
                    paths, pred = test_loaded
                    row, changed = test_diff_row(name, paths, pred, anchor, test_meta)
                    diff_rows.append(row)
                    if not changed.empty:
                        changes.append(changed)
            elif path.suffix.lower() == ".csv":
                sub = pd.read_csv(path)
                if {"path", "prediction"}.issubset(sub.columns) and len(sub) == len(anchor):
                    row, changed = test_diff_row(
                        name,
                        sub["path"].astype(str).to_numpy(),
                        sub["prediction"].astype(int).to_numpy(),
                        anchor,
                        test_meta,
                    )
                    diff_rows.append(row)
                    if not changed.empty:
                        changes.append(changed)
        except Exception as exc:
            print(f"skip_error={path} error={exc}")

    metrics = pd.DataFrame(metric_rows)
    diffs = pd.DataFrame(diff_rows)
    changes_frame = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame()

    if not metrics.empty:
        metrics = metrics.sort_values(
            ["hard_acc", "public_like_acc", "all_acc"],
            ascending=[False, False, False],
            na_position="last",
        )
    if not diffs.empty:
        diffs = diffs.sort_values(
            ["known_lb", "diff_vs_anchor_public201", "diff_vs_anchor_all"],
            ascending=[False, True, True],
            na_position="last",
        )

    metrics_path = outputs / f"{args.output_prefix}_metrics.csv"
    diffs_path = outputs / f"{args.output_prefix}_diffs.csv"
    changes_path = outputs / f"{args.output_prefix}_changes.csv"
    metrics.to_csv(metrics_path, index=False)
    diffs.to_csv(diffs_path, index=False)
    changes_frame.to_csv(changes_path, index=False)

    print("Top train metrics:")
    print(metrics.head(20).to_string(index=False) if not metrics.empty else "none")
    print("\nTop submission diffs:")
    columns = [
        "candidate",
        "known_lb",
        "diff_vs_anchor_public201",
        "diff_vs_anchor_hidden204",
        "public_diff_by_date",
    ]
    available = [column for column in columns if column in diffs.columns]
    print(diffs[available].head(30).to_string(index=False) if not diffs.empty else "none")
    print("\nWrote:")
    print(metrics_path)
    print(diffs_path)
    print(changes_path)


if __name__ == "__main__":
    main()
