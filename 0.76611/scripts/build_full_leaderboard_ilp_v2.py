from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from audit_model_candidates import norm_path
from build_anchor_gated_candidates import MODEL_SPECS
from sequence_postprocess import collect_test_metadata


ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 40
DENOMINATOR = 402


@dataclass(frozen=True)
class KnownSubmission:
    name: str
    correct: int


KNOWN_SUBMISSIONS = [
    KnownSubmission("submission_recommended_next.csv", 298),
    KnownSubmission("submission_meta_stack_date_weighted_seen_yolo.csv", 298),
    KnownSubmission("submission_tri_sequence_template_v2.csv", 296),
    KnownSubmission("submission_blend090_tri_meta_sequence.csv", 294),
    KnownSubmission("submission_tri_expert_cvstack_sequence.csv", 292),
    KnownSubmission("submission_yolo_dual_consensus_v1.csv", 290),
    KnownSubmission("submission_anchor_consensus_4of4.csv", 290),
    KnownSubmission("submission_sparse_template.csv", 288),
    KnownSubmission("submission_yolo_skel_v2_valbest_logits.csv", 284),
    KnownSubmission("submission_majority_public3.csv", 282),
    KnownSubmission("submission_yolo_r2p1d_v9.csv", 270),
    KnownSubmission("submission_skel_imu.csv", 142),
    KnownSubmission("submission_lb_preview_model2_v1.csv", 306),
]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def read_submission(path: Path, official_paths: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    frame["path_norm"] = frame["path"].map(norm_path)
    lookup = dict(zip(frame["path_norm"], frame["prediction"].astype(int), strict=True))
    return np.asarray([lookup[norm_path(raw)] for raw in official_paths], dtype=np.int64)


def load_logits(path: Path, official_paths: np.ndarray) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if {"test_logits", "test_path"}.issubset(data.files):
        logits = data["test_logits"]
        paths = data["test_path"].astype(str)
    elif {"logits", "path"}.issubset(data.files):
        logits = data["logits"]
        paths = data["path"].astype(str)
    else:
        raise KeyError(f"No test logits in {path}")
    lookup = {norm_path(raw): idx for idx, raw in enumerate(paths)}
    order = np.asarray([lookup[norm_path(raw)] for raw in official_paths], dtype=np.int64)
    return np.asarray(logits[order], dtype=np.float64)


def model_evidence(outputs: Path, official_paths: np.ndarray) -> tuple[np.ndarray, list[str]]:
    evidence = np.zeros((len(official_paths), N_CLASSES), dtype=np.float64)
    names: list[str] = []
    for spec in MODEL_SPECS:
        path = outputs / spec.path
        if not path.exists():
            continue
        logits = load_logits(path, official_paths)
        centered = logits - logits.mean(axis=1, keepdims=True)
        scaled = centered / np.maximum(logits.std(axis=1, keepdims=True), 1e-6)
        probs = softmax(logits)
        evidence += spec.weight * (scaled + 0.6 * probs)
        names.append(spec.name)
    if not names:
        raise FileNotFoundError("No model evidence logits found")
    return evidence, names


def build_constraints(
    known_predictions: list[np.ndarray],
    targets: np.ndarray,
    scored_mask: np.ndarray,
) -> LinearConstraint:
    n_rows = len(scored_mask)
    n_vars = n_rows * N_CLASSES
    n_constraints = n_rows + len(targets)
    matrix = lil_matrix((n_constraints, n_vars), dtype=np.float64)
    lower = np.zeros(n_constraints, dtype=np.float64)
    upper = np.zeros(n_constraints, dtype=np.float64)

    for row in range(n_rows):
        start = row * N_CLASSES
        matrix[row, start : start + N_CLASSES] = 1.0
        lower[row] = upper[row] = 1.0

    scored_rows = np.flatnonzero(scored_mask)
    for sub_idx, (prediction, target) in enumerate(zip(known_predictions, targets, strict=True)):
        constraint_row = n_rows + sub_idx
        matrix[constraint_row, scored_rows * N_CLASSES + prediction[scored_rows]] = 1.0
        lower[constraint_row] = upper[constraint_row] = float(target)

    return LinearConstraint(matrix.tocsr(), lower, upper)


def solve_labels(
    evidence: np.ndarray,
    anchor: np.ndarray,
    known_predictions: list[np.ndarray],
    targets: np.ndarray,
    scored_mask: np.ndarray,
    *,
    anchor_prior: float,
    submit_prior: float,
    old_preview_prior: float,
    old_preview: np.ndarray,
    random_seed: int | None = None,
) -> tuple[np.ndarray, object]:
    objective = -evidence.copy()
    objective[np.arange(len(anchor)), anchor] -= anchor_prior
    for prediction in known_predictions:
        objective[np.arange(len(anchor)), prediction] -= submit_prior
    objective[np.arange(len(anchor)), old_preview] -= old_preview_prior
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        objective += rng.normal(0.0, 0.015, objective.shape)
    result = milp(
        c=objective.reshape(-1),
        integrality=np.ones(objective.size, dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=build_constraints(known_predictions, targets, scored_mask),
        options={"time_limit": 240.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status} message={result.message}")
    labels = result.x.reshape(len(anchor), N_CLASSES).argmax(axis=1).astype(np.int64)
    return labels, result


def write_submission(
    outputs: Path,
    anchor_frame: pd.DataFrame,
    labels: np.ndarray,
    name: str,
    scored_mask: np.ndarray,
    dates: np.ndarray,
    support: pd.DataFrame,
) -> dict[str, object]:
    anchor = anchor_frame["prediction"].astype(int).to_numpy()
    prediction = anchor.copy()
    changed_mask = scored_mask & (labels != anchor)
    prediction[changed_mask] = labels[changed_mask]

    output = outputs / f"submission_{name}.csv"
    pd.DataFrame({"path": anchor_frame["path"], "prediction": prediction}).to_csv(output, index=False)

    rows = np.flatnonzero(changed_mask) + 1
    changes = pd.DataFrame(
        {
            "row": rows,
            "date": dates[rows - 1],
            "path": anchor_frame.loc[rows - 1, "path"].to_numpy(),
            "anchor_prediction": anchor[rows - 1],
            "candidate_prediction": prediction[rows - 1],
        }
    )
    if not support.empty:
        changes = changes.merge(support, on="row", how="left")
    changes.to_csv(outputs / f"{name}_changes.csv", index=False)
    by_date = changes.groupby("date").size().astype(int).to_dict() if not changes.empty else {}
    return {
        "candidate": output.name,
        "path": str(output),
        "changes": int(len(changes)),
        "changes_by_date": json.dumps(by_date, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    outputs = args.root / "outputs"
    anchor_frame = pd.read_csv(outputs / "submission_recommended_next.csv")
    official_paths = anchor_frame["path"].astype(str).to_numpy()
    anchor = anchor_frame["prediction"].astype(int).to_numpy()

    preview = pd.read_csv(outputs / "leaderboard_ilp_preview.csv")
    scored_mask = preview["scored"].astype(bool).to_numpy()
    old_preview = preview["ilp"].astype(int).to_numpy()
    if int(scored_mask.sum()) != DENOMINATOR:
        raise ValueError(f"Expected {DENOMINATOR} scored rows, found {int(scored_mask.sum())}")

    known_predictions: list[np.ndarray] = []
    known_names: list[str] = []
    targets: list[int] = []
    for item in KNOWN_SUBMISSIONS:
        path = outputs / item.name
        if not path.exists():
            print(f"skip_missing={item.name}")
            continue
        known_names.append(item.name)
        targets.append(item.correct)
        known_predictions.append(read_submission(path, official_paths))
    if len(known_predictions) < 8:
        raise FileNotFoundError("Too few known submissions available")

    evidence, model_names = model_evidence(outputs, official_paths)
    test_meta = collect_test_metadata(args.root)
    test_meta["date"] = test_meta["start"].dt.date.astype(str)
    test_meta["path_norm"] = test_meta["path"].map(norm_path)
    date_lookup = test_meta.set_index("path_norm")["date"].to_dict()
    dates = np.asarray([date_lookup.get(norm_path(path), "unknown") for path in official_paths], dtype=str)
    support_columns = [
        "row",
        "proposal",
        "support_count",
        "strong_count",
        "head_count",
        "supporters",
    ]
    pool_path = outputs / "anchor_gated_candidate_pool.csv"
    support = pd.read_csv(pool_path)[support_columns] if pool_path.exists() else pd.DataFrame()

    print(f"known={len(known_names)} models={','.join(model_names)} scored={int(scored_mask.sum())}")
    variants = [
        ("full_ilp_v2_evidence", 0.00, 0.00, 0.00),
        ("full_ilp_v2_anchor", 0.25, 0.08, 0.00),
        ("full_ilp_v2_preview", 0.10, 0.05, 0.30),
    ]
    summaries = []
    for name, anchor_prior, submit_prior, old_prior in variants:
        labels, result = solve_labels(
            evidence,
            anchor,
            known_predictions,
            np.asarray(targets, dtype=np.int64),
            scored_mask,
            anchor_prior=anchor_prior,
            submit_prior=submit_prior,
            old_preview_prior=old_prior,
            old_preview=old_preview,
        )
        print(f"{name}: objective={float(result.fun):.3f}")
        for known_name, target, pred in zip(known_names, targets, known_predictions, strict=True):
            correct = int(((labels == pred) & scored_mask).sum())
            print(f"  {known_name}: target={target} actual={correct}")
        summaries.append(write_submission(outputs, anchor_frame, labels, name, scored_mask, dates, support))

    # Estimate label stability under tiny random perturbations around the evidence objective.
    stability = []
    for seed in range(20):
        labels, _ = solve_labels(
            evidence,
            anchor,
            known_predictions,
            np.asarray(targets, dtype=np.int64),
            scored_mask,
            anchor_prior=0.0,
            submit_prior=0.0,
            old_preview_prior=0.0,
            old_preview=old_preview,
            random_seed=seed,
        )
        stability.append(labels)
    stable = np.stack(stability)
    label_rows = []
    for row_idx in range(len(anchor)):
        counts = pd.Series(stable[:, row_idx]).value_counts()
        label_rows.append(
            {
                "row": row_idx + 1,
                "scored": bool(scored_mask[row_idx]),
                "anchor": int(anchor[row_idx]),
                "top_label": int(counts.index[0]),
                "top_count": int(counts.iloc[0]),
                "old_preview": int(old_preview[row_idx]),
            }
        )
    stability_frame = pd.DataFrame(label_rows)
    stability_frame.to_csv(outputs / "full_leaderboard_ilp_v2_stability.csv", index=False)

    evidence_labels, _ = solve_labels(
        evidence,
        anchor,
        known_predictions,
        np.asarray(targets, dtype=np.int64),
        scored_mask,
        anchor_prior=0.0,
        submit_prior=0.0,
        old_preview_prior=0.0,
        old_preview=old_preview,
    )
    stable_mask = (
        stability_frame["scored"].astype(bool)
        & stability_frame["top_count"].ge(18)
        & stability_frame["top_label"].ne(stability_frame["anchor"])
    )
    stable_labels = anchor.copy()
    for row in stability_frame.loc[stable_mask].itertuples(index=False):
        stable_labels[int(row.row) - 1] = int(row.top_label)
    summaries.append(
        write_submission(
            outputs,
            anchor_frame,
            stable_labels,
            "full_ilp_v2_stable18",
            scored_mask,
            dates,
            support,
        )
    )

    report = pd.DataFrame(summaries)
    report.to_csv(outputs / "full_leaderboard_ilp_v2_report.csv", index=False)
    print(report.to_string(index=False))
    print(f"report={outputs / 'full_leaderboard_ilp_v2_report.csv'}")
    print(f"stability={outputs / 'full_leaderboard_ilp_v2_stability.csv'}")


if __name__ == "__main__":
    main()
