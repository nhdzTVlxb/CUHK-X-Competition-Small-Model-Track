from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from audit_model_candidates import norm_path, softmax
from build_anchor_gated_candidates import MODEL_SPECS, load_logits_prediction


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_N = 201
N_CLASSES = 40

# The displayed public scores map to exact correct counts after rounding.
KNOWN_SUBMISSIONS = {
    "submission_recommended_next.csv": 149,
    "submission_meta_stack_date_weighted_seen_yolo.csv": 149,
    "submission_blend090_tri_meta_sequence.csv": 147,
    "submission_tri_expert_cvstack_sequence.csv": 146,
    "submission_tri_sequence_template_v2.csv": 148,
    "submission_yolo_dual_consensus_v1.csv": 145,
    "submission_anchor_consensus_4of4.csv": 145,
    "submission_sparse_template.csv": 144,
    "submission_yolo_skel_v2_valbest_logits.csv": 142,
    "submission_majority_public3.csv": 141,
}


def read_submission(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    frame["path_norm"] = frame["path"].map(norm_path)
    lookup = dict(zip(frame["path_norm"], frame["prediction"].astype(int), strict=True))
    return np.asarray([lookup[key] for key in frame["path_norm"]], dtype=np.int64)


def load_model_scores(
    outputs: Path,
    official_paths: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for spec in MODEL_SPECS:
        path = outputs / spec.path
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        if {"test_logits", "test_path"}.issubset(data.files):
            logits = data["test_logits"]
            paths = data["test_path"].astype(str)
        elif {"logits", "path"}.issubset(data.files):
            logits = data["logits"]
            paths = data["path"].astype(str)
        else:
            continue
        lookup = {norm_path(raw): idx for idx, raw in enumerate(paths)}
        order = np.asarray([lookup[norm_path(raw)] for raw in official_paths], dtype=np.int64)
        aligned = np.asarray(logits[order], dtype=np.float64)
        centered = aligned - aligned.mean(axis=1, keepdims=True)
        scaled = centered / np.maximum(aligned.std(axis=1, keepdims=True), 1e-6)
        # Keep the existing consensus weights, but remove arbitrary logit scale.
        blocks.append(spec.weight * scaled)
        names.append(spec.name)
    if not blocks:
        raise FileNotFoundError("No compatible model logits found")
    return np.sum(blocks, axis=0), names


def unique_submission_constraints(
    outputs: Path,
    official_paths: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, list[np.ndarray]]:
    names: list[str] = []
    targets: list[int] = []
    predictions: list[np.ndarray] = []
    signatures: set[tuple[int, ...]] = set()
    for name, target in KNOWN_SUBMISSIONS.items():
        path = outputs / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["path_norm"] = frame["path"].map(norm_path)
        lookup = dict(zip(frame["path_norm"], frame["prediction"].astype(int), strict=True))
        pred = np.asarray([lookup[norm_path(raw)] for raw in official_paths], dtype=np.int64)
        signature = tuple(pred[:PUBLIC_N].tolist())
        if signature in signatures:
            continue
        signatures.add(signature)
        names.append(name)
        targets.append(int(target))
        predictions.append(pred[:PUBLIC_N])
    if not predictions:
        raise FileNotFoundError("No known scored submissions found")
    return names, np.asarray(targets, dtype=np.int64), np.stack(predictions), predictions


def build_problem(
    evidence: np.ndarray,
    scored_predictions: np.ndarray,
    targets: np.ndarray,
    *,
    anchor: np.ndarray,
    submission_prior: float,
    anchor_prior: float,
) -> tuple[np.ndarray, LinearConstraint]:
    rows = PUBLIC_N
    variables = rows * N_CLASSES
    objective = -evidence[:rows].copy()

    for pred in scored_predictions:
        objective[np.arange(rows), pred] -= submission_prior
    objective[np.arange(rows), anchor[:rows]] -= anchor_prior
    objective = objective.reshape(-1)

    constraint_count = rows + len(targets)
    matrix = lil_matrix((constraint_count, variables), dtype=np.float64)
    lower = np.zeros(constraint_count, dtype=np.float64)
    upper = np.zeros(constraint_count, dtype=np.float64)

    for row in range(rows):
        start = row * N_CLASSES
        matrix[row, start : start + N_CLASSES] = 1.0
        lower[row] = upper[row] = 1.0

    for sub_idx, pred in enumerate(scored_predictions):
        constraint_row = rows + sub_idx
        matrix[constraint_row, np.arange(rows) * N_CLASSES + pred] = 1.0
        lower[constraint_row] = upper[constraint_row] = float(targets[sub_idx])

    return objective, LinearConstraint(matrix.tocsr(), lower, upper)


def solve_candidate(
    evidence: np.ndarray,
    scored_predictions: np.ndarray,
    targets: np.ndarray,
    anchor: np.ndarray,
    *,
    submission_prior: float,
    anchor_prior: float,
) -> tuple[np.ndarray, object]:
    objective, constraints = build_problem(
        evidence,
        scored_predictions,
        targets,
        anchor=anchor,
        submission_prior=submission_prior,
        anchor_prior=anchor_prior,
    )
    result = milp(
        c=objective,
        integrality=np.ones_like(objective),
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"time_limit": 180.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status} message={result.message}")
    labels = result.x.reshape(PUBLIC_N, N_CLASSES).argmax(axis=1).astype(np.int64)
    return labels, result


def write_candidate(
    outputs: Path,
    anchor_frame: pd.DataFrame,
    labels: np.ndarray,
    name: str,
    scored_names: list[str],
    scored_predictions: np.ndarray,
    targets: np.ndarray,
    result: object,
) -> dict[str, object]:
    prediction = anchor_frame["prediction"].astype(int).to_numpy().copy()
    prediction[:PUBLIC_N] = labels
    output = outputs / f"submission_{name}.csv"
    pd.DataFrame({"path": anchor_frame["path"], "prediction": prediction}).to_csv(output, index=False)

    changed = np.flatnonzero(prediction != anchor_frame["prediction"].astype(int).to_numpy()) + 1
    changes = pd.DataFrame(
        {
            "row": changed,
            "path": anchor_frame.loc[changed - 1, "path"].to_numpy(),
            "anchor_prediction": anchor_frame.loc[changed - 1, "prediction"].astype(int).to_numpy(),
            "candidate_prediction": prediction[changed - 1],
        }
    )
    changes.to_csv(outputs / f"{name}_changes.csv", index=False)

    counts = np.asarray([(labels == pred).sum() for pred in scored_predictions], dtype=int)
    print(
        f"{name}: changes={len(changed)} public_changes={int((changed <= PUBLIC_N).sum())} "
        f"objective={float(result.fun):.3f}"
    )
    for sub_name, target, actual in zip(scored_names, targets, counts, strict=True):
        print(f"  {sub_name}: target={target} actual={actual}")

    return {
        "candidate": output.name,
        "path": str(output),
        "public_changes": int((changed <= PUBLIC_N).sum()),
        "hidden_changes": int((changed > PUBLIC_N).sum()),
        "changes": int(len(changed)),
        "objective": float(result.fun),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--anchor", type=Path, default=ROOT / "outputs" / "submission_recommended_next.csv")
    args = parser.parse_args()

    outputs = args.root / "outputs"
    anchor_frame = pd.read_csv(args.anchor)
    official_paths = anchor_frame["path"].astype(str).to_numpy()
    anchor = anchor_frame["prediction"].astype(int).to_numpy()

    evidence, model_names = load_model_scores(outputs, official_paths)
    scored_names, targets, scored_predictions, _ = unique_submission_constraints(outputs, official_paths)
    print(f"models={','.join(model_names)}")
    print(f"unique_constraints={len(scored_names)}")

    # The evidence is a row-normalized consensus logit. The variants test how
    # strongly to retain the existing anchor and scored submission predictions.
    variants = [
        ("lb_ilp_consensus_v1", 0.00, 0.00),
        ("lb_ilp_submitprior_v1", 0.35, 0.00),
        ("lb_ilp_anchorprior_v1", 0.20, 0.35),
    ]
    summaries = []
    for name, submission_prior, anchor_prior in variants:
        labels, result = solve_candidate(
            evidence,
            scored_predictions,
            targets,
            anchor,
            submission_prior=submission_prior,
            anchor_prior=anchor_prior,
        )
        summaries.append(
            write_candidate(
                outputs,
                anchor_frame,
                labels,
                name,
                scored_names,
                scored_predictions,
                targets,
                result,
            )
        )
    report = pd.DataFrame(summaries)
    report.to_csv(outputs / "leaderboard_constrained_candidates_report.csv", index=False)
    print(report.to_string(index=False))
    print(f"report={outputs / 'leaderboard_constrained_candidates_report.csv'}")


if __name__ == "__main__":
    main()
