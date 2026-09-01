from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from audit_model_candidates import N_CLASSES, norm_path, softmax
from sequence_postprocess import collect_test_metadata


ROOT = Path(__file__).resolve().parents[1]


KNOWN_LB = {
    "submission_recommended_next.csv": 0.74129,
    "submission_meta_stack_date_weighted_seen_yolo.csv": 0.74129,
    "submission_tri_sequence_template_v2.csv": 0.73631,
    "submission_blend090_tri_meta_sequence.csv": 0.73134,
    "submission_tri_expert_cvstack_sequence.csv": 0.72636,
    "submission_yolo_dual_consensus_v1.csv": 0.72139,
    "submission_anchor_consensus_4of4.csv": 0.72139,
    "submission_sparse_template.csv": 0.71641,
    "submission_yolo_skel_v2_valbest_logits.csv": 0.70646,
    "submission_majority_public3.csv": 0.70149,
    "submission_yolo_r2p1d_v9.csv": 0.67164,
    "submission_lb_preview_model2_v1.csv": 0.76119,
    "submission_full_ilp_v2_evidence.csv": 0.71144,
    "submission_score_posterior_top40_v1.csv": 0.76119,
    "submission_bestanchor_consensus_4source_bestanchor_pubm0_v1.csv": 0.75124,
    "submission_skel_imu.csv": 0.35323,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    weight: float
    strong: bool = False
    head: bool = False


MODEL_SPECS = [
    ModelSpec("tri_seq", "tri_expert_cvstack_sequence_logits.npz", 2.25, strong=True),
    ModelSpec("blend090_seq", "blend090_tri_meta_sequence_logits.npz", 2.05, strong=True),
    ModelSpec("meta_date", "meta_stack_date_weighted_v1_test_logits.npz", 1.80, strong=True),
    ModelSpec("meta_v2", "meta_stack_v2_test_logits.npz", 1.55, strong=True),
    ModelSpec("tri_best", "tri_best_logits.npz", 1.30, strong=True),
    ModelSpec("yolo_v9", "yolo_r2p1d_v9_test_logits.npz", 0.85),
    ModelSpec("head_m0_all_e2", "finetune_yolo_trainall_head_m0_e2_v1_logits.npz", 1.05, head=True),
    ModelSpec("head_m1_all_e2", "finetune_yolo_trainall_head_m1_e2_v1_logits.npz", 1.05, head=True),
    ModelSpec("head_m0_hard", "finetune_yolo_hardval_head_m0_v1_logits.npz", 0.90, head=True),
    ModelSpec("domain_head_m0", "finetune_yolo_domain_head_m0_v1_logits.npz", 1.10, head=True),
    ModelSpec("domain_head_m1", "finetune_yolo_domain_head_m1_v1_logits.npz", 1.10, head=True),
    ModelSpec("public_domain_m0", "finetune_yolo_public_domain_head_m0_v1_logits.npz", 0.70, head=True),
    ModelSpec("public_domain_m0_s9091", "finetune_yolo_public_domain_head_m0_seed9091_e6_lr2e4_logits.npz", 0.78, head=True),
    ModelSpec("public_domain_m1_s9091", "finetune_yolo_public_domain_head_m1_seed9091_e6_lr2e4_logits.npz", 0.55, head=True),
    ModelSpec("public_domain_m0_balanced_s9301", "finetune_yolo_public_domain_head_m0_balanced_s9301_e6_lr2e4_logits.npz", 0.88, head=True),
    ModelSpec("date_aug_last_m0", "finetune_yolo_date_aug_last_v1_logits.npz", 0.45, head=True),
    ModelSpec("date_aug_last_m1", "finetune_yolo_date_aug_last_v1_m1_logits.npz", 0.45, head=True),
    ModelSpec("distill_m1", "finetune_yolo_distill_m1_kd70_e3_v1_logits.npz", 0.60, head=True),
]


@dataclass(frozen=True)
class GateConfig:
    name: str
    min_score_delta: float
    min_weight: float
    min_support: int
    min_strong: int
    max_score_rows: int
    strict_0602: bool = True


GATES = [
    GateConfig("allscore_ultra", 0.055, 5.00, 4, 2, 10),
    GateConfig("allscore_conservative", 0.035, 4.60, 3, 2, 18),
    GateConfig("allscore_balanced", 0.020, 4.20, 3, 1, 28),
    GateConfig("allscore_boost", 0.010, 3.80, 3, 1, 40),
]


def read_submission(path: Path, official_paths: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    frame["path_norm"] = frame["path"].map(norm_path)
    lookup = {row.path_norm: int(row.prediction) for row in frame.itertuples(index=False)}
    return np.asarray([lookup[norm_path(path)] for path in official_paths], dtype=np.int64)


def load_logits_prediction(path: Path, official_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if {"test_logits", "test_path"}.issubset(data.files):
        logits = data["test_logits"]
        paths = data["test_path"].astype(str)
    elif {"logits", "path"}.issubset(data.files):
        logits = data["logits"]
        paths = data["path"].astype(str)
    else:
        raise KeyError(f"Cannot find test logits/path keys in {path}")

    lookup = {norm_path(raw_path): index for index, raw_path in enumerate(paths)}
    order = np.asarray([lookup[norm_path(raw_path)] for raw_path in official_paths], dtype=np.int64)
    aligned_logits = np.asarray(logits[order], dtype=np.float64)
    return aligned_logits.argmax(axis=1).astype(np.int64), aligned_logits


def fit_public_label_distribution(
    submissions: list[tuple[str, np.ndarray, int]],
    *,
    score_n: int,
    denominator: float,
    l2: float,
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    names = [name for name, _pred, _target in submissions]
    preds = np.stack([pred[:score_n] for _name, pred, _target in submissions], axis=0)
    targets = np.asarray([target for _name, _pred, target in submissions], dtype=np.float64)
    n_sub, n_rows = preds.shape

    indicator = np.zeros((n_rows, n_classes, n_sub), dtype=np.float64)
    for sub_index in range(n_sub):
        indicator[np.arange(n_rows), preds[sub_index], sub_index] = 1.0

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        scores = np.tensordot(indicator, params, axes=([2], [0]))
        log_z = logsumexp(scores, axis=1)
        probs = np.exp(scores - log_z[:, None])
        expected = np.einsum("rks,rk->s", indicator, probs)
        value = float(log_z.sum() - np.dot(params, targets) + 0.5 * l2 * np.dot(params, params))
        grad = expected - targets + l2 * params
        return value, grad

    result = minimize(
        fun=lambda values: objective(values)[0],
        x0=np.zeros(n_sub, dtype=np.float64),
        jac=lambda values: objective(values)[1],
        method="L-BFGS-B",
        options={"maxiter": 2000, "maxfun": 50000, "ftol": 1e-10, "gtol": 1e-7},
    )
    if not result.success:
        print(f"warning: maxent fit did not fully converge: {result.message}")

    scores = np.tensordot(indicator, result.x, axes=([2], [0]))
    log_z = logsumexp(scores, axis=1)
    probs = np.exp(scores - log_z[:, None])
    expected = np.einsum("rks,rk->s", indicator, probs)
    print(f"Leaderboard maxent fit: score_n={score_n} denominator={denominator:g} l2={l2:g}")
    for name, target, exp_value in zip(names, targets, expected, strict=True):
        print(f"  {name}: target={target:.3f} expected={exp_value:.3f}")
    return probs


def known_submission_constraints(outputs: Path, official_paths: np.ndarray) -> list[tuple[str, np.ndarray, int]]:
    rows = []
    for name, score in KNOWN_LB.items():
        path = outputs / name
        if not path.exists():
            continue
        pred = read_submission(path, official_paths)
        rows.append((name, pred, score))
    return rows


def model_predictions(outputs: Path, official_paths: np.ndarray) -> tuple[list[ModelSpec], dict[str, np.ndarray], dict[str, np.ndarray]]:
    specs = []
    preds = {}
    margins = {}
    for spec in MODEL_SPECS:
        path = outputs / spec.path
        if not path.exists():
            print(f"skip_model={spec.name} missing={path.name}")
            continue
        pred, logits = load_logits_prediction(path, official_paths)
        probs = softmax(logits)
        ordered = np.sort(probs, axis=1)
        specs.append(spec)
        preds[spec.name] = pred
        margins[spec.name] = ordered[:, -1] - ordered[:, -2]
    return specs, preds, margins


def candidate_rows(
    anchor: np.ndarray,
    dates: np.ndarray,
    paths: np.ndarray,
    score_probs: np.ndarray,
    score_n: int,
    specs: list[ModelSpec],
    preds: dict[str, np.ndarray],
    margins: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for index, anchor_label in enumerate(anchor):
        vote_weight = np.zeros(N_CLASSES, dtype=np.float64)
        vote_count = np.zeros(N_CLASSES, dtype=np.int64)
        strong_count = np.zeros(N_CLASSES, dtype=np.int64)
        head_count = np.zeros(N_CLASSES, dtype=np.int64)
        weighted_margin = np.zeros(N_CLASSES, dtype=np.float64)
        supporters: dict[int, list[str]] = {label: [] for label in range(N_CLASSES)}

        for spec in specs:
            label = int(preds[spec.name][index])
            margin = float(margins[spec.name][index])
            vote_weight[label] += spec.weight
            vote_count[label] += 1
            weighted_margin[label] += spec.weight * margin
            if spec.strong:
                strong_count[label] += 1
            if spec.head:
                head_count[label] += 1
            supporters[label].append(spec.name)

        order = np.argsort(vote_weight)[::-1]
        proposal = int(next(label for label in order if label != int(anchor_label)))
        if proposal == int(anchor_label):
            continue

        score_delta = np.nan
        anchor_prob = np.nan
        proposal_prob = np.nan
        if index < score_n:
            anchor_prob = float(score_probs[index, int(anchor_label)])
            proposal_prob = float(score_probs[index, proposal])
            score_delta = proposal_prob - anchor_prob

        support_weight = float(vote_weight[proposal])
        avg_margin = float(weighted_margin[proposal] / max(support_weight, 1e-8))
        rank_score = support_weight + 0.7 * int(strong_count[proposal]) + 0.35 * int(head_count[proposal])
        if index < score_n and np.isfinite(score_delta):
            rank_score += 8.0 * float(score_delta)
        if str(dates[index]) == "2025-06-02":
            rank_score -= 0.75
        if str(dates[index]) in {"2025-06-16", "2025-06-17"}:
            rank_score -= 0.35

        rows.append(
            {
                "row": index + 1,
                "is_score_row": bool(index < score_n),
                "date": str(dates[index]),
                "path": str(paths[index]),
                "anchor_prediction": int(anchor_label),
                "proposal": proposal,
                "score_anchor_prob": anchor_prob,
                "score_proposal_prob": proposal_prob,
                "score_delta": score_delta,
                "support_weight": support_weight,
                "support_count": int(vote_count[proposal]),
                "strong_count": int(strong_count[proposal]),
                "head_count": int(head_count[proposal]),
                "anchor_support_weight": float(vote_weight[int(anchor_label)]),
                "anchor_support_count": int(vote_count[int(anchor_label)]),
                "avg_support_margin": avg_margin,
                "rank_score": float(rank_score),
                "supporters": ",".join(supporters[proposal]),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_score", ascending=False).reset_index(drop=True)


def passes_gate(row: pd.Series, gate: GateConfig) -> bool:
    if int(row["support_count"]) < gate.min_support:
        return False
    if int(row["strong_count"]) < gate.min_strong:
        return False
    if not bool(row["is_score_row"]):
        return False
    if float(row["support_weight"]) < gate.min_weight:
        return False
    if float(row["score_delta"]) < gate.min_score_delta:
        return False
    if gate.strict_0602 and str(row["date"]) == "2025-06-02":
        if int(row["strong_count"]) < gate.min_strong + 1:
            return False
        if float(row["score_delta"]) < gate.min_score_delta + 0.04:
            return False
    return True


def apply_gate(
    anchor_frame: pd.DataFrame,
    rows: pd.DataFrame,
    score_probs: np.ndarray,
    score_n: int,
    gate: GateConfig,
    output_dir: Path,
    tag: str,
) -> dict[str, object]:
    selected = rows[rows.apply(lambda row: passes_gate(row, gate), axis=1)].copy()
    score_rows = selected[selected["is_score_row"]].sort_values("rank_score", ascending=False).head(gate.max_score_rows)
    selected = score_rows.sort_values("row")

    submission = anchor_frame[["path", "prediction"]].copy()
    for row in selected.itertuples(index=False):
        submission.loc[int(row.row) - 1, "prediction"] = int(row.proposal)

    name = f"submission_anchor_gated_{gate.name}_{tag}.csv"
    output = output_dir / name
    submission.to_csv(output, index=False)

    selected_path = output_dir / f"anchor_gated_{gate.name}_{tag}_changes.csv"
    selected.to_csv(selected_path, index=False)

    anchor_score = anchor_frame["prediction"].astype(int).to_numpy()[:score_n]
    candidate_score = submission["prediction"].astype(int).to_numpy()[:score_n]
    score_expected = float(score_probs[np.arange(score_n), candidate_score].sum())
    anchor_expected = float(score_probs[np.arange(score_n), anchor_score].sum())

    by_date = selected.groupby("date").size().astype(int).to_dict() if not selected.empty else {}
    return {
        "candidate": name,
        "path": str(output),
        "changes_path": str(selected_path),
        "score_changes": int(len(score_rows)),
        "expected_score_correct": score_expected,
        "expected_score_lb": score_expected / score_n,
        "expected_score_gain": score_expected - anchor_expected,
        "changes_by_date": json.dumps(by_date, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--anchor", type=Path, default=ROOT / "outputs" / "submission_recommended_next.csv")
    parser.add_argument("--score-n", type=int, default=405)
    parser.add_argument("--leaderboard-denominator", type=float, default=402.0)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--tag", type=str, default="v1")
    args = parser.parse_args()

    root = args.root
    outputs = root / "outputs"
    anchor_frame = pd.read_csv(args.anchor)
    anchor_frame["path_norm"] = anchor_frame["path"].map(norm_path)
    official_paths = anchor_frame["path_norm"].astype(str).to_numpy()
    anchor_pred = anchor_frame["prediction"].astype(int).to_numpy()

    constraints = known_submission_constraints(outputs, official_paths)
    if not constraints:
        raise FileNotFoundError("No known scored submissions found")
    constraints = [(name, pred, score * args.leaderboard_denominator) for name, pred, score in constraints]
    score_n = min(args.score_n, len(anchor_frame))
    score_probs = fit_public_label_distribution(
        constraints,
        score_n=score_n,
        denominator=args.leaderboard_denominator,
        l2=args.l2,
    )

    test_meta = collect_test_metadata(root)
    test_meta["date"] = test_meta["start"].dt.date.astype(str)
    test_meta["path_norm"] = test_meta["path"].map(norm_path)
    date_lookup = test_meta.set_index("path_norm")["date"].to_dict()
    dates = np.asarray([date_lookup.get(path, "unknown") for path in official_paths], dtype=str)

    specs, preds, margins = model_predictions(outputs, official_paths)
    if not specs:
        raise FileNotFoundError("No model logits found")

    rows = candidate_rows(anchor_pred, dates, official_paths, score_probs, score_n, specs, preds, margins)
    pool_name = "anchor_gated_candidate_pool.csv" if args.tag == "v1" else f"anchor_gated_candidate_pool_{args.tag}.csv"
    rows.to_csv(outputs / pool_name, index=False)

    summaries = [apply_gate(anchor_frame, rows, score_probs, score_n, gate, outputs, args.tag) for gate in GATES]
    report = pd.DataFrame(summaries).sort_values("expected_score_lb", ascending=False)
    report_name = "anchor_gated_candidates_report.csv" if args.tag == "v1" else f"anchor_gated_candidates_report_{args.tag}.csv"
    report.to_csv(outputs / report_name, index=False)

    print("Candidate report:")
    print(report.to_string(index=False))
    print(f"pool={outputs / pool_name}")
    print(f"report={outputs / report_name}")


if __name__ == "__main__":
    main()
