from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from audit_model_candidates import norm_path
from sequence_postprocess import collect_test_metadata


ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 40


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
    KnownSubmission("submission_full_ilp_v2_evidence.csv", 286),
    KnownSubmission("submission_score_posterior_top40_v1.csv", 306),
    KnownSubmission("submission_bestanchor_consensus_4source_bestanchor_pubm0_v1.csv", 302),
]


def read_submission(path: Path, official_paths: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    frame["path_norm"] = frame["path"].map(norm_path)
    lookup = dict(zip(frame["path_norm"], frame["prediction"].astype(int), strict=True))
    return np.asarray([lookup[norm_path(raw)] for raw in official_paths], dtype=np.int64)


def fit_posterior(predictions: np.ndarray, targets: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray]:
    n_sub, n_rows = predictions.shape
    indicator = np.zeros((n_rows, N_CLASSES, n_sub), dtype=np.float64)
    for sub_idx in range(n_sub):
        indicator[np.arange(n_rows), predictions[sub_idx], sub_idx] = 1.0

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
        options={"maxiter": 5000, "maxfun": 100000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        print(f"warning: posterior fit status={result.status} message={result.message}")
    scores = np.tensordot(indicator, result.x, axes=([2], [0]))
    probs = np.exp(scores - logsumexp(scores, axis=1)[:, None])
    expected = np.einsum("rks,rk->s", indicator, probs)
    return probs, expected


def support_lookup(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)[
        [
            "row",
            "proposal",
            "support_weight",
            "support_count",
            "strong_count",
            "head_count",
            "supporters",
        ]
    ]


def write_candidate(
    outputs: Path,
    anchor_frame: pd.DataFrame,
    posterior: pd.DataFrame,
    selected: pd.DataFrame,
    name: str,
    expected_anchor: float,
) -> dict[str, object]:
    prediction = anchor_frame["prediction"].astype(int).to_numpy().copy()
    for row in selected.itertuples(index=False):
        prediction[int(row.row) - 1] = int(row.posterior_label)
    output = outputs / f"submission_{name}.csv"
    pd.DataFrame({"path": anchor_frame["path"], "prediction": prediction}).to_csv(output, index=False)
    changes = selected.sort_values("row")
    changes.to_csv(outputs / f"{name}_changes.csv", index=False)
    expected_score = expected_anchor + float(selected["posterior_delta"].sum())
    by_date = changes.groupby("date").size().astype(int).to_dict() if not changes.empty else {}
    return {
        "candidate": output.name,
        "path": str(output),
        "changes": int(len(changes)),
        "expected_correct": expected_score,
        "expected_lb": expected_score / 402.0,
        "expected_gain": expected_score - expected_anchor,
        "changes_by_date": json.dumps(by_date, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--anchor", type=Path, default=ROOT / "outputs" / "submission_recommended_next.csv")
    parser.add_argument("--support-pool", type=Path, default=ROOT / "outputs" / "anchor_gated_candidate_pool.csv")
    parser.add_argument("--tag", type=str, default="v1")
    parser.add_argument("--l2", type=float, default=0.05)
    args = parser.parse_args()

    outputs = args.root / "outputs"
    anchor_frame = pd.read_csv(args.anchor)
    official_paths = anchor_frame["path"].astype(str).to_numpy()
    anchor = anchor_frame["prediction"].astype(int).to_numpy()
    preview = pd.read_csv(outputs / "leaderboard_ilp_preview.csv")
    scored_mask = preview["scored"].astype(bool).to_numpy()
    scored_rows = np.flatnonzero(scored_mask)

    names: list[str] = []
    targets: list[int] = []
    predictions: list[np.ndarray] = []
    for item in KNOWN_SUBMISSIONS:
        path = outputs / item.name
        if not path.exists():
            print(f"skip_missing={item.name}")
            continue
        names.append(item.name)
        targets.append(item.correct)
        predictions.append(read_submission(path, official_paths)[scored_rows])
    pred_matrix = np.stack(predictions)
    target_arr = np.asarray(targets, dtype=np.float64)
    probs, expected = fit_posterior(pred_matrix, target_arr, args.l2)
    print(f"fit rows={len(scored_rows)} submissions={len(names)} l2={args.l2}")
    for name, target, exp_value in zip(names, targets, expected, strict=True):
        print(f"  {name}: target={target} expected={exp_value:.3f}")

    full_probs = np.zeros((len(anchor), N_CLASSES), dtype=np.float64)
    full_probs[:, :] = 1.0 / N_CLASSES
    full_probs[scored_rows] = probs
    top_label = full_probs.argmax(axis=1).astype(int)
    anchor_prob = full_probs[np.arange(len(anchor)), anchor]
    top_prob = full_probs[np.arange(len(anchor)), top_label]
    delta = top_prob - anchor_prob
    test_meta = collect_test_metadata(args.root)
    test_meta["date"] = test_meta["start"].dt.date.astype(str)
    test_meta["path_norm"] = test_meta["path"].map(norm_path)
    date_lookup = test_meta.set_index("path_norm")["date"].to_dict()

    posterior = pd.DataFrame(
        {
            "row": np.arange(len(anchor)) + 1,
            "scored": scored_mask,
            "date": [date_lookup.get(norm_path(path), "unknown") for path in official_paths],
            "path": official_paths,
            "anchor_prediction": anchor,
            "posterior_label": top_label,
            "anchor_prob": anchor_prob,
            "posterior_prob": top_prob,
            "posterior_delta": delta,
        }
    )
    support = support_lookup(args.support_pool)
    if not support.empty:
        posterior = posterior.merge(support, left_on=["row", "posterior_label"], right_on=["row", "proposal"], how="left")
        for column in ["support_weight", "support_count", "strong_count", "head_count"]:
            posterior[column] = posterior[column].fillna(0)
        posterior["supporters"] = posterior["supporters"].fillna("")
    else:
        posterior["support_weight"] = 0.0
        posterior["support_count"] = 0
        posterior["strong_count"] = 0
        posterior["head_count"] = 0
        posterior["supporters"] = ""

    posterior = posterior.sort_values("posterior_delta", ascending=False)
    rows_name = "score_posterior_rows.csv" if args.tag == "v1" else f"score_posterior_rows_{args.tag}.csv"
    posterior.to_csv(outputs / rows_name, index=False)
    expected_anchor = float(anchor_prob[scored_mask].sum())

    base_mask = posterior["scored"] & posterior["posterior_label"].ne(posterior["anchor_prediction"]) & posterior["posterior_delta"].gt(0)
    suffix = args.tag if args.tag != "v1" else "v1"
    rules = {
        f"score_posterior_top12_{suffix}": posterior.loc[base_mask].head(12),
        f"score_posterior_top24_{suffix}": posterior.loc[base_mask].head(24),
        f"score_posterior_top40_{suffix}": posterior.loc[base_mask].head(40),
        f"score_posterior_model2_{suffix}": posterior.loc[base_mask & posterior["support_count"].ge(2)].head(50),
        f"score_posterior_strong_{suffix}": posterior.loc[base_mask & posterior["strong_count"].ge(2)].head(35),
    }
    summaries = [
        write_candidate(outputs, anchor_frame, posterior, selected, name, expected_anchor)
        for name, selected in rules.items()
    ]
    report = pd.DataFrame(summaries).sort_values("expected_lb", ascending=False)
    report_name = "score_posterior_candidates_report.csv" if args.tag == "v1" else f"score_posterior_candidates_report_{args.tag}.csv"
    report.to_csv(outputs / report_name, index=False)
    print(report.to_string(index=False))
    print(f"rows={outputs / rows_name}")
    print(f"report={outputs / report_name}")


if __name__ == "__main__":
    main()
