from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sequence_postprocess import collect_test_metadata, collect_train_metadata  # noqa: E402


N_CLASSES = 40
DEFAULT_BEST = ROOT / "outputs" / "raw_submissions" / "55848407_submission_public.csv"
DEFAULT_TEST_LOGITS = ROOT / "outputs" / "yolo_r2p1d_v9_logits.npz"
DEFAULT_VAL_LOGITS = ROOT / "outputs" / "yolo_r2p1d_val_logits.npz"


def validation_rows(root: Path, values: np.lib.npyio.NpzFile) -> pd.DataFrame:
    train = collect_train_metadata(root).set_index("clip_id")
    rows: list[dict[str, object]] = []
    for position, clip_id in enumerate(values["clip_id"].astype(str)):
        row = train.loc[clip_id].to_dict()
        row.update(
            {
                "clip_id": clip_id,
                "_position": position,
                "label": int(values["y"][position]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def sequence_templates(
    train: pd.DataFrame,
    excluded_users: set[str],
) -> Counter[tuple[int, ...]]:
    sequences: Counter[tuple[int, ...]] = Counter()
    subset = train[~train["user"].isin(excluded_users)]
    for _key, group in subset.groupby(["user", "trial"], sort=False):
        sequence = tuple(group.sort_values("start")["label"].astype(int))
        sequences[sequence] += 1
    return sequences


def template_vote(
    observed: np.ndarray,
    templates: Counter[tuple[int, ...]],
    max_distance: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    candidates: list[tuple[int, tuple[int, ...], int]] = []
    for sequence, frequency in templates.items():
        if len(sequence) != len(observed):
            continue
        distance = int(np.sum(observed != np.asarray(sequence)))
        if distance <= max_distance:
            candidates.append((distance, sequence, frequency))

    if not candidates:
        return observed.copy(), np.zeros(len(observed), dtype=np.float64), -1, 0

    weights = np.asarray(
        [frequency * np.exp(-distance / temperature) for distance, _seq, frequency in candidates],
        dtype=np.float64,
    )
    weights /= np.maximum(weights.sum(), 1e-12)
    votes = np.zeros((len(observed), N_CLASSES), dtype=np.float64)
    for weight, (_distance, sequence, _frequency) in zip(weights, candidates, strict=True):
        for position, label in enumerate(sequence):
            votes[position, label] += weight

    proposal = votes.argmax(axis=1).astype(np.int64)
    confidence = votes.max(axis=1)
    min_distance = min(item[0] for item in candidates)
    return proposal, confidence, min_distance, len(candidates)


def apply_sparse_templates(
    frame: pd.DataFrame,
    model_logits: np.ndarray,
    templates: Counter[tuple[int, ...]],
    base_predictions: np.ndarray,
    *,
    max_distance: int,
    temperature: float,
    vote_threshold: float,
    margin_threshold: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    model_predictions = model_logits.argmax(axis=1).astype(np.int64)
    margins = np.sort(model_logits, axis=1)[:, -1] - np.sort(model_logits, axis=1)[:, -2]
    output = base_predictions.astype(np.int64).copy()
    changes: list[dict[str, object]] = []
    metadata = frame.set_index("_position")

    for group_key, group in frame.groupby("group", sort=False):
        group = group.sort_values("start")
        positions = group["_position"].astype(int).to_numpy()
        observed = model_predictions[positions]
        proposal, confidence, min_distance, candidate_count = template_vote(
            observed, templates, max_distance, temperature
        )
        if min_distance < 0:
            continue

        for local, position in enumerate(positions):
            old = int(output[position])
            new = int(proposal[local])
            accepted = (
                new != old
                and confidence[local] >= vote_threshold
                and margins[position] <= margin_threshold
            )
            if accepted:
                output[position] = new
            if new != old:
                changes.append(
                    {
                        "group": str(group_key),
                        "position": int(position),
                        "clip_id": str(metadata.loc[position, "clip_id"]),
                        "base": old,
                        "model_prediction": int(model_predictions[position]),
                        "proposal": new,
                        "vote_confidence": float(confidence[local]),
                        "model_margin": float(margins[position]),
                        "template_min_distance": int(min_distance),
                        "template_candidates": int(candidate_count),
                        "accepted": bool(accepted),
                    }
                )
    return output, pd.DataFrame(changes)


def evaluate_validation(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    values = np.load(args.val_logits, allow_pickle=True)
    frame = validation_rows(root, values)
    frame["group"] = frame["user"].astype(str) + "__" + frame["trial"].astype(str)
    model_logits = values["per_model_logits"][0] if "per_model_logits" in values else values["logits"]
    model_predictions = model_logits.argmax(axis=1).astype(np.int64)

    # Mimic a user-unseen setting: templates never use the held-out users.
    rows: list[dict[str, object]] = []
    for excluded_user in sorted(frame["user"].astype(str).unique()):
        subset = frame["user"].astype(str) == excluded_user
        heldout = frame.loc[subset].copy()
        heldout["_position"] = np.arange(len(heldout), dtype=int)
        templates = sequence_templates(
            collect_train_metadata(root), {excluded_user}
        )
        base = model_predictions[subset.to_numpy()]
        logits = model_logits[subset.to_numpy()]
        pred, changes = apply_sparse_templates(
            heldout,
            logits,
            templates,
            base,
            max_distance=args.max_distance,
            temperature=args.temperature,
            vote_threshold=args.vote_threshold,
            margin_threshold=args.margin_threshold,
        )
        truth = values["y"][subset.to_numpy()]
        rows.append(
            {
                "excluded_user": excluded_user,
                "baseline_accuracy": float(np.mean(base == truth)),
                "template_accuracy": float(np.mean(pred == truth)),
                "changed": int(np.sum(pred != base)),
                "accepted_proposals": int(changes["accepted"].sum()) if not changes.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def load_test_frame(root: Path, values: np.lib.npyio.NpzFile) -> pd.DataFrame:
    frame = collect_test_metadata(root)
    order = {str(clip_id): index for index, clip_id in enumerate(values["clip_id"])}
    frame["_position"] = frame["clip_id"].map(order).astype(int)
    frame["group"] = frame["segment"].astype(str)
    return frame.sort_values("start").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--best-submission", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--test-logits", type=Path, default=DEFAULT_TEST_LOGITS)
    parser.add_argument("--val-logits", type=Path, default=DEFAULT_VAL_LOGITS)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_sparse_template.csv")
    parser.add_argument("--changes", type=Path, default=ROOT / "outputs" / "sparse_template_changes.csv")
    parser.add_argument("--validation-report", type=Path, default=ROOT / "outputs" / "sparse_template_validation.csv")
    parser.add_argument("--max-distance", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--vote-threshold", type=float, default=0.9)
    parser.add_argument("--margin-threshold", type=float, default=2.0)
    args = parser.parse_args()

    root = args.root
    validation = evaluate_validation(root, args)
    validation.to_csv(args.validation_report, index=False)

    train = collect_train_metadata(root)
    test_values = np.load(args.test_logits, allow_pickle=True)
    test_frame = load_test_frame(root, test_values)
    test_templates = sequence_templates(train, set())
    best = pd.read_csv(args.best_submission)
    path_to_base = dict(zip(best["path"].astype(str), best["prediction"].astype(int), strict=True))
    base = np.asarray([path_to_base[str(path)] for path in test_values["path"]], dtype=np.int64)
    pred, changes = apply_sparse_templates(
        test_frame,
        test_values["per_model_logits"][0]
        if "per_model_logits" in test_values
        else test_values["logits"],
        test_templates,
        base,
        max_distance=args.max_distance,
        temperature=args.temperature,
        vote_threshold=args.vote_threshold,
        margin_threshold=args.margin_threshold,
    )
    output = pd.DataFrame({"path": test_values["path"].astype(str), "prediction": pred})
    output.to_csv(args.output, index=False)
    changes.to_csv(args.changes, index=False)

    print(validation.to_string(index=False))
    print(
        f"validation mean baseline={validation['baseline_accuracy'].mean():.6f} "
        f"template={validation['template_accuracy'].mean():.6f}"
    )
    print(f"test changed={int(np.sum(pred != base))}")
    print(f"submission={args.output}")
    print(f"changes={args.changes}")
    print(f"validation={args.validation_report}")


if __name__ == "__main__":
    main()
