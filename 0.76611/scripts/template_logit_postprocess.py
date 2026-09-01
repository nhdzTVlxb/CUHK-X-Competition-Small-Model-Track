from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sequence_postprocess import collect_test_metadata, collect_train_metadata  # noqa: E402


N_CLASSES = 40
DEFAULT_BASE = ROOT / "outputs" / "raw_submissions" / "55848407_submission_public.csv"
DEFAULT_TEST = ROOT / "outputs" / "yolo_r2p1d_test_per_model_logits.npz"
DEFAULT_VAL = ROOT / "outputs" / "yolo_r2p1d_val_logits.npz"


def sequence_templates(train: pd.DataFrame, excluded_users: set[str]) -> Counter[tuple[int, ...]]:
    subset = train[~train["user"].astype(str).isin(excluded_users)]
    counts: Counter[tuple[int, ...]] = Counter()
    for _key, group in subset.groupby(["user", "trial"], sort=False):
        sequence = tuple(group.sort_values("start")["label"].astype(int))
        counts[sequence] += 1
    return counts


def frame_for_validation(root: Path, values: np.lib.npyio.NpzFile) -> pd.DataFrame:
    train = collect_train_metadata(root).set_index("clip_id")
    rows: list[dict[str, object]] = []
    for position, clip_id in enumerate(values["clip_id"].astype(str)):
        row = train.loc[clip_id].to_dict()
        rows.append(
            {
                **row,
                "clip_id": clip_id,
                "_position": position,
                "label": int(values["y"][position]),
                "group": f"{row['user']}__{row['trial']}",
            }
        )
    return pd.DataFrame(rows)


def frame_for_test(root: Path, values: np.lib.npyio.NpzFile) -> pd.DataFrame:
    frame = collect_test_metadata(root)
    order = {str(clip_id): position for position, clip_id in enumerate(values["clip_id"])}
    frame["_position"] = frame["clip_id"].map(order).astype(int)
    frame["group"] = frame["segment"].astype(str)
    return frame.sort_values(["group", "start"]).reset_index(drop=True)


def normalize_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return (logits - logits.mean(axis=1, keepdims=True)) / np.maximum(
        logits.std(axis=1, keepdims=True), 1e-6
    )


def candidate_sequences(
    observed: np.ndarray,
    templates: Counter[tuple[int, ...]],
    max_distance: int,
    min_frequency: int,
) -> list[tuple[tuple[int, ...], int, int]]:
    result: list[tuple[tuple[int, ...], int, int]] = []
    for sequence, frequency in templates.items():
        if frequency < min_frequency or len(sequence) != len(observed):
            continue
        distance = int(np.sum(observed != np.asarray(sequence)))
        if distance <= max_distance:
            result.append((sequence, frequency, distance))
    return result


def score_sequence(
    emissions: np.ndarray,
    sequence: tuple[int, ...],
    frequency: int,
    distance: int,
    *,
    frequency_weight: float,
    distance_penalty: float,
) -> float:
    labels = np.asarray(sequence, dtype=np.int64)
    row_score = float(emissions[np.arange(len(labels)), labels].sum())
    return (
        row_score
        + frequency_weight * float(np.log1p(frequency))
        - distance_penalty * float(distance)
    )


def decode_frame(
    frame: pd.DataFrame,
    logits: np.ndarray,
    templates: Counter[tuple[int, ...]],
    base_predictions: np.ndarray,
    *,
    max_distance: int,
    min_frequency: int,
    margin_threshold: float,
    min_gain: float,
    frequency_weight: float,
    distance_penalty: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    emissions = normalize_rows(logits)
    output = np.asarray(base_predictions, dtype=np.int64).copy()
    records: list[dict[str, object]] = []

    for group_key, group in frame.groupby("group", sort=False):
        group = group.sort_values("start")
        positions = group["_position"].astype(int).to_numpy()
        observed = output[positions].copy()
        candidates = candidate_sequences(observed, templates, max_distance, min_frequency)
        if not candidates:
            continue

        baseline_sequence = tuple(int(value) for value in observed)
        baseline_score = score_sequence(
            emissions[positions],
            baseline_sequence,
            frequency=0,
            distance=0,
            frequency_weight=frequency_weight,
            distance_penalty=distance_penalty,
        )
        ranked = sorted(
            (
                score_sequence(
                    emissions[positions],
                    sequence,
                    frequency,
                    distance,
                    frequency_weight=frequency_weight,
                    distance_penalty=distance_penalty,
                ),
                sequence,
                frequency,
                distance,
            )
            for sequence, frequency, distance in candidates
        )[::-1]
        best_score, best_sequence, best_frequency, best_distance = ranked[0]
        gain = float(best_score - baseline_score)
        proposed = np.asarray(best_sequence, dtype=np.int64)
        changed = proposed != observed
        margins = np.sort(logits[positions], axis=1)[:, -1] - np.sort(logits[positions], axis=1)[:, -2]
        accepted = changed & (margins <= margin_threshold) & (gain >= min_gain)

        for local, position in enumerate(positions):
            if not changed[local]:
                continue
            if accepted[local]:
                output[position] = proposed[local]
            records.append(
                {
                    "group": str(group_key),
                    "position": int(position),
                    "clip_id": str(frame.loc[frame["_position"] == position, "clip_id"].iloc[0]),
                    "base": int(observed[local]),
                    "proposal": int(proposed[local]),
                    "model_margin": float(margins[local]),
                    "sequence_gain": gain,
                    "template_frequency": int(best_frequency),
                    "template_distance": int(best_distance),
                    "candidate_count": int(len(candidates)),
                    "accepted": bool(accepted[local]),
                }
            )

    return output, pd.DataFrame(records)


def evaluate_grid(
    root: Path,
    values: np.lib.npyio.NpzFile,
    *,
    max_distances: list[int],
    min_frequencies: list[int],
    margins: list[float],
    gains: list[float],
    frequency_weight: float,
    distance_penalty: float,
) -> pd.DataFrame:
    train = collect_train_metadata(root)
    frame = frame_for_validation(root, values)
    model_logits = values["per_model_logits"][0] if "per_model_logits" in values else values["logits"]
    base = model_logits.argmax(axis=1).astype(np.int64)
    rows: list[dict[str, object]] = []

    for max_distance in max_distances:
        for min_frequency in min_frequencies:
            for margin_threshold in margins:
                for min_gain in gains:
                    user_results = []
                    for excluded_user in sorted(frame["user"].astype(str).unique()):
                        mask = frame["user"].astype(str).eq(excluded_user).to_numpy()
                        heldout = frame.loc[mask].copy()
                        heldout["_position"] = np.arange(len(heldout), dtype=int)
                        templates = sequence_templates(train, {excluded_user})
                        pred, changes = decode_frame(
                            heldout,
                            model_logits[mask],
                            templates,
                            base[mask],
                            max_distance=max_distance,
                            min_frequency=min_frequency,
                            margin_threshold=margin_threshold,
                            min_gain=min_gain,
                            frequency_weight=frequency_weight,
                            distance_penalty=distance_penalty,
                        )
                        truth = values["y"][mask]
                        user_results.append(
                            {
                                "baseline": float(np.mean(base[mask] == truth)),
                                "decoded": float(np.mean(pred == truth)),
                                "changed": int(np.sum(pred != base[mask])),
                                "accepted": int(changes["accepted"].sum()) if not changes.empty else 0,
                            }
                        )
                    rows.append(
                        {
                            "max_distance": max_distance,
                            "min_frequency": min_frequency,
                            "margin_threshold": margin_threshold,
                            "min_gain": min_gain,
                            "mean_baseline": float(np.mean([item["baseline"] for item in user_results])),
                            "mean_decoded": float(np.mean([item["decoded"] for item in user_results])),
                            "global_baseline": float(np.mean(base == values["y"])),
                            "global_decoded": float(
                                np.mean(
                                    [
                                        item["decoded"]
                                        for item in user_results
                                    ]
                                )
                            ),
                            "mean_changed": float(np.mean([item["changed"] for item in user_results])),
                            "mean_accepted": float(np.mean([item["accepted"] for item in user_results])),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["mean_decoded", "mean_changed"], ascending=[False, True]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--base-submission", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--test-logits", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--val-logits", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_template_logit_v1.csv")
    parser.add_argument("--changes", type=Path, default=ROOT / "outputs" / "template_logit_v1_changes.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "template_logit_validation_v1.csv")
    parser.add_argument("--max-distance", type=int, default=1)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--margin-threshold", type=float, default=4.0)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--frequency-weight", type=float, default=0.0)
    parser.add_argument("--distance-penalty", type=float, default=0.0)
    args = parser.parse_args()

    val_values = np.load(args.val_logits, allow_pickle=True)
    report = evaluate_grid(
        args.root,
        val_values,
        max_distances=[0, 1, 2],
        min_frequencies=[1, 2],
        margins=[1.0, 2.0, 4.0, 8.0],
        gains=[0.0, 0.5, 1.0, 2.0],
        frequency_weight=args.frequency_weight,
        distance_penalty=args.distance_penalty,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)

    test_values = np.load(args.test_logits, allow_pickle=True)
    test_frame = frame_for_test(args.root, test_values)
    train = collect_train_metadata(args.root)
    templates = sequence_templates(train, set())
    test_logits = (
        test_values["per_model_logits"][0]
        if "per_model_logits" in test_values
        else test_values["logits"]
    )
    base_submission = pd.read_csv(args.base_submission)
    base_lookup = dict(
        zip(
            base_submission["path"].astype(str),
            base_submission["prediction"].astype(int),
            strict=True,
        )
    )
    base = np.asarray([base_lookup[str(path)] for path in test_values["path"]], dtype=np.int64)
    pred, changes = decode_frame(
        test_frame,
        test_logits,
        templates,
        base,
        max_distance=args.max_distance,
        min_frequency=args.min_frequency,
        margin_threshold=args.margin_threshold,
        min_gain=args.min_gain,
        frequency_weight=args.frequency_weight,
        distance_penalty=args.distance_penalty,
    )
    pd.DataFrame({"path": test_values["path"].astype(str), "prediction": pred}).to_csv(
        args.output, index=False
    )
    changes.to_csv(args.changes, index=False)

    print(report.head(12).to_string(index=False))
    print(f"validation={args.report}")
    print(f"test changed={int(np.sum(pred != base))}")
    print(f"submission={args.output}")
    print(f"changes={args.changes}")


if __name__ == "__main__":
    main()
