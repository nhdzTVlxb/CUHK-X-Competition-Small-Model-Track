from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_postprocess import collect_test_metadata, collect_train_metadata
from sequence_template_time_v2 import add_time_segments

ROOT = Path(__file__).resolve().parents[1]


def row_normalize(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return (logits - logits.mean(axis=1, keepdims=True)) / np.maximum(
        logits.std(axis=1, keepdims=True), 1e-6
    )


def make_templates(
    train: pd.DataFrame, excluded_users: set[str]
) -> dict[int, list[tuple[np.ndarray, int]]]:
    counters: dict[int, Counter[tuple[int, ...]]] = {}
    subset = train[~train["user"].astype(str).isin(excluded_users)]
    for _key, group in subset.groupby(["user", "date", "segment"], sort=False):
        sequence = tuple(group.sort_values("start")["label"].astype(int))
        counters.setdefault(len(sequence), Counter())[sequence] += 1
    return {
        length: [(np.asarray(sequence, dtype=np.int64), frequency) for sequence, frequency in counter.items()]
        for length, counter in counters.items()
    }


def frame_for_validation(root: Path, values: np.lib.npyio.NpzFile) -> pd.DataFrame:
    train = add_time_segments(collect_train_metadata(root)).set_index("clip_id")
    rows: list[dict[str, object]] = []
    for position, clip_id in enumerate(values["val_clip_id"].astype(str)):
        if clip_id not in train.index:
            continue
        row = train.loc[clip_id].to_dict()
        rows.append(
            {
                **row,
                "clip_id": clip_id,
                "_position": position,
                "label": int(values["val_y"][position]),
                "group": f"{row['user']}__{row['date']}__{row['segment']}",
            }
        )
    return pd.DataFrame(rows)


def frame_for_test(root: Path, values: np.lib.npyio.NpzFile) -> pd.DataFrame:
    frame = collect_test_metadata(root)
    frame["_position"] = frame["clip_id"].map(
        {str(clip_id): position for position, clip_id in enumerate(values["test_clip_id"])}
    ).astype(int)
    frame["group"] = frame["segment"].astype(str)
    return frame.sort_values(["group", "start"]).reset_index(drop=True)


def decode(
    frame: pd.DataFrame,
    logits: np.ndarray,
    templates_by_length: dict[int, list[tuple[np.ndarray, int]]],
    *,
    max_distance: int,
    max_rank: int,
    margin_limit: float,
    min_gain: float,
    frequency_weight: float,
    distance_penalty: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    output = logits.argmax(axis=1).astype(np.int64)
    emissions = row_normalize(logits)
    margins = np.sort(logits, axis=1)[:, -1] - np.sort(logits, axis=1)[:, -2]
    order = np.argsort(logits, axis=1)[:, ::-1]
    records: list[dict[str, object]] = []

    for group_key, group in frame.groupby("group", sort=False):
        group = group.sort_values("start")
        positions = group["_position"].astype(int).to_numpy()
        observed = output[positions]
        candidates = templates_by_length.get(len(positions), [])
        if not candidates:
            continue

        baseline_score = float(emissions[positions, observed].sum())
        best: tuple[float, np.ndarray, int, int] | None = None
        for sequence, frequency in candidates:
            distance = int(np.sum(sequence != observed))
            if distance > max_distance:
                continue
            score = (
                float(emissions[positions, sequence].sum())
                + frequency_weight * float(np.log1p(frequency))
                - distance_penalty * distance
            )
            if best is None or score > best[0]:
                best = (score, sequence, distance, frequency)
        if best is None:
            continue

        best_score, sequence, distance, frequency = best
        gain = float(best_score - baseline_score)
        ranks = np.asarray(
            [np.where(row == label)[0][0] + 1 for row, label in zip(order[positions], sequence)],
            dtype=np.int64,
        )
        changed = sequence != output[positions]
        accepted = (
            changed
            & (ranks <= max_rank)
            & (margins[positions] <= margin_limit)
            & (gain >= min_gain)
        )
        output[positions[accepted]] = sequence[accepted]

        clip_ids = group["clip_id"].astype(str).to_numpy()
        for local, position in enumerate(positions):
            if not changed[local]:
                continue
            records.append(
                {
                    "group": str(group_key),
                    "position": int(position),
                    "clip_id": str(clip_ids[local]),
                    "base": int(logits[position].argmax()),
                    "proposal": int(sequence[local]),
                    "accepted": bool(accepted[local]),
                    "rank": int(ranks[local]),
                    "margin": float(margins[position]),
                    "gain": gain,
                    "template_frequency": int(frequency),
                    "template_distance": int(distance),
                    "group_base": " ".join(map(str, observed.tolist())),
                    "group_proposal": " ".join(map(str, sequence.tolist())),
                }
            )
    return output, pd.DataFrame(records)


def validate(root: Path, values: np.lib.npyio.NpzFile, args: argparse.Namespace) -> tuple[float, pd.DataFrame]:
    train = add_time_segments(collect_train_metadata(root))
    frame = frame_for_validation(root, values)
    output = values["val_logits"].argmax(axis=1).astype(np.int64)
    records: list[pd.DataFrame] = []

    for user in sorted(frame["user"].astype(str).unique()):
        heldout = frame.loc[frame["user"].astype(str).eq(user)].copy()
        positions = heldout["_position"].astype(int).to_numpy()
        heldout["_position"] = np.arange(len(positions), dtype=np.int64)
        heldout["group"] = (
            heldout["user"].astype(str)
            + "__"
            + heldout["date"].astype(str)
            + "__"
            + heldout["segment"].astype(str)
        )
        decoded, changes = decode(
            heldout,
            values["val_logits"][positions],
            make_templates(train, {user}),
            max_distance=args.max_distance,
            max_rank=args.max_rank,
            margin_limit=args.margin_limit,
            min_gain=args.min_gain,
            frequency_weight=args.frequency_weight,
            distance_penalty=args.distance_penalty,
        )
        output[positions] = decoded
        if not changes.empty:
            changes["user"] = user
            records.append(changes)

    report = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return float(np.mean(output == values["val_y"])), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--logits", type=Path, default=ROOT / "outputs" / "tri_best_logits.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_tri_sequence_logit_time_v3.csv")
    parser.add_argument("--changes", type=Path, default=ROOT / "outputs" / "sequence_logit_time_v3_changes.csv")
    parser.add_argument("--validation-report", type=Path, default=ROOT / "outputs" / "sequence_logit_time_v3_validation_changes.csv")
    parser.add_argument("--max-distance", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--margin-limit", type=float, default=4.0)
    parser.add_argument("--min-gain", type=float, default=-2.0)
    parser.add_argument("--frequency-weight", type=float, default=1.0)
    parser.add_argument("--distance-penalty", type=float, default=0.0)
    args = parser.parse_args()

    values = np.load(args.logits, allow_pickle=True)
    train = add_time_segments(collect_train_metadata(args.root))
    prediction, changes = decode(
        frame_for_test(args.root, values),
        values["test_logits"],
        make_templates(train, set()),
        max_distance=args.max_distance,
        max_rank=args.max_rank,
        margin_limit=args.margin_limit,
        min_gain=args.min_gain,
        frequency_weight=args.frequency_weight,
        distance_penalty=args.distance_penalty,
    )
    pd.DataFrame(
        {"path": values["test_path"].astype(str), "prediction": prediction.astype(int)}
    ).to_csv(args.output, index=False)
    changes.to_csv(args.changes, index=False)

    validation_acc, validation_changes = validate(args.root, values, args)
    validation_changes.to_csv(args.validation_report, index=False)

    print(f"validation_accuracy={validation_acc:.6f}")
    print(f"validation_changes={len(validation_changes)}")
    print(f"test_candidates={len(changes)}")
    print(f"test_accepted={int(changes['accepted'].sum()) if not changes.empty else 0}")
    print(f"submission={args.output}")
    print(f"changes={args.changes}")
    print(f"validation_report={args.validation_report}")


if __name__ == "__main__":
    main()
