from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_postprocess import collect_test_metadata, collect_train_metadata

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 40
VAL_USERS = {"user8", "user9", "user23", "user24"}


def add_time_segments(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["user", "start"]).reset_index(drop=True).copy()
    frame["date"] = frame["start"].dt.date.astype(str)
    frame["gap"] = (
        frame.groupby(["user", "date"])["start"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )
    frame["segment"] = frame.groupby(["user", "date"])["gap"].transform(
        lambda values: values.gt(20.0).cumsum()
    )
    return frame


def make_templates(
    train: pd.DataFrame, excluded_users: set[str]
) -> Counter[tuple[int, ...]]:
    templates: Counter[tuple[int, ...]] = Counter()
    subset = train[~train["user"].astype(str).isin(excluded_users)]
    for _key, group in subset.groupby(["user", "date", "segment"], sort=False):
        sequence = tuple(group.sort_values("start")["label"].astype(int))
        templates[sequence] += 1
    return templates


def top_margin(logits: np.ndarray) -> np.ndarray:
    ordered = np.sort(logits, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def propose_sequence(
    observed: np.ndarray,
    templates: Counter[tuple[int, ...]],
    max_distance: int,
) -> tuple[np.ndarray | None, float, int, int]:
    candidates: list[tuple[np.ndarray, int, int]] = []
    for sequence, frequency in templates.items():
        if len(sequence) != len(observed):
            continue
        distance = int(np.sum(observed != np.asarray(sequence)))
        if distance <= max_distance:
            candidates.append((np.asarray(sequence, dtype=np.int64), frequency, distance))
    if not candidates:
        return None, 0.0, -1, 0

    weights = np.asarray(
        [frequency * np.exp(-distance) for _sequence, frequency, distance in candidates],
        dtype=np.float64,
    )
    weights /= np.maximum(weights.sum(), 1e-12)
    votes = np.zeros((len(observed), N_CLASSES), dtype=np.float64)
    for weight, (sequence, _frequency, _distance) in zip(weights, candidates, strict=True):
        votes[np.arange(len(observed)), sequence] += weight
    return (
        votes.argmax(axis=1).astype(np.int64),
        float(votes.max(axis=1).min()),
        min(item[2] for item in candidates),
        len(candidates),
    )


def apply_templates(
    frame: pd.DataFrame,
    logits: np.ndarray,
    templates: Counter[tuple[int, ...]],
    *,
    max_distance: int = 1,
    max_rank: int = 4,
    margin_limit: float = 3.0,
    vote_limit: float = 0.5,
) -> tuple[np.ndarray, pd.DataFrame]:
    baseline = logits.argmax(axis=1).astype(np.int64)
    output = baseline.copy()
    margins = top_margin(logits)
    records: list[dict[str, object]] = []

    for group_key, group in frame.groupby("group", sort=False):
        group = group.sort_values("start")
        positions = group["_position"].astype(int).to_numpy()
        observed = baseline[positions]
        proposal, _min_vote, distance, candidate_count = propose_sequence(
            observed, templates, max_distance
        )
        if proposal is None:
            continue

        order = np.argsort(logits[positions], axis=1)[:, ::-1]
        ranks = np.asarray(
            [np.where(row == label)[0][0] + 1 for row, label in zip(order, proposal)],
            dtype=np.int64,
        )
        changed = proposal != baseline[positions]
        accepted = (
            changed
            & (ranks <= max_rank)
            & (margins[positions] <= margin_limit)
        )
        output[positions[accepted]] = proposal[accepted]
        clip_ids = group["clip_id"].astype(str).to_numpy()
        for local, position in enumerate(positions):
            if not changed[local]:
                continue
            records.append(
                {
                    "group": str(group_key),
                    "position": int(position),
                    "clip_id": str(clip_ids[local]),
                    "base": int(baseline[position]),
                    "proposal": int(proposal[local]),
                    "accepted": bool(accepted[local]),
                    "rank": int(ranks[local]),
                    "margin": float(margins[position]),
                    "candidate_count": int(candidate_count),
                    "template_distance": int(distance),
                }
            )
    return output, pd.DataFrame(records)


def validation_frame(root: Path, values: np.lib.npyio.NpzFile) -> pd.DataFrame:
    train = add_time_segments(collect_train_metadata(root)).set_index("clip_id")
    rows: list[dict[str, object]] = []
    for position, clip_id in enumerate(values["val_clip_id"].astype(str)):
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


def evaluate(
    root: Path,
    values: np.lib.npyio.NpzFile,
    *,
    max_distance: int,
    max_rank: int,
    margin_limit: float,
) -> tuple[float, pd.DataFrame]:
    train = add_time_segments(collect_train_metadata(root))
    frame = validation_frame(root, values)
    logits = values["val_logits"]
    baseline = logits.argmax(axis=1)
    output = baseline.copy()
    records: list[dict[str, object]] = []

    for user in sorted(frame["user"].astype(str).unique()):
        mask = frame["user"].astype(str).eq(user).to_numpy()
        heldout = frame.loc[mask].copy()
        heldout["_position"] = np.arange(mask.sum(), dtype=np.int64)
        heldout["group"] = (
            heldout["user"].astype(str)
            + "__"
            + heldout["date"].astype(str)
            + "__"
            + heldout["segment"].astype(str)
        )
        decoded, changes = apply_templates(
            heldout,
            logits[mask],
            make_templates(train, {user}),
            max_distance=max_distance,
            max_rank=max_rank,
            margin_limit=margin_limit,
        )
        output[mask] = decoded
        if not changes.empty:
            changes["user"] = user
            records.append(changes)
    report = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return float(np.mean(output == values["val_y"])), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--logits", type=Path, default=ROOT / "outputs" / "tri_best_logits.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_tri_sequence_template_v2.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "sequence_template_time_v2_report.csv")
    parser.add_argument("--max-distance", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=4)
    parser.add_argument("--margin-limit", type=float, default=3.0)
    args = parser.parse_args()

    values = np.load(args.logits, allow_pickle=True)
    train = add_time_segments(collect_train_metadata(args.root))
    test_values = values
    test_frame = collect_test_metadata(args.root)
    test_frame["_position"] = test_frame["clip_id"].map(
        {str(clip_id): i for i, clip_id in enumerate(test_values["test_clip_id"])}
    ).astype(int)
    test_frame["group"] = test_frame["segment"].astype(str)
    test_frame = test_frame.sort_values(["group", "start"]).reset_index(drop=True)

    test_logits = test_values["test_logits"]
    test_templates = make_templates(train, set())
    prediction, changes = apply_templates(
        test_frame,
        test_logits,
        test_templates,
        max_distance=args.max_distance,
        max_rank=args.max_rank,
        margin_limit=args.margin_limit,
    )
    output = pd.DataFrame(
        {"path": test_values["test_path"].astype(str), "prediction": prediction}
    )
    output.to_csv(args.output, index=False)
    changes.to_csv(args.report, index=False)

    validation_acc, validation_changes = evaluate(
        args.root,
        values,
        max_distance=args.max_distance,
        max_rank=args.max_rank,
        margin_limit=args.margin_limit,
    )
    print(f"validation_accuracy={validation_acc:.6f}")
    print(f"validation_changes={len(validation_changes)}")
    print(f"test_candidates={len(changes)}")
    print(f"test_accepted={int(changes['accepted'].sum()) if not changes.empty else 0}")
    print(f"submission={args.output}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
