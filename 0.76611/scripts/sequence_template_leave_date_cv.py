from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_postprocess import collect_test_metadata, collect_train_metadata_complete
from sequence_template_time_v2 import add_time_segments


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_LIKE_DATES = [
    "2025-05-31",
    "2025-06-01",
    "2025-06-02",
    "2025-06-12",
    "2025-06-13",
]
TEST_DATE_WEIGHTS = {
    "2025-05-31": 26,
    "2025-06-01": 32,
    "2025-06-02": 39,
    "2025-06-12": 13,
    "2025-06-13": 29,
}


def row_normalize(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), 1e-6
    )


def top_margin(logits: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(logits), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def make_templates(
    train: pd.DataFrame, excluded_dates: set[str]
) -> dict[int, list[tuple[np.ndarray, int]]]:
    counters: dict[int, Counter[tuple[int, ...]]] = {}
    subset = train[~train["date"].astype(str).isin(excluded_dates)]
    for _key, group in subset.groupby(["user", "date", "segment"], sort=False):
        sequence = tuple(group.sort_values("start")["label"].astype(int))
        counters.setdefault(len(sequence), Counter())[sequence] += 1
    return {
        length: [
            (np.asarray(sequence, dtype=np.int64), int(frequency))
            for sequence, frequency in counter.items()
        ]
        for length, counter in counters.items()
    }


def make_frame(
    root: Path, values: np.lib.npyio.NpzFile, *, test: bool = False
) -> pd.DataFrame:
    if test:
        frame = collect_test_metadata(root)
        positions = {
            str(clip_id): position
            for position, clip_id in enumerate(values["test_clip_id"].astype(str))
        }
        frame["_position"] = frame["clip_id"].map(positions).astype(int)
        frame["group"] = frame["segment"].astype(str)
        return frame.sort_values(["group", "start"]).reset_index(drop=True)

    train = add_time_segments(collect_train_metadata_complete(root)).set_index("clip_id")
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
    baseline = logits.argmax(axis=1).astype(np.int64)
    output = baseline.copy()
    emissions = row_normalize(logits)
    margins = top_margin(logits)
    order = np.argsort(logits, axis=1)[:, ::-1]
    records: list[dict[str, object]] = []

    for group_key, group in frame.groupby("group", sort=False):
        group = group.sort_values("start")
        positions = group["_position"].astype(int).to_numpy()
        observed = baseline[positions]
        candidates = templates_by_length.get(len(positions), [])
        best: tuple[float, np.ndarray, int, int] | None = None
        baseline_score = float(emissions[positions, observed].sum())
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
            [
                np.where(row == label)[0][0] + 1
                for row, label in zip(order[positions], sequence, strict=True)
            ],
            dtype=np.int64,
        )
        changed = sequence != observed
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
                    "base": int(baseline[position]),
                    "proposal": int(sequence[local]),
                    "accepted": bool(accepted[local]),
                    "rank": int(ranks[local]),
                    "margin": float(margins[position]),
                    "gain": gain,
                    "template_frequency": int(frequency),
                    "template_distance": int(distance),
                }
            )
    return output, pd.DataFrame(records)


def evaluate_date(
    frame: pd.DataFrame,
    logits: np.ndarray,
    truth: np.ndarray,
    train: pd.DataFrame,
    date: str,
    params: dict[str, float | int],
) -> tuple[dict[str, object], pd.DataFrame]:
    mask = frame["date"].astype(str).eq(date).to_numpy()
    heldout = frame.loc[mask].copy()
    heldout["_position"] = np.arange(mask.sum(), dtype=np.int64)
    decoded, changes = decode(
        heldout,
        logits[mask],
        make_templates(train, {date}),
        max_distance=int(params["max_distance"]),
        max_rank=int(params["max_rank"]),
        margin_limit=float(params["margin_limit"]),
        min_gain=float(params["min_gain"]),
        frequency_weight=float(params["frequency_weight"]),
        distance_penalty=float(params["distance_penalty"]),
    )
    baseline = logits[mask].argmax(axis=1)
    truth_date = truth[mask]
    row = {
        "date": date,
        "n": int(mask.sum()),
        "baseline_acc": float(np.mean(baseline == truth_date)),
        "decoded_acc": float(np.mean(decoded == truth_date)),
        "gain": float(np.mean(decoded == truth_date) - np.mean(baseline == truth_date)),
        "changed": int(np.sum(decoded != baseline)),
        "accepted_records": int(changes["accepted"].sum()) if not changes.empty else 0,
    }
    if not changes.empty:
        changes = changes.copy()
        changes["date"] = date
    return row, changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--logits",
        type=Path,
        default=ROOT / "outputs" / "tri_expert_cvstack_sequence_logits.npz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "sequence_template_leave_date_cv.csv",
    )
    parser.add_argument(
        "--changes",
        type=Path,
        default=ROOT / "outputs" / "sequence_template_leave_date_changes.csv",
    )
    args = parser.parse_args()

    values = np.load(args.logits, allow_pickle=True)
    train = add_time_segments(collect_train_metadata_complete(args.root))
    frame = make_frame(args.root, values)
    logits = values["val_logits"]
    truth = values["val_y"].astype(int)

    grid = [
        {
            "max_distance": distance,
            "max_rank": rank,
            "margin_limit": margin,
            "min_gain": gain,
            "frequency_weight": frequency,
            "distance_penalty": penalty,
        }
        for distance in [1, 2]
        for rank in [2, 3, 4]
        for margin in [0.75, 1.5, 2.5, 4.0]
        for gain in [-1.0, -0.25, 0.0]
        for frequency in [0.0, 0.5, 1.0]
        for penalty in [0.0, 0.5]
    ]

    rows: list[dict[str, object]] = []
    all_changes: list[pd.DataFrame] = []
    for index, params in enumerate(grid, start=1):
        date_rows: list[dict[str, object]] = []
        date_changes: list[pd.DataFrame] = []
        for date in PUBLIC_LIKE_DATES:
            row, changes = evaluate_date(frame, logits, truth, train, date, params)
            date_rows.append(row)
            if not changes.empty:
                date_changes.append(changes)
        weighted_gain = sum(
            row["gain"] * TEST_DATE_WEIGHTS[row["date"]]
            for row in date_rows
        ) / sum(TEST_DATE_WEIGHTS.values())
        mean_gain = float(np.mean([row["gain"] for row in date_rows]))
        worst_gain = float(min(row["gain"] for row in date_rows))
        result = {
            **params,
            "weighted_gain": float(weighted_gain),
            "mean_gain": mean_gain,
            "worst_gain": worst_gain,
            "changed_total": int(sum(row["changed"] for row in date_rows)),
            "accepted_total": int(sum(row["accepted_records"] for row in date_rows)),
        }
        for row in date_rows:
            result[f"{row['date']}_gain"] = row["gain"]
            result[f"{row['date']}_changed"] = row["changed"]
        rows.append(result)
        if index % 100 == 0:
            print(f"evaluated={index}/{len(grid)}")

    report = pd.DataFrame(rows).sort_values(
        ["weighted_gain", "worst_gain", "mean_gain", "changed_total"],
        ascending=[False, False, False, True],
    )
    report.to_csv(args.report, index=False)

    best_params = report.iloc[0].to_dict()
    best_changes: list[pd.DataFrame] = []
    best_date_rows = []
    for date in PUBLIC_LIKE_DATES:
        row, changes = evaluate_date(
            frame,
            logits,
            truth,
            train,
            date,
            best_params,
        )
        best_date_rows.append(row)
        if not changes.empty:
            best_changes.append(changes)
    if best_changes:
        pd.concat(best_changes, ignore_index=True).to_csv(args.changes, index=False)
    else:
        pd.DataFrame().to_csv(args.changes, index=False)

    print("baseline_acc=", f"{np.mean(logits.argmax(axis=1) == truth):.6f}")
    print(report.head(12).to_string(index=False))
    print("best_date_rows")
    print(pd.DataFrame(best_date_rows).to_string(index=False))
    print(f"report={args.report}")
    print(f"changes={args.changes}")


if __name__ == "__main__":
    main()
