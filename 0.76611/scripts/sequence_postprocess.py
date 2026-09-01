from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

N_CLASSES = 40
FRAME_RE = re.compile(
    r"(?P<date>\d{4}-\d\d-\d\d)_(?P<time>\d\d-\d\d-\d\d\.\d{3})_(?P<frame>\d+)(?:_|\.|$)"
)
FRAME_ONLY_RE = re.compile(r"(?P<frame>\d+)(?:_|\.|$)")


def parse_frame_time(path: Path) -> tuple[pd.Timestamp, int] | None:
    match = FRAME_RE.search(path.name)
    if match is None:
        return None
    time_text = match.group("time").replace("-", ":", 2)
    return pd.Timestamp(f"{match.group('date')} {time_text}"), int(match.group("frame"))


def _parse_skeleton_metadata(trial_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp, int, int] | None:
    """Recover clip timing when Depth_Color filenames contain only frame numbers."""
    predictions_dir = trial_dir.parent.parent.parent.parent / "Skeleton" / trial_dir.parent.parent.name / trial_dir.parent.name / trial_dir.name / "predictions"
    if not predictions_dir.exists():
        return None

    timestamped: list[tuple[pd.Timestamp, int]] = []
    frame_only: list[int] = []
    for path in predictions_dir.iterdir():
        if not path.is_file():
            continue
        parsed = parse_frame_time(path)
        if parsed is not None:
            timestamped.append(parsed)
            continue
        match = re.search(r"_(?P<frame>\d+)(?:\.json|$)", path.name)
        if match is not None:
            frame_only.append(int(match.group("frame")))
    if not timestamped:
        return None

    timestamped.sort(key=lambda item: item[1])
    first_time, first_frame = timestamped[0]
    last_time, last_frame = timestamped[-1]
    all_frames = frame_only + [frame for _time, frame in timestamped]
    f0, f1 = min(all_frames), max(all_frames)
    # The source streams are nominally 10 FPS. Use timestamped frames as
    # anchors so mixed old/new filename conventions remain chronologically ordered.
    start = first_time + pd.to_timedelta((f0 - first_frame) / 10.0, unit="s")
    end = last_time + pd.to_timedelta((f1 - last_frame) / 10.0, unit="s")
    return start, end, f0, f1


def collect_train_metadata(root: Path) -> pd.DataFrame:
    image_root = root / "data" / "Training" / "data" / "HAR" / "data" / "Depth_Color"
    rows: list[dict[str, object]] = []
    for action_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        label = int(action_dir.name.split("_", 1)[0])
        for user_dir in sorted(p for p in action_dir.iterdir() if p.is_dir()):
            for trial_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
                parsed = [parse_frame_time(p) for p in trial_dir.iterdir() if p.is_file()]
                parsed = [item for item in parsed if item is not None]
                if not parsed:
                    continue
                parsed.sort()
                rows.append(
                    {
                        "clip_id": f"{action_dir.name}__{user_dir.name}__{trial_dir.name}",
                        "label": label,
                        "user": user_dir.name,
                        "trial": trial_dir.name,
                        "start": parsed[0][0],
                        "end": parsed[-1][0],
                        "f0": parsed[0][1],
                        "f1": parsed[-1][1],
                    }
                )
    return pd.DataFrame(rows).sort_values(["user", "trial", "start"]).reset_index(drop=True)


def collect_train_metadata_complete(root: Path) -> pd.DataFrame:
    """Collect all train clips, including clips with timestamp-free depth names."""
    frame = collect_train_metadata(root)
    known = set(frame["clip_id"].astype(str)) if not frame.empty else set()
    image_root = root / "data" / "Training" / "data" / "HAR" / "data" / "Depth_Color"
    rows: list[dict[str, object]] = []
    for action_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        label = int(action_dir.name.split("_", 1)[0])
        for user_dir in sorted(p for p in action_dir.iterdir() if p.is_dir()):
            for trial_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
                clip_id = f"{action_dir.name}__{user_dir.name}__{trial_dir.name}"
                if clip_id in known:
                    continue
                recovered = _parse_skeleton_metadata(trial_dir)
                if recovered is None:
                    continue
                start, end, f0, f1 = recovered
                rows.append(
                    {
                        "clip_id": clip_id,
                        "label": label,
                        "user": user_dir.name,
                        "trial": trial_dir.name,
                        "start": start,
                        "end": end,
                        "f0": f0,
                        "f1": f1,
                    }
                )
    if rows:
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    return frame.sort_values(["user", "trial", "start"]).reset_index(drop=True)


def collect_test_metadata(root: Path) -> pd.DataFrame:
    test_root = root / "data" / "Testing" / "data" / "small_model_track_test"
    test_csv = root / "data" / "Testing" / "test.csv"
    rows: list[dict[str, object]] = []
    for raw_path in pd.read_csv(test_csv)["path"].astype(str):
        clip_id = raw_path.strip("/\\").split("/")[-1]
        image_dir = test_root / clip_id / "Depth_Color"
        parsed = [parse_frame_time(p) for p in image_dir.iterdir() if p.is_file()]
        parsed = [item for item in parsed if item is not None]
        if not parsed:
            continue
        parsed.sort()
        rows.append(
            {
                "clip_id": clip_id,
                "path": raw_path,
                "start": parsed[0][0],
                "end": parsed[-1][0],
                "f0": parsed[0][1],
                "f1": parsed[-1][1],
            }
        )
    frame = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    frame["gap_s"] = frame["start"].diff().dt.total_seconds().fillna(0.0)
    frame["segment"] = frame["gap_s"].gt(20.0).cumsum().astype(int)
    return frame


def transition_log_probs(
    train: pd.DataFrame,
    excluded_users: set[str],
    smoothing: float,
) -> np.ndarray:
    counts = np.full((N_CLASSES, N_CLASSES), smoothing, dtype=np.float64)
    subset = train[~train["user"].isin(excluded_users)]
    for _, group in subset.groupby(["user", "trial"], sort=False):
        labels = group.sort_values("start")["label"].astype(int).tolist()
        for left, right in zip(labels, labels[1:]):
            counts[left, right] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs)


def row_normalize(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    center = logits.mean(axis=1, keepdims=True)
    scale = logits.std(axis=1, keepdims=True)
    return (logits - center) / np.maximum(scale, 1e-6)


def viterbi(logits: np.ndarray, transition: np.ndarray, weight: float) -> np.ndarray:
    emissions = np.asarray(logits, dtype=np.float64)
    length = emissions.shape[0]
    scores = np.empty((length, N_CLASSES), dtype=np.float64)
    back = np.zeros((length, N_CLASSES), dtype=np.int16)
    scores[0] = emissions[0]
    for index in range(1, length):
        candidates = scores[index - 1][:, None] + weight * transition
        back[index] = candidates.argmax(axis=0).astype(np.int16)
        scores[index] = emissions[index] + candidates.max(axis=0)
    output = np.empty(length, dtype=np.int64)
    output[-1] = scores[-1].argmax()
    for index in range(length - 1, 0, -1):
        output[index - 1] = back[index, output[index]]
    return output


def row_margin(logits: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(logits), axis=1)
    return ordered[:, -1] - ordered[:, -2]


def decode_groups(
    frame: pd.DataFrame,
    logits: np.ndarray,
    transition: np.ndarray,
    weight: float,
    margin_limit: float | None,
    require_disagreement: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    baseline = logits.argmax(axis=1).astype(np.int64)
    emissions = row_normalize(logits)
    output = baseline.copy()
    records: list[dict[str, object]] = []

    for group_key, group in frame.groupby("group", sort=False):
        group = group.sort_values("start")
        positions = group["_position"].astype(int).to_numpy()
        proposed = viterbi(emissions[positions], transition, weight)
        changed = proposed != baseline[positions]
        margins = row_margin(emissions[positions])
        allowed = np.ones(len(positions), dtype=bool)
        if margin_limit is not None:
            allowed &= margins <= margin_limit
        if require_disagreement:
            allowed &= proposed != baseline[positions]
        accepted = changed & allowed
        output[positions[accepted]] = proposed[accepted]
        for pos, proposal, margin, is_changed, is_accepted in zip(
            positions, proposed, margins, changed, accepted, strict=True
        ):
            if is_changed:
                records.append(
                    {
                        "group": group_key,
                        "position": int(pos),
                        "clip_id": str(frame.iloc[pos]["clip_id"]),
                        "baseline": int(baseline[pos]),
                        "proposal": int(proposal),
                        "margin": float(margin),
                        "accepted": bool(is_accepted),
                    }
                )
    return output, pd.DataFrame(records)


def validation_frame(root: Path, val_logits: np.ndarray) -> pd.DataFrame:
    values = np.load(root / "outputs" / "yolo_r2p1d_val_logits.npz", allow_pickle=True)
    frame = collect_train_metadata(root)
    lookup = frame.set_index("clip_id")
    rows = []
    for position, clip_id in enumerate(values["clip_id"].astype(str)):
        row = lookup.loc[clip_id].to_dict()
        row["clip_id"] = clip_id
        row["label"] = int(values["y"][position])
        row["_position"] = position
        row["group"] = f"{row['user']}__{row['trial']}"
        rows.append(row)
    return pd.DataFrame(rows)


def score(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(truth)))


def load_test_logits(root: Path, path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    values = np.load(path, allow_pickle=True)
    logits = values["logits"]
    frame = collect_test_metadata(root)
    order = {str(clip_id): index for index, clip_id in enumerate(values["clip_id"])}
    frame["_position"] = frame["clip_id"].map(order).astype(int)
    frame = frame.sort_values("start").reset_index(drop=True)
    frame["group"] = frame["segment"].astype(str)
    # The output is indexed in official manifest order, while groups are temporal.
    return logits, frame


def write_submission(root: Path, preds: np.ndarray, output: Path) -> None:
    manifest = pd.read_csv(root / "cache" / "yolo_r2p1d_v9" / "test_manifest.csv")
    pd.DataFrame({"path": manifest["path"].astype(str), "prediction": preds.astype(int)}).to_csv(
        output, index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--test-logits",
        type=Path,
        default=None,
        help="Defaults to outputs/yolo_r2p1d_v9_logits.npz.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--weight", type=float, default=0.8)
    parser.add_argument("--margin-limit", type=float, default=1.5)
    parser.add_argument("--all-changes", action="store_true")
    args = parser.parse_args()

    root = args.root
    train = collect_train_metadata(root)
    val_values = np.load(root / "outputs" / "yolo_r2p1d_val_logits.npz", allow_pickle=True)
    val_logits = val_values["logits"]
    val_frame = validation_frame(root, val_logits)
    val_transition = transition_log_probs(train, {"user8", "user9", "user23", "user24"}, args.smoothing)

    baseline = val_logits.argmax(axis=1)
    rows = []
    for weight in np.linspace(0.0, 2.0, 21):
        for margin in [None, 0.5, 1.0, 1.5, 2.0, 3.0]:
            pred, changes = decode_groups(
                val_frame,
                val_logits,
                val_transition,
                float(weight),
                margin,
                require_disagreement=False,
            )
            rows.append(
                {
                    "weight": float(weight),
                    "margin_limit": margin if margin is not None else -1.0,
                    "accuracy": score(pred, val_values["y"]),
                    "changed": int(np.sum(pred != baseline)),
                    "proposed_changes": len(changes),
                }
            )
    report = pd.DataFrame(rows).sort_values(["accuracy", "changed"], ascending=[False, True])
    print("validation baseline=", f"{score(baseline, val_values['y']):.6f}")
    print(report.head(12).to_string(index=False))

    if args.test_logits is None:
        args.test_logits = root / "outputs" / "yolo_r2p1d_v9_logits.npz"
    test_logits, test_frame = load_test_logits(root, args.test_logits)
    test_transition = transition_log_probs(train, set(), args.smoothing)
    chosen = report.iloc[0]
    margin = None if float(chosen["margin_limit"]) < 0 else float(chosen["margin_limit"])
    test_pred, test_changes = decode_groups(
        test_frame,
        test_logits,
        test_transition,
        float(chosen["weight"]),
        margin if not args.all_changes else None,
        require_disagreement=False,
    )
    output = args.output or root / "outputs" / "submission_yolo_r2p1d_sequence.csv"
    write_submission(root, test_pred, output)
    change_path = root / "outputs" / "sequence_changes.csv"
    test_changes.to_csv(change_path, index=False)
    report_path = args.report or root / "outputs" / "sequence_validation_report.csv"
    report.to_csv(report_path, index=False)
    print(f"test rows={len(test_pred)} changed={int(np.sum(test_pred != test_logits.argmax(1)))}")
    print(f"submission={output}")
    print(f"changes={change_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
