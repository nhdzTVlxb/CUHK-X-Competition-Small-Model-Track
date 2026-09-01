from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "outputs" / "submission_tri_best.csv",
    )
    parser.add_argument(
        "--logits",
        type=Path,
        default=ROOT / "outputs" / "tri_best_logits.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
    )
    args = parser.parse_args()

    base = pd.read_csv(args.base)
    base_paths = base["path"].astype(str).tolist()
    sources = {
        "phuong": args.root / "kernel_outputs" / "phuongncn_0711" / "submission.csv",
        "kunal": args.root / "kernel_outputs" / "kunal_yolo_v8" / "submission.csv",
        "skel": args.root / "outputs" / "submission_skel_imu_v2.csv",
        "welsh": args.root / "external_models" / "welsh_submission" / "submission.csv",
    }
    predictions: dict[str, np.ndarray] = {}
    for name, path in sources.items():
        frame = pd.read_csv(path)
        lookup = frame.set_index("path")["prediction"]
        missing = sorted(set(base_paths) - set(lookup.index.astype(str)))
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} base paths")
        predictions[name] = lookup.reindex(base_paths).astype(int).to_numpy()

    external = np.stack(list(predictions.values()))
    main_prediction = base["prediction"].astype(int).to_numpy()
    values = np.empty(len(base), dtype=np.int64)
    votes = np.zeros(len(base), dtype=np.int64)
    for index in range(len(base)):
        unique, counts = np.unique(external[:, index], return_counts=True)
        winner = int(np.argmax(counts))
        values[index] = int(unique[winner])
        votes[index] = int(counts[winner])

    logits = np.load(args.logits, allow_pickle=True)["test_logits"]
    order = np.argsort(logits, axis=1)[:, ::-1]
    raw_margin = logits[np.arange(len(logits)), order[:, 0]] - logits[
        np.arange(len(logits)), order[:, 1]
    ]

    report = base[["path", "prediction"]].copy()
    report["consensus_prediction"] = values
    report["consensus_votes"] = votes
    report["main_margin"] = raw_margin
    for name, prediction in predictions.items():
        report[name] = prediction
    report["consensus_differs"] = (values != main_prediction) & (votes >= 3)
    report.sort_values(
        ["consensus_differs", "consensus_votes", "main_margin"],
        ascending=[False, False, True],
    ).to_csv(args.output_dir / "external_consensus_report.csv", index=False)

    outputs = []
    for label, threshold in [
        ("all", float("inf")),
        ("margin0p25", 0.25),
        ("margin0p5", 0.5),
        ("margin1p0", 1.0),
        ("margin1p5", 1.5),
        ("margin2p0", 2.0),
    ]:
        mask = (votes >= 3) & (values != main_prediction) & (raw_margin <= threshold)
        prediction = main_prediction.copy()
        prediction[mask] = values[mask]
        output = args.output_dir / f"submission_tri_external_consensus_{label}.csv"
        pd.DataFrame({"path": base["path"], "prediction": prediction}).to_csv(
            output, index=False
        )
        outputs.append(
            {
                "candidate": label,
                "overrides": int(mask.sum()),
                "output": str(output),
            }
        )

    summary = pd.DataFrame(outputs)
    summary.to_csv(args.output_dir / "external_consensus_candidates.csv", index=False)
    print(summary.to_string(index=False))
    print("3/4-or-4/4 consensus disagreements:")
    print(
        report.loc[
            report["consensus_differs"],
            ["path", "prediction", "consensus_prediction", "consensus_votes", "main_margin"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
