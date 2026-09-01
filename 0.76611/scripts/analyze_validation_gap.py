from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sequence_postprocess import collect_test_metadata, collect_train_metadata
from sequence_template_time_v2 import add_time_segments

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_N = 201


KNOWN_LB = {
    "submission_recommended_next.csv": 0.74129,
    "submission_blend090_tri_meta_sequence.csv": 0.73134,
    "submission_tri_expert_cvstack_sequence.csv": 0.72636,
}


LOCAL_CV = {
    "submission_recommended_next.csv": 0.9418238993710691,
    "submission_blend090_tri_meta_sequence.csv": 0.959740702831798,
    "submission_tri_expert_cvstack_sequence.csv": 0.9593995223473218,
    "submission_blend090_tri_meta_raw.csv": 0.9556465369839645,
}


def infer_public_correct(score: float) -> int:
    return int(round(score * PUBLIC_N))


def load_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "prediction" not in frame:
        raise ValueError(f"Missing prediction column: {path}")
    frame["prediction"] = frame["prediction"].astype(int)
    return frame


def main() -> None:
    outputs = ROOT / "outputs"
    anchor_name = "submission_recommended_next.csv"
    anchor = load_submission(outputs / anchor_name)

    submission_files = sorted(outputs.glob("submission_*.csv"))
    rows = []
    change_rows = []
    distribution_rows = []
    for path in submission_files:
        try:
            frame = load_submission(path)
        except Exception:
            continue
        if len(frame) != len(anchor):
            continue

        diff = frame["prediction"].to_numpy() != anchor["prediction"].to_numpy()
        name = path.name
        lb = KNOWN_LB.get(name)
        local = LOCAL_CV.get(name)
        rows.append(
            {
                "file": name,
                "known_lb": lb,
                "public_correct_if_known": infer_public_correct(lb) if lb is not None else np.nan,
                "local_cv": local,
                "cv_minus_lb": local - lb if lb is not None and local is not None else np.nan,
                "diff_vs_anchor_all": int(diff.sum()),
                "diff_vs_anchor_public201": int(diff[:PUBLIC_N].sum()),
                "diff_vs_anchor_hidden204": int(diff[PUBLIC_N:].sum()),
            }
        )

        public = frame.iloc[:PUBLIC_N]
        for label, count in public["prediction"].value_counts().sort_index().items():
            distribution_rows.append(
                {
                    "file": name,
                    "split": "public201",
                    "prediction": int(label),
                    "count": int(count),
                }
            )
        for idx in np.flatnonzero(diff[:PUBLIC_N]):
            change_rows.append(
                {
                    "file": name,
                    "row": int(idx + 1),
                    "path": anchor.loc[idx, "path"],
                    "anchor_prediction": int(anchor.loc[idx, "prediction"]),
                    "candidate_prediction": int(frame.loc[idx, "prediction"]),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["known_lb", "local_cv", "diff_vs_anchor_public201"],
        ascending=[False, False, True],
        na_position="last",
    )
    changes = pd.DataFrame(change_rows)
    distributions = pd.DataFrame(distribution_rows)

    train = add_time_segments(collect_train_metadata(ROOT))
    test = collect_test_metadata(ROOT)
    train["date"] = train["start"].dt.date.astype(str)
    test["date"] = test["start"].dt.date.astype(str)
    test["is_public201"] = test["clip_id"].str.extract(r"SM_test_(\d+)")[0].astype(int).le(PUBLIC_N)

    date_summary = pd.concat(
        [
            train.groupby("date").size().rename("train_count"),
            test.groupby("date").size().rename("test_count"),
            test.groupby(["date", "is_public201"]).size().unstack(fill_value=0).rename(
                columns={False: "test_hidden204", True: "test_public201"}
            ),
        ],
        axis=1,
    ).fillna(0).astype(int)

    summary.to_csv(outputs / "validation_gap_submission_summary.csv", index=False)
    changes.to_csv(outputs / "validation_gap_public201_changes.csv", index=False)
    distributions.to_csv(outputs / "validation_gap_public201_distribution.csv", index=False)
    date_summary.to_csv(outputs / "validation_gap_date_summary.csv")

    print("Known LB uses public_n=201:")
    for name, score in KNOWN_LB.items():
        print(f"  {name}: {infer_public_correct(score)}/{PUBLIC_N} = {score:.5f}")
    print("\nTop summary:")
    print(summary.head(20).to_string(index=False))
    print("\nDate summary:")
    print(date_summary.to_string())
    print("\nWrote:")
    print(outputs / "validation_gap_submission_summary.csv")
    print(outputs / "validation_gap_public201_changes.csv")
    print(outputs / "validation_gap_public201_distribution.csv")
    print(outputs / "validation_gap_date_summary.csv")


if __name__ == "__main__":
    main()
