from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from audit_model_candidates import norm_path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_N = 201


def parse_top(value: object) -> tuple[int, int]:
    try:
        parsed = ast.literal_eval(str(value))
        if not parsed:
            return -1, 0
        label, count = parsed[0]
        return int(label), int(count)
    except (ValueError, SyntaxError, TypeError):
        return -1, 0


def main() -> None:
    root = ROOT
    outputs = root / "outputs"
    anchor_path = outputs / "submission_recommended_next.csv"
    anchor = pd.read_csv(anchor_path)
    anchor["path_norm"] = anchor["path"].map(norm_path)

    stability = pd.read_csv(outputs / "leaderboard_ilp_stability.csv")
    preview = pd.read_csv(outputs / "leaderboard_ilp_preview.csv")
    pool = pd.read_csv(outputs / "anchor_gated_candidate_pool.csv")
    stability[["ilp_label", "ilp_frequency"]] = stability["top"].apply(
        lambda value: pd.Series(parse_top(value))
    )
    frame = (
        stability.merge(
            preview[["row", "ilp"]],
            on="row",
            how="left",
        )
        .merge(
            pool[
                [
                    "row",
                    "proposal",
                    "support_count",
                    "strong_count",
                    "head_count",
                    "supporters",
                ]
            ],
            on="row",
            how="left",
        )
    )
    frame["row_index"] = frame["row"].astype(int) - 1
    frame["anchor_prediction"] = anchor.loc[frame["row_index"], "prediction"].to_numpy()
    frame["is_public201"] = frame["row"].le(PUBLIC_N)
    frame["model_matches_ilp"] = frame["proposal"].eq(frame["ilp_label"])
    frame["ilp_is_stable"] = frame["nonunknown"].ge(49) & frame["ilp_frequency"].ge(40)

    rules = {
        "ilp_stable_public": (
            frame["is_public201"]
            & frame["ilp_is_stable"]
            & frame["ilp_label"].ge(0)
            & frame["ilp_label"].ne(frame["anchor_prediction"])
        ),
        "ilp_model2_public": (
            frame["is_public201"]
            & frame["ilp_is_stable"]
            & frame["model_matches_ilp"]
            & frame["support_count"].ge(2)
            & frame["ilp_label"].ge(0)
            & frame["ilp_label"].ne(frame["anchor_prediction"])
        ),
        "ilp_strong_public": (
            frame["is_public201"]
            & frame["ilp_is_stable"]
            & frame["model_matches_ilp"]
            & frame["support_count"].ge(4)
            & frame["strong_count"].ge(2)
            & frame["ilp_label"].ge(0)
            & frame["ilp_label"].ne(frame["anchor_prediction"])
        ),
        "ilp_strong3_public": (
            frame["is_public201"]
            & frame["ilp_is_stable"]
            & frame["model_matches_ilp"]
            & frame["support_count"].ge(3)
            & frame["strong_count"].ge(3)
            & frame["ilp_label"].ge(0)
            & frame["ilp_label"].ne(frame["anchor_prediction"])
        ),
    }

    summaries = []
    for name, mask in rules.items():
        prediction = anchor["prediction"].astype(int).to_numpy().copy()
        selected = frame.loc[mask].sort_values("row")
        for row in selected.itertuples(index=False):
            prediction[int(row.row) - 1] = int(row.ilp_label)
        output = outputs / f"submission_{name}_v1.csv"
        pd.DataFrame({"path": anchor["path"], "prediction": prediction}).to_csv(
            output, index=False
        )
        changes = selected[
            [
                "row",
                "date",
                "anchor_prediction",
                "ilp_label",
                "ilp_frequency",
                "support_count",
                "strong_count",
                "proposal",
                "supporters",
            ]
        ].copy()
        changes.to_csv(outputs / f"{name}_changes.csv", index=False)
        summaries.append(
            {
                "candidate": output.name,
                "path": str(output),
                "public_changes": int(mask.sum()),
                "hidden_changes": 0,
            }
        )

    frame.to_csv(outputs / "ilp_consensus_audit.csv", index=False)
    report = pd.DataFrame(summaries)
    report.to_csv(outputs / "ilp_consensus_candidates_report.csv", index=False)
    print(report.to_string(index=False))
    print("Selected rows:")
    print(
        frame.loc[
            frame["is_public201"]
            & frame["ilp_label"].ne(frame["anchor_prediction"])
            & frame["ilp_is_stable"],
            [
                "row",
                "date",
                "anchor_prediction",
                "ilp_label",
                "ilp_frequency",
                "support_count",
                "strong_count",
                "proposal",
                "supporters",
            ],
        ]
        .sort_values("row")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
