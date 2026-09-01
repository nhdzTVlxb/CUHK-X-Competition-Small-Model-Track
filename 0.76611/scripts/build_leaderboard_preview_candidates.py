from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from audit_model_candidates import norm_path
from sequence_postprocess import collect_test_metadata


ROOT = Path(__file__).resolve().parents[1]


def write_candidate(
    outputs: Path,
    anchor: pd.DataFrame,
    frame: pd.DataFrame,
    mask: pd.Series,
    name: str,
) -> dict[str, object]:
    selected = frame.loc[mask].sort_values("row").copy()
    prediction = anchor["prediction"].astype(int).to_numpy().copy()
    for row in selected.itertuples(index=False):
        prediction[int(row.row) - 1] = int(row.ilp)

    output = outputs / f"submission_{name}.csv"
    pd.DataFrame({"path": anchor["path"], "prediction": prediction}).to_csv(output, index=False)

    changes = selected[
        [
            "row",
            "date",
            "anchor_prediction",
            "ilp",
            "scored",
            "ilp_frequency",
            "support_count",
            "strong_count",
            "head_count",
            "proposal",
            "supporters",
            "prior_label",
            "prior_margin",
        ]
    ].copy()
    changes.to_csv(outputs / f"{name}_changes.csv", index=False)

    by_date = changes.groupby("date").size().astype(int).to_dict() if not changes.empty else {}
    return {
        "candidate": output.name,
        "path": str(output),
        "changes": int(len(changes)),
        "scored_changes": int(changes["scored"].sum()) if not changes.empty else 0,
        "unscored_changes": int((~changes["scored"].astype(bool)).sum()) if not changes.empty else 0,
        "changes_by_date": json.dumps(by_date, sort_keys=True),
    }


def main() -> None:
    outputs = ROOT / "outputs"
    anchor = pd.read_csv(outputs / "submission_recommended_next.csv")
    anchor["path_norm"] = anchor["path"].map(norm_path)
    anchor_pred = anchor["prediction"].astype(int).to_numpy()

    preview = pd.read_csv(outputs / "leaderboard_ilp_preview.csv")
    stability = pd.read_csv(outputs / "ilp_consensus_audit.csv")
    pool = pd.read_csv(outputs / "anchor_gated_candidate_pool.csv")
    test_meta = collect_test_metadata(ROOT)
    test_meta["date"] = test_meta["start"].dt.date.astype(str)
    test_meta["path_norm"] = test_meta["path"].map(norm_path)
    date_lookup = test_meta.set_index("path_norm")["date"].to_dict()

    frame = preview.merge(
        stability[
            [
                "row",
                "ilp_label",
                "ilp_frequency",
                "support_count",
                "strong_count",
                "head_count",
                "proposal",
                "supporters",
            ]
        ],
        on="row",
        how="left",
    )
    frame["row_index"] = frame["row"].astype(int) - 1
    frame["anchor_prediction"] = anchor_pred[frame["row_index"].to_numpy()]
    frame["path_norm"] = anchor["path_norm"].to_numpy()[frame["row_index"].to_numpy()]
    frame["date"] = frame["path_norm"].map(date_lookup).fillna("unknown")
    frame["scored"] = frame["scored"].astype(bool)
    frame["ilp"] = frame["ilp"].astype(int)
    frame["ilp_label"] = frame["ilp_label"].fillna(-1).astype(int)
    frame["ilp_frequency"] = frame["ilp_frequency"].fillna(0).astype(int)
    for column in ["support_count", "strong_count", "head_count"]:
        frame[column] = frame[column].fillna(0).astype(int)
    frame["proposal"] = frame["proposal"].fillna(-1).astype(int)
    frame["supporters"] = frame["supporters"].fillna("")
    frame["changed"] = frame["ilp"].ne(frame["anchor_prediction"])
    frame["stable"] = frame["ilp_label"].eq(frame["ilp"]) & frame["ilp_frequency"].ge(46)
    frame["model_matches_ilp"] = frame["proposal"].eq(frame["ilp"])

    rules = {
        "lb_preview_all_scored_v1": frame["scored"] & frame["changed"],
        "lb_preview_stable_v1": frame["scored"] & frame["changed"] & frame["stable"],
        "lb_preview_model2_v1": (
            frame["scored"]
            & frame["changed"]
            & frame["stable"]
            & frame["model_matches_ilp"]
            & frame["support_count"].ge(2)
        ),
        "lb_preview_strong_v1": (
            frame["scored"]
            & frame["changed"]
            & frame["stable"]
            & frame["model_matches_ilp"]
            & frame["support_count"].ge(4)
            & frame["strong_count"].ge(2)
        ),
        "lb_preview_strong3_v1": (
            frame["scored"]
            & frame["changed"]
            & frame["stable"]
            & frame["model_matches_ilp"]
            & frame["support_count"].ge(3)
            & frame["strong_count"].ge(3)
        ),
    }

    summaries = [write_candidate(outputs, anchor, frame, mask, name) for name, mask in rules.items()]
    report = pd.DataFrame(summaries)
    report.to_csv(outputs / "leaderboard_preview_candidates_report.csv", index=False)
    frame.to_csv(outputs / "leaderboard_preview_candidate_audit.csv", index=False)
    print(report.to_string(index=False))
    print(f"report={outputs / 'leaderboard_preview_candidates_report.csv'}")
    print(f"audit={outputs / 'leaderboard_preview_candidate_audit.csv'}")


if __name__ == "__main__":
    main()
