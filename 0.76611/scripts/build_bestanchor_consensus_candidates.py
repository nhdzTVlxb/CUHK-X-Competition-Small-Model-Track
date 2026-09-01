from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from audit_model_candidates import norm_path
from sequence_postprocess import collect_test_metadata


ROOT = Path(__file__).resolve().parents[1]


def read_diff(anchor: pd.DataFrame, path: Path) -> dict[tuple[int, int], str]:
    frame = pd.read_csv(path)
    out = {}
    for index, (old, new) in enumerate(zip(anchor["prediction"].astype(int), frame["prediction"].astype(int), strict=True), start=1):
        if old != new:
            out[(index, int(new))] = path.name
    return out


def write_candidate(
    outputs: Path,
    anchor: pd.DataFrame,
    selected: pd.DataFrame,
    name: str,
) -> dict[str, object]:
    pred = anchor["prediction"].astype(int).to_numpy().copy()
    for row in selected.itertuples(index=False):
        pred[int(row.row) - 1] = int(row.label)
    output = outputs / f"submission_{name}.csv"
    pd.DataFrame({"path": anchor["path"], "prediction": pred}).to_csv(output, index=False)
    changes = selected.sort_values("row")
    changes.to_csv(outputs / f"{name}_changes.csv", index=False)
    by_date = changes.groupby("date").size().astype(int).to_dict() if not changes.empty else {}
    expected_gain = float(changes["posterior_delta"].sum()) if "posterior_delta" in changes else 0.0
    return {
        "candidate": output.name,
        "path": str(output),
        "changes": int(len(changes)),
        "expected_gain": expected_gain,
        "expected_lb_from_306": (306.0 + expected_gain) / 402.0,
        "changes_by_date": json.dumps(by_date, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--anchor", type=Path, default=ROOT / "outputs" / "submission_lb_preview_model2_v1.csv")
    parser.add_argument("--tag", type=str, default="bestanchor_pubm0_v1")
    args = parser.parse_args()

    outputs = args.root / "outputs"
    anchor = pd.read_csv(args.anchor)
    anchor["path_norm"] = anchor["path"].map(norm_path)

    families = {
        "gated_boost": outputs / f"submission_anchor_gated_allscore_boost_{args.tag}.csv",
        "gated_ultra": outputs / f"submission_anchor_gated_allscore_ultra_{args.tag}.csv",
        "posterior_top12": outputs / f"submission_score_posterior_top12_{args.tag}.csv",
        "posterior_top24": outputs / f"submission_score_posterior_top24_{args.tag}.csv",
        "posterior_top40": outputs / f"submission_score_posterior_top40_{args.tag}.csv",
        "posterior_model2": outputs / f"submission_score_posterior_model2_{args.tag}.csv",
        "posterior_strong": outputs / f"submission_score_posterior_strong_{args.tag}.csv",
    }
    diffs = {name: read_diff(anchor, path) for name, path in families.items() if path.exists()}
    if not diffs:
        raise FileNotFoundError(f"No candidate submissions found for tag={args.tag}")

    rows = []
    for key in sorted(set().union(*[set(value) for value in diffs.values()])):
        sources = [name for name, value in diffs.items() if key in value]
        rows.append({"row": key[0], "label": key[1], "sources": ",".join(sources), "source_count": len(sources)})
    frame = pd.DataFrame(rows)

    test_meta = collect_test_metadata(args.root)
    test_meta["date"] = test_meta["start"].dt.date.astype(str)
    test_meta["path_norm"] = test_meta["path"].map(norm_path)
    date_lookup = test_meta.set_index("path_norm")["date"].to_dict()
    frame["path"] = frame["row"].map(lambda row: anchor.loc[int(row) - 1, "path"])
    frame["date"] = frame["path"].map(lambda value: date_lookup.get(norm_path(value), "unknown"))
    frame["anchor_prediction"] = frame["row"].map(lambda row: int(anchor.loc[int(row) - 1, "prediction"]))

    posterior_path = outputs / f"score_posterior_rows_{args.tag}.csv"
    if posterior_path.exists():
        posterior = pd.read_csv(posterior_path)[["row", "posterior_label", "posterior_delta", "posterior_prob", "anchor_prob", "support_count", "strong_count", "head_count", "supporters"]]
        frame = frame.merge(posterior, left_on=["row", "label"], right_on=["row", "posterior_label"], how="left")
        for column in ["posterior_delta", "posterior_prob", "anchor_prob", "support_count", "strong_count", "head_count"]:
            frame[column] = frame[column].fillna(0)
        frame["supporters"] = frame["supporters"].fillna("")
    else:
        frame["posterior_delta"] = 0.0
        frame["support_count"] = 0
        frame["strong_count"] = 0
        frame["head_count"] = 0
        frame["supporters"] = ""

    frame = frame.sort_values(["source_count", "posterior_delta"], ascending=False)
    frame.to_csv(outputs / f"bestanchor_consensus_pool_{args.tag}.csv", index=False)

    def has_all(row: pd.Series, names: set[str]) -> bool:
        sources = set(str(row["sources"]).split(","))
        return names.issubset(sources)

    rules = {
        f"bestanchor_consensus_top12_or_ultra_{args.tag}": frame[
            frame.apply(lambda row: has_all(row, {"posterior_top12"}) or has_all(row, {"gated_ultra"}), axis=1)
        ],
        f"bestanchor_consensus_gated_and_posterior_{args.tag}": frame[
            frame.apply(lambda row: bool({"gated_boost", "gated_ultra"} & set(str(row["sources"]).split(","))) and bool({"posterior_top24", "posterior_top40", "posterior_model2", "posterior_strong"} & set(str(row["sources"]).split(","))), axis=1)
        ],
        f"bestanchor_consensus_3source_{args.tag}": frame[frame["source_count"].ge(3)],
        f"bestanchor_consensus_4source_{args.tag}": frame[frame["source_count"].ge(4)],
        f"bestanchor_consensus_positive_model2_{args.tag}": frame[
            frame["sources"].str.contains("posterior_model2", regex=False)
            & frame["posterior_delta"].gt(0)
            & frame["support_count"].ge(2)
        ],
    }
    summaries = [write_candidate(outputs, anchor, selected, name) for name, selected in rules.items()]
    report = pd.DataFrame(summaries).sort_values("expected_lb_from_306", ascending=False)
    report.to_csv(outputs / f"bestanchor_consensus_report_{args.tag}.csv", index=False)
    print(report.to_string(index=False))
    print(f"pool={outputs / f'bestanchor_consensus_pool_{args.tag}.csv'}")
    print(f"report={outputs / f'bestanchor_consensus_report_{args.tag}.csv'}")


if __name__ == "__main__":
    main()
