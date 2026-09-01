from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def margins(logits: np.ndarray) -> np.ndarray:
    order = np.argsort(logits, axis=1)[:, ::-1]
    return logits[np.arange(len(logits)), order[:, 0]] - logits[
        np.arange(len(logits)), order[:, 1]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--views",
        type=Path,
        default=ROOT / "outputs" / "yolo_view_tta_logits.npz",
    )
    parser.add_argument(
        "--tri",
        type=Path,
        default=ROOT / "outputs" / "tri_best_logits.npz",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "outputs" / "submission_tri_best.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
    )
    args = parser.parse_args()

    view_data = np.load(args.views, allow_pickle=True)
    tri_data = np.load(args.tri, allow_pickle=True)
    base_submission = pd.read_csv(args.base)
    base_prediction = base_submission["prediction"].astype(int).to_numpy()

    val_tri = tri_data["val_logits"]
    test_tri = tri_data["test_logits"]
    val_view = view_data["val_model0_zoom_in_base"]
    test_view = view_data["test_model0_zoom_in_base"]
    val_y = tri_data["val_y"]

    val_tri_prediction = val_tri.argmax(axis=1)
    test_tri_prediction = test_tri.argmax(axis=1)
    val_view_prediction = val_view.argmax(axis=1)
    test_view_prediction = test_view.argmax(axis=1)
    val_tri_margin = margins(val_tri)
    test_tri_margin = margins(test_tri)
    val_view_margin = margins(val_view)
    test_view_margin = margins(test_view)

    rules = {
        "b2_v3": (2.0, 3.0),
        "b2_v2": (2.0, 2.0),
        "b3_v2": (3.0, 2.0),
        "b2p9_v1p8": (2.9, 1.8),
    }
    rows: list[dict[str, object]] = []
    for name, (base_limit, view_limit) in rules.items():
        val_mask = (
            (val_tri_prediction != val_view_prediction)
            & (val_tri_margin <= base_limit)
            & (val_view_margin >= view_limit)
        )
        test_mask = (
            (test_tri_prediction != test_view_prediction)
            & (test_tri_margin <= base_limit)
            & (test_view_margin >= view_limit)
        )
        val_prediction = val_tri_prediction.copy()
        val_prediction[val_mask] = val_view_prediction[val_mask]
        test_prediction = test_tri_prediction.copy()
        test_prediction[test_mask] = test_view_prediction[test_mask]
        output = args.output_dir / f"submission_tri_view_m0zin_{name}.csv"
        pd.DataFrame(
            {"path": base_submission["path"], "prediction": test_prediction}
        ).to_csv(output, index=False)
        rows.append(
            {
                "candidate": name,
                "base_limit": base_limit,
                "view_limit": view_limit,
                "val_overrides": int(val_mask.sum()),
                "val_accuracy": float(np.mean(val_prediction == val_y)),
                "test_overrides": int(test_mask.sum()),
                "output": str(output),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "view_tta_candidates.csv", index=False)

    report = pd.DataFrame(
        {
            "path": base_submission["path"],
            "base": base_prediction,
            "view_prediction": test_view_prediction,
            "base_margin": test_tri_margin,
            "view_margin": test_view_margin,
            "changed_by_view": test_tri_prediction != test_view_prediction,
        }
    )
    report.sort_values(["changed_by_view", "base_margin"], ascending=[False, True]).to_csv(
        args.output_dir / "view_tta_test_report.csv", index=False
    )
    print(summary.to_string(index=False))
    print("test disagreements:")
    print(
        report.loc[
            report["changed_by_view"],
            ["path", "base", "view_prediction", "base_margin", "view_margin"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
