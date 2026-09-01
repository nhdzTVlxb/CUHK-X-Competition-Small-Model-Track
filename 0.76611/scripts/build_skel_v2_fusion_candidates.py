from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x.astype(np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def top_stats(logits: np.ndarray, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prob = softmax(logits, temperature)
    order = np.argsort(prob, axis=1)[:, ::-1]
    top = order[:, 0]
    conf = prob[np.arange(len(prob)), top]
    margin = conf - prob[np.arange(len(prob)), order[:, 1]]
    return top.astype(int), conf, margin


def class_precision(val_logits: np.ndarray, val_y: np.ndarray) -> pd.DataFrame:
    pred, conf, margin = top_stats(val_logits)
    rows = []
    for cls in range(40):
        mask = pred == cls
        support = int(mask.sum())
        correct = int((val_y[mask] == cls).sum()) if support else 0
        truth = int((val_y == cls).sum())
        rows.append(
            {
                "class": cls,
                "pred_support": support,
                "truth_support": truth,
                "precision": correct / support if support else 0.0,
                "recall": correct / truth if truth else 0.0,
                "mean_conf": float(conf[mask].mean()) if support else 0.0,
                "mean_margin": float(margin[mask].mean()) if support else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_submission(base: pd.DataFrame, pred: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": base["path"].astype(str), "prediction": pred.astype(int)}).to_csv(output, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.root

    base = pd.read_csv(root / "outputs" / "submission_yolo_r2p1d.csv")
    yolo_npz = np.load(root / "outputs" / "yolo_r2p1d_logits.npz")
    skel_test_npz = np.load(root / "outputs" / "skel_imu_v2_test_logits.npz")
    skel_val_npz = np.load(root / "outputs" / "skel_imu_v2_val_logits.npz")
    skel_csv = pd.read_csv(root / "outputs" / "submission_skel_imu_v2.csv")

    yolo_pred, yolo_conf, yolo_margin = top_stats(yolo_npz["logits"], temperature=2.0)
    skel_pred, skel_conf, skel_margin = top_stats(skel_test_npz["logits"], temperature=1.4)
    if not np.array_equal(base["prediction"].astype(int).to_numpy(), yolo_pred):
        print("warning: yolo logits argmax differs from base csv")
        yolo_pred = base["prediction"].astype(int).to_numpy()
    if not np.array_equal(skel_csv["prediction"].astype(int).to_numpy(), skel_pred):
        print("warning: skel logits argmax differs from skel csv")
        skel_pred = skel_csv["prediction"].astype(int).to_numpy()

    val_precision = class_precision(skel_val_npz["logits"], skel_val_npz["y"])
    val_precision.to_csv(root / "outputs" / "skel_imu_v2_class_precision.csv", index=False)

    extra = {}
    for name, path in {
        "kunal": root / "kernel_outputs" / "kunal_yolo_v8" / "submission.csv",
        "welsh": root / "external_models" / "welsh_submission" / "submission.csv",
        "old_skel": root / "outputs" / "submission_skel_imu.csv",
    }.items():
        if path.exists():
            extra[name] = pd.read_csv(path)["prediction"].astype(int).to_numpy()

    report = base[["path"]].copy()
    report["yolo_pred"] = yolo_pred
    report["yolo_conf_t2"] = yolo_conf
    report["yolo_margin_t2"] = yolo_margin
    report["skel_pred"] = skel_pred
    report["skel_conf_t14"] = skel_conf
    report["skel_margin_t14"] = skel_margin
    report["skel_val_precision"] = val_precision.set_index("class").loc[skel_pred, "precision"].to_numpy()
    report["skel_val_support"] = val_precision.set_index("class").loc[skel_pred, "pred_support"].to_numpy()
    for name, preds in extra.items():
        report[name] = preds
        report[f"skel_agrees_{name}"] = preds == skel_pred
        report[f"yolo_agrees_{name}"] = preds == yolo_pred
    report["skel_differs"] = skel_pred != yolo_pred
    report["rank_score"] = (
        report["skel_conf_t14"]
        + 0.7 * report["skel_margin_t14"]
        + 0.8 * report["skel_val_precision"]
        - 0.8 * report["yolo_conf_t2"]
        - 0.8 * report["yolo_margin_t2"]
    )

    # Conservative, auditable candidates. These are intentionally sparse; the
    # public visual model remains the anchor because it is much stronger.
    masks: dict[str, pd.Series] = {}
    masks["strict_consensus"] = (
        report["skel_differs"]
        & (report["skel_val_precision"] >= 0.72)
        & (report["skel_val_support"] >= 5)
        & (report["skel_conf_t14"] >= 0.42)
        & (report["skel_margin_t14"] >= 0.16)
        & (report["yolo_margin_t2"] <= 0.32)
        & (report[[c for c in report.columns if c.startswith("skel_agrees_")]].any(axis=1))
    )
    masks["precision_only"] = (
        report["skel_differs"]
        & (report["skel_val_precision"] >= 0.80)
        & (report["skel_val_support"] >= 4)
        & (report["skel_conf_t14"] >= 0.36)
        & (report["yolo_margin_t2"] <= 0.25)
    )
    masks["top8_ranked"] = pd.Series(False, index=report.index)
    ranked_idx = report.loc[
        report["skel_differs"]
        & (report["skel_val_precision"] >= 0.65)
        & (report["skel_conf_t14"] >= 0.35)
        & (report["yolo_margin_t2"] <= 0.35)
    ].sort_values("rank_score", ascending=False).head(8).index
    masks["top8_ranked"].loc[ranked_idx] = True

    summary = []
    for name, mask in masks.items():
        pred = yolo_pred.copy()
        pred[mask.to_numpy()] = skel_pred[mask.to_numpy()]
        out = root / "outputs" / f"submission_yolo_skel_v2_{name}.csv"
        write_submission(base, pred, out)
        report[f"override_{name}"] = mask
        summary.append({"candidate": name, "overrides": int(mask.sum()), "output": str(out)})

    report.sort_values(["rank_score"], ascending=False).to_csv(root / "outputs" / "skel_imu_v2_fusion_report.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(root / "outputs" / "skel_imu_v2_fusion_candidates.csv", index=False)
    print(summary_df.to_string(index=False))
    print("class precision:")
    print(val_precision.sort_values("precision", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
