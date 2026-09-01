from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
N_CLASSES = 40
RADAR_COLUMNS = ("x", "y", "z", "v", "snr", "noise")


def read_clip(folder: Path) -> np.ndarray:
    parts = []
    for path in sorted(folder.glob("*.csv")):
        try:
            frame = pd.read_csv(path)
            if all(column in frame for column in RADAR_COLUMNS):
                values = frame[list(RADAR_COLUMNS)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                if len(values):
                    parts.append(values)
        except Exception:
            continue
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, len(RADAR_COLUMNS)), dtype=np.float32)


def summarize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(60 + 32 * 12 + 3, dtype=np.float32)
    base = []
    for column in range(values.shape[1]):
        vector = values[:, column]
        base.extend(
            [
                float(vector.mean()),
                float(vector.std()),
                float(np.min(vector)),
                float(np.max(vector)),
                float(np.quantile(vector, 0.1)),
                float(np.quantile(vector, 0.25)),
                float(np.quantile(vector, 0.5)),
                float(np.quantile(vector, 0.75)),
                float(np.quantile(vector, 0.9)),
                float(np.mean(np.abs(vector))),
            ]
        )
    temporal = []
    for chunk in np.array_split(values, 32):
        if len(chunk):
            temporal.append(np.concatenate([chunk.mean(axis=0), chunk.std(axis=0)]))
        else:
            temporal.append(np.zeros(12, dtype=np.float32))
    return np.concatenate(
        [
            np.asarray(base, dtype=np.float32),
            np.asarray([1.0, np.log1p(len(values)), np.log1p(np.unique(values[:, 0]).size)], dtype=np.float32),
            np.asarray(temporal, dtype=np.float32).reshape(-1),
        ]
    )


def clip_records(root: Path, split: str) -> list[dict[str, object]]:
    if split == "train":
        base = root / "data" / "Training" / "data" / "HAR" / "data"
        records = []
        for action in sorted((base / "Skeleton").iterdir()):
            if not action.is_dir():
                continue
            label = int(action.name.split("_", 1)[0])
            for user in sorted(p for p in action.iterdir() if p.is_dir()):
                for trial in sorted(p for p in user.iterdir() if p.is_dir()):
                    records.append(
                        {
                            "clip_id": f"{action.name}__{user.name}__{trial.name}",
                            "path": f"{action.name}/{user.name}/{trial.name}",
                            "folder": base / "Radar" / action.name / user.name / trial.name,
                            "label": label,
                            "user": user.name,
                        }
                    )
        return records

    test_root = root / "data" / "Testing" / "data" / "small_model_track_test"
    test_csv = root / "data" / "Testing" / "test.csv"
    records = []
    for raw_path in pd.read_csv(test_csv)["path"].astype(str):
        clip_id = raw_path.strip("/\\").split("/")[-1]
        records.append(
            {
                "clip_id": clip_id,
                "path": raw_path,
                "folder": test_root / clip_id / "Radar",
            }
        )
    return records


def build_features(root: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    records = clip_records(root, split)
    features = []
    for record in tqdm(records, desc=f"radar {split}"):
        values = read_clip(Path(record["folder"]))
        features.append(summarize(values))
    frame = pd.DataFrame(records)
    return np.asarray(features, dtype=np.float32), frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--C", type=float, default=0.01)
    args = parser.parse_args()

    x_train, train = build_features(args.root, "train")
    x_test, test = build_features(args.root, "test")
    y = train["label"].astype(int).to_numpy()
    groups = train["user"].astype(str).to_numpy()
    model_factory = lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.C, max_iter=5000, solver="lbfgs", multi_class="multinomial"),
    )
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros((len(train), N_CLASSES), dtype=np.float32)
    fold_scores = []
    for fold, (tr, va) in enumerate(splitter.split(x_train, y, groups=groups), start=1):
        model = model_factory()
        model.fit(x_train[tr], y[tr])
        oof[va] = model.predict_proba(x_train[va]).astype(np.float32)
        score = float((oof[va].argmax(1) == y[va]).mean())
        fold_scores.append(score)
        print(f"fold={fold} acc={score:.4f}")
    print(f"mean_fold_acc={np.mean(fold_scores):.4f} oof_acc={(oof.argmax(1) == y).mean():.4f} dim={x_train.shape[1]}")

    model = model_factory()
    model.fit(x_train, y)
    test_proba = model.predict_proba(x_test).astype(np.float32)
    out = args.root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "radar_stats_train_logits.npz", logits=oof, y=y, clip_id=train["clip_id"].astype(str).to_numpy(), user=groups)
    np.savez_compressed(out / "radar_stats_test_logits.npz", logits=test_proba, path=test["path"].astype(str).to_numpy(), clip_id=test["clip_id"].astype(str).to_numpy())
    pd.DataFrame({"path": test["path"].astype(str), "prediction": test_proba.argmax(1).astype(int)}).to_csv(out / "submission_radar_stats.csv", index=False)
    joblib.dump(model, out / "radar_stats_model.joblib")
    np.savez_compressed(out / "radar_stats_features.npz", train=x_train, test=x_test)


if __name__ == "__main__":
    main()
