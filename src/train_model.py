"""Learn from the saved readings, then save the finished model.

This is a batch job: 
it opens the CSV, reads all rows at once, trains Isolation Forest a single time, and exits. 
It does not stay running.

Three things happen here. 
1) The model learns the shape of normal readings,
2) its result is measured, 
and 3) it is written to disk as a joblib file. 
That file is the model. 
app.py later loads it and serves it.

The is_anomaly column is not used for training. 
Isolation Forest only sees temperature, humidity, and sound. 
The column is kept as an answer key.

sklearn's score is lower when a reading is more unusual. 
This project flips the sign, so a higher score always means more anomalous.
"""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from src import config


def load_training_table(path: Path = config.TRAIN_CSV) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    X = np.array([[float(row[name]) for name in config.FEATURES] for row in rows], dtype=float)
    y = np.array([int(row["is_anomaly"]) for row in rows], dtype=int)
    return X, y


def anomaly_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher means more anomalous (opposite of sklearn's score_samples)."""
    return -model.score_samples(X)


def train_model(X: np.ndarray) -> IsolationForest:
    model = IsolationForest(
        n_estimators=config.N_ESTIMATORS,
        contamination=config.CONTAMINATION,
        random_state=config.RANDOM_SEED,
    )
    model.fit(X)
    return model


def evaluate(model: IsolationForest, X: np.ndarray, y: np.ndarray) -> str:
    scores = anomaly_score(model, X)
    flagged = (model.predict(X) == -1).astype(int)

    n = len(y)
    n_true = int(y.sum())
    n_flagged = int(flagged.sum())
    tn, fp, fn, tp = confusion_matrix(y, flagged).ravel()

    lines = [
        "Isolation Forest — simulated factory sensors",
        f"Rows: {n}",
        f"Features: {', '.join(config.FEATURES)}",
        f"Trees: {config.N_ESTIMATORS}",
        f"Contamination: {config.CONTAMINATION}",
        "",
        "Training does not use is_anomaly. That column is only an answer key.",
        "",
        f"Simulator faults (is_anomaly=1): {n_true} ({100 * n_true / n:.1f}%)",
        f"Model flagged (predict=-1):     {n_flagged} ({100 * n_flagged / n:.1f}%)",
        "",
        f"True positives:  {tp}",
        f"False positives: {fp}",
        f"False negatives: {fn}",
        f"True negatives:  {tn}",
        "",
        "Mean anomaly score (higher = more odd):",
        f"  simulator-normal rows:  {scores[y == 0].mean():.4f}",
        f"  simulator-fault rows:   {scores[y == 1].mean():.4f}",
        "",
        classification_report(
            y,
            flagged,
            target_names=["normal", "anomaly"],
            digits=3,
        ),
        "Sanity checks (fresh readings, not in the CSV):",
    ]

    checks = [
        ("typical hall", np.array([[22.0, 45.0, 62.0]])),
        ("overheating spike", np.array([[92.0, 84.0, 104.0]])),
    ]
    for label, sample in checks:
        score = float(anomaly_score(model, sample)[0])
        pred = "anomaly" if model.predict(sample)[0] == -1 else "normal"
        lines.append(f"  {label}: score={score:.4f}  flag={pred}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    X, y = load_training_table()
    model = train_model(X)
    report = evaluate(model, X, y)

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": config.FEATURES,
            "score": "higher means more anomalous; score = -IsolationForest.score_samples",
        },
        config.MODEL_PATH,
    )
    config.METRICS_PATH.write_text(report)

    print(report)
    print(f"Saved model:   {config.MODEL_PATH}")
    print(f"Saved metrics: {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
