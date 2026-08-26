"""Serve the trained model over HTTP.

This is the service that holds it and puts it behind an address other programs can call.

Unlike train_model.py, this does not finish. It starts and then waits. 
When a reading arrives it answers with a score, then waits again. 
The model is loaded once at start-up, not on every request, so answers stay fast.

It reports an anomaly; it does not act on one.

Endpoints:
    POST /predict   one reading  -> anomaly score and flag
    GET  /health    is the service up and is a model loaded
    GET  /metrics   simple counters for monitoring
"""

from __future__ import annotations

import time
from threading import Lock

import joblib
import numpy as np
from flask import Flask, jsonify, request

from src import config

app = Flask(__name__)

_bundle = None
_model = None
_load_error: str | None = None

_stats_lock = Lock()
_stats = {
    "requests": 0,
    "anomalies": 0,
    "errors": 0,
    "last_score": None,
    "last_flag": None,
    "started_at": time.time(),
}


def load_model() -> None:
    """Read the saved model from disk once, at start-up."""
    global _bundle, _model, _load_error
    try:
        _bundle = joblib.load(config.MODEL_PATH)
        _model = _bundle["model"]
        _load_error = None
    except FileNotFoundError:
        _load_error = f"No model at {config.MODEL_PATH}. Run: python -m src.train_model"
    except Exception as exc:
        _load_error = f"Could not load model: {exc}"


def read_features(payload: dict) -> np.ndarray:
    """Turn one JSON reading into the row the model expects.

    Raises ValueError when a sensor value is missing or not a number.
    """
    values = []
    for name in config.FEATURES:
        if name not in payload:
            raise ValueError(f"missing field: {name}")
        try:
            values.append(float(payload[name]))
        except (TypeError, ValueError):
            raise ValueError(f"field '{name}' must be a number")
    return np.array([values], dtype=float)


@app.post("/predict")
def predict():
    if _model is None:
        with _stats_lock:
            _stats["errors"] += 1
        return jsonify({"error": _load_error or "model not loaded"}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        with _stats_lock:
            _stats["errors"] += 1
        return jsonify({"error": "body must be a JSON object"}), 400

    try:
        features = read_features(payload)
    except ValueError as exc:
        with _stats_lock:
            _stats["errors"] += 1
        return jsonify({"error": str(exc)}), 400

    started = time.perf_counter()
    score = float(-_model.score_samples(features)[0])
    is_anomaly = bool(_model.predict(features)[0] == -1)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    with _stats_lock:
        _stats["requests"] += 1
        _stats["anomalies"] += int(is_anomaly)
        _stats["last_score"] = round(score, 4)
        _stats["last_flag"] = "anomaly" if is_anomaly else "normal"

    return jsonify(
        {
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "latency_ms": latency_ms,
            "model": "IsolationForest",
        }
    )


@app.get("/health")
def health():
    if _model is None:
        return jsonify({"status": "unhealthy", "reason": _load_error}), 503
    return jsonify(
        {
            "status": "ok",
            "model_loaded": True,
            "features": config.FEATURES,
            "uptime_seconds": round(time.time() - _stats["started_at"], 1),
        }
    )


@app.get("/metrics")
def metrics():
    with _stats_lock:
        snapshot = dict(_stats)
    served = snapshot["requests"]
    snapshot["anomaly_rate"] = round(snapshot["anomalies"] / served, 4) if served else None
    snapshot["uptime_seconds"] = round(time.time() - snapshot.pop("started_at"), 1)
    return jsonify(snapshot)


load_model()

if __name__ == "__main__":
    if _load_error:
        print(_load_error)
    print(f"Serving on {config.API_URL}")
    app.run(host=config.API_HOST, port=config.API_PORT)
