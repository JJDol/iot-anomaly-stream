"""Shared settings for the whole project. Values only, no logic and no work.

Four groups live here:
    file locations   where the CSV, the saved model, and the metrics report go
    data recipe      normal ranges, fault ranges, how often faults appear
    model options    which columns are features, number of trees, contamination
    service address  host and port, so the API and the stream agree on it

Several files need the same values. 
Keeping them in one place means a change cannot be applied to one file and forgotten in another.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "sensor_train.csv"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = RESULTS_DIR / "models" / "isolation_forest.joblib"
METRICS_PATH = RESULTS_DIR / "metrics.txt"

FEATURES = ["temperature", "humidity", "sound"]

# Typical production-hall readings (not outdoor weather)
TEMP_MEAN = 22.0
TEMP_STD = 1.5
HUMIDITY_MEAN = 45.0
HUMIDITY_STD = 4.0
SOUND_MEAN = 62.0
SOUND_STD = 3.0

# Faulty-item spikes shop-floor staff would notice
ANOMALY_TEMP = (78.0, 95.0)
ANOMALY_HUMIDITY = (80.0, 95.0)
ANOMALY_SOUND = (92.0, 110.0)

ANOMALY_RATE = 0.05
TRAIN_ROWS = 2000
RANDOM_SEED = 42
STREAM_INTERVAL_SEC = 1.0

N_ESTIMATORS = 100
CONTAMINATION = ANOMALY_RATE

API_HOST = "127.0.0.1"
API_PORT = 5001
API_URL = f"http://{API_HOST}:{API_PORT}"
