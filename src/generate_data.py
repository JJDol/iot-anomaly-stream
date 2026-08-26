"""Create the fictional pile of sensor readings and save it as a CSV.

the values are invented here: 
mostly normal readings of temperature, humidity, and sound, plus a small share of faults.
The result is the history that train_model.py learns from.

The seed in config.py is fixed, so every run produces the same file. 
Anyone who clones the repository gets the same data and therefore the same model.

"""

from __future__ import annotations

import argparse
import csv
import random

from src import config


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sample_normal(rng: random.Random) -> dict[str, float]:
    return {
        "temperature": round(_clip(rng.gauss(config.TEMP_MEAN, config.TEMP_STD), 15.0, 30.0), 2),
        "humidity": round(_clip(rng.gauss(config.HUMIDITY_MEAN, config.HUMIDITY_STD), 25.0, 70.0), 2),
        "sound": round(_clip(rng.gauss(config.SOUND_MEAN, config.SOUND_STD), 45.0, 80.0), 2),
    }


def sample_anomaly(rng: random.Random) -> dict[str, float]:
    return {
        "temperature": round(rng.uniform(*config.ANOMALY_TEMP), 2),
        "humidity": round(rng.uniform(*config.ANOMALY_HUMIDITY), 2),
        "sound": round(rng.uniform(*config.ANOMALY_SOUND), 2),
    }


def generate_reading(rng: random.Random, anomaly_rate: float = config.ANOMALY_RATE) -> dict:
    is_anomaly = rng.random() < anomaly_rate
    reading = sample_anomaly(rng) if is_anomaly else sample_normal(rng)
    reading["is_anomaly"] = int(is_anomaly)
    return reading


def generate_dataset(
    n: int = config.TRAIN_ROWS,
    anomaly_rate: float = config.ANOMALY_RATE,
    seed: int = config.RANDOM_SEED,
) -> list[dict]:
    rng = random.Random(seed)
    return [generate_reading(rng, anomaly_rate) for _ in range(n)]


def save_csv(rows: list[dict], path=config.TRAIN_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["temperature", "humidity", "sound", "is_anomaly"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fictional sensor data")
    parser.add_argument("--rows", type=int, default=config.TRAIN_ROWS)
    args = parser.parse_args()

    rows = generate_dataset(n=args.rows)
    save_csv(rows)

    n_anom = sum(row["is_anomaly"] for row in rows)
    print(f"Wrote {len(rows)} rows to {config.TRAIN_CSV}")
    print(f"Anomalies: {n_anom} ({100 * n_anom / len(rows):.1f}%)")
    print("Example normal:", next(row for row in rows if row["is_anomaly"] == 0))
    print("Example anomaly:", next(row for row in rows if row["is_anomaly"] == 1))


if __name__ == "__main__":
    main()
