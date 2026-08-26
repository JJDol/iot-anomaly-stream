"""Play the part of the sensors on the production line.

This is a client, not a service. It produces readings and hands them onward.
it never waits to be called. In the architecture it sits where the real sensors would be.

The values come from generate_data.py, but no seed is fixed here. 
Thus each run produces different readings. 
They are new to the model, which is what makes the demo a genuine test rather than a replay of the training file.

This file controls timing only: one reading, then a pause, then the next. 
Thus, invented numbers behave like a live feed instead of a finished table.

Each reading is sent to the API and the returned score is printed, so the
whole chain can be watched live. Only the three sensor values are sent. Real
sensors do not know whether an item is faulty, so is_anomaly stays here and is
shown next to the answer only to make the demo easy to follow.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Iterator

import requests

from src import config
from src.generate_data import generate_reading

PREDICT_URL = f"{config.API_URL}/predict"


def iter_stream(
    interval: float = config.STREAM_INTERVAL_SEC,
    anomaly_rate: float = config.ANOMALY_RATE,
    seed: int | None = None,
) -> Iterator[dict]:
    rng = random.Random(seed)
    while True:
        yield generate_reading(rng, anomaly_rate)
        time.sleep(interval)


def send_reading(session: requests.Session, reading: dict, timeout: float = 2.0) -> dict:
    """Send the three sensor values to the API and return its answer."""
    payload = {name: reading[name] for name in config.FEATURES}
    response = session.post(PREDICT_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def format_line(reading: dict, answer: dict) -> str:
    sensors = (
        f"{reading['temperature']:6.2f}C "
        f"{reading['humidity']:6.2f}% "
        f"{reading['sound']:6.2f}dB"
    )
    verdict = "ANOMALY" if answer["is_anomaly"] else "normal "
    expected = "fault" if reading["is_anomaly"] else "ok"
    return f"{sensors} -> score {answer['anomaly_score']:.4f}  {verdict}  (simulated: {expected})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream sensor readings into the API")
    parser.add_argument("--interval", type=float, default=config.STREAM_INTERVAL_SEC)
    parser.add_argument("--limit", type=int, default=None, help="stop after this many readings")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="print readings without calling the API",
    )
    args = parser.parse_args()

    if args.print_only:
        print("Streaming sensor readings. Stop with Ctrl+C.", flush=True)
        try:
            for count, reading in enumerate(iter_stream(interval=args.interval), start=1):
                print(reading, flush=True)
                if args.limit and count >= args.limit:
                    break
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
        return

    print(f"Sending readings to {PREDICT_URL}. Stop with Ctrl+C.", flush=True)
    sent = 0
    flagged = 0

    with requests.Session() as session:
        try:
            for reading in iter_stream(interval=args.interval):
                try:
                    answer = send_reading(session, reading)
                except requests.exceptions.ConnectionError:
                    print("API unreachable. Start it with: python -m src.app", flush=True)
                    break
                except requests.exceptions.RequestException as exc:
                    print(f"Request failed: {exc}", flush=True)
                    continue

                sent += 1
                flagged += int(answer["is_anomaly"])
                print(format_line(reading, answer), flush=True)

                if args.limit and sent >= args.limit:
                    break
        except KeyboardInterrupt:
            print("", flush=True)

    print(f"\nSent {sent} readings, {flagged} flagged as anomalies.", flush=True)


if __name__ == "__main__":
    main()
