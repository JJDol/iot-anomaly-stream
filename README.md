# Factory sensor anomaly service (IU DLBDSMTP01)

This is the code for my IU course **DLBDSMTP01 – Project: From Model to Production**, Task 1 (stream processing).

The story is a plant that makes wind-turbine parts. Sensors measure **temperature**, **humidity**, and **sound**. This project serves a simple Isolation Forest as a REST API, so each new reading gets an **anomaly score** as it arrives.

There is no real factory. `generate_data.py` invents a training CSV. `stream_simulator.py` invents **new** readings one by one and sends them to the API. The 2,000 training rows are **not** replayed as the stream.

## What the files do

| File | Role |
|---|---|
| `src/config.py` | Shared settings (paths, data recipe, model options, API address) |
| `src/generate_data.py` | Creates the fictional CSV; also defines `generate_reading` for one row |
| `src/train_model.py` | Batch job: train Isolation Forest, write `results/models/isolation_forest.joblib` |
| `src/app.py` | REST service. Loads the joblib file and waits. `POST /predict`, `GET /health`, `GET /metrics` |
| `src/stream_simulator.py` | Client that pretends to be the sensors. Sends one reading at a time to the API |

The joblib file is the **model**. `app.py` is the **service** that holds it.

## Setup

```bash
cd iot-anomaly-stream
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Commands below assume you are in `iot-anomaly-stream/` with the virtual environment active. Prefix them with `PYTHONPATH=.` so Python can import `src`.

## Run

**1. Create training data** (optional if `data/sensor_train.csv` is already there):

```bash
PYTHONPATH=. python -m src.generate_data
```

**2. Train and save the model:**

```bash
PYTHONPATH=. python -m src.train_model
```

Metrics are written to `results/metrics.txt`. The model file is not in git (it is large); train locally after clone.

**3. Start the service** (leave this terminal open):

```bash
PYTHONPATH=. python -m src.app
```

It serves at [http://127.0.0.1:5001](http://127.0.0.1:5001).

**4. In a second terminal, send a live stream:**

```bash
PYTHONPATH=. python -m src.stream_simulator
```

Stop with Ctrl+C. Short test: `--limit 15`. Faster: `--interval 0.2`.

**5. Check monitoring:**

```bash
curl http://127.0.0.1:5001/health
curl http://127.0.0.1:5001/metrics
```

One reading by hand:

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 92, "humidity": 84, "sound": 104}'
```

Higher `anomaly_score` means more unusual. `is_anomaly` is true or false.

The oral is submitted to IU as a PDF plus a 15-minute recording. This repo is the code the examiner can reproduce: https://github.com/JJDol/iot-anomaly-stream
