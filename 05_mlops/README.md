# 05 — MLOps Basics

Demonstrates MLOps fundamentals: encapsulating preprocessing + model into a **sklearn Pipeline**,
saving/loading models with **joblib**, and exposing a **CLI training script**.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Defines a reusable `build_pipeline()` function |
| `train.py` | Trains the pipeline, cross-validates, saves to `models/` |
| `predict.py` | Loads a saved model and runs inference |

## Usage

```bash
# Train and save
python train.py
python train.py --n-estimators 300 --output models/rf_v2.joblib

# Predict
python predict.py
python predict.py --model models/rf_v2.joblib
```

## Why Pipelines?

A sklearn `Pipeline` chains preprocessing and model into one object:
- **No leakage**: scaler is fit only on training data, automatically
- **Production-ready**: call `pipeline.predict(raw_data)` — no manual scaling step
- **Reproducible**: one object to save, version, and deploy
