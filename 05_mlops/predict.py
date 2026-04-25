"""
predict.py -- Load a saved pipeline and run inference.

Usage:
    python predict.py                             # demo with 5 random samples
    python predict.py --model models/rf_pipeline.joblib
    python predict.py --data path/to/features.csv
"""

import argparse
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def predict(model_path: str = "models/rf_pipeline.joblib",
            data_path: Optional[str] = None) -> None:
    pipeline = joblib.load(model_path)
    print(f"Loaded pipeline from:  {model_path}")
    print(f"Pipeline steps:        {[name for name, _ in pipeline.steps]}")

    raw = load_breast_cancer()

    if data_path:
        X = pd.read_csv(data_path).values
        true_labels = None
    else:
        # Demo: use 5 random held-out samples
        rng = np.random.default_rng(seed=0)
        idx = rng.integers(0, len(raw.data), size=5)
        X = raw.data[idx]
        true_labels = [raw.target_names[raw.target[i]] for i in idx]
        print(f"\nTrue labels:      {true_labels}")

    predictions = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]
    pred_labels = [raw.target_names[p] for p in predictions]

    print(f"Predicted labels: {pred_labels}")
    print(f"Benign prob:      {[f'{p:.2f}' for p in probs]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a saved pipeline")
    parser.add_argument("--model", default="models/rf_pipeline.joblib")
    parser.add_argument("--data", default=None,
                        help="CSV file with feature columns (30 features, no header)")
    args = parser.parse_args()
    predict(args.model, args.data)
