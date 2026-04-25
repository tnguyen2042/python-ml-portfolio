"""
train.py -- Train the pipeline on breast cancer data and save to disk.

Usage:
    python train.py
    python train.py --output models/rf_pipeline.joblib --n-estimators 300
"""

import argparse
import pathlib

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from pipeline import build_pipeline


def train(output_path: str = "models/rf_pipeline.joblib",
          n_estimators: int = 200) -> None:
    raw = load_breast_cancer()
    X, y = raw.data, raw.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(n_estimators=n_estimators)

    # Cross-validate before committing to a final fit
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"5-fold CV ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print(f"Test ROC-AUC:      {roc_auc_score(y_test, y_prob):.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=raw.target_names))

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    print(f"Model saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train breast cancer classifier")
    parser.add_argument("--output", default="models/rf_pipeline.joblib",
                        help="Path to save the trained pipeline")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of trees in the Random Forest")
    args = parser.parse_args()
    train(args.output, args.n_estimators)
