"""
pipeline.py -- sklearn Pipeline definition for breast cancer classification.

Encapsulates StandardScaler + RandomForestClassifier into a single object.
The same preprocessing applied at training time is automatically applied
at inference time, eliminating a common source of production bugs.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def build_pipeline(n_estimators: int = 200, random_state: int = 42) -> Pipeline:
    """Return a fitted-ready Pipeline: StandardScaler -> RandomForestClassifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
