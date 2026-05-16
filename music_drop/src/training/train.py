import numpy as np
from typing import List

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .data import Sample

def train_model(labeled_samples: List[Sample]) -> Pipeline:
    X = np.vstack([s.x for s in labeled_samples])
    y = np.array([s.y for s in labeled_samples], dtype=int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)
    return model