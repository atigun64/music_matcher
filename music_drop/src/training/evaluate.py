import numpy as np
from typing import List

from sklearn.metrics import average_precision_score

from .data import Sample

def evaluate_model(model, val_samples: List[Sample]) -> float:
    X_val = np.vstack([s.x for s in val_samples])
    y_val = np.array([s.y for s in val_samples], dtype=int)

    prob = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, prob)
    return ap