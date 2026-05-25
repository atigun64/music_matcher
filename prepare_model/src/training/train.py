import numpy as np
from typing import List
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .model import model

from .data import Sample
from .utils import sample_to_vector

def train_model(labeled_samples: List[Sample]) -> Pipeline:
    X = np.stack([
        sample_to_vector(s)
        for s in labeled_samples
    ])
    y = np.array([s.y for s in labeled_samples], dtype=int)

    model.fit(X, y)
    dump(model, "drop_model.joblib")
    return model
