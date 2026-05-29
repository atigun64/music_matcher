from __future__ import annotations

from functools import lru_cache

import numpy as np
from joblib import load

from .window import build_feature_window_ml
from .heuristic_candidates import detect_candidates


@lru_cache(maxsize=4)
def _load_model(model_path: str = "drop_model.joblib"):
    return load(model_path)


def _build_ml_vector(E, O, C, F, B, beat_idx: int, hscore: float) -> np.ndarray:
    window = build_feature_window_ml(E, O, C, F, B, beat_idx)
    return np.concatenate(([float(hscore)], window.reshape(-1)))


def get_ml_candidates(
    E,
    O,
    C,
    F,
    B,
    beat_times,
    model_path: str = "drop_model.joblib",
    heuristic_threshold: float = 0.5,
    min_score: float = 0.6,
    min_gap_sec: float = 30.0,
    max_candidates: int = 10,
):
    """
    1) get heuristic candidates from detect_candidates
    2) score them with ML
    3) keep only strong ones
    4) suppress close duplicates
    """
    E = np.asarray(E, dtype=float)
    O = np.asarray(O, dtype=float)
    C = np.asarray(C, dtype=float)
    B = np.asarray(B, dtype=float)
    beat_times = np.asarray(beat_times, dtype=float)

    n = min(len(E), len(O), len(C), len(B), len(beat_times))
    E, O, C, B, beat_times = E[:n], O[:n], C[:n], B[:n], beat_times[:n]

    # 1) heuristic candidates
    heuristic_cands = detect_candidates(E, O, C, B, beat_times, 0.0, 200)
    cand = [
        c for c in heuristic_cands
        if c[2] > heuristic_threshold
    ]

    print(len(cand))

    if not cand:
        return []
    
    # 2) build features for only heuristic candidates
    model = _load_model(model_path)

    X = []
    meta = []  # (beat_idx, beat_time, hscore)
    for beat_idx, beat_time, hscore in cand:
        if beat_idx is None:
            continue
        X.append(_build_ml_vector(E, O, C, F, B, int(beat_idx), float(hscore)))
        meta.append((int(beat_idx), float(beat_time), float(hscore)))

    if not X:
        return []

    X = np.stack(X)
    probs = model.predict_proba(X)[:, 1]

    # 3) keep only strong ML candidates
    scored = []
    for (beat_idx, beat_time, hscore), p in zip(meta, probs):
        if p >= min_score:
            scored.append((beat_idx, beat_time, float(p), hscore))

    if not scored:
        return []

    # 4) sort by ML score and suppress close drops
    scored.sort(key=lambda x: x[2], reverse=True)

    selected = []
    for beat_idx, beat_time, mscore, hscore in scored:
        if len(selected) >= max_candidates:
            break

        too_close = False
        for _, prev_time, _, _ in selected:
            if abs(beat_time - prev_time) < min_gap_sec:
                too_close = True
                break

        if too_close:
            continue

        selected.append((beat_idx, beat_time, mscore, hscore))

    print(len(selected))
    return [(i, bt, ms) for i, bt, ms, _ in selected]
