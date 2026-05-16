from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from music_drop.src.cache import AudioFeatureCache

from music_core import detect_candidates, score_drops

@dataclass
class Sample:
    track_id: str
    beat_idx: int
    x: np.ndarray
    y: Optional[int] = None   # label, 0/1
    source: str = ""          # "heuristic" or "background"
    hscore: float = 0.0       # heuristic score

def make_sample(track_id, beat_idx, source, hscore):
    return Sample(
        track_id=track_id,
        beat_idx=beat_idx,
        source=source,
        hscore=hscore,
    )

def heuristic_candidates(track_id):
    cache = AudioFeatureCache()
    payload = cache.get_by_id(track_id)

    candidates = detect_candidates(payload["E"], payload["O"], payload["C"], payload["B"], payload["beat_times"])
    return candidates


def sample_random_beats(track_id, background_per_track, avoid):
    cache = AudioFeatureCache()
    payload = cache.get_by_id(track_id)

    scores = np.array(score_drops(payload["E"], payload["O"], payload["C"], payload["B"]))

    # indices we are allowed to pick from
    all_indices = np.arange(len(scores))
    avoid = set(avoid or [])
    candidates = np.array([i for i in all_indices if i not in avoid])

    if len(candidates) == 0:
        return []

    # soften the weighting so it isn't too dramatic
    candidate_scores = scores[candidates].astype(float)

    # make sure weights are non-negative
    candidate_scores = np.maximum(candidate_scores, 0)

    # softer emphasis on larger scores
    weights = np.sqrt(candidate_scores + 1e-9)   # or np.log1p(candidate_scores)

    # if all weights are zero, fall back to uniform sampling
    if weights.sum() == 0:
        weights = np.ones_like(weights, dtype=float)

    weights = weights / weights.sum()

    k = min(background_per_track, len(candidates))
    chosen = np.random.choice(candidates, size=k, replace=False, p=weights)

    return chosen

def build_pool(track_ids, heuristic_per_track=10, background_per_track=10):
    """
    Build unlabeled pool from:
    - heuristic candidates
    - random background beats
    """
    pool = []

    for tid in track_ids:
        cand = heuristic_candidates(tid)[:heuristic_per_track]
        cand_beats = set()

        for beat_idx, _, hscore in cand:
            pool.append(make_sample(tid, beat_idx, source="heuristic", hscore=hscore))
            cand_beats.add(beat_idx)


        bg_beats = sample_random_beats(tid, background_per_track, avoid=cand_beats)
        for beat_idx in bg_beats:
            pool.append(make_sample(tid, beat_idx, source="background", hscore=0.0))

    return pool



def select_queries(model, pool: List[Sample], batch_size=20):
    """
    Select the most useful samples to label next.
    Strategy:
    - 70% from heuristic candidates with highest uncertainty
    - 30% from background samples with highest uncertainty
    """
    if len(pool) == 0:
        return []

    X = np.vstack([s.x for s in pool])
    p = model.predict_proba(X)[:, 1]

    # uncertainty is highest near 0.5
    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0  # 1.0 = most uncertain, 0.0 = most sure

    heuristic_idx = [i for i, s in enumerate(pool) if s.source == "heuristic"]
    background_idx = [i for i, s in enumerate(pool) if s.source == "background"]

    k_h = int(batch_size * 0.7)
    k_b = batch_size - k_h

    # sort by uncertainty descending
    heuristic_idx = sorted(heuristic_idx, key=lambda i: uncertainty[i], reverse=True)
    background_idx = sorted(background_idx, key=lambda i: uncertainty[i], reverse=True)

    chosen_idx = heuristic_idx[:k_h] + background_idx[:k_b]

    # if not enough from one group, fill from the other
    if len(chosen_idx) < batch_size:
        remaining = [i for i in range(len(pool)) if i not in chosen_idx]
        remaining = sorted(remaining, key=lambda i: uncertainty[i], reverse=True)
        chosen_idx += remaining[:(batch_size - len(chosen_idx))]

    chosen = [pool[i] for i in chosen_idx]
    return chosen