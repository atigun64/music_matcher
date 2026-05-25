from typing import List
import numpy as np

from .data import Sample

from music_drop.src.cache import AudioFeatureCache

from music_core import detect_candidates, score_drops, build_feature_window
from .utils import sample_to_vector


_cache = AudioFeatureCache()


def make_sample(track_id, beat_idx, source, hscore, mscore=0.0, payload=None):
    if payload is None:
        print("track_id:", track_id)
        payload = _cache.get_by_id(track_id)
        print("payload is None?", payload is None)

    return Sample(
        track_id=track_id,
        beat_idx=beat_idx,
        x=build_feature_window(payload["E"], payload["O"], payload["B"], payload["C"], beat_idx),
        source=source,
        hscore=hscore,
        mscore=mscore,
    )



def heuristic_candidates(payload):
    return detect_candidates(
        payload["E"],
        payload["O"],
        payload["C"],
        payload["B"],
        payload["beat_times"],
        threshold=0.64,
        max_candidates=7,
    )


def sample_background_beats(payload, background_per_track, avoid):
    scores = np.asarray(score_drops(payload["E"], payload["O"], payload["C"], payload["B"]))

    all_indices = np.arange(len(scores))
    avoid = set(avoid or [])
    candidates = np.array([i for i in all_indices if i not in avoid])

    if len(candidates) == 0:
        return []

    candidate_scores = np.maximum(scores[candidates].astype(float), 0.0)
    weights = np.sqrt(candidate_scores + 1e-9)

    if weights.sum() == 0:
        weights = np.ones_like(weights, dtype=float)

    weights /= weights.sum()

    k = min(background_per_track, len(candidates))
    chosen = np.random.choice(candidates, size=k, replace=False, p=weights)
    return chosen.tolist()


HEURISTIC_THRESHOLD = 0.5

def build_pool(track_ids, heuristic_per_track=3, background_per_track=0):
    pool = []

    for track_id in track_ids:
        payload = _cache.get_by_id(track_id)

        if payload is None:
            print(f"WARNING: no payload for track_id={track_id}, skipping")
            continue

        cand = [
            c for c in heuristic_candidates(payload)
            if c[2] >= HEURISTIC_THRESHOLD
        ][:heuristic_per_track]

        for beat_idx, _, hscore in cand:
            pool.append(
                make_sample(
                    track_id,
                    beat_idx,
                    source="heuristic",
                    hscore=hscore,
                    payload=payload,
                )
            )

        # optional later:
        # bg_beats = sample_background_beats(payload, background_per_track, avoid={c[0] for c in cand})
        # for beat_idx in bg_beats:
        #     pool.append(make_sample(track_id, beat_idx, source="background", hscore=0.0, payload=payload))

    return pool



def select_queries(model, pool: List[Sample], batch_size=20):
    if len(pool) == 0:
        return []

    # sklearn-style tabular input
    X = np.stack([sample_to_vector(s) for s in pool])
    p = model.predict_proba(X)[:, 1]

    # save model score into each pool item
    for s, ms in zip(pool, p):
        s.mscore = float(ms)

    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0

    heuristic_idx = [i for i, s in enumerate(pool) if s.source == "heuristic"]
    background_idx = [i for i, s in enumerate(pool) if s.source == "background"]

    k_h = int(batch_size * 0.7)
    k_b = batch_size - k_h

    heuristic_idx = sorted(heuristic_idx, key=lambda i: uncertainty[i], reverse=True)
    background_idx = sorted(background_idx, key=lambda i: uncertainty[i], reverse=True)

    chosen_idx = heuristic_idx[:k_h] + background_idx[:k_b]

    if len(chosen_idx) < batch_size:
        remaining = [i for i in range(len(pool)) if i not in chosen_idx]
        remaining = sorted(remaining, key=lambda i: uncertainty[i], reverse=True)
        chosen_idx += remaining[: batch_size - len(chosen_idx)]

    return [pool[i] for i in chosen_idx]
