from typing import List
import numpy as np

from .data import Sample

from music_drop.src.cache import AudioFeatureCache

from music_core import detect_candidates, build_feature_window_ml, score_drops
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
        x=build_feature_window_ml(payload["E"], payload["O"], payload["C"], payload["F"], payload["B"], beat_idx),
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

def build_pool(track_ids):
    pool = []

    for track_id in track_ids:
        payload = _cache.get_by_id(track_id)

        if payload is None:
            print(f"WARNING: no payload for track_id={track_id}, skipping")
            continue
            
        if len(payload["E"]) > 1000:   # sanity check to avoid loading huge files
            print(f"WARNING: payload for track_id={track_id} has {len(payload['E'])} frames, skipping")
            continue
        
        scores = score_drops(
            payload["E"],
            payload["O"],
            payload["C"],
            payload["B"],
        )

        for beat_idx, hscore in enumerate(scores):
            if hscore > 0.6:   # heuristic threshold for pool inclusion
                pool.append(
                    make_sample(
                        track_id=track_id,
                        beat_idx=beat_idx,
                        source="heuristic",
                        hscore=hscore,
                        payload=payload,
                    )
                )

    return pool

def pool_filter(pool, labeled_samples):
    """
        Filter out samples from the pool which are near already labeled samples.
        This is to avoid showing the same or very similar samples to the annotator multiple times.
        If a label is positive we filter [-4, +4] beat indices around it.
        Otherwise we filter [-2, +2] beat indices around it.
    """
    filtered = []

    for sample in pool:
        too_close = False
        for labeled in labeled_samples:
            if sample.track_id == labeled.track_id:
                dist = abs(sample.beat_idx - labeled.beat_idx)
                if labeled.y == 1 and dist <= 2:
                    too_close = True
                    break
                elif labeled.y == 0 and dist <= 4:
                    too_close = True
                    break

        if not too_close:
            filtered.append(sample)

    return filtered

def select_queries(
    model,
    pool: List[Sample],
    batch_size=20,
    frac_uncertain=0.4,
    frac_high_pos=0.3,
    frac_high_neg=0.3,
    heuristic_frac=0.7,
    suppress_radius_uncertain=2,
    suppress_radius_high_pos=2,
    suppress_radius_high_neg=3,
    max_per_track=3,
):
    if len(pool) == 0:
        return []

    # sklearn-style tabular input
    X = np.stack([sample_to_vector(s) for s in pool])
    p = model.predict_proba(X)[:, 1]

    for s, ms in zip(pool, p):
        s.mscore = float(ms)

    confidence = np.abs(p - 0.5) * 2.0
    uncertainty = 1.0 - confidence

    idx_all = np.arange(len(pool), dtype=int)
    heuristic_idx = np.array([i for i, s in enumerate(pool) if s.source == "heuristic"], dtype=int)

    def rank_indices(indices, mode):
        indices = np.asarray(indices, dtype=int)
        if len(indices) == 0:
            return []

        if mode == "uncertain":
            order = np.argsort(-uncertainty[indices])
        elif mode == "high_pos":
            order = np.argsort(-p[indices])
        elif mode == "high_neg":
            order = np.argsort(p[indices])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return indices[order].tolist()

    def suppress_remaining(available_set, chosen_sample, radius):
        """
        Remove samples from the same track within ±radius beats.
        """
        for j in list(available_set):
            s = pool[j]
            if s.track_id == chosen_sample.track_id:
                if abs(s.beat_idx - chosen_sample.beat_idx) <= radius:
                    available_set.discard(j)

    def select_bucket(k, mode, radius):
        """
        Greedy selection with neighborhood suppression.
        """
        if k <= 0:
            return []

        # Prefer heuristic items first, then fill from everything
        k_h = min(len(heuristic_idx), int(round(k * heuristic_frac)))
        ranked_heur = rank_indices(heuristic_idx, mode)
        ranked_all = rank_indices(idx_all, mode)

        # Build a combined ranked list without duplicates
        ranked = []
        seen = set()

        for i in ranked_heur + ranked_all:
            if i not in seen:
                ranked.append(i)
                seen.add(i)

        available = set(ranked)
        chosen = []
        per_track_counts = {}

        for i in ranked:
            if len(chosen) >= k:
                break
            if i not in available:
                continue

            s = pool[i]
            cnt = per_track_counts.get(s.track_id, 0)
            if cnt >= max_per_track:
                continue

            chosen.append(i)
            per_track_counts[s.track_id] = cnt + 1

            # suppress neighborhood around this chosen point
            suppress_remaining(available, s, radius)

            # also remove the chosen one itself
            available.discard(i)

        return chosen

    k_u = int(round(batch_size * frac_uncertain))
    k_hp = int(round(batch_size * frac_high_pos))
    k_hn = batch_size - k_u - k_hp

    chosen = []
    chosen_set = set()

    def add_bucket(bucket_indices):
        for i in bucket_indices:
            if i not in chosen_set and len(chosen) < batch_size:
                chosen.append(i)
                chosen_set.add(i)

    add_bucket(select_bucket(k_u, "uncertain", suppress_radius_uncertain))
    add_bucket(select_bucket(k_hp, "high_pos", suppress_radius_high_pos))
    add_bucket(select_bucket(k_hn, "high_neg", suppress_radius_high_neg))

    # Final fill if we still have room
    if len(chosen) < batch_size:
        remaining = [i for i in idx_all if i not in chosen_set]

        # Diversity-aware fill: prefer things not too close to already chosen points
        def diversity_score(i):
            s = pool[i]
            # Penalize same-track closeness to chosen items
            penalty = 0.0
            for j in chosen:
                sj = pool[j]
                if sj.track_id == s.track_id:
                    d = abs(sj.beat_idx - s.beat_idx)
                    if d < 10:
                        penalty += (10 - d)
            # Prefer uncertainty/extremes
            return max(uncertainty[i], confidence[i]) - 0.05 * penalty

        remaining = sorted(remaining, key=diversity_score, reverse=True)
        for i in remaining:
            if len(chosen) >= batch_size:
                break
            s = pool[i]
            cnt = sum(1 for j in chosen if pool[j].track_id == s.track_id)
            if cnt < max_per_track:
                chosen.append(i)
                chosen_set.add(i)

    return [pool[i] for i in chosen[:batch_size]]
