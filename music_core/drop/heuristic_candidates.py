from .drop_heuristic import score_drops
from scipy.signal import find_peaks
import numpy as np

MAX_CANDIDATES_PER_SONG = 5
PEAK_DISTANCE = 4
PEAK_PROMINENCE = 0.08
THRESHOLD = 0.64

NEIGHBOR_OFFSETS = (-5, 0, 5)

def detect_candidates(E, O, C, B, beat_times, threshold=THRESHOLD, max_candidates=MAX_CANDIDATES_PER_SONG):
    score = score_drops(E, O, C, B)
    peaks, _ = find_peaks(score, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)

    if len(peaks) == 0:
        return []

    ranked_peaks = sorted(peaks, key=lambda i: score[i], reverse=True)

    selected = set()
    for p in ranked_peaks:
        if score[p] >= threshold:
            for off in NEIGHBOR_OFFSETS:
                j = p + off
                if 0 <= j < len(score):
                    selected.add(j)

        if len(selected) >= max_candidates * len(NEIGHBOR_OFFSETS):
            break

    ranked = sorted(selected, key=lambda i: score[i], reverse=True)

    out = []
    for i in ranked:
        if 0 <= i < len(beat_times):
            out.append((i, beat_times[i], score[i]))

    return out
