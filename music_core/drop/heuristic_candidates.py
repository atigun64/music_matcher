from .drop_heuristic import score_drops
from scipy.signal import find_peaks

MAX_CANDIDATES_PER_SONG = 5
PEAK_DISTANCE = 4
PEAK_PROMINENCE = 0.08

def detect_candidates(E, O, C, B, beat_times):
    score = score_drops(E, O, C, B)
    peaks, _ = find_peaks(score, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)

    if len(peaks) == 0:
        return []

    ranked = sorted(peaks, key=lambda i: score[i], reverse=True)[:MAX_CANDIDATES_PER_SONG]
    out = []
    for i in ranked:
        if 0 <= i < len(beat_times):
            out.append((i, beat_times[i], score[i]))
    return out