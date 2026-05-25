from optimizer.models import Alignment
from .utils import clamp01, track_duration

def score_average_preference(alignment: Alignment):
    """
    Duration-weighted average track preference.
    Assumes track.preference is in [0,1].
    Returns [0,1].
    """
    total = 0.0
    total_duration = 0.0

    for track in alignment.tracks:
        duration = track_duration(track)
        if duration <= 0:
            continue

        preference = clamp01(track.preference)
        total += preference * duration
        total_duration += duration

    if total_duration <= 0:
        return 0.5

    return clamp01(total / total_duration)
