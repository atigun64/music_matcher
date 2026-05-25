from optimizer.models import AssignedTrack, Alignment
from .utils import track_duration, range_score, clamp01

from .config import (
    VIDEO_EDGE_FRAC,
    BPM_EDGE_MIN,
    BPM_EDGE_MAX,
    BPM_MID_MIN,
    BPM_MID_MAX,
    BPM_FALLOFF,
)

def score_bpm(track: AssignedTrack, video_length: float):
    """
    Returns [0,1].

    Beginning/end prefer lower BPM.
    Middle prefers higher BPM.

    Uses effective BPM = BPM * speed.
    """
    if track.BPM is None or track.start_time is None or track.length is None:
        return 0.5

    if track.speed is None or track.speed <= 0:
        return 0.5

    if video_length is None or video_length <= 0:
        return 0.5

    duration = track_duration(track)
    if duration <= 0:
        return 0.5

    track_center = track.start_time + duration / 2.0
    rel_pos = track_center / video_length

    effective_bpm = track.BPM * track.speed

    # Beginning or end
    if rel_pos < VIDEO_EDGE_FRAC or rel_pos > 1.0 - VIDEO_EDGE_FRAC:
        return range_score(
            effective_bpm,
            BPM_EDGE_MIN,
            BPM_EDGE_MAX,
            BPM_FALLOFF
        )

    # Middle
    return range_score(
        effective_bpm,
        BPM_MID_MIN,
        BPM_MID_MAX,
        BPM_FALLOFF
    )


def score_average_bpm(alignment: Alignment, video_length: float):
    """
    Duration-weighted average BPM score.
    Returns [0,1].
    """
    total = 0.0
    total_duration = 0.0

    for track in alignment.tracks:
        duration = track_duration(track)
        if duration <= 0:
            continue

        total += score_bpm(track, video_length) * duration
        total_duration += duration

    if total_duration <= 0:
        return 0.5

    return clamp01(total / total_duration)
