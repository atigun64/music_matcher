from optimizer.models import AssignedTrack, Alignment
from .utils import clamp01, track_end_time

from .config import (
    MAX_ACCEPTABLE_OVERLAP,
    AVERAGE_OVERLAP_WEIGHT,
    WORST_OVERLAP_WEIGHT
)


def overlap_quality_from_duration(overlap_duration: float):
    """
    Returns [0,1].
    1.0 = overlap is acceptable or tiny
    0.0 = overlap is very bad
    """
    if overlap_duration <= MAX_ACCEPTABLE_OVERLAP:
        return 1.0

    # linearly decay from 1.0 down to 0.0 over one more threshold length
    return clamp01(1.0 - (overlap_duration - MAX_ACCEPTABLE_OVERLAP) / MAX_ACCEPTABLE_OVERLAP)


def pair_overlap_duration(track1: AssignedTrack, track2: AssignedTrack):
    """
    Returns overlap duration in seconds.
    """
    if track1.start_time is None or track1.length is None:
        return 0.0
    if track2.start_time is None or track2.length is None:
        return 0.0

    end1 = track_end_time(track1)
    end2 = track_end_time(track2)

    if end1 is None or end2 is None:
        return 0.0

    latest_start = max(track1.start_time, track2.start_time)
    earliest_end = min(end1, end2)

    return max(0.0, earliest_end - latest_start)


def _score_overlap_quality_from_tracks(tracks):
    """
    Shared helper for partial/final overlap scoring.
    Returns [0,1].
    """
    if len(tracks) < 2:
        return 1.0

    pair_scores = []

    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            overlap_sec = pair_overlap_duration(tracks[i], tracks[j])
            pair_scores.append(overlap_quality_from_duration(overlap_sec))

    if not pair_scores:
        return 1.0

    avg_overlap_quality = sum(pair_scores) / len(pair_scores)
    worst_overlap_quality = min(pair_scores)

    final = (
        avg_overlap_quality * AVERAGE_OVERLAP_WEIGHT +
        worst_overlap_quality * WORST_OVERLAP_WEIGHT
    )

    return clamp01(final)


def score_overlap_quality(alignment: Alignment):
    """
    Partial overlap score for beam search.
    Returns [0,1].

    This scores the overlap of tracks already in the alignment.
    """
    tracks = [
        t for t in alignment.tracks
        if t.start_time is not None and t.length is not None
    ]

    if not tracks:
        return 0.5

    return _score_overlap_quality_from_tracks(tracks)
