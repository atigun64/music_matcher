from optimizer.models import Alignment
from .utils import clamp01, track_end_time

from .config import (
    MAX_ACCEPTABLE_GAP,
    WORST_GAP_WEIGHT,
    AVERAGE_GAP_WEIGHT,
)


def gap_quality(gap: float):
    """
    Returns [0,1].
    1.0 = gap is acceptable or tiny
    0.0 = gap is very bad
    """
    if gap <= MAX_ACCEPTABLE_GAP:
        return 1.0

    # linearly decay from 1.0 down to 0.0 over one more threshold length
    return clamp01(1.0 - (gap - MAX_ACCEPTABLE_GAP) / MAX_ACCEPTABLE_GAP)


def _merged_intervals(tracks, horizon: float):
    """
    Return merged [start, end] intervals clipped to [0, horizon].
    """
    intervals = []

    for t in tracks:
        if t.start_time is None or t.length is None:
            continue

        end = track_end_time(t)
        if end is None:
            continue

        if end <= 0 or t.start_time >= horizon:
            continue

        start = max(0.0, t.start_time)
        end = min(horizon, end)

        if end > start:
            intervals.append((start, end))

    intervals.sort(key=lambda x: x[0])

    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return merged


def _score_gaps_from_intervals(merged_intervals, horizon: float, include_tail_gap: bool):
    """
    Convert merged intervals into a [0,1] gap quality score.
    """
    if horizon <= 0:
        return 0.5

    gap_scores = []
    last_end = 0.0

    for start, end in merged_intervals:
        if start > last_end:
            gap = start - last_end
            gap_scores.append(gap_quality(gap))
        last_end = max(last_end, end)

    if include_tail_gap and last_end < horizon:
        gap = horizon - last_end
        gap_scores.append(gap_quality(gap))

    if not gap_scores:
        return 1.0

    avg_gap_quality = sum(gap_scores) / len(gap_scores)
    worst_gap_quality = min(gap_scores)

    final = (
        AVERAGE_GAP_WEIGHT * avg_gap_quality +
        WORST_GAP_WEIGHT * worst_gap_quality
    )

    return clamp01(final)


def score_gap_quality_partial(alignment: Alignment, video_length: float):
    """
    Partial gap score for beam search.
    Returns [0,1].

    Scores only the already-known covered part.
    Ignores future tail gap after current coverage.
    """
    if video_length is None or video_length <= 0:
        return 0.5

    tracks = [
        t for t in alignment.tracks
        if t.start_time is not None and t.length is not None
    ]

    if not tracks:
        return 0.5

    covered_ends = []
    for t in tracks:
        end = track_end_time(t)
        if end is not None:
            covered_ends.append(end)

    if not covered_ends:
        return 0.5

    # Only score up to the currently covered end, not the unknown future.
    horizon = min(video_length, max(covered_ends))

    merged = _merged_intervals(tracks, horizon)
    return _score_gaps_from_intervals(merged, horizon, include_tail_gap=False)


def score_gap_quality_final(alignment: Alignment, video_length: float):
    """
    Final gap score for completed alignment.
    Returns [0,1].

    Penalizes:
    - total uncovered time
    - biggest single gap
    """
    if video_length is None or video_length <= 0:
        return 0.5

    tracks = [
        t for t in alignment.tracks
        if t.start_time is not None and t.length is not None
    ]

    if not tracks:
        return 0.0

    merged = _merged_intervals(tracks, video_length)
    return _score_gaps_from_intervals(merged, video_length, include_tail_gap=True)
