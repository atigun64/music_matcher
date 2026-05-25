from optimizer.models import TrackSignature, Alignment
from .utils import clamp01, track_duration
import math

def score_style(signature1: TrackSignature, signature2: TrackSignature):
    """
    Euclidean similarity between two normalized signatures.
    Assumes each component is in [0,1].
    Returns [0,1].
    """
    if signature1 is None or signature2 is None:
        return 0.5

    if len(signature1.sig) != len(signature2.sig):
        return 0.5

    if len(signature1.sig) == 0:
        return 0.5

    distance = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(signature1.sig, signature2.sig))
    )

    max_distance = math.sqrt(len(signature1.sig))
    similarity = 1.0 - (distance / max_distance)

    return clamp01(similarity)

def score_alignment_style(alignment: Alignment, video_signature: TrackSignature):
    """
    Style score for the whole alignment.

    Idea:
    - average style should be good
    - one very off track should still hurt

    Returns [0,1].
    """
    if video_signature is None:
        return 0.5

    scores = []
    durations = []

    for track in alignment.tracks:
        duration = track_duration(track)
        if duration <= 0:
            continue

        s = score_style(track.signature, video_signature)
        scores.append(s)
        durations.append(duration)

    if not scores:
        return 0.5

    total_duration = sum(durations)
    avg = sum(s * d for s, d in zip(scores, durations)) / total_duration
    worst = min(scores)

    # Mostly average, but one bad track still matters.
    final = 0.7 * avg + 0.3 * worst
    return clamp01(final)

