from optimizer.models import PointAnnotation, AssignedTrack, Alignment
from .utils import track_end_time, clamp01
from .config import HERTZ
import math


# --------------------------------------------------
# Tuning
# --------------------------------------------------

# Stronger requests count more in aggregation
STRENGTH_POWER = 2.0

# Timing behavior in seconds
TIMING_GOOD_SEC = 2.0      # 0-2s: basically same
TIMING_MILD_SEC = 3.0      # 2-3s: slightly worse
TIMING_BAD_SEC = 5.0       # 3-5s: clearly worse
TIMING_TERRIBLE_SEC = 8.0  # 5-8s: very bad

# Small floor so geometric mean does not become exact zero unless you want it to
EPS = 1e-9


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def importance(strength: float) -> float:
    """
    Convert request strength [0,1] into an aggregation weight.
    """
    strength = clamp01(strength)
    return strength ** STRENGTH_POWER


def timing_quality(time_diff_ticks: float) -> float:
    """
    Returns [0,1].

    Behavior:
    - 0-2s: almost identical / perfect
    - 2-3s: slightly worse
    - 3-5s: much worse
    - 5-8s: very bad
    - 8s+: catastrophic
    """
    d = abs(float(time_diff_ticks)) / HERTZ  # convert ticks -> seconds

    if d <= TIMING_GOOD_SEC:
        return 1.0

    if d <= TIMING_MILD_SEC:
        return 0.95

    if d <= TIMING_BAD_SEC:
        # 0.95 -> 0.35
        x = (d - TIMING_MILD_SEC) / (TIMING_BAD_SEC - TIMING_MILD_SEC)
        return 0.95 * (1.0 - x) + 0.35 * x

    if d <= TIMING_TERRIBLE_SEC:
        # 0.35 -> 0.05
        x = (d - TIMING_BAD_SEC) / (TIMING_TERRIBLE_SEC - TIMING_BAD_SEC)
        return 0.35 * (1.0 - x) + 0.05 * x

    # beyond terrible, decay toward zero
    extra = d - TIMING_TERRIBLE_SEC
    return max(0.0001, 0.05 * (0.75 ** extra))


def weighted_geometric_mean(values: list[float], weights: list[float]) -> float:
    """
    Weighted geometric mean in [0,1].

    If any value is tiny, the result becomes tiny.
    That is exactly what you want for "one bad drop ruins it".
    """
    if not values:
        return 0.5

    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 0.5

    log_sum = 0.0
    for v, w in zip(values, weights):
        log_sum += w * math.log(max(EPS, clamp01(v)))

    return clamp01(math.exp(log_sum / total_weight))


# --------------------------------------------------
# Per-request scoring against one track
# --------------------------------------------------

def request_quality_against_track(request: PointAnnotation, track: AssignedTrack) -> float:
    """
    Returns [0,1].

    1.0 = request is matched very well by this track
    0.0 = no useful match
    """
    if request.time is None or track.start_time is None:
        return 0.0

    if track.speed is None or track.speed <= 0:
        return 0.0

    track_end = track_end_time(track)
    if track_end is None:
        return 0.0

    # This track must cover the request point to be considered.
    if request.time < track.start_time or request.time > track_end:
        return 0.0

    req_strength = clamp01(request.strength)

    track_drops = [
        a for a in track.annotations
        if isinstance(a, PointAnnotation)
        and a.label == "drop"
        and a.time is not None
    ]

    if not track_drops:
        return 0.0

    # Find best internal drop for this request
    candidates = []
    for d in track_drops:
        drop_strength = clamp01(d.strength)
        video_drop_time = track.start_time + (d.time / track.speed)
        candidates.append((video_drop_time, drop_strength))

    best_time, best_strength = min(
        candidates,
        key=lambda x: abs(request.time - x[0])
    )

    time_diff = abs(request.time - best_time)
    tq = timing_quality(time_diff)

    # If the timing is terrible, this request-track pair is bad.
    if tq <= 0.0:
        return 0.0

    # Strength agreement between requested and actual drop
    strength_match = 1.0 - abs(req_strength - best_strength)
    strength_match = clamp01(strength_match)

    # Strong actual drops should matter more.
    # This keeps weak drops from looking too good for important requests.
    quality = tq * best_strength * strength_match

    return clamp01(quality)


# --------------------------------------------------
# Best track quality for one request
# --------------------------------------------------

def best_request_quality(request: PointAnnotation, tracks: list[AssignedTrack]) -> float:
    """
    Returns the best quality among all tracks that cover this request.
    """
    best = 0.0
    for track in tracks:
        q = request_quality_against_track(request, track)
        if q > best:
            best = q
    return clamp01(best)


# --------------------------------------------------
# Alignment-level scoring
# --------------------------------------------------

def score_alignment_drops_final(alignment: Alignment, requests: list[PointAnnotation]) -> float:
    """
    Final drop score for a completed alignment.
    Returns [0,1].

    This is intentionally non-additive:
    one bad request should hurt a lot.
    """
    tracks = alignment.tracks

    values: list[float] = []
    weights: list[float] = []

    for request in requests:
        if request.time is None:
            continue

        w = importance(request.strength)
        q = best_request_quality(request, tracks)

        values.append(q)
        weights.append(w)

    if not values:
        return 0.5

    return weighted_geometric_mean(values, weights)


def score_alignment_drops_partial(alignment: Alignment, requests: list[PointAnnotation]) -> float:
    """
    Partial drop score for beam search.
    Future requests are ignored.
    Past uncovered requests are heavily punished.
    Returns [0,1].
    """
    tracks = alignment.tracks

    if not tracks:
        return 0.5

    valid_ends = []
    for track in tracks:
        end = track_end_time(track)
        if end is not None:
            valid_ends.append(end)

    if not valid_ends:
        return 0.5

    covered_end = max(valid_ends)

    values: list[float] = []
    weights: list[float] = []

    for request in requests:
        if request.time is None:
            continue

        # ignore future requests for partial beam states
        if request.time > covered_end:
            continue

        w = importance(request.strength)
        q = best_request_quality(request, tracks)

        values.append(q)
        weights.append(w)

    if not values:
        return 0.5

    return weighted_geometric_mean(values, weights)
