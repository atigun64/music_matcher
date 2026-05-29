import numpy as np

from music_core.preprocessing import delta_k, percentile_rank, previous_mean, robust_z

LEFT_CONTEXT = 20
RIGHT_CONTEXT = 10

def get_window(beat_idx, left=LEFT_CONTEXT, right=RIGHT_CONTEXT):
    return slice(beat_idx - left, beat_idx + right + 1)


def build_feature_window(E, O, B, C, beat_idx, left=20, right=10):
    """
    Build a feature window around a beat index.
    
    Args:
        E, O, B, C: Feature arrays
        beat_idx: Center beat index
        left: Beats to look left
        right: Beats to look right
    
    Returns:
        Feature window array of shape (left+right+1, 4)
    """
    w = get_window(int(beat_idx), left=left, right=right)
    window_size = w.stop - w.start

    x = np.zeros((window_size, 4), dtype=np.float32)

    src_start = max(0, w.start)
    src_stop = min(len(E), w.stop)

    if src_start < src_stop:
        dst_start = src_start - w.start
        dst_stop = dst_start + (src_stop - src_start)

        x[dst_start:dst_stop, 0] = E[src_start:src_stop]
        x[dst_start:dst_stop, 1] = O[src_start:src_stop]
        x[dst_start:dst_stop, 2] = C[src_start:src_stop]
        x[dst_start:dst_stop, 3] = B[src_start:src_stop]

    return x


def build_feature_window_ml(E, O, C, F, B, beat_idx, left=20, right=10, add_mask=True, add_pos=True):
    E = np.asarray(E, dtype=float)
    O = np.asarray(O, dtype=float)
    C = np.asarray(C, dtype=float)
    F = np.asarray(F, dtype=float)
    B = np.asarray(B, dtype=float)

    n = min(len(E), len(O), len(C), len(F), len(B))
    E, O, C, F, B = E[:n], O[:n], C[:n], F[:n], B[:n]

    # Raw stack
    raw = np.column_stack([E, O, C, F, B])

    # Track-wise robust normalization
    Z = np.column_stack([robust_z(raw[:, i]) for i in range(raw.shape[1])])

    # Percentile-like "how high is this in the track?"
    P = np.column_stack([percentile_rank(raw[:, i]) for i in range(raw.shape[1])]) - 0.5

    # Multi-scale deltas
    D1 = delta_k(Z, 1)
    D2 = delta_k(Z, 2)
    D4 = delta_k(Z, 4)

    # Contrast to short history
    C4 = Z - previous_mean(Z, 4)
    C8 = Z - previous_mean(Z, 8)

    track_X = np.concatenate([Z, P, D1, D2, D4, C4, C8], axis=1).astype(np.float32)

    w = get_window(beat_idx, left=left, right=right)
    window_size = w.stop - w.start

    extra = int(add_mask) + int(add_pos)
    x = np.zeros((window_size, track_X.shape[1] + extra), dtype=np.float32)

    src_start = max(0, w.start)
    src_stop = min(n, w.stop)

    if src_start < src_stop:
        dst_start = src_start - w.start
        dst_stop = dst_start + (src_stop - src_start)

        x[dst_start:dst_stop, :track_X.shape[1]] = track_X[src_start:src_stop]

        col = track_X.shape[1]

        if add_mask:
            x[dst_start:dst_stop, col] = 1.0
            col += 1

        if add_pos:
            rel = (np.arange(w.start, w.stop) - beat_idx).astype(np.float32)
            rel /= float(max(left, right))
            x[:, col] = rel

    return x

def window_times(beat_idx, beat_times):
    beat_times = np.asarray(beat_times, dtype=float)

    w = get_window(beat_idx)

    src_start = max(0, w.start)
    src_stop = min(len(beat_times), w.stop)

    if src_start >= src_stop:
        return None

    start_time = beat_times[src_start]

    # Prefer the next beat timestamp as end.
    if src_stop < len(beat_times):
        end_time = beat_times[src_stop]
    else:
        # Estimate final beat duration.
        if len(beat_times) >= 2:
            beat_dur = np.median(np.diff(beat_times))
        else:
            beat_dur = 0.5
        end_time = beat_times[src_stop - 1] + beat_dur

    return start_time, end_time
