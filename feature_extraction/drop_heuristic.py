import numpy as np

def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-9


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    scale = robust_scale(x)
    return (x - med) / scale

DROP_LEAD_BEATS = -3  # keep your current setting

def rate_window(window, candidate_idx=20, lead_beats=DROP_LEAD_BEATS):
    """
    Rate one feature window.

    Parameters
    ----------
    window : np.ndarray
        Shape (N, 4) or (N, >=4)
        Columns: normalized [energy, onset, centroid, bass]
    candidate_idx : int
        Index of the candidate / beat-0 inside the window.
        For your saved [-20, +10] windows, this is usually 20. (off set 0)
    lead_beats : int
        Your lead/offset correction.

    Returns
    -------
    float
        Score in roughly [0, 1].
    """
    window = np.asarray(window, dtype=float)

    if window.ndim != 2 or window.shape[1] < 4:
        raise ValueError("window must have shape (N, >=4) with columns [E, O, C, B]")

    E = window[:, 0]
    O = window[:, 1]
    C = window[:, 2]
    B = window[:, 3]

    n = len(E)

    # same slicing logic as your working detector,
    # but relative to the window's candidate index
    pre = slice(candidate_idx - 16 - lead_beats, candidate_idx - 10 - lead_beats)
    build = slice(candidate_idx - 10 - lead_beats, candidate_idx - 4 - lead_beats)
    drop = slice(candidate_idx - 4 - lead_beats, candidate_idx + 4 - lead_beats)

    # bounds check
    if (
        pre.start < 0 or drop.stop > n or
        build.start < 0 or build.stop > n
    ):
        return 0.0

    pre_E = np.mean(E[pre])
    build_E = np.mean(E[build])
    drop_E = np.mean(E[drop])

    pre_O = np.mean(O[pre])
    build_O = np.mean(O[build])
    drop_O = np.mean(O[drop])

    pre_C = np.mean(C[pre])
    build_C = np.mean(C[build])
    drop_C = np.mean(C[drop])

    pre_B = np.mean(B[pre])
    build_B = np.mean(B[build])
    drop_B = np.mean(B[drop])

    build_up = (
        0.40 * sigmoid(build_E - pre_E) +
        0.15 * sigmoid(build_O - pre_O) +
        0.15 * sigmoid(build_C - pre_C) +
        0.30 * sigmoid(build_B - pre_B)
    )

    drop_jump = (
        0.40 * sigmoid(drop_E - build_E) +
        0.15 * sigmoid(drop_O - build_O) +
        0.10 * sigmoid(drop_C - build_C) +
        0.35 * sigmoid(drop_B - build_B)
    )

    total_contrast = (
        0.35 * sigmoid(drop_E - pre_E) +
        0.15 * sigmoid(drop_O - pre_O) +
        0.10 * sigmoid(drop_C - pre_C) +
        0.40 * sigmoid(drop_B - pre_B)
    )

    score = (
        0.55 * drop_jump +
        0.30 * build_up +
        0.15 * total_contrast
    )

    return float(score)


def score_drops(E, O, C, B):
    E = np.asarray(E, dtype=float)
    O = np.asarray(O, dtype=float)
    C = np.asarray(C, dtype=float)
    B = np.asarray(B, dtype=float)

    n = min(len(E), len(O), len(C), len(B))
    E = E[:n]
    O = O[:n]
    C = C[:n]
    B = B[:n]

    Ez = robust_z(E)
    Oz = robust_z(O)
    Cz = robust_z(C)
    Bz = robust_z(B)

    score = np.zeros(n, dtype=float)

    for i in range(20, n - 10):
        window = np.column_stack([
            Ez[i - 20 : i + 10 + 1],
            Oz[i - 20 : i + 10 + 1],
            Cz[i - 20 : i + 10 + 1],
            Bz[i - 20 : i + 10 + 1],
        ])
        score[i] = rate_window(window)

    return score