import numpy as np
from scipy.ndimage import gaussian_filter1d

def smooth(x, sigma=1.0):
    return gaussian_filter1d(x, sigma=sigma) if len(x) > 5 else x

def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-9

def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    scale = robust_scale(x)
    return (x - med) / scale

def sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def percentile_rank(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks / (len(x) - 1)

def delta_k(X, k=1):
    X = np.asarray(X, dtype=float)
    out = np.zeros_like(X)
    if len(X) > k:
        out[k:] = X[k:] - X[:-k]
    return out

def previous_mean(X, k=4):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    T, D = X.shape
    out = np.zeros_like(X)
    csum = np.vstack([np.zeros((1, D)), np.cumsum(X, axis=0)])

    for t in range(T):
        start = max(0, t - k)
        stop = t
        if stop > start:
            out[t] = (csum[stop] - csum[start]) / (stop - start)
        else:
            out[t] = X[t]
    return out
