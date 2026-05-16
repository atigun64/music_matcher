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
    return 1.0 / (1.0 + np.exp(-x))
