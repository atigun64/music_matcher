import numpy as np

def beat_sync(x, beat_times, frame_times):
    """
      Takes an array x where x[i] represent a feature at frame i.
      And makes an array vals[i], 
      which represent the same feature for beat i
      which is average of x[beat_times[i]: beat_times[i+1]]
    """
    vals = []
    for i, t0 in enumerate(beat_times):
        t1 = beat_times[i + 1] if i < len(beat_times) - 1 else frame_times[-1]
        mask = (frame_times >= t0) & (frame_times < t1)
        vals.append(np.mean(x[mask]) if np.any(mask) else np.nan)
    vals = np.array(vals, dtype=float)

    # fill gaps
    if np.any(np.isnan(vals)):
        good = np.where(~np.isnan(vals))[0]
        bad = np.where(np.isnan(vals))[0]
        vals[bad] = np.interp(bad, good, vals[good])

    return vals