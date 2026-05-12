import numpy as np
import librosa
import os
from matplotlib import pyplot as plt
import pandas as pd
import essentia.standard as es
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

audio_path = os.path.join("musics", "Jordan Schor, Harley Bird - Home [NCS Release].mp3")

# ----------------------------
# 1) Load audio with librosa
# ----------------------------
y, sr = librosa.load(audio_path, sr=44100, mono=True)

# ----------------------------
# 2) Beat tracking with Essentia
# ----------------------------
# Essentia loader
loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
y_es = loader()

# Rhythm extraction
rhythm = es.RhythmExtractor2013(method="multifeature")
out = rhythm(y_es)

# Usually: out[0] = bpm, out[1] = beat timestamps
bpm = float(out[0])
beat_times = np.array(out[1], dtype=float)


if len(beat_times) < 8:
    raise RuntimeError("Not enough beats detected. Try using librosa beat tracking fallback.")

# ----------------------------
# 3) Librosa features
# ----------------------------
hop = 512
n_fft = 2048

rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)[0]
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

bass_band = (freqs >= 20) & (freqs <= 150)
bass_energy = S[bass_band].mean(axis=0)
full_energy = S.mean(axis=0) + 1e-9
bass_ratio = bass_energy / full_energy

frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

def beat_sync(x):
    vals = []
    for i, t0 in enumerate(beat_times):
        print(t0)
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

E = beat_sync(rms)
O = beat_sync(onset_env)
C = beat_sync(centroid)
F = beat_sync(flatness)
B = beat_sync(bass_ratio)


# ----------------------------
# 4) Smooth features
# ----------------------------
def smooth(x, sigma=1.0):
    return gaussian_filter1d(x, sigma=sigma) if len(x) > 5 else x

E = smooth(E)
O = smooth(O)
C = smooth(C)
F = smooth(F)
B = smooth(B)

# ----------------------------
# 5) Derivative / trend features
# ----------------------------
def z(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-9)

dE = np.diff(E, prepend=E[0])
dO = np.diff(O, prepend=O[0])
dC = np.diff(C, prepend=C[0])
dB = np.diff(B, prepend=B[0])

def rolling_slope(x, win=8):
    out = np.full(len(x), np.nan, dtype=float)
    xs = np.arange(win, dtype=float)
    xs = xs - xs.mean()
    denom = np.sum(xs ** 2)

    for i in range(win - 1, len(x)):
        ywin = x[i - win + 1:i + 1]
        out[i] = np.dot(xs, ywin - ywin.mean()) / denom
    return out

trend_E = rolling_slope(E, win=8)
trend_O = rolling_slope(O, win=8)
trend_C = rolling_slope(C, win=8)
trend_F = rolling_slope(F, win=8)

# ----------------------------
# 6) Heuristic score
# ----------------------------
# Drop: sudden jump in energy, onset, bass
drop_score = (
    1.5 * z(dE) +
    1.2 * z(dO) +
    1.2 * z(dB) +
    1 * z(E)
)

# ----------------------------
# 7) Peak picking
# ----------------------------

drop_peaks, _ = find_peaks(drop_score, distance=4, prominence=0.5)

def top_candidates(peaks, score, n=5):
    if len(peaks) == 0:
        return []
    ranked = sorted(peaks, key=lambda i: score[i], reverse=True)[:n]
    rows = []
    for i in ranked:
        rows.append({
            "time_sec": round(float(beat_times[i]), 2),
            "time_min:sec": f"{int(beat_times[i]//60)}:{int(beat_times[i]%60):02d}",
            "score": round(float(score[i]), 3),
            "beat_idx": int(i),
        })
    return rows

drop_rows = top_candidates(drop_peaks, drop_score, n=5)

print(f"\nEstimated BPM: {bpm:.2f}\n")

print("\n=== Likely DROP points ===")
print(pd.DataFrame(drop_rows).to_string(index=False) if drop_rows else "None found")