from pathlib import Path
import random
import numpy as np
from scipy.signal import find_peaks

from feature_extraction.extract_features import extract_features
from feature_extraction.drop_heuristic import score_drops

DATASET_ROOT = Path("ncs_dataset")
THRESHOLD = 0

def get_random_song():
    files = []
    for p in DATASET_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
            files.append(p)

    if not files:
        raise RuntimeError(f"No audio files found in {DATASET_ROOT}")

    return random.choice(files)

def suppress_close_peaks(peaks, score, beat_times, min_sep_sec=10.0):
    """
    Keep only the best peak in any time neighborhood smaller than min_sep_sec.
    """
    if len(peaks) == 0:
        return []

    # sort peaks by score descending
    ranked = sorted(peaks, key=lambda i: score[i], reverse=True)

    kept = []
    kept_times = []

    for i in ranked:
        t = float(beat_times[i]) if i < len(beat_times) else float(i)

        # reject if too close to an already kept peak
        too_close = False
        for kt in kept_times:
            if abs(t - kt) < min_sep_sec:
                too_close = True
                break

        if not too_close:
            kept.append(i)
            kept_times.append(t)

    # sort final peaks in time order
    kept = sorted(kept, key=lambda i: beat_times[i] if i < len(beat_times) else i)
    return kept


def main():
    song_path = get_random_song()
    print(f"Selected song: {song_path}")

    E, O, C, F, B, bpm, beat_times, frame_times = extract_features(str(song_path))
    score = score_drops(E, O, B, C)

    peaks, props = find_peaks(score, height=THRESHOLD, distance=4)
    peaks = suppress_close_peaks(peaks, score, beat_times, min_sep_sec=10.0)

    if len(peaks) == 0:
        print("No drop candidates found above threshold.")
        return

    print(f"\nEstimated BPM: {bpm:.2f}")
    print("\n=== Likely DROP points ===")

    ranked = sorted(peaks, key=lambda i: score[i], reverse=True)

    for i in ranked[:10]:
        t = float(beat_times[i]) if i < len(beat_times) else float(i)
        print(
            f"time_sec={t:.2f}  "
            f"time_min:sec={int(t//60)}:{int(t%60):02d}  "
            f"score={score[i]:.3f}  "
            f"beat_idx={i}"
        )

if __name__ == "__main__":
    main()