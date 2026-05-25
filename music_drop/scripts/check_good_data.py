from pathlib import Path
import random
import numpy as np
from music_drop.src.training.labeling import load_labeled_samples, save_labeled_samples
from music_drop.src.training.sampling import build_pool, sample_to_vector
from music_drop.src.training.train import train_model
from music_drop.src.ui import UISample, UI
from music_drop.src.cache import AudioFeatureCache
from music_core import window_times


DATASET_ROOT = Path("music_drop", "data", "music_dataset")
LABEL_SPLIT = "train"
MODEL_THRESHOLD = (0.6, 1.0)

HEURISTIC_PER_TRACK = 20
BACKGROUND_PER_TRACK = 0


def get_track_ids():
    return sorted([
        p.name
        for p in DATASET_ROOT.iterdir()
        if p.is_dir()
    ])


def sample_to_ui(sample, cache: AudioFeatureCache, model_score: float) -> UISample:
    payload = cache.get_by_id(sample.track_id)
    beat_times = payload["beat_times"]
    audio_path = cache.audio_path_from_id(sample.track_id)

    left_idx = max(0, sample.beat_idx - 5)
    right_idx = min(len(beat_times) - 1, sample.beat_idx + 5)

    return UISample(
        track_path=audio_path,
        key_point=beat_times[sample.beat_idx],
        time_window=window_times(beat_idx=sample.beat_idx, beat_times=beat_times),
        tolerance_window=(beat_times[left_idx], beat_times[right_idx]),
        model_score=float(model_score),
    )


def main():
    cache = AudioFeatureCache()

    labeled_samples = load_labeled_samples(split=LABEL_SPLIT)
    if len(labeled_samples) == 0:
        print(f"No labeled samples found in split='{LABEL_SPLIT}'.")
        return

    labeled_keys = {(s.track_id, s.beat_idx) for s in labeled_samples}
    print(f"Loaded {len(labeled_samples)} labeled samples.")

    model = train_model(labeled_samples)
    print("Model trained.")

    track_ids = get_track_ids()
    pool = build_pool(
        track_ids,
        heuristic_per_track=HEURISTIC_PER_TRACK,
        background_per_track=BACKGROUND_PER_TRACK,
    )

    pool = [
        s for s in pool
        if (s.track_id, s.beat_idx) not in labeled_keys
    ]

    if not pool:
        print("No unlabeled samples in the pool.")
        return

    X = np.stack([sample_to_vector(s) for s in pool])
    probs = model.predict_proba(X)[:, 1]

    # save model scores on samples
    for s, p in zip(pool, probs):
        s.mscore = float(p)

    # pick strong candidates
    selected_samples = [
        s for s, p in zip(pool, probs)
        if MODEL_THRESHOLD[0] <= s.mscore < MODEL_THRESHOLD[1]
    ]

    random.shuffle(selected_samples)

    print(f"Selected {len(selected_samples)} samples with model_score in range {MODEL_THRESHOLD}")

    if not selected_samples:
        print("No samples passed threshold.")
        return

    ui_samples = [
        sample_to_ui(s, cache, s.mscore)
        for s in selected_samples
    ]

    # UI returns labels in the same order
    labels = UI.sample(ui_samples)

    # write labels back to samples
    newly_labeled = []
    for s, y in zip(selected_samples, labels):
        if y is None:
            continue
        s.y = int(y)
        newly_labeled.append(s)

    save_labeled_samples(newly_labeled, split=LABEL_SPLIT)
    print(f"Saved {len(newly_labeled)} new labels.")


if __name__ == "__main__":
    main()
