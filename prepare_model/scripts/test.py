from pathlib import Path

from music_drop.src.training.sampling import heuristic_candidates, make_sample
from music_drop.src.training.labeling import label_batch, save_labeled_samples
from music_drop.src.cache import AudioFeatureCache


DATASET_ROOT = Path("music_drop", "data", "music_dataset")
MAX_PER_TRACK = 10   # optional cap, change or remove if you want


def get_track_ids():
    return sorted([
        p.name
        for p in DATASET_ROOT.iterdir()
        if p.is_dir()
    ])


def build_high_heuristic_batch(track_ids):
    cache = AudioFeatureCache()
    samples = []

    for track_id in track_ids:
        payload = cache.get_by_id(track_id)
        if payload is None:
            print(f"Skipping {track_id}: no payload")
            continue

        cand = heuristic_candidates(payload)

        # optional: take only top N per track
        cand = cand[:MAX_PER_TRACK]

        for beat_idx, _, hscore in cand:
            samples.append(
                make_sample(
                    track_id=track_id,
                    beat_idx=beat_idx,
                    source="heuristic",
                    hscore=hscore,
                    payload=payload,
                )
            )

    return samples


if __name__ == "__main__":
    track_ids = get_track_ids()
    samples = build_high_heuristic_batch(track_ids)

    print(f"Collected {len(samples)} heuristic samples")

    labeled = label_batch(samples)
    save_labeled_samples(labeled, split="train")
