from music_drop.src.training.active_learning import active_learning_loop
from pathlib import Path

DATASET_ROOT = Path("music_drop", "data", "music_dataset")

if __name__ == "__main__":
    train_track_ids = sorted([
        p.name
        for p in DATASET_ROOT.iterdir()
        if p.is_dir()
    ])

    model, labeled_samples = active_learning_loop(
        train_track_ids=train_track_ids,
        rounds=10,
        initial_label_count=50,
        batch_size=10,
        split="train",
    )
