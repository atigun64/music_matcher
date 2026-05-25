from pathlib import Path
from music_drop.src.training.labeling import load_labeled_samples
from music_drop.src.training.train import train_model


DATASET_ROOT = Path("music_drop", "data", "music_dataset")
LABEL_SPLIT = "train"

def main():
    labeled_samples = load_labeled_samples(split=LABEL_SPLIT)
    if len(labeled_samples) == 0:
        print(f"No labeled samples found in split='{LABEL_SPLIT}'.")
        return

    print(f"Loaded {len(labeled_samples)} labeled samples.")

    train_model(labeled_samples)
    print("Model trained.")


if __name__ == "__main__":
    main()
