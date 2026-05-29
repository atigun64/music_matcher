from pathlib import Path
import random
from mutagen import File as MutagenFile

from music_core import extract_features
from music_core.drop.drop_ml import get_ml_candidates

DATASET_ROOT = Path("music_drop/data/music_dataset")
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}
NUM_TRACKS = 5
MAX_DURATION_SEC = 5 * 60


def get_audio_files():
    return [
        p for p in DATASET_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]


def get_duration_sec(path: Path) -> float:
    audio = MutagenFile(path)
    if audio is None or not hasattr(audio, "info"):
        return 0.0
    return float(audio.info.length)


def main():
    files = get_audio_files()
    if not files:
        print(f"No audio files found in {DATASET_ROOT}")
        return

    random.shuffle(files)
    chosen = files[:NUM_TRACKS]

    print(f"Selected {len(chosen)} random tracks\n")

    for path in chosen:
        print("=" * 80)
        print(f"Track: {path.name}")

        try:
            if get_duration_sec(path) > MAX_DURATION_SEC:
                print("Skipped: longer than 5 minutes")
                continue

            E, O, C, F, B, _, beat_times, _ = extract_features(str(path))

            cands = get_ml_candidates(E, O, C, F, B, beat_times)

            if not cands:
                print("No candidates found.")
                continue

            print(f"Candidates ({len(cands)}):")
            for beat_idx, beat_time, score in cands:
                print(f"  beat={beat_idx:5d}  time={beat_time:8.2f}s  score={score:.3f}")

        except Exception as e:
            print(f"Failed: {e}")


if __name__ == "__main__":
    main()
