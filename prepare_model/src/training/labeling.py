from typing import List
from pathlib import Path
import json

from .data import Sample
from .sampling import make_sample

from music_core import window_times
from music_drop.src.cache import AudioFeatureCache
from music_drop.src.ui import UISample, UI


LABEL_DIR = Path("music_drop/data/labeled_samples")

def _label_file(split: str) -> Path:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    return LABEL_DIR / f"{split}.jsonl"


def _sample_key(track_id: str, beat_idx: int) -> str:
    return f"{track_id}:{beat_idx}"


def label_batch(samples: List[Sample]) -> List[Sample]:
    ui_samples: list[UISample] = []

    cache = AudioFeatureCache()

    for s in samples:
        audio_path = cache.audio_path_from_id(s.track_id)
        beat_times = cache.get_by_id(s.track_id)["beat_times"]

        left_idx = max(0, s.beat_idx - 5)
        right_idx = min(len(beat_times) - 1, s.beat_idx + 5)

        left_time = beat_times[left_idx]
        right_time = beat_times[right_idx]

        ui_samples.append(
            UISample(
                track_path=audio_path,
                key_point=beat_times[s.beat_idx],
                time_window=window_times(beat_idx=s.beat_idx, beat_times=beat_times),
                tolerance_window=(left_time, right_time),
                model_score=s.mscore,
            )
        )


    labels = UI.sample(ui_samples)

    labeled_samples: List[Sample] = []
    for idx, s in enumerate(samples):
        if labels[idx] is not None:
            s.y = int(labels[idx])
            labeled_samples.append(s)

    return labeled_samples


def save_labeled_samples(samples: List[Sample], split: str = "train") -> None:
    """
    Save labeled samples to disk.
    Uses upsert behavior:
    - existing (track_id, beat_idx) entries get replaced
    - new ones are appended
    """
    path = _label_file(split)

    existing: dict[str, dict] = {}

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = _sample_key(rec["track_id"], rec["beat_idx"])
                existing[key] = rec

    for s in samples:
        if s.y is None:
            continue

        rec = {
            "track_id": s.track_id,
            "beat_idx": int(s.beat_idx),
            "y": int(s.y),
            "source": s.source,
            "hscore": float(s.hscore),
            "mscore": float(s.mscore),
        }
        existing[_sample_key(s.track_id, s.beat_idx)] = rec

    with path.open("w", encoding="utf-8") as f:
        for rec in existing.values():
            f.write(json.dumps(rec) + "\n")


def rewrite_labeled_samples(samples: List[Sample], split: str = "train") -> None:
    """
    Rewrite all the samples on disk
    clears the existing file and writes only the provided samples.
    """
    path = _label_file(split)

    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            if s.y is None:
                continue

            rec = {
                "track_id": s.track_id,
                "beat_idx": int(s.beat_idx),
                "y": int(s.y),
                "source": s.source,
                "hscore": float(s.hscore),
                "mscore": float(s.mscore),
            }
            f.write(json.dumps(rec) + "\n")
    

def load_labeled_samples(split: str = "train") -> List[Sample]:
    """
    Load labeled samples from disk and rebuild their x features.
    """
    path = _label_file(split)
    if not path.exists():
        return []

    samples: List[Sample] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            s = make_sample(
                track_id=rec["track_id"],
                beat_idx=int(rec["beat_idx"]),
                source=rec.get("source", ""),
                hscore=float(rec.get("hscore", 0.0)),
                mscore=float(rec.get("mscore", 0.0)),
            )
            s.y = int(rec["y"])
            samples.append(s)

    return samples