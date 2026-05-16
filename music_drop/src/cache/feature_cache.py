from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any

import numpy as np

from .config import CACHE_ROOT, AUDIO_EXTS, DATASET_ROOT, TIMEOUT_PER_SONG_SEC

CACHE_ROOT.mkdir(parents=True, exist_ok=True)

FeaturePayload = dict[str, Any]

# --------------------------------------------------
# FILE DISCOVERY
# --------------------------------------------------

def collect_audio_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def feature_cache_path(
    audio_path: Path,
    dataset_root: Path,
    cache_root: Path = CACHE_ROOT,
) -> Path:
    rel = audio_path.relative_to(dataset_root)
    safe = "_".join(rel.with_suffix("").parts).replace(" ", "_")
    return cache_root / f"{safe}.npz"


# --------------------------------------------------
# CACHE IO
# --------------------------------------------------

def _make_payload(E, O, C, F, B, bpm, beat_times, frame_times) -> FeaturePayload:
    return {
        "E": np.asarray(E),
        "O": np.asarray(O),
        "C": np.asarray(C),
        "F": np.asarray(F),
        "B": np.asarray(B),
        "bpm": float(bpm),
        "beat_times": np.asarray(beat_times),
        "frame_times": np.asarray(frame_times),
    }


def load_cache_if_valid(audio_path: Path, cache_path: Path):
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=False)
        cached_mtime = float(data["source_mtime"])
        current_mtime = audio_path.stat().st_mtime

        if abs(cached_mtime - current_mtime) > 1e-6:
            return None

        return {
            "E": data["E"],
            "O": data["O"],
            "C": data["C"],
            "F": data["F"],
            "B": data["B"],
            "bpm": float(data["bpm"]),
            "beat_times": data["beat_times"],
            "frame_times": data["frame_times"],
        }
    except Exception:
        return None


def save_cache(
    cache_path: Path,
    source_path: Path,
    E,
    O,
    C,
    F,
    B,
    bpm,
    beat_times,
    frame_times,
):
    np.savez_compressed(
        cache_path,
        source_path=str(source_path),
        source_mtime=np.float64(source_path.stat().st_mtime),
        E=np.asarray(E, dtype=np.float32),
        O=np.asarray(O, dtype=np.float32),
        C=np.asarray(C, dtype=np.float32),
        F=np.asarray(F, dtype=np.float32),
        B=np.asarray(B, dtype=np.float32),
        bpm=np.float32(bpm),
        beat_times=np.asarray(beat_times, dtype=np.float32),
        frame_times=np.asarray(frame_times, dtype=np.float32),
    )


def load_cached_features(
    audio_path: Path,
    dataset_root: Path = DATASET_ROOT,
    cache_root: Path = CACHE_ROOT,
):
    cache_path = feature_cache_path(audio_path, dataset_root, cache_root)
    return load_cache_if_valid(audio_path, cache_path)


def save_cached_features(
    audio_path: Path,
    payload: FeaturePayload,
    dataset_root: Path = DATASET_ROOT,
    cache_root: Path = CACHE_ROOT,
):
    cache_path = feature_cache_path(audio_path, dataset_root, cache_root)
    save_cache(
        cache_path=cache_path,
        source_path=audio_path,
        E=payload["E"],
        O=payload["O"],
        C=payload["C"],
        F=payload["F"],
        B=payload["B"],
        bpm=payload["bpm"],
        beat_times=payload["beat_times"],
        frame_times=payload["frame_times"],
    )


# --------------------------------------------------
# PUBLIC API CLASS
# --------------------------------------------------

class AudioFeatureCache:
    """
    Public API for reading/saving cached audio features.

    Cache format and file naming are unchanged, so existing caches remain valid.
    """

    def __init__(
        self,
        dataset_root: Path = DATASET_ROOT,
        cache_root: Path = CACHE_ROOT,
    ):
        self.dataset_root = Path(dataset_root)
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def cache_path(self, audio_path: Path) -> Path:
        return feature_cache_path(audio_path, self.dataset_root, self.cache_root)

    def get(self, audio_path: Path):
        """Load cached features if the cache is valid, otherwise return None."""
        return load_cache_if_valid(audio_path, self.cache_path(audio_path))

    def load(self, audio_path: Path):
        """Alias for get()."""
        return self.get(audio_path)

    def has(self, audio_path: Path) -> bool:
        return self.get(audio_path) is not None

    def save(self, audio_path: Path, payload: FeaturePayload):
        """Save one feature payload to cache."""
        save_cache(
            cache_path=self.cache_path(audio_path),
            source_path=audio_path,
            E=payload["E"],
            O=payload["O"],
            C=payload["C"],
            F=payload["F"],
            B=payload["B"],
            bpm=payload["bpm"],
            beat_times=payload["beat_times"],
            frame_times=payload["frame_times"],
        )

    def extract(self, audio_path: Path) -> FeaturePayload:
        """Extract features for one audio file, but do not save."""
        from music_core.features.extract_features import extract_features

        E, O, C, F, B, bpm, beat_times, frame_times = extract_features(str(audio_path))
        return _make_payload(E, O, C, F, B, bpm, beat_times, frame_times)

    def get_or_extract(self, audio_path: Path) -> FeaturePayload:
        """
        Return cached features if present and valid.
        Otherwise extract them, save them, and return them.
        """
        cached = self.get(audio_path)
        if cached is not None:
            return cached

        payload = self.extract(audio_path)
        self.save(audio_path, payload)
        return payload

    def collect_audio_files(self, root: Path | None = None):
        return collect_audio_files(self.dataset_root if root is None else root)
    
    def audio_path_from_id(self, music_id: str) -> Path | None:
        folder = self.dataset_root / music_id
        if not folder.exists():
            return None

        files = collect_audio_files(folder)
        if not files:
            return None

        # If each folder has one song, this is enough:
        return files[0]

    def get_by_id(self, music_id: str):
        """Load cached features using the direct parent folder name as ID."""
        audio_path = self.audio_path_from_id(music_id)
        if audio_path is None:
            return None
        return self.get(audio_path)

    def get_or_extract_by_id(self, music_id: str):
        """Load cached features by ID, or extract/save if missing."""
        audio_path = self.audio_path_from_id(music_id)
        if audio_path is None:
            raise FileNotFoundError(f"No audio file found for ID: {music_id}")

        return self.get_or_extract(audio_path)


# --------------------------------------------------
# WORKER PROCESS
# --------------------------------------------------

def worker_loop(in_q: mp.Queue, out_q: mp.Queue):
    """
    Persistent worker:
    - imports extractor once
    - processes many songs
    """
    try:
        from music_core.features.extract_features import extract_features
    except Exception as e:
        out_q.put(("fatal_import_error", repr(e)))
        return

    while True:
        item = in_q.get()
        if item is None:
            break

        job_id, audio_path_str = item
        try:
            E, O, C, F, B, bpm, beat_times, frame_times = extract_features(audio_path_str)
            out_q.put((
                "ok",
                job_id,
                _make_payload(E, O, C, F, B, bpm, beat_times, frame_times),
            ))
        except Exception as e:
            out_q.put(("error", job_id, repr(e)))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    cache = AudioFeatureCache(DATASET_ROOT, CACHE_ROOT)

    if not cache.dataset_root.exists():
        raise RuntimeError(f"Dataset root not found: {cache.dataset_root}")

    files = cache.collect_audio_files()
    print(f"Found {len(files)} audio files")

    # First pass: filter out valid caches
    todo = []
    for path in files:
        cached = cache.get(path)
        if cached is not None:
            print(f"[cache hit] {path}")
        else:
            todo.append(path)

    print(f"{len(todo)} files need extraction")

    if not todo:
        print("All files already cached.")
        return

    ctx = mp.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue()

    proc = ctx.Process(target=worker_loop, args=(in_q, out_q), daemon=True)
    proc.start()

    # Send all jobs
    for job_id, path in enumerate(todo):
        in_q.put((job_id, str(path)))

    # Stop marker
    in_q.put(None)

    completed = 0
    failed = 0
    start_time = time.time()

    # Receive results
    results = {}
    expected = len(todo)

    while completed + failed < expected:
        try:
            msg = out_q.get(timeout=TIMEOUT_PER_SONG_SEC)
        except queue.Empty:
            print("Timeout waiting for worker. Terminating.")
            proc.terminate()
            proc.join()
            break

        tag = msg[0]

        if tag == "fatal_import_error":
            print(f"FATAL: extractor import failed: {msg[1]}")
            proc.terminate()
            proc.join()
            return

        if tag == "ok":
            _, job_id, payload = msg
            results[job_id] = payload
            completed += 1
            print(f"[{completed + failed}/{expected}] extracted ok")

        elif tag == "error":
            _, job_id, err = msg
            results[job_id] = None
            failed += 1
            print(f"[{completed + failed}/{expected}] FAILED: {err}")

    proc.join(timeout=5)

    # Save results
    print("Saving caches...")
    for job_id, path in enumerate(todo):
        payload = results.get(job_id)
        if payload is None:
            continue

        cache_path = cache.cache_path(path)
        try:
            save_cache(
                cache_path=cache_path,
                source_path=path,
                E=payload["E"],
                O=payload["O"],
                C=payload["C"],
                F=payload["F"],
                B=payload["B"],
                bpm=payload["bpm"],
                beat_times=payload["beat_times"],
                frame_times=payload["frame_times"],
            )
            print(f"  saved: {cache_path}")
        except Exception as e:
            print(f"  save failed for {path}: {e}")

    elapsed = time.time() - start_time
    print(f"Done. Elapsed: {elapsed:.1f}s")


__all__ = [
    "AudioFeatureCache",
    "collect_audio_files",
    "feature_cache_path",
    "load_cache_if_valid",
    "save_cache",
    "load_cached_features",
    "save_cached_features",
    "main",
]


if __name__ == "__main__":
    mp.freeze_support()
    main()
