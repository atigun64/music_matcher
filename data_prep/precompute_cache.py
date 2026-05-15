from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue
import time

import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATASET_ROOT = Path("ncs_dataset")
CACHE_ROOT = Path("feature_cache")
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

TIMEOUT_PER_SONG_SEC = 240  # adjust if needed

CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FILE DISCOVERY
# --------------------------------------------------

def collect_audio_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def feature_cache_path(audio_path: Path, dataset_root: Path) -> Path:
    rel = audio_path.relative_to(dataset_root)
    safe = "_".join(rel.with_suffix("").parts).replace(" ", "_")
    return CACHE_ROOT / f"{safe}.npz"


# --------------------------------------------------
# CACHE IO
# --------------------------------------------------

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
        from feature_extraction.extract_features import extract_features
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
                {
                    "E": np.asarray(E),
                    "O": np.asarray(O),
                    "C": np.asarray(C),
                    "F": np.asarray(F),
                    "B": np.asarray(B),
                    "bpm": float(bpm),
                    "beat_times": np.asarray(beat_times),
                    "frame_times": np.asarray(frame_times),
                }
            ))
        except Exception as e:
            out_q.put(("error", job_id, repr(e)))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    if not DATASET_ROOT.exists():
        raise RuntimeError(f"Dataset root not found: {DATASET_ROOT}")

    files = collect_audio_files(DATASET_ROOT)
    print(f"Found {len(files)} audio files")

    # First pass: filter out valid caches
    todo = []
    for path in files:
        cache_path = feature_cache_path(path, DATASET_ROOT)
        cached = load_cache_if_valid(path, cache_path)
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

        cache_path = feature_cache_path(path, DATASET_ROOT)
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


if __name__ == "__main__":
    mp.freeze_support()
    main()
