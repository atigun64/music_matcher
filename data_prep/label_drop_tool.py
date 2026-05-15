from __future__ import annotations

import json
import subprocess
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import ttk


# ==================================================
# CONFIG
# ==================================================

DATASET_ROOT = Path("ncs_dataset")
CACHE_ROOT = Path("feature_cache")
SAVE_ROOT = Path("drop_labels")

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# Window around candidate:
WINDOW_BEFORE_BEATS = 20
WINDOW_AFTER_BEATS = 10

# Candidate detection:
MAX_CANDIDATES_PER_SONG = 5
PEAK_DISTANCE = 4
PEAK_PROMINENCE = 0.08

# Save classes 0..5
NUM_CLASSES = 6

FEATURE_NAMES = ["energy", "onset", "centroid", "flatness", "bass"]

# Playback refresh
PLAYBACK_UPDATE_MS = 50


# ==================================================
# HELPERS
# ==================================================

def robust_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + 1e-9


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_stem(path: Path) -> str:
    parts = path.with_suffix("").parts
    return "_".join(parts[-3:]).replace(" ", "_")


def collect_audio_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def create_dirs():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for c in range(NUM_CLASSES):
        (SAVE_ROOT / f"class_{c}").mkdir(parents=True, exist_ok=True)


def feature_cache_path(audio_path: Path, dataset_root: Path) -> Path:
    rel = audio_path.relative_to(dataset_root)
    safe = "_".join(rel.with_suffix("").parts).replace(" ", "_")
    return CACHE_ROOT / f"{safe}.npz"


def play_clip_external(audio_path: Path, start_sec: float, end_sec: float):
    ffplay = shutil.which("ffplay")
    if ffplay is None:
        raise RuntimeError("ffplay not found. Install it with: sudo apt install ffmpeg")

    dur = max(0.05, end_sec - start_sec)

    cmd = [
        ffplay,
        "-nodisp",
        "-autoexit",
        "-loglevel", "error",
        "-ss", str(start_sec),
        "-t", str(dur),
        str(audio_path),
    ]

    return subprocess.Popen(cmd)


# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass
class Candidate:
    beat_idx: int
    time_sec: float
    score: float


@dataclass
class SongData:
    path: Path
    E: np.ndarray
    O: np.ndarray
    C: np.ndarray
    F: np.ndarray
    B: np.ndarray
    bpm: float
    beat_times: np.ndarray
    frame_times: np.ndarray
    candidates: list[Candidate]


class PlaybackState:
    def __init__(self):
        self.active = False
        self.song: Optional[SongData] = None
        self.candidate: Optional[Candidate] = None
        self.window_start_sec = 0.0
        self.window_end_sec = 0.0
        self.play_start_real = 0.0
        self.window_start_idx = 0
        self.window_end_idx = 0
        self.proc = None


# ==================================================
# CACHE IO
# ==================================================

def load_cached_features(path: Path, dataset_root: Path):
    cache_path = feature_cache_path(path, dataset_root)
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=False)
        current_mtime = path.stat().st_mtime
        cached_mtime = float(data["source_mtime"])
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


# ==================================================
# CANDIDATE DETECTION
# ==================================================

def robust_z(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    scale = robust_scale(x)
    return (x - med) / scale


DROP_LEAD_BEATS = -3

def build_candidate_score(E, O, C, B):
    E = np.asarray(E, dtype=float)
    O = np.asarray(O, dtype=float)
    C = np.asarray(C, dtype=float)
    B = np.asarray(B, dtype=float)

    n = min(len(E), len(O), len(C), len(B))
    E = E[:n]
    O = O[:n]
    C = C[:n]
    B = B[:n]

    Ez = robust_z(E)
    Oz = robust_z(O)
    Cz = robust_z(C)
    Bz = robust_z(B)

    score = np.zeros(n, dtype=float)

    for i in range(WINDOW_BEFORE_BEATS, n - WINDOW_AFTER_BEATS):
        pre = slice(i - 16 - DROP_LEAD_BEATS, i - 10 - DROP_LEAD_BEATS)
        build = slice(i - 10 - DROP_LEAD_BEATS, i - 4 - DROP_LEAD_BEATS)
        drop = slice(i - 4 - DROP_LEAD_BEATS, i + 4 - DROP_LEAD_BEATS)

        pre_E = np.mean(Ez[pre])
        build_E = np.mean(Ez[build])
        drop_E = np.mean(Ez[drop])

        pre_O = np.mean(Oz[pre])
        build_O = np.mean(Oz[build])
        drop_O = np.mean(Oz[drop])

        pre_C = np.mean(Cz[pre])
        build_C = np.mean(Cz[build])
        drop_C = np.mean(Cz[drop])

        pre_B = np.mean(Bz[pre])
        build_B = np.mean(Bz[build])
        drop_B = np.mean(Bz[drop])

        build_up = (
            0.40 * sigmoid(build_E - pre_E) +
            0.15 * sigmoid(build_O - pre_O) +
            0.15 * sigmoid(build_C - pre_C) +
            0.30 * sigmoid(build_B - pre_B)
        )

        drop_jump = (
            0.40 * sigmoid(drop_E - build_E) +
            0.15 * sigmoid(drop_O - build_O) +
            0.10 * sigmoid(drop_C - build_C) +
            0.35 * sigmoid(drop_B - build_B)
        )

        total_contrast = (
            0.35 * sigmoid(drop_E - pre_E) +
            0.15 * sigmoid(drop_O - pre_O) +
            0.10 * sigmoid(drop_C - pre_C) +
            0.40 * sigmoid(drop_B - pre_B)
        )

        score[i] = (
            0.55 * drop_jump +
            0.30 * build_up +
            0.15 * total_contrast
        )

    return score




def detect_candidates(E, O, C, B, beat_times):
    score = build_candidate_score(E, O, C, B)
    peaks, _ = find_peaks(score, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)

    if len(peaks) == 0:
        return []

    ranked = sorted(peaks, key=lambda i: score[i], reverse=True)[:MAX_CANDIDATES_PER_SONG]
    out = []
    for i in ranked:
        if 0 <= i < len(beat_times):
            out.append(Candidate(int(i), float(beat_times[i]), float(score[i])))
    return out



# ==================================================
# WINDOW EXTRACTION
# ==================================================

def extract_window(E, O, C, F, B, beat_times, center_idx):
    n = min(len(E), len(O), len(C), len(F), len(B), len(beat_times))
    E = np.asarray(E[:n], dtype=float)
    O = np.asarray(O[:n], dtype=float)
    C = np.asarray(C[:n], dtype=float)
    F = np.asarray(F[:n], dtype=float)
    B = np.asarray(B[:n], dtype=float)
    beat_times = np.asarray(beat_times[:n], dtype=float)

    start_idx = center_idx - WINDOW_BEFORE_BEATS
    end_idx = center_idx + WINDOW_AFTER_BEATS
    if start_idx < 0 or end_idx >= n:
        raise ValueError("Candidate too close to edge for window.")

    offsets = np.arange(-WINDOW_BEFORE_BEATS, WINDOW_AFTER_BEATS + 1, dtype=int)
    rows = []
    bt = []
    for off in offsets:
        idx = center_idx + off
        rows.append([E[idx], O[idx], C[idx], F[idx], B[idx]])
        bt.append(float(beat_times[idx]))

    return (
        offsets.astype(np.int32),
        np.asarray(rows, dtype=np.float32),
        np.asarray(bt, dtype=np.float32),
    )


def window_bounds_from_candidate(beat_times: np.ndarray, center_idx: int):
    start_idx = center_idx - WINDOW_BEFORE_BEATS
    end_idx = center_idx + WINDOW_AFTER_BEATS
    if start_idx < 0 or end_idx >= len(beat_times):
        raise ValueError("Candidate too close to edge.")
    return float(beat_times[start_idx]), float(beat_times[end_idx]), start_idx, end_idx


# ==================================================
# SONG LOADING
# ==================================================

def load_song_data(path: Path) -> Optional[SongData]:
    cached = load_cached_features(path, DATASET_ROOT)
    if cached is None:
        return None

    E = np.asarray(cached["E"], dtype=float)
    O = np.asarray(cached["O"], dtype=float)
    C = np.asarray(cached["C"], dtype=float)
    F = np.asarray(cached["F"], dtype=float)
    B = np.asarray(cached["B"], dtype=float)
    beat_times = np.asarray(cached["beat_times"], dtype=float)
    frame_times = np.asarray(cached["frame_times"], dtype=float)
    bpm = float(cached["bpm"])

    n = min(len(E), len(O), len(C), len(F), len(B), len(beat_times))
    if n < WINDOW_BEFORE_BEATS + WINDOW_AFTER_BEATS + 10:
        return None

    E = E[:n]
    O = O[:n]
    C = C[:n]
    F = F[:n]
    B = B[:n]
    beat_times = beat_times[:n]
    candidates = detect_candidates(E, O, C, B, beat_times)

    if len(candidates) == 0:
        return None

    return SongData(
        path=path,
        E=E,
        O=O,
        C=C,
        F=F,
        B=B,
        bpm=bpm,
        beat_times=beat_times,
        frame_times=frame_times,
        candidates=candidates,
    )


# ==================================================
# GUI APP
# ==================================================

class DropLabelTool:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.audio_files = collect_audio_files(dataset_root)
        if not self.audio_files:
            raise RuntimeError(f"No audio files found under {dataset_root}")

        create_dirs()
        self.manifest_path = SAVE_ROOT / "manifest.jsonl"

        self.song_idx = 0
        self.cand_idx = 0
        self.current_song: Optional[SongData] = None
        self.current_candidate: Optional[Candidate] = None

        self.playback = PlaybackState()

        self.root = tk.Tk()
        self.root.title("Drop Label Tool")

        self.file_var = tk.StringVar()
        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.play_var = tk.StringVar()

        self._build_ui()
        self._bind_keys()

        self.load_song(self.song_idx)
        self._tick()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, textvariable=self.file_var, wraplength=900, justify="left").pack(anchor="w")
        ttk.Label(top, textvariable=self.info_var, wraplength=900, justify="left").pack(anchor="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.status_var, wraplength=900, justify="left", foreground="blue").pack(anchor="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.play_var, wraplength=900, justify="left", foreground="green").pack(anchor="w", pady=(4, 8))

        self.canvas = tk.Canvas(top, width=900, height=140, bg="white")
        self.canvas.pack(fill="x", pady=(0, 10))

        btn_row = ttk.Frame(top)
        btn_row.pack(fill="x", pady=4)

        ttk.Button(btn_row, text="Prev Cand", command=self.prev_candidate).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Next Cand", command=self.next_candidate).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Play", command=self.play_candidate).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Stop", command=self.stop_playback).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Next Song", command=self.next_song).pack(side="left", padx=2)

        rate_row = ttk.Frame(top)
        rate_row.pack(fill="x", pady=(8, 4))

        ttk.Label(rate_row, text="Label:").pack(side="left", padx=(0, 8))
        for label in range(NUM_CLASSES):
            ttk.Button(
                rate_row,
                text=str(label),
                command=lambda l=label: self.rate_candidate(l),
                width=4
            ).pack(side="left", padx=3)

        ttk.Label(
            top,
            text="Keys: 0-5 label | Left/Right candidate | Space play | S stop | N next song",
            foreground="gray"
        ).pack(anchor="w", pady=(8, 0))

    def _bind_keys(self):
        for i in range(NUM_CLASSES):
            self.root.bind(str(i), lambda e, l=i: self.rate_candidate(l))
        self.root.bind("<Left>", lambda e: self.prev_candidate())
        self.root.bind("<Right>", lambda e: self.next_candidate())
        self.root.bind("<space>", lambda e: self.play_candidate())
        self.root.bind("s", lambda e: self.stop_playback())
        self.root.bind("n", lambda e: self.next_song())

    def run(self):
        self.root.mainloop()

    def load_song(self, idx: int):
        while 0 <= idx < len(self.audio_files):
            path = self.audio_files[idx]
            self.status_var.set(f"Loading cached features: {path}")
            self.root.update_idletasks()

            song = load_song_data(path)
            if song is None:
                idx += 1
                continue

            self.song_idx = idx
            self.current_song = song
            self.cand_idx = 0
            self.current_candidate = song.candidates[0]
            self.stop_playback()
            self.update_view()
            return

        self.current_song = None
        self.current_candidate = None
        self.file_var.set("No more songs.")
        self.info_var.set("")
        self.status_var.set("Finished all songs.")
        self.play_var.set("")
        self.canvas.delete("all")

    def next_song(self):
        self.load_song(self.song_idx + 1)

    def next_candidate(self):
        if not self.current_song:
            return

        if self.cand_idx + 1 < len(self.current_song.candidates):
            self.cand_idx += 1
            self.current_candidate = self.current_song.candidates[self.cand_idx]
            self.stop_playback()
            self.update_view()
        else:
            self.next_song()

    def prev_candidate(self):
        if not self.current_song:
            return

        if self.cand_idx > 0:
            self.cand_idx -= 1
            self.current_candidate = self.current_song.candidates[self.cand_idx]
            self.stop_playback()
            self.update_view()

    def rate_candidate(self, label: int):
        if not self.current_song or not self.current_candidate:
            return

        self.save_candidate(label)
        self.status_var.set(f"Saved label {label}.")
        self.next_candidate()

    def save_candidate(self, label: int):
        song = self.current_song
        cand = self.current_candidate
        assert song is not None and cand is not None

        offsets, window, beat_times_window = extract_window(
            song.E, song.O, song.C, song.F, song.B, song.beat_times, cand.beat_idx
        )
        vector = window.astype(np.float32).reshape(-1)

        rel = song.path.relative_to(self.dataset_root)
        base = safe_stem(rel)

        out_dir = SAVE_ROOT / f"class_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{base}__cand{cand.beat_idx:04d}__t{cand.time_sec:.2f}__s{cand.score:.3f}.npz"
        out_path = out_dir / out_name

        np.savez_compressed(
            out_path,
            label=np.int32(label),
            song_path=str(song.path),
            rel_path=str(rel),
            beat_idx=np.int32(cand.beat_idx),
            time_sec=np.float32(cand.time_sec),
            heuristic_score=np.float32(cand.score),
            bpm=np.float32(song.bpm),
            feature_names=np.array(FEATURE_NAMES, dtype=object),
            offsets=offsets,
            beat_times_window=beat_times_window,
            window=window,
            vector=vector,
        )

        record = {
            "label": int(label),
            "song_path": str(song.path),
            "rel_path": str(rel),
            "beat_idx": int(cand.beat_idx),
            "time_sec": float(cand.time_sec),
            "heuristic_score": float(cand.score),
            "bpm": float(song.bpm),
            "save_path": str(out_path),
        }
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def play_candidate(self):
        if not self.current_song or not self.current_candidate:
            return

        song = self.current_song
        cand = self.current_candidate

        try:
            start_sec, end_sec, start_idx, end_idx = window_bounds_from_candidate(song.beat_times, cand.beat_idx)
        except ValueError as e:
            self.status_var.set(str(e))
            return

        self.stop_playback()

        try:
            proc = play_clip_external(song.path, start_sec, end_sec)
        except Exception as e:
            self.status_var.set(f"Playback failed: {e}")
            return

        self.playback.active = True
        self.playback.song = song
        self.playback.candidate = cand
        self.playback.window_start_sec = start_sec
        self.playback.window_end_sec = end_sec
        self.playback.play_start_real = time.time()
        self.playback.window_start_idx = start_idx
        self.playback.window_end_idx = end_idx
        self.playback.proc = proc

        self.status_var.set(
            f"Playing exact window: {start_sec:.2f}s -> {end_sec:.2f}s "
            f"(beat 0 = candidate)"
        )

    def stop_playback(self):
        if self.playback.proc is not None:
            try:
                self.playback.proc.terminate()
            except Exception:
                pass
            self.playback.proc = None

        self.playback.active = False
        self.play_var.set("Stopped.")
        self._draw_window()

    def update_view(self):
        if not self.current_song or not self.current_candidate:
            return

        song = self.current_song
        cand = self.current_candidate

        self.file_var.set(f"Song {self.song_idx + 1}/{len(self.audio_files)}: {song.path}")
        self.info_var.set(
            f"Candidate {self.cand_idx + 1}/{len(song.candidates)} | "
            f"time={cand.time_sec:.2f}s | beat_idx={cand.beat_idx} | "
            f"heuristic_score={cand.score:.3f} | bpm={song.bpm:.2f}"
        )
        self.status_var.set(
            f"Window: [-{WINDOW_BEFORE_BEATS}, +{WINDOW_AFTER_BEATS}] beats. "
            f"Beat 0 = candidate."
        )
        self._draw_window()

    def _draw_window(self):
        self.canvas.delete("all")

        width = 900
        height = 140
        left_pad = 40
        right_pad = 40
        y = 70

        self.canvas.create_line(left_pad, y, width - right_pad, y, fill="#222", width=3)

        total_beats = WINDOW_BEFORE_BEATS + WINDOW_AFTER_BEATS
        start_x = left_pad
        end_x = width - right_pad
        cand_x = left_pad + (WINDOW_BEFORE_BEATS / total_beats) * (width - left_pad - right_pad)

        self.canvas.create_line(start_x, y - 20, start_x, y + 20, fill="blue", width=2)
        self.canvas.create_text(start_x, y + 30, text=f"-{WINDOW_BEFORE_BEATS}", fill="blue")

        self.canvas.create_line(cand_x, y - 30, cand_x, y + 30, fill="red", width=3)
        self.canvas.create_text(cand_x, y - 38, text="0", fill="red")
        self.canvas.create_text(cand_x, y + 30, text="candidate", fill="red")

        self.canvas.create_line(end_x, y - 20, end_x, y + 20, fill="green", width=2)
        self.canvas.create_text(end_x, y + 30, text=f"+{WINDOW_AFTER_BEATS}", fill="green")

        for off in [-20, -10, -5, 0, 5, 10]:
            x = left_pad + ((off + WINDOW_BEFORE_BEATS) / total_beats) * (width - left_pad - right_pad)
            self.canvas.create_line(x, y - 10, x, y + 10, fill="#777")
            self.canvas.create_text(x, y - 18, text=str(off), fill="#444")

        # Playhead line
        if self.playback.active and self.playback.proc is not None:
            if self.playback.proc.poll() is not None:
                self.playback.active = False
                self.playback.proc = None
                self.play_var.set("Stopped.")
                self._draw_window()
                return

            elapsed = time.time() - self.playback.play_start_real
            current_abs = self.playback.window_start_sec + elapsed

            if current_abs >= self.playback.window_end_sec:
                current_abs = self.playback.window_end_sec
                self.playback.active = False

            song = self.playback.song
            start_idx = self.playback.window_start_idx
            end_idx = self.playback.window_end_idx

            if song is not None:
                beat_window = song.beat_times[start_idx:end_idx + 1]
                if len(beat_window) > 0:
                    nearest = int(np.argmin(np.abs(beat_window - current_abs)))
                    current_offset = nearest - WINDOW_BEFORE_BEATS
                    self.play_var.set(
                        f"Playing... beat offset ≈ {current_offset:+d} "
                        f"(time {current_abs:.2f}s / {self.playback.window_end_sec:.2f}s)"
                    )
                else:
                    self.play_var.set(
                        f"Playing... time {current_abs:.2f}s / {self.playback.window_end_sec:.2f}s"
                    )

            frac = 0.0
            if self.playback.window_end_sec > self.playback.window_start_sec:
                frac = (current_abs - self.playback.window_start_sec) / (
                    self.playback.window_end_sec - self.playback.window_start_sec
                )
            frac = max(0.0, min(1.0, frac))

            play_x = left_pad + frac * (width - left_pad - right_pad)
            self.canvas.create_line(play_x, y - 40, play_x, y + 40, fill="black", width=2)
            self.canvas.create_oval(play_x - 5, y - 5, play_x + 5, y + 5, fill="black", outline="black")
        else:
            self.play_var.set("Stopped.")

    def _tick(self):
        if self.current_song and self.current_candidate:
            self._draw_window()
        self.root.after(PLAYBACK_UPDATE_MS, self._tick)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    create_dirs()
    tool = DropLabelTool(DATASET_ROOT)
    tool.run()
