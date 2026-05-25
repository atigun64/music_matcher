from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
import soundfile as sf

from .data import UISample

class UI:
    @staticmethod
    def sample(samples: Sequence[UISample]) -> list[Optional[int]]:
        app = _SampleApp(list(samples), read_only=False)
        return app.run()

    @staticmethod
    def view(samples: Sequence[UISample]) -> None:
        app = _SampleApp(list(samples), read_only=True)
        app.run()


class _SampleApp:
    def __init__(self, samples: list[UISample], read_only: bool):
        self.samples = samples
        self.read_only = read_only
        self.labels: list[Optional[int]] = [None] * len(samples)

        self.index = 0
        self._window_cache: dict[int, tuple[np.ndarray, int]] = {}

        self._playing = False
        self._play_started_at = 0.0
        self._play_duration = 0.0
        self._tick_job = None

        self.root = tk.Tk()
        self.root.title("Drop Labeling UI" if not read_only else "Drop Viewer")
        self.root.geometry("900x420")
        self.root.minsize(760, 360)

        self._build_widgets()
        self._bind_keys()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.samples:
            self._show_current()
        else:
            self._set_status("No samples.")

    def run(self):
        self.root.mainloop()
        if self.read_only:
            return None
        return self.labels

    # ---------------- UI layout ----------------

    def _build_widgets(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        self.title_var = tk.StringVar()
        self.meta_var = tk.StringVar()
        self.window_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.label_var = tk.StringVar()

        ttk.Label(
            outer,
            textvariable=self.title_var,
            font=("TkDefaultFont", 12, "bold")
        ).pack(anchor="w")

        ttk.Label(
            outer,
            textvariable=self.meta_var
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(
            outer,
            textvariable=self.window_var
        ).pack(anchor="w", pady=(2, 10))

        self.canvas = tk.Canvas(
            outer,
            height=120,
            background="#111111",
            highlightthickness=1,
            highlightbackground="#444444"
        )
        self.canvas.pack(fill="x", pady=(0, 12))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))

        self.prev_btn = ttk.Button(controls, text="⟨ Prev", command=self._prev_sample)
        self.prev_btn.pack(side="left")

        self.play_btn = ttk.Button(controls, text="Play", command=self._toggle_play)
        self.play_btn.pack(side="left", padx=(8, 0))

        self.stop_btn = ttk.Button(controls, text="Stop", command=self._stop_audio)
        self.stop_btn.pack(side="left", padx=(8, 0))

        self.next_btn = ttk.Button(controls, text="Next ⟩", command=self._next_sample)
        self.next_btn.pack(side="left", padx=(8, 0))

        ttk.Separator(outer).pack(fill="x", pady=8)

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x")

        if not self.read_only:
            label_frame = ttk.Frame(bottom)
            label_frame.pack(side="left")

            ttk.Button(label_frame, text="Label 0", command=lambda: self._set_label(0)).pack(side="left")
            ttk.Button(label_frame, text="Clear", command=self._clear_label).pack(side="left", padx=(8, 0))
            ttk.Button(label_frame, text="Label 1", command=lambda: self._set_label(1)).pack(side="left", padx=(8, 0))

            ttk.Label(bottom, textvariable=self.label_var).pack(side="left", padx=(16, 0))

        ttk.Label(bottom, textvariable=self.status_var).pack(side="right")

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self._prev_sample())
        self.root.bind("<Right>", lambda e: self._next_sample())
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Escape>", lambda e: self._on_close())

        if not self.read_only:
            self.root.bind("0", lambda e: self._set_label(0))
            self.root.bind("1", lambda e: self._set_label(1))
            self.root.bind("<BackSpace>", lambda e: self._clear_label())

    # ---------------- sample display ----------------

    def _show_current(self):
        if not self.samples:
            return

        sample = self.samples[self.index]
        start_sec, end_sec = sample.time_window
        duration = max(0.0, end_sec - start_sec)

        self.title_var.set(
            f"[{self.index + 1}/{len(self.samples)}] {sample.track_path.name}"
        )
        self.meta_var.set(
            f"Path: {sample.track_path}"
        )
        self.window_var.set(
            f"Window: {self._fmt_time(start_sec)} → {self._fmt_time(end_sec)}   "
            f"(duration {duration:.2f}s)    "
            f"Key point: {self._fmt_time(sample.key_point)}    "
            f"Tolerance: {self._fmt_time(sample.tolerance_window[0])} → {self._fmt_time(sample.tolerance_window[1])}    "
            f"Model score: {getattr(sample, 'model_score', 0.0):.3f}"
        )



        if self.read_only:
            self.label_var.set("")
        else:
            current_label = self.labels[self.index]
            if current_label is None:
                self.label_var.set("Current label: None")
            else:
                self.label_var.set(f"Current label: {current_label}")

        self._set_status(
            f"Labeled: {sum(v is not None for v in self.labels)}/{len(self.labels)}"
            if not self.read_only else
            f"Viewing {self.index + 1}/{len(self.samples)}"
        )

        self._draw_timeline(playhead_sec=None)

    def _draw_timeline(self, playhead_sec: Optional[float]):
        self.canvas.delete("all")

        if not self.samples:
            return

        sample = self.samples[self.index]
        start_sec, end_sec = sample.time_window
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 60)

        pad_x = 30
        y0 = 25
        y1 = height - 30
        left = pad_x
        right = width - pad_x

        self.canvas.create_rectangle(left, y0, right, y1, fill="#222222", outline="#666666")

        duration = end_sec - start_sec
        if duration <= 0:
            return

        # key point marker
        key_rel = (sample.key_point - start_sec) / duration
        key_rel = min(max(key_rel, 0.0), 1.0)
        key_x = left + key_rel * (right - left)

        self.canvas.create_line(key_x, y0, key_x, y1, fill="#ff4d4d", width=3)
        self.canvas.create_text(key_x, y0 - 10, text="key", fill="#ff8080")
        
        # tolerance window markers
        tol_start_sec, tol_end_sec = sample.tolerance_window

        tol_start_rel = (tol_start_sec - start_sec) / duration
        tol_end_rel = (tol_end_sec - start_sec) / duration
        tol_start_rel = min(max(tol_start_rel, 0.0), 1.0)
        tol_end_rel = min(max(tol_end_rel, 0.0), 1.0)

        tol_start_x = left + tol_start_rel * (right - left)
        tol_end_x = left + tol_end_rel * (right - left)

        self.canvas.create_line(
            tol_start_x, y0, tol_start_x, y1,
            fill="#d9a441", width=2, dash=(4, 2)
        )
        self.canvas.create_line(
            tol_end_x, y0, tol_end_x, y1,
            fill="#d9a441", width=2, dash=(4, 2)
        )

        self.canvas.create_text(tol_start_x, y0 - 10, text="tol-", fill="#d9a441")
        self.canvas.create_text(tol_end_x, y0 - 10, text="tol+", fill="#d9a441")

        # playhead marker
        if playhead_sec is not None:
            play_rel = (playhead_sec - start_sec) / duration
            play_rel = min(max(play_rel, 0.0), 1.0)
            play_x = left + play_rel * (right - left)
            self.canvas.create_line(play_x, y0, play_x, y1, fill="#4da6ff", width=2)
            self.canvas.create_text(play_x, y1 + 12, text="play", fill="#80c1ff")

        self.canvas.create_text(left, y1 + 12, text=self._fmt_time(start_sec), fill="#dddddd", anchor="w")
        self.canvas.create_text(right, y1 + 12, text=self._fmt_time(end_sec), fill="#dddddd", anchor="e")

    # ---------------- playback ----------------

    def _toggle_play(self):
        if self._playing:
            self._stop_audio()
        else:
            self._play_current()

    def _play_current(self):
        if not self.samples:
            return

        self._stop_audio()

        try:
            audio, sr = self._get_window_audio(self.index)
        except Exception as e:
            self._set_status(f"Audio load failed: {e}")
            return

        if audio.size == 0:
            self._set_status("Empty audio window.")
            return

        sd.play(audio, sr, blocking=False)

        self._playing = True
        self._play_started_at = time.perf_counter()
        self._play_duration = len(audio) / float(sr)

        self.play_btn.configure(text="Pause/Stop")
        self._schedule_tick()

    def _stop_audio(self):
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None

        sd.stop()
        self._playing = False
        self.play_btn.configure(text="Play")

        if self.samples:
            self._draw_timeline(playhead_sec=None)

    def _schedule_tick(self):
        self._tick_job = self.root.after(30, self._tick_playback)

    def _tick_playback(self):
        if not self._playing or not self.samples:
            return

        elapsed = time.perf_counter() - self._play_started_at
        sample = self.samples[self.index]
        start_sec, _ = sample.time_window

        if elapsed >= self._play_duration:
            self._stop_audio()
            return

        self._draw_timeline(playhead_sec=start_sec + elapsed)
        self._schedule_tick()

    def _get_window_audio(self, idx: int) -> tuple[np.ndarray, int]:
        if idx in self._window_cache:
            return self._window_cache[idx]

        sample = self.samples[idx]
        audio, sr = self._load_audio_window(sample.track_path, sample.time_window)
        self._window_cache[idx] = (audio, sr)
        return audio, sr

    @staticmethod
    def _load_audio_window(track_path: Path, time_window: tuple[float, float]) -> tuple[np.ndarray, int]:
        """
        Reads only the requested [start, end] audio window from disk.

        Works well for wav/flac/ogg with soundfile.
        If your files are mp3 and this fails, replace this function.
        """
        start_sec, end_sec = time_window
        start_sec = max(0.0, float(start_sec))
        end_sec = max(start_sec, float(end_sec))

        with sf.SoundFile(str(track_path), "r") as f:
            sr = f.samplerate
            start_frame = int(start_sec * sr)
            end_frame = int(end_sec * sr)

            total_frames = len(f)
            start_frame = max(0, min(start_frame, total_frames))
            end_frame = max(start_frame, min(end_frame, total_frames))

            f.seek(start_frame)
            frames_to_read = end_frame - start_frame
            audio = f.read(frames_to_read, dtype="float32", always_2d=False)

        if isinstance(audio, np.ndarray) and audio.ndim == 0:
            audio = np.array([], dtype=np.float32)

        return audio, sr

    # ---------------- labeling ----------------

    def _set_label(self, value: int):
        if self.read_only or not self.samples:
            return

        self.labels[self.index] = int(value)
        self._show_current()

        # auto-advance except on last item
        if self.index < len(self.samples) - 1:
            self._next_sample()

    def _clear_label(self):
        if self.read_only or not self.samples:
            return

        self.labels[self.index] = None
        self._show_current()

    # ---------------- navigation ----------------

    def _prev_sample(self):
        if not self.samples or self.index == 0:
            return
        self._stop_audio()
        self.index -= 1
        self._show_current()

    def _next_sample(self):
        if not self.samples or self.index >= len(self.samples) - 1:
            return
        self._stop_audio()
        self.index += 1
        self._show_current()

    # ---------------- misc ----------------

    def _set_status(self, text: str):
        self.status_var.set(text)

    def _on_close(self):
        self._stop_audio()
        self.root.quit()
        self.root.destroy()

    @staticmethod
    def _fmt_time(sec: float) -> str:
        sec = float(sec)
        mins = int(sec // 60)
        secs = sec - mins * 60
        return f"{mins:02d}:{secs:05.2f}"
