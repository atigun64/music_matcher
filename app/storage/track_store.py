from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import AnnotationPoint, TrackMeta, TrackRecord
from .config import TRACKS_ROOT


class TrackStore:
    """
    Persists TrackRecord objects to disk.

    On disk layout:
      data/track_library/
        _index.json
        <track_id>/
          meta.json
          annotations.json
          audio_path.txt
    """

    def __init__(self, tracks_root: Path = TRACKS_ROOT):
        self.root = Path(tracks_root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.index_path = self.root / "_index.json"
        self.index: dict[str, str] = self._load_index()

    # ---------------------------
    # index
    # ---------------------------

    def _load_index(self) -> dict[str, str]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return {str(k): str(v) for k, v in data.items()}
        except Exception:
            return {}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    @staticmethod
    def _norm_path(p: Path) -> str:
        return str(Path(p).expanduser().resolve())

    # ---------------------------
    # paths
    # ---------------------------

    def _track_dir(self, track_id: str) -> Path:
        return self.root / str(track_id)

    def _meta_path(self, track_id: str) -> Path:
        return self._track_dir(track_id) / "meta.json"

    def _ann_path(self, track_id: str) -> Path:
        return self._track_dir(track_id) / "annotations.json"

    def _audio_path_file(self, track_id: str) -> Path:
        return self._track_dir(track_id) / "audio_path.txt"

    def _next_id(self) -> str:
        used: set[int] = set()
        for p in self.root.iterdir():
            if p.is_dir():
                try:
                    used.add(int(p.name))
                except ValueError:
                    pass

        i = 0
        while i in used:
            i += 1
        return str(i)

    # ---------------------------
    # serialization helpers
    # ---------------------------

    @staticmethod
    def _annotation_to_dict(a: AnnotationPoint) -> Dict[str, Any]:
        return {
            "label": a.label,
            "time_ticks": float(a.time_ticks),
            "strength": float(a.strength),
        }

    @staticmethod
    def _annotation_from_dict(d: Dict[str, Any]) -> AnnotationPoint:
        return AnnotationPoint(
            label=str(d["label"]),
            time_ticks=float(d["time_ticks"]),
            strength=float(d.get("strength", 1.0)),
        )

    @staticmethod
    def _meta_to_dict(meta: TrackMeta) -> Dict[str, Any]:
        return {
            "length_ticks": float(meta.length_ticks),
            "bpm": float(meta.bpm),
            "signature": [float(x) for x in meta.signature],
            "preference": float(meta.preference),
            "min_speed": float(meta.min_speed),
            "max_speed": float(meta.max_speed),
        }

    @staticmethod
    def _meta_from_dict(d: Dict[str, Any]) -> TrackMeta:
        return TrackMeta(
            length_ticks=float(d["length_ticks"]),
            bpm=float(d["bpm"]),
            signature=[float(x) for x in d["signature"]],
            preference=float(d.get("preference", 1.0)),
            min_speed=float(d.get("min_speed", 0.98)),
            max_speed=float(d.get("max_speed", 1.20)),
        )

    # ---------------------------
    # public API
    # ---------------------------

    def list_track_ids(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def list_tracks(self) -> List[Dict[str, Any]]:
        """
        Lightweight info for UI lists.
        """
        out: List[Dict[str, Any]] = []
        for tid in self.list_track_ids():
            rec = self.load_track(tid)
            display_name = Path(rec.audio_path).name if rec.audio_path else tid
            out.append(
                {
                    "track_id": tid,
                    "display_name": display_name,
                    "audio_path": rec.audio_path,
                }
            )
        return out

    def save_track(
        self,
        track: TrackRecord,
        *,
        overwrite: bool = True,
    ) -> None:
        """
        Save a TrackRecord.
        """
        tdir = self._track_dir(track.track_id)
        if tdir.exists() and not overwrite:
            raise FileExistsError(f"Track exists: {tdir}")

        tdir.mkdir(parents=True, exist_ok=True)

        # write audio path
        if track.audio_path:
            self._audio_path_file(track.track_id).write_text(
                str(Path(track.audio_path).expanduser().resolve()),
                encoding="utf-8",
            )

            self.index[self._norm_path(Path(track.audio_path))] = str(track.track_id)
            self._save_index()

        # write meta and annotations
        self._meta_path(track.track_id).write_text(
            json.dumps(self._meta_to_dict(track.meta), indent=2),
            encoding="utf-8",
        )
        self._ann_path(track.track_id).write_text(
            json.dumps([self._annotation_to_dict(a) for a in track.annotations], indent=2),
            encoding="utf-8",
        )

    def save_track_for_audio_path(
        self,
        audio_path: Path,
        *,
        meta: TrackMeta,
        annotations: List[AnnotationPoint],
        overwrite: bool = True,
    ) -> str:
        """
        Deduplicate by audio_path:
          - if already known, overwrite that track
          - else create next numeric id
        """
        audio_path = Path(audio_path)
        key = self._norm_path(audio_path)

        existing_id = self.index.get(key)
        if existing_id is not None:
            self.save_track(
                TrackRecord(
                    track_id=existing_id,
                    audio_path=str(audio_path),
                    meta=meta,
                    annotations=annotations,
                ),
                overwrite=overwrite,
            )
            return existing_id

        track_id = self._next_id()
        self.save_track(
            TrackRecord(
                track_id=track_id,
                audio_path=str(audio_path),
                meta=meta,
                annotations=annotations,
            ),
            overwrite=True,
        )
        return track_id

    def delete_track(self, track_id: str) -> None:
        tdir = self._track_dir(track_id)
        if not tdir.exists():
            raise FileNotFoundError(f"Track not found: {tdir}")

        # remove from index
        audio_path_file = self._audio_path_file(track_id)
        if audio_path_file.exists():
            audio_path = audio_path_file.read_text(encoding="utf-8").strip()
            key = self._norm_path(Path(audio_path))
            if key in self.index:
                del self.index[key]
                self._save_index()

        # remove track directory
        for p in tdir.iterdir():
            p.unlink()
        tdir.rmdir()


    def load_track(self, track_id: str) -> TrackRecord:
        tdir = self._track_dir(track_id)
        if not tdir.exists():
            raise FileNotFoundError(f"Track not found: {tdir}")

        meta = self._meta_from_dict(
            json.loads(self._meta_path(track_id).read_text(encoding="utf-8"))
        )

        ann_rows = json.loads(self._ann_path(track_id).read_text(encoding="utf-8"))
        annotations = [self._annotation_from_dict(a) for a in ann_rows]

        audio_path = ""
        ap = self._audio_path_file(track_id)
        if ap.exists():
            audio_path = ap.read_text(encoding="utf-8").strip()

        return TrackRecord(
            track_id=str(track_id),
            audio_path=audio_path,
            meta=meta,
            annotations=annotations,
        )

    def load_tracks(self, track_ids: List[str]) -> List[TrackRecord]:
        return [self.load_track(tid) for tid in track_ids]
