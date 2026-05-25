from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import (
    AlignmentSpec,
    AlignmentTrack,
    AnnotationPoint,
    QuerySpec,
    StudioMeta,
    StudioSession,
)
from .config import STUDIOS_ROOT


class StudioStore:
    """
    Persists StudioSession / QuerySpec / AlignmentSpec to disk.

    On disk layout:
      data/studios/
        <studio_id>/
          meta.json
          query.json
          alignment.json
    """

    def __init__(self, studios_root: Path = STUDIOS_ROOT):
        self.root = Path(studios_root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # paths
    # ---------------------------

    def _studio_dir(self, studio_id: str) -> Path:
        return self.root / str(studio_id)

    def _meta_path(self, studio_id: str) -> Path:
        return self._studio_dir(studio_id) / "meta.json"

    def _query_path(self, studio_id: str) -> Path:
        return self._studio_dir(studio_id) / "query.json"

    def _alignment_path(self, studio_id: str) -> Path:
        return self._studio_dir(studio_id) / "alignment.json"

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
    def _query_to_dict(query: QuerySpec) -> Dict[str, Any]:
        return {
            "length_ticks": float(query.length_ticks),
            "signature": [float(x) for x in query.signature],
            "requested_points": [StudioStore._annotation_to_dict(p) for p in query.requested_points],
        }

    @staticmethod
    def _query_from_dict(d: Dict[str, Any]) -> QuerySpec:
        return QuerySpec(
            length_ticks=float(d["length_ticks"]),
            signature=[float(x) for x in d["signature"]],
            requested_points=[StudioStore._annotation_from_dict(p) for p in d.get("requested_points", [])],
        )

    @staticmethod
    def _alignment_to_dict(al: AlignmentSpec) -> Dict[str, Any]:
        return {
            "score": float(al.score),
            "tracks": [
                {
                    "track_id": t.track_id,
                    "start_time_ticks": float(t.start_time_ticks),
                    "speed": float(t.speed),
                    "placed_points": [StudioStore._annotation_to_dict(p) for p in t.placed_points],
                }
                for t in al.tracks
            ],
        }

    @staticmethod
    def _alignment_from_dict(d: Dict[str, Any]) -> AlignmentSpec:
        tracks = []
        for t in d.get("tracks", []):
            tracks.append(
                AlignmentTrack(
                    track_id=str(t["track_id"]),
                    start_time_ticks=float(t["start_time_ticks"]),
                    speed=float(t.get("speed", 1.0)),
                    placed_points=[StudioStore._annotation_from_dict(p) for p in t.get("placed_points", [])],
                )
            )
        return AlignmentSpec(score=float(d["score"]), tracks=tracks)

    @staticmethod
    def _meta_to_dict(meta: StudioMeta) -> Dict[str, Any]:
        return {
            "source": meta.source,
            "video_path": meta.video_path,
            "notes": meta.notes,
        }

    @staticmethod
    def _meta_from_dict(d: Dict[str, Any], studio_id: str) -> StudioMeta:
        return StudioMeta(
            source=str(d.get("source", "silent")),
            video_path=d.get("video_path", None),
            notes=str(d.get("notes", "")),
        )

    # ---------------------------
    # public API
    # ---------------------------

    def list_studio_ids(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def create_studio(self, studio_id: Optional[str] = None, *, overwrite: bool = False) -> str:
        if studio_id is None:
            studio_id = self._next_id()

        sdir = self._studio_dir(studio_id)
        if sdir.exists() and not overwrite:
            raise FileExistsError(f"Studio exists: {sdir}")

        sdir.mkdir(parents=True, exist_ok=True)
        self._meta_path(studio_id).write_text(
            json.dumps({"source": "silent", "video_path": None, "notes": ""}, indent=2),
            encoding="utf-8",
        )
        return str(studio_id)

    def delete_studio(self, studio_id: str) -> None:
        sdir = self._studio_dir(studio_id)
        if sdir.exists() and sdir.is_dir():
            for child in sdir.iterdir():
                child.unlink()
            sdir.rmdir()

    def save_meta(self, studio_id: str, meta: StudioMeta, *, overwrite: bool = True) -> None:
        path = self._meta_path(studio_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Studio meta exists: {path}")
        self._studio_dir(studio_id).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._meta_to_dict(meta), indent=2), encoding="utf-8")

    def load_meta(self, studio_id: str) -> StudioMeta:
        return self._meta_from_dict(
            json.loads(self._meta_path(studio_id).read_text(encoding="utf-8")),
            studio_id=studio_id,
        )

    def save_query(self, studio_id: str, query: QuerySpec, *, overwrite: bool = True) -> None:
        path = self._query_path(studio_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Query exists: {path}")
        self._studio_dir(studio_id).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._query_to_dict(query), indent=2), encoding="utf-8")

    def load_query(self, studio_id: str) -> QuerySpec:
        return self._query_from_dict(
            json.loads(self._query_path(studio_id).read_text(encoding="utf-8"))
        )

    def save_alignment(self, studio_id: str, alignment: AlignmentSpec, *, overwrite: bool = True) -> None:
        path = self._alignment_path(studio_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Alignment exists: {path}")
        self._studio_dir(studio_id).mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._alignment_to_dict(alignment), indent=2), encoding="utf-8")

    def load_alignment(self, studio_id: str) -> AlignmentSpec:
        return self._alignment_from_dict(
            json.loads(self._alignment_path(studio_id).read_text(encoding="utf-8"))
        )

    def save_session(self, session: StudioSession, *, overwrite: bool = True) -> None:
        """
        Convenience: save meta/query/alignment from a StudioSession.
        """
        self.save_meta(session.studio_id, session.meta, overwrite=overwrite)
        if session.query is not None:
            self.save_query(session.studio_id, session.query, overwrite=overwrite)
        if session.alignment is not None:
            self.save_alignment(session.studio_id, session.alignment, overwrite=overwrite)

    def load_session(self, studio_id: str) -> StudioSession:
        """
        Load a studio session if query/alignment exist.
        """
        meta = self.load_meta(studio_id)

        query = None
        qpath = self._query_path(studio_id)
        if qpath.exists():
            query = self.load_query(studio_id)

        alignment = None
        apath = self._alignment_path(studio_id)
        if apath.exists():
            alignment = self.load_alignment(studio_id)

        return StudioSession(
            studio_id=str(studio_id),
            meta=meta,
            query=query,
            alignment=alignment,
        )
