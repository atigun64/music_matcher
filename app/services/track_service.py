from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.storage import TRACK_STORE
from app.storage.track_store import TrackStore
from app.models import TrackMeta, TrackRecord, AnnotationPoint
from music_core import get_ml_candidates, extract_features


class TrackService:
    def __init__(self, track_store: TrackStore = TRACK_STORE):
        self.track_store = track_store

    def upload_track(self, track_path: str | Path) -> TrackRecord:
        """
        Upload a track, extract metadata and annotations, then save it.

        Returns:
            TrackRecord
        """
        track_path = Path(track_path)

        feats = extract_features(str(track_path))
        E, O, C, F, B, bpm, beat_times, frame_times = feats

        length_sec = float(frame_times[-1]) if len(frame_times) > 0 else 0.0
        length_ticks = length_sec

        meta = TrackMeta(
            length_ticks=length_ticks,
            bpm=float(bpm),
            signature=[0.6, 0.4, 0.8, 0.3],  # TODO compute real signature
            preference=1.0,
            min_speed=0.98,
            max_speed=1.20,
        )

        candidates = get_ml_candidates(E, O, C, F, B, beat_times)

        annotations: List[AnnotationPoint] = []
        for _beat_idx, beat_time_sec, score in candidates:
            annotations.append(
                AnnotationPoint(
                    label="drop",
                    time_ticks=float(beat_time_sec),
                    strength=float(score),
                )
            )

        track_id = self.track_store.save_track_for_audio_path(
            audio_path=track_path,
            meta=meta,
            annotations=annotations,
            overwrite=True,
        )

        return self.track_store.load_track(track_id)

    def edit_track_annotations(
        self,
        track_id: str,
        annotations: List[AnnotationPoint],
    ) -> None:
        """
        Replace annotations for an existing track.
        """
        record = self.track_store.load_track(track_id)

        updated = TrackRecord(
            track_id=record.track_id,
            audio_path=record.audio_path,
            meta=record.meta,
            annotations=annotations,
        )

        self.track_store.save_track(updated, overwrite=True)

    def edit_track_meta(
        self,
        track_id: str,
        meta: TrackMeta,
    ) -> None:
        """
        Replace metadata for an existing track.
        """
        record = self.track_store.load_track(track_id)

        updated = TrackRecord(
            track_id=record.track_id,
            audio_path=record.audio_path,
            meta=meta,
            annotations=record.annotations,
        )

        self.track_store.save_track(updated, overwrite=True)
      
    def delete_track(self, track_id: str) -> None:
        """
        Delete a track from the library.
        """
        self.track_store.delete_track(track_id)

    def list_tracks(self) -> List[Dict[str, Any]]:
        """
        Lightweight UI-friendly listing.
        Returns list of dicts from TrackStore.
        """
        return self.track_store.list_tracks()
    
    def list_track_ids(self) -> List[str]:
        return self.track_store.list_track_ids()

    def load_track(self, track_id: str) -> TrackRecord:
        """
        Load a full TrackRecord.
        """
        return self.track_store.load_track(track_id)

    def load_all_tracks(self) -> List[TrackRecord]:
        """
        Load all tracks in the library.
        """
        track_ids = self.track_store.list_track_ids()
        return self.track_store.load_tracks(track_ids)

    def load_tracks(self, track_ids: List[str]) -> List[TrackRecord]:
        return self.track_store.load_tracks(track_ids)
