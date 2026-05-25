# This file contains the main logic for converting between app models and optimizer models,
# running the optimizer, and converting results back. Used by StudioService to run optimization for a studio session.

from __future__ import annotations

from typing import List

from optimizer.models import (
    Query as O_Query,
    Track as O_Track,
    TrackLibrary as O_TrackLibrary,
    PointAnnotation as O_PointAnnotation,
    TrackSignature as O_TrackSignature,
)
from optimizer.core import BeamSearch

from app.services.track_service import TrackService
from app.models import (
    AlignmentSpec,
    AlignmentTrack,
    AnnotationPoint,
    QuerySpec,
    TrackRecord,
)


def _queryspec_to_optimizer(query: QuerySpec) -> O_Query:
    oq = O_Query()
    oq.set_length(query.length_ticks)
    oq.set_signature(O_TrackSignature(list(query.signature)))

    for p in query.requested_points:
        oq.add_annotation(
            O_PointAnnotation(
                p.label,
                time=p.time_ticks,
                strength=p.strength,
            )
        )

    return oq


def _trackrecord_to_optimizer(record: TrackRecord) -> O_Track:
    ot = O_Track()
    ot.set_length(record.meta.length_ticks)
    ot.set_BPM(record.meta.bpm)
    ot.set_signature(O_TrackSignature(list(record.meta.signature)))

    ot.preference = record.meta.preference
    ot.min_speed = record.meta.min_speed
    ot.max_speed = record.meta.max_speed

    # Keep track_id on the optimizer object for later conversion back.
    # The optimizer classes don't need to know about it, but Python allows it.
    ot.track_id = record.track_id
    ot.path = record.audio_path

    for ann in record.annotations:
        ot.add_annotation(
            O_PointAnnotation(
                ann.label,
                time=ann.time_ticks,
                strength=ann.strength,
            )
        )

    return ot


def _alignment_to_app(alignment) -> AlignmentSpec:
    """
    Convert optimizer alignment -> app AlignmentSpec.

    This assumes optimizer alignment.tracks items have:
      - start_time
      - speed
      - annotations
      - optionally track_id
    """
    placed_tracks: List[AlignmentTrack] = []

    for tr in alignment.tracks:
        placed_points: List[AnnotationPoint] = []

        for ann in tr.annotations:
            if ann.label != "drop":
                continue

            placed_points.append(
                AnnotationPoint(
                    label=ann.label,
                    time_ticks=float(ann.time),
                    strength=float(getattr(ann, "strength", 1.0)),
                )
            )

        track_id = str(getattr(tr, "track_id", "unknown"))

        placed_tracks.append(
            AlignmentTrack(
                track_id=track_id,
                start_time_ticks=float(tr.start_time),
                speed=float(tr.speed),
                placed_points=placed_points,
            )
        )

    return AlignmentSpec(
        score=float(alignment.score),
        tracks=placed_tracks,
    )


def run_optimizer(query: QuerySpec, track_service: TrackService) -> AlignmentSpec:
    """
    Convert app QuerySpec + saved tracks -> optimizer objects,
    run BeamSearch, then convert result back to app AlignmentSpec.
    """
    print("\n[run_optimizer] starting...", flush=True)

    # Load app track records from storage
    track_records = track_service.load_all_tracks()
    
    for i, record in enumerate(track_records[:10], start=1):
      print(
        f"[run_optimizer] track {i}: id={record.track_id}, "
        f"length_ticks={record.meta.length_ticks}, "
        f"bpm={record.meta.bpm}, "
        f"annotations={len(record.annotations)}",
        flush=True,
      )

    for j, ann in enumerate(record.annotations[:10], start=1):
        print(
            f"    ann {j}: label={ann.label}, "
            f"time_ticks={ann.time_ticks}, "
            f"strength={ann.strength}",
            flush=True,
        )

    print(f"[run_optimizer] loaded {len(track_records)} tracks from storage", flush=True)

    if len(track_records) == 0:
        print("[run_optimizer] WARNING: no tracks available in track library", flush=True)

    # Debug track summary
    for i, record in enumerate(track_records[:10], start=1):
        print(
            f"[run_optimizer] track {i}: id={record.track_id}, "
            f"length_ticks={record.meta.length_ticks}, "
            f"bpm={record.meta.bpm}, "
            f"annotations={len(record.annotations)}",
            flush=True,
        )

    # Convert query
    print(
        f"[run_optimizer] query: length_ticks={query.length_ticks}, "
        f"signature={query.signature}, "
        f"requested_points={len(query.requested_points)}",
        flush=True,
    )

    o_query = _queryspec_to_optimizer(query)

    # Convert tracks
    o_library = O_TrackLibrary()
    for record in track_records:
        o_track = _trackrecord_to_optimizer(record)
        o_library.add_track(o_track)

    print(f"[run_optimizer] optimizer track library size: {len(o_library.tracks)}", flush=True)

    # Run optimizer
    optimizer = BeamSearch(beam_width=100, max_steps=10)
    alignment = optimizer.optimize(o_query, o_library)

    print(
        f"[run_optimizer] alignment finished: score={alignment.score}, "
        f"tracks={len(alignment.tracks)}",
        flush=True,
    )

    for i, tr in enumerate(alignment.tracks[:10], start=1):
        print(
            f"[run_optimizer] aligned track {i}: "
            f"track_id={getattr(tr, 'track_id', 'unknown')}, "
            f"start_time={getattr(tr, 'start_time', None)}, "
            f"speed={getattr(tr, 'speed', None)}, "
            f"annotations={len(getattr(tr, 'annotations', []))}",
            flush=True,
        )

    # Convert back to app model
    app_alignment = _alignment_to_app(alignment)

    print("[run_optimizer] done.\n", flush=True)
    return app_alignment
