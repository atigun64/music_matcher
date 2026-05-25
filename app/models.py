from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ------------------------------------------------------------
# Shared point model
# ------------------------------------------------------------

@dataclass
class AnnotationPoint:
    """
    A generic point on a timeline.

    Used for:
    - track annotations
    - query requested points
    - aligned / placed points
    """
    label: str
    time_ticks: float
    strength: float = 1.0


# ------------------------------------------------------------
# Track library models
# ------------------------------------------------------------

@dataclass
class TrackMeta:
    """
    Metadata / constraints for one track.
    """
    length_ticks: float
    bpm: float
    signature: List[float]
    preference: float = 1.0
    min_speed: float = 0.98
    max_speed: float = 1.20


@dataclass
class TrackRecord:
    """
    One track in the app's track library.
    """
    track_id: str
    audio_path: str
    meta: TrackMeta
    annotations: List[AnnotationPoint] = field(default_factory=list)


# ------------------------------------------------------------
# Query / studio models
# ------------------------------------------------------------

@dataclass
class QuerySpec:
    """
    The requested timeline for the studio/compiler.
    """
    length_ticks: float
    signature: List[float]
    requested_points: List[AnnotationPoint] = field(default_factory=list)


@dataclass
class StudioMeta:
    """
    Studio session metadata.

    Note:
    - keep this focused on the session itself
    """
    source: str = "silent"          # "silent" or "video"
    video_path: Optional[str] = None
    notes: str = ""


# ------------------------------------------------------------
# Alignment models
# ------------------------------------------------------------

@dataclass
class AlignmentTrack:
    """
    One track placed by the optimizer into the studio timeline.
    """
    track_id: str
    start_time_ticks: float
    speed: float = 1.0
    placed_points: List[AnnotationPoint] = field(default_factory=list)


@dataclass
class AlignmentSpec:
    """
    The optimizer result for a studio.
    """
    score: float
    tracks: List[AlignmentTrack] = field(default_factory=list)


# ------------------------------------------------------------
# Studio wrapper
# ------------------------------------------------------------

@dataclass
class StudioSession:
    """
    One studio compile session.

    This is the app-level wrapper that ties together:
    - studio_id
    - studio metadata
    - query
    - alignment result
    """
    studio_id: str
    meta: StudioMeta
    query: Optional[QuerySpec] = None
    alignment: Optional[AlignmentSpec] = None
