from dataclasses import dataclass, field
from typing import List
import math

from optimizer.models import Query, TrackLibrary, Alignment, AssignedTrack, Track, PointAnnotation
from optimizer.scores import score_alignment_partial, score_alignment_final
from .config import DROP_MISS_TOLERANCE, MAX_ACCEPTABLE_OVERLAP


@dataclass
class BeamState:
    alignment: Alignment
    score: float
    covered_end: float = 0.0
    used_track_ids: set[int] = field(default_factory=set)


class Optimizer:
    def optimize(self, query: Query, tracks: TrackLibrary) -> Alignment:
        raise NotImplementedError


class BeamSearch(Optimizer):
    def __init__(self, beam_width: int = 10, max_steps: int = 20):
        self.beam_width = beam_width
        self.max_steps = max_steps

        # how many request anchors to try each step
        self.request_anchor_limit = 3

    def optimize(self, query: Query, tracks: TrackLibrary) -> Alignment:
        initial_alignment = Alignment()
        initial_alignment.tracks = []

        initial_state = BeamState(
            alignment=initial_alignment,
            score=score_alignment_partial(initial_alignment, query),
            covered_end=0.0,
            used_track_ids=set(),
        )

        beam: list[BeamState] = [initial_state]

        for _ in range(self.max_steps):
            new_beam: list[BeamState] = []

            for state in beam:
                candidates = self._candidate_extensions(state, query, tracks)

                for candidate in candidates:
                    new_state = self._extend_state(state, candidate, query)
                    new_beam.append(new_state)

            if not new_beam:
                break

            new_beam.sort(key=lambda s: s.score, reverse=True)
            beam = new_beam[: self.beam_width]

        if not beam:
            return initial_alignment

        best_state = max(beam, key=lambda s: score_alignment_final(s.alignment, query))
        best_alignment = best_state.alignment
        best_alignment.score = score_alignment_final(best_alignment, query)
        return best_alignment

    # --------------------------------------------------
    # Candidate generation
    # --------------------------------------------------

    def _candidate_extensions(
        self,
        state: BeamState,
        query: Query,
        tracks: TrackLibrary
    ) -> list[tuple[Track, float, float]]:
        """
        Returns candidate placements:
            (track, start_time, speed)

        Strategy:
        1) try to anchor track drops to the next strongest nearby requests
        2) if nothing good appears, fall back to sequential placement
        """
        candidates: list[tuple[Track, float, float]] = []
        seen: set[tuple[int, float, float]] = set()

        requested_drops = self._requested_drops(query)

        if not requested_drops:
            return self._sequential_candidates(state, query, tracks)

        # Try requests that are not obviously in the past.
        # We focus on requests at or after the current frontier, with a small overlap window.
        frontier = state.covered_end
        overlap_slack = MAX_ACCEPTABLE_OVERLAP

        future_targets = [
            r for r in requested_drops
            if r.time is not None and r.time >= frontier - overlap_slack
        ]

        # If nothing is found near the frontier, still try the strongest remaining requests.
        if not future_targets:
            future_targets = requested_drops[:]

        # Strongest first, then earlier first.
        future_targets.sort(key=lambda r: (-self._strength_of(r), r.time))
        future_targets = future_targets[: self.request_anchor_limit]

        for request in future_targets:
            if request.time is None:
                continue

            for track in tracks.get_tracks():
                track_id = id(track)
                if track_id in state.used_track_ids:
                    continue

                if track.length is None:
                    continue

                drop_anns = self._track_drops(track)
                if not drop_anns:
                    continue

                for drop in drop_anns:
                    if drop.time is None:
                        continue

                    for speed in self._speed_candidates(track):
                        if speed <= 0:
                            continue

                        # exact anchor placement
                        base_start = request.time - (drop.time / speed)

                        # a few tiny nudges around the exact anchor
                        for delta in self._anchor_offsets():
                            start_time = base_start + delta
                            end_time = start_time + (track.length / speed)

                            if start_time < 0:
                                continue

                            if query.length is not None and end_time > query.length:
                                continue

                            # Keep mostly left-to-right behavior.
                            # Allow a small overlap, but not huge backtracking.
                            if start_time < frontier - overlap_slack:
                                continue

                            key = (track_id, round(start_time, 2), round(speed, 3))
                            if key in seen:
                                continue
                            seen.add(key)

                            candidates.append((track, start_time, speed))

        # If anchoring gives too few candidates, add sequential fallback.
        if len(candidates) < self.beam_width:
            candidates.extend(self._sequential_candidates(state, query, tracks, seen))

        return candidates

    def _requested_drops(self, query: Query) -> list[PointAnnotation]:
        return [
            a for a in query.annotations
            if isinstance(a, PointAnnotation)
            and a.label == "drop"
            and a.time is not None
        ]

    def _track_drops(self, track: Track) -> list[PointAnnotation]:
        return [
            a for a in track.annotations
            if isinstance(a, PointAnnotation)
            and a.label == "drop"
            and a.time is not None
        ]

    def _strength_of(self, ann: PointAnnotation) -> float:
        s = ann.strength if ann.strength is not None else 0.0
        return max(0.0, min(1.0, float(s)))

    def _speed_candidates(self, track: Track) -> list[float]:
        min_speed = getattr(track, "min_speed", 0.98)
        max_speed = getattr(track, "max_speed", 1.20)

        if min_speed > max_speed:
            min_speed, max_speed = max_speed, min_speed

        # simple 3-point search
        mid = (min_speed + max_speed) / 2.0

        speeds = [min_speed, (min_speed + mid) / 2.0 , mid, (max_speed + mid) / 2.0, max_speed]

        # dedupe
        out: list[float] = []
        seen: set[float] = set()
        for s in speeds:
            if s not in seen:
                seen.add(s)
                out.append(s)

        return out

    def _anchor_offsets(self) -> list[float]:
        """
        Small offsets around the exact anchor.

        If your times are ticks, these are ticks.
        """
        tol = int(DROP_MISS_TOLERANCE)
        step = max(1, tol // 3)

        return [
            -step,
            0,
            step,
        ]

    def _sequential_candidates(
        self,
        state: BeamState,
        query: Query,
        tracks: TrackLibrary,
        seen: set[tuple[int, float, float]] | None = None
    ) -> list[tuple[Track, float, float]]:
        """
        Fallback: place a new track after the current frontier.
        """
        candidates: list[tuple[Track, float, float]] = []
        seen = seen if seen is not None else set()

        frontier = state.covered_end
        query_length = query.length

        for track in tracks.get_tracks():
            track_id = id(track)
            if track_id in state.used_track_ids:
                continue

            if track.length is None:
                continue

            for speed in self._speed_candidates(track):
                start_time = frontier
                end_time = start_time + (track.length / speed)

                if query_length is not None and end_time > query_length:
                    continue

                key = (track_id, round(start_time, 2), round(speed, 3))
                if key in seen:
                    continue
                seen.add(key)

                candidates.append((track, start_time, speed))

        return candidates

    # --------------------------------------------------
    # State extension
    # --------------------------------------------------

    def _extend_state(
        self,
        state: BeamState,
        candidate: tuple[Track, float, float],
        query: Query
    ) -> BeamState:
        track, start_time, speed = candidate

        assigned = self._assign_track(track, start_time, speed)

        new_alignment = Alignment()
        new_alignment.tracks = list(state.alignment.tracks)
        new_alignment.tracks.append(assigned)

        new_used_ids = set(state.used_track_ids)
        new_used_ids.add(id(track))

        new_end = max(state.covered_end, start_time + self._assigned_duration(assigned))

        new_score = score_alignment_partial(new_alignment, query)

        return BeamState(
            alignment=new_alignment,
            score=new_score,
            covered_end=new_end,
            used_track_ids=new_used_ids,
        )

    def _assign_track(self, track: Track, start_time: float, speed: float) -> AssignedTrack:
        assigned = AssignedTrack()

        assigned.track_id = track.track_id
        assigned.length = track.length
        assigned.BPM = track.BPM
        assigned.signature = track.signature
        assigned.annotations = list(track.annotations)

        assigned.preference = getattr(track, "preference", 1.0)
        assigned.min_speed = getattr(track, "min_speed", 0.98)
        assigned.max_speed = getattr(track, "max_speed", 1.20)

        assigned.start_time = start_time
        assigned.speed = speed

        return assigned

    def _assigned_duration(self, track: AssignedTrack) -> float:
        if track.length is None or track.speed is None or track.speed <= 0:
            return 0.0
        return track.length / track.speed
