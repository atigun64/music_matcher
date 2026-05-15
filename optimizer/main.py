from models import Query, Track, TrackLibrary, PointAnnotation, TrackSignature
from core import BeamSearch

from config import HERTZ
from utils import sec_to_ticks, ticks_to_sec

def build_test_query() -> Query:
    q = Query()
    q.set_length(sec_to_ticks(180.0))

    # Video signature (normalized)
    q.set_signature(TrackSignature([0.6, 0.4, 0.8, 0.3]))

    # Requested drops in seconds -> convert to ticks
    q.add_annotation(PointAnnotation("drop", time=sec_to_ticks(30.0), strength=1.0))
    q.add_annotation(PointAnnotation("drop", time=sec_to_ticks(95.0), strength=1.0))
    q.add_annotation(PointAnnotation("drop", time=sec_to_ticks(150.0), strength=0.1))

    return q


def build_test_tracks() -> TrackLibrary:
    tracks = TrackLibrary()

    # Track A: good if placed so its 35s drop lands near 30s
    a = Track()
    a.set_length(sec_to_ticks(60.0))
    a.set_BPM(128)
    a.set_signature(TrackSignature([0.62, 0.38, 0.79, 0.28]))
    a.preference = 0.9
    a.min_speed = 0.98
    a.max_speed = 1.20
    a.add_annotation(PointAnnotation("drop", time=sec_to_ticks(12.0), strength=0.9))
    a.add_annotation(PointAnnotation("drop", time=sec_to_ticks(35.0), strength=0.7))
    tracks.add_track(a)

    # Track B: good if placed so its 20s drop lands near 95s
    b = Track()
    b.set_length(sec_to_ticks(75.0))
    b.set_BPM(174)
    b.set_signature(TrackSignature([0.58, 0.45, 0.75, 0.35]))
    b.preference = 0.8
    b.min_speed = 0.98
    b.max_speed = 1.20
    b.add_annotation(PointAnnotation("drop", time=sec_to_ticks(20.0), strength=1.0))
    b.add_annotation(PointAnnotation("drop", time=sec_to_ticks(50.0), strength=0.8))
    tracks.add_track(b)

    # Track C: good if placed so its 12s drop lands near 150s
    c = Track()
    c.set_length(sec_to_ticks(50.0))
    c.set_BPM(110)
    c.set_signature(TrackSignature([0.40, 0.60, 0.55, 0.20]))
    c.preference = 0.6
    c.min_speed = 0.98
    c.max_speed = 1.20
    c.add_annotation(PointAnnotation("drop", time=sec_to_ticks(12.0), strength=0.6))
    tracks.add_track(c)

    return tracks


def best_request_match(request_time_ticks: float, alignment):
    """
    Returns:
        (best_drop_time_ticks, abs_error_ticks)
    Finds the closest placed track drop to the request time.
    """
    best_drop_time = None
    best_error = None

    for track in alignment.tracks:
        if track.start_time is None or track.speed is None:
            continue

        for ann in track.annotations:
            if ann.label != "drop" or ann.time is None:
                continue

            placed_time = track.start_time + (ann.time / track.speed)
            error = abs(request_time_ticks - placed_time)

            if best_error is None or error < best_error:
                best_error = error
                best_drop_time = placed_time

    if best_drop_time is None:
        return (None, None)

    return best_drop_time, best_error


def main():
    query = build_test_query()
    tracks = build_test_tracks()

    optimizer = BeamSearch(beam_width=100, max_steps=10)
    alignment = optimizer.optimize(query, tracks)

    print("==== RESULT ====")
    print("Alignment score:", alignment.score)
    print("Placed tracks:", len(alignment.tracks))

    print("\n---- TRACKS ----")
    for i, track in enumerate(alignment.tracks, start=1):
        print(f"\nTrack {i}")
        print("  start_time:", ticks_to_sec(track.start_time))
        print("  length:", ticks_to_sec(track.length))
        print("  BPM:", track.BPM)
        print("  speed:", track.speed)
        print("  preference:", track.preference)
        print("  signature:", track.signature.sig if track.signature else None)

        for ann in track.annotations:
            if ann.label == "drop":
                placed_drop = track.start_time + (ann.time / track.speed)
                print(
                    "   drop:",
                    ticks_to_sec(placed_drop),
                    "strength:",
                    ann.strength
                )

    print("\n---- REQUEST MATCHES ----")
    requested_drops = [
        a for a in query.annotations
        if a.label == "drop" and a.time is not None
    ]

    for req in requested_drops:
        best_drop_time, best_error = best_request_match(req.time, alignment)

        if best_drop_time is None:
            print("Request", ticks_to_sec(req.time), "-> no match found")
        else:
            print(
                "Request",
                ticks_to_sec(req.time),
                "-> best placed drop at",
                ticks_to_sec(best_drop_time),
                "error =",
                ticks_to_sec(best_error),
                "sec"
            )


if __name__ == "__main__":
    main()
