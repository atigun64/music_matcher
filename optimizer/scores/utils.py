from models import AssignedTrack

def clamp01(x):
    if x is None:
        return 0.0
    return max(0.0, min(1.0, float(x)))

def track_end_time(track: AssignedTrack):
    if track.start_time is None or track.length is None or track.speed is None or track.speed <= 0:
        return None
    return track.start_time + track.length / track.speed

def track_duration(track: AssignedTrack):
    if track.length is None or track.speed is None or track.speed <= 0:
        return 0.0
    return track.length / track.speed


def range_score(value, low, high, falloff):
    if value is None:
        return 0.5

    if low <= value <= high:
        return 1.0

    if value < low:
        return max(0.0, 1.0 - (low - value) / falloff)

    return max(0.0, 1.0 - (value - high) / falloff)
