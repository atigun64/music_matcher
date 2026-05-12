from config import HERTZ

def sec_to_ticks(seconds: float) -> float:
    return seconds * HERTZ

def ticks_to_sec(ticks: float) -> float:
    return ticks / HERTZ
