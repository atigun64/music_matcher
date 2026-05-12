from config import HERTZ

DROP_MISS_TOLERANCE_SEC = 2.0 # Tolerance in seconds for drop annotations
MAX_ACCEPTABLE_GAP_SEC = 5.0       # seconds
MAX_ACCEPTABLE_OVERLAP_SEC = 5.0    # seconds

VIDEO_EDGE_FRAC = 0.20   # first/last 20% of the video are edge zones

BPM_EDGE_MIN = 85
BPM_EDGE_MAX = 130       # good BPM for beginning/end

BPM_MID_MIN = 125
BPM_MID_MAX = 180        # good BPM for middle

BPM_FALLOFF = 20.0       # how fast score drops outside the preferred range

# Gap quality parameters
AVERAGE_GAP_WEIGHT = 0.6
WORST_GAP_WEIGHT = 0.4

# Overlap quality parameters
AVERAGE_OVERLAP_WEIGHT = 0.6
WORST_OVERLAP_WEIGHT = 0.4

# Weights for different score components
DROP_UNCOVERED_PENALTY = 2.0 # Penalty factor for uncovered drops, relative to total request strength
WEIGHT_DROPS = 9.9
WEIGHT_BPM = 0.02
WEIGHT_PREFERENCE = 0.02
WEIGHT_STYLE = 0.02
WEIGHT_GAP = 0.02
WEIGHT_OVERLAP = 0.02


DROP_MISS_TOLERANCE = DROP_MISS_TOLERANCE_SEC * HERTZ # Convert tolerance to number of annotations
MAX_ACCEPTABLE_GAP = MAX_ACCEPTABLE_GAP_SEC * HERTZ
MAX_ACCEPTABLE_OVERLAP = MAX_ACCEPTABLE_OVERLAP_SEC * HERTZ