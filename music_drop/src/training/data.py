from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class Sample:
    track_id: str
    beat_idx: int
    x: np.ndarray
    y: Optional[int] = None   # label, 0/1
    source: str = ""          # "heuristic" or "background"
    hscore: float = 0.0       # heuristic score