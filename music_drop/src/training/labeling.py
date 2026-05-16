import numpy as np
from typing import List

from .data import Sample

def label_batch(samples: List[Sample]) -> List[Sample]:
    """
    Ask the human to label each sample.
    Replace this with your GUI / annotation tool if you have one.
    """
    for s in samples:
        print(f"\nTrack: {s.track_id}")
        print(f"Beat: {s.beat_idx}")
        print(f"Heuristic score: {s.hscore:.3f}")
        while True:
            ans = input("Drop? [1=yes, 0=no]: ").strip()
            if ans in ("0", "1"):
                s.y = int(ans)
                break
            print("Please type 1 or 0.")
    return samples
