import numpy as np
from .data import Sample

def sample_to_vector(s: Sample) -> np.ndarray:
    return np.concatenate(([s.hscore], s.x.reshape(-1)))
