from pathlib import Path

class UISample:
    def __init__(self, track_path: Path, key_point, time_window, tolerance_window, model_score = 0.0):
        self.track_path = track_path
        self.key_point = key_point
        self.time_window = time_window
        self.tolerance_window = tolerance_window
        self.model_score :float = model_score
