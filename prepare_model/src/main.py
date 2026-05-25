from pathlib import Path
from ui.labeling_ui import UI
from ui.data import UISample

samples = [
    UISample(
        track_path=Path("music_drop/data/ncs_dataset/_2oVvLVZMHU/Itro - All For You (feat. SILIAS) ｜ DnB ｜ NCS - Copyright Free Music.mp3"),
        key_point=121.2,
        time_window=(116.0, 126.0),
    ),
    UISample(
        track_path=Path("music_drop/data/ncs_dataset/_2oVvLVZMHU/Itro - All For You (feat. SILIAS) ｜ DnB ｜ NCS - Copyright Free Music.mp3"),
        key_point=121.2,
        time_window=(116.0, 126.0),
    ),
    UISample(
        track_path=Path("music_drop/data/ncs_dataset/_2oVvLVZMHU/Itro - All For You (feat. SILIAS) ｜ DnB ｜ NCS - Copyright Free Music.mp3"),
        key_point=121.2,
        time_window=(126.0, 160.0),
    ),
]

labels = UI.sample(samples)
print(labels)   # e.g. [1, None]

UI.view(samples)
