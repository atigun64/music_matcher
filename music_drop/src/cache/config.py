from pathlib import Path

DATASET_ROOT = Path("music_drop", "data", "ncs_dataset")
CACHE_ROOT = Path("music_drop", "data", "feature_cache")
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

TIMEOUT_PER_SONG_SEC = 240  # adjust if needed
