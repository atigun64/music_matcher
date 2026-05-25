import json
from pathlib import Path
from yt_dlp import YoutubeDL

from .config import OUTPUT_DIR, ARCHIVE_FILE, NCS_URL, MAX_DOWNLOADS


OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# HELPERS
# =========================

def load_archive_ids() -> set[str]:
    """Read already-downloaded IDs from the archive file."""
    ids = set()
    if ARCHIVE_FILE.exists():
        with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0] == "youtube":
                    ids.add(parts[1])
    return ids

def seed_archive_from_existing_files():
    """
    If you already have some downloaded files in OUTPUT_DIR,
    add them to the archive so yt-dlp skips them.
    """
    ids = load_archive_ids()

    # Case 1: folders named by video id, with mp3 inside
    for track_dir in OUTPUT_DIR.iterdir():
        if not track_dir.is_dir():
            continue

        if any(track_dir.glob("*.mp3")):
            ids.add(track_dir.name)

        # Case 2: metadata.json exists, extract youtube id if possible
        meta_path = track_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                url = meta.get("youtube_url", "")
                if "v=" in url:
                    video_id = url.split("v=")[-1].split("&")[0]
                    ids.add(video_id)
            except Exception:
                pass

    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        for video_id in sorted(ids):
            f.write(f"youtube {video_id}\n")

# Seed archive from what you already have
seed_archive_from_existing_files()

# =========================
# YT-DLP OPTIONS
# =========================

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": str(OUTPUT_DIR / "%(id)s" / "%(title)s.%(ext)s"),
    "ignoreerrors": True,
    "quiet": False,

    # This makes yt-dlp skip anything already in downloaded.txt
    "download_archive": str(ARCHIVE_FILE),

    # Limit how many entries from the channel it will process
    "playlistend": MAX_DOWNLOADS,

    # Saves metadata automatically as .info.json next to the file
    "writeinfojson": True,

    # Convert to mp3
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}

# =========================
# DOWNLOAD
# =========================

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([NCS_URL])

print("\nDone.")
