from __future__ import annotations

from pathlib import Path
from typing import List

from app.services.track_service import TrackService
from app.services.studio_service import StudioService
from app.storage import TRACK_STORE, STUDIO_STORE

from app.models import AnnotationPoint, QuerySpec, StudioMeta

# ------------------------------------------------------------
# Track helpers
# ------------------------------------------------------------

def print_tracks(service: TrackService):
    tracks = service.list_tracks()

    if not tracks:
        print("\nNo tracks saved.\n")
        return

    print("\nSaved tracks:")
    print("-" * 100)
    for item in tracks:
        track_id = item.get("track_id", "")
        display_name = item.get("display_name", "")
        audio_path = item.get("audio_path", "")
        print(f"ID: {track_id:>4} | {display_name} | {audio_path}")
    print("-" * 100)
    print()


def upload_track_interactive(service: TrackService):
    raw = input("Enter audio file path: ").strip().strip('"').strip("'")
    if not raw:
        print("No path entered.")
        return

    path = Path(raw)
    if not path.exists():
        print(f"File does not exist: {path}")
        return

    try:
        record = service.upload_track(path)
        print("\nUploaded successfully.")
        print(f"Track ID: {record.track_id}")
        print(f"Audio path: {record.audio_path}")
        print(f"Length ticks: {record.meta.length_ticks}")
        print(f"BPM: {record.meta.bpm}")
        print(f"Signature: {record.meta.signature}")
        print(f"Annotations: {len(record.annotations)}")
        print()
    except Exception as e:
        print(f"\nUpload failed: {e}\n")


def view_track_interactive(service: TrackService):
    track_id = input("Enter track id: ").strip()
    if not track_id:
        print("No track id entered.")
        return

    try:
        record = service.load_track(track_id)
        print("\nTrack loaded:")
        print(f"Track ID: {record.track_id}")
        print(f"Audio path: {record.audio_path}")
        print(f"Length ticks: {record.meta.length_ticks}")
        print(f"BPM: {record.meta.bpm}")
        print(f"Signature: {record.meta.signature}")
        print(f"Preference: {record.meta.preference}")
        print(f"Min speed: {record.meta.min_speed}")
        print(f"Max speed: {record.meta.max_speed}")
        print("\nAnnotations:")
        if not record.annotations:
            print("  (none)")
        else:
            for i, ann in enumerate(record.annotations, start=1):
                print(
                    f"  {i:02d}. {ann.label} | time={ann.time_ticks} | strength={ann.strength}"
                )
        print()
    except Exception as e:
        print(f"\nCould not load track: {e}\n")


def delete_track_interactive(service: TrackService):
    track_id = input("Enter track id to delete: ").strip()
    if not track_id:
        print("No track id entered.")
        return

    confirm = input(f"Delete track '{track_id}'? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        service.delete_track(track_id)
        print(f"Track {track_id} deleted.\n")
    except Exception as e:
        print(f"\nDelete failed: {e}\n")


# ------------------------------------------------------------
# Studio helpers
# ------------------------------------------------------------

def print_studios(service: StudioService):
    studio_ids = service.list_studio_ids()

    if not studio_ids:
        print("\nNo studios saved.\n")
        return

    print("\nSaved studios:")
    print("-" * 100)
    for sid in studio_ids:
        print(f"Studio ID: {sid}")
    print("-" * 100)
    print()


def create_studio_interactive(service: StudioService):
    try:
        studio_id = service.create_studio()
        print(f"\nCreated studio: {studio_id}\n")
    except Exception as e:
        print(f"\nCould not create studio: {e}\n")


def view_studio_session_interactive(service: StudioService):
    studio_id = input("Enter studio id: ").strip()
    if not studio_id:
        print("No studio id entered.")
        return

    try:
        session = service.get_studio_session(studio_id)
        print("\nStudio loaded:")
        print(f"Studio ID: {session.studio_id}")
        print(f"Meta source: {session.meta.source}")
        print(f"Meta video path: {session.meta.video_path}")
        print(f"Meta notes: {session.meta.notes}")

        if session.query is None:
            print("\nQuery: (none)")
        else:
            print("\nQuery:")
            print(f"  Length ticks: {session.query.length_ticks}")
            print(f"  Signature: {session.query.signature}")
            print(f"  Requested points: {len(session.query.requested_points)}")
            for i, p in enumerate(session.query.requested_points, start=1):
                print(
                    f"    {i:02d}. {p.label} | time={p.time_ticks} | strength={p.strength}"
                )

        if session.alignment is None:
            print("\nAlignment: (none)")
        else:
            print("\nAlignment:")
            print(f"  Score: {session.alignment.score}")
            print(f"  Tracks: {len(session.alignment.tracks)}")
            for i, tr in enumerate(session.alignment.tracks, start=1):
                print(
                    f"    {i:02d}. track_id={tr.track_id} | start={tr.start_time_ticks} | speed={tr.speed} | placed_points={len(tr.placed_points)}"
                )
                for j, p in enumerate(tr.placed_points, start=1):
                    print(
                        f"       {j:02d}. {p.label} | time={p.time_ticks} | strength={p.strength}"
                    )

        print()
    except Exception as e:
        print(f"\nCould not load studio session: {e}\n")


def edit_studio_meta_interactive(service: StudioService):
    studio_id = input("Enter studio id: ").strip()
    if not studio_id:
        print("No studio id entered.")
        return

    try:
        source = input("Source [silent/video]: ").strip().lower()
        if source not in ("silent", "video"):
            print("Invalid source, using silent.")
            source = "silent"

        video_path = None
        if source == "video":
            vp = input("Enter video path (or empty): ").strip().strip('"').strip("'")
            video_path = vp if vp else None

        notes = input("Notes (optional): ").strip()

        meta = StudioMeta(
            source=source,
            video_path=video_path,
            notes=notes,
        )

        session = service.get_studio_session(studio_id)
        session.meta = meta
        service.save_studio_session(session)

        print("Studio meta saved.\n")
    except Exception as e:
        print(f"\nCould not edit studio meta: {e}\n")


def create_query_interactive(service: StudioService):
    studio_id = input("Enter studio id: ").strip()
    if not studio_id:
        print("No studio id entered.")
        return

    try:
        length_sec = float(input("Query length in seconds: ").strip())
        signature_raw = input("Signature [4 floats comma-separated, default 0.6,0.4,0.8,0.3]: ").strip()
        if signature_raw:
            signature = [float(x.strip()) for x in signature_raw.split(",")]
        else:
            signature = [0.6, 0.4, 0.8, 0.3]

        point_count = int(input("How many requested drop points? ").strip())

        requested_points: List[AnnotationPoint] = []
        for i in range(point_count):
            print(f"\nPoint {i + 1}/{point_count}")
            point_sec = float(input("  Time in seconds: ").strip())
            strength = float(input("  Strength [0..1, e.g. 1.0 / 0.66 / 0.33]: ").strip())
            requested_points.append(
                AnnotationPoint(
                    label="drop",
                    time_ticks=float(point_sec),
                    strength=strength,
                )
            )

        query = QuerySpec(
            length_ticks=float(length_sec),
            signature=signature,
            requested_points=requested_points,
        )

        service.save_query(studio_id, query)
        print("\nQuery saved.\n")
    except Exception as e:
        print(f"\nCould not create/save query: {e}\n")


def run_optimizer_interactive(service: StudioService):
    studio_id = input("Enter studio id: ").strip()
    if not studio_id:
        print("No studio id entered.")
        return

    try:
        alignment = service.run_optimizer(studio_id)
        print("\nOptimizer finished.")
        print(f"Alignment score: {alignment.score}")
        print(f"Placed tracks: {len(alignment.tracks)}")
        for i, tr in enumerate(alignment.tracks, start=1):
            print(
                f"  {i:02d}. track_id={tr.track_id} | start_ticks={tr.start_time_ticks} | speed={tr.speed} | placed_points={len(tr.placed_points)}"
            )
        print()
    except Exception as e:
        print(f"\nOptimizer failed: {e}\n")


def delete_studio_interactive(service: StudioService):
    studio_id = input("Enter studio id to delete: ").strip()
    if not studio_id:
        print("No studio id entered.")
        return

    confirm = input(f"Delete studio '{studio_id}'? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    try:
        # If you haven’t implemented delete yet, this will fail.
        service.studio_store.delete_studio(studio_id)
        print(f"Studio {studio_id} deleted.\n")
    except Exception as e:
        print(f"\nDelete failed: {e}\n")


# ------------------------------------------------------------
# Menus
# ------------------------------------------------------------

def track_menu(track_service: TrackService):
    while True:
        print("==== TRACK MENU ====")
        print("1) List tracks")
        print("2) Upload track")
        print("3) View track")
        print("4) Delete track")
        print("5) Back")
        choice = input("> ").strip()

        if choice == "1":
            print_tracks(track_service)
        elif choice == "2":
            upload_track_interactive(track_service)
        elif choice == "3":
            view_track_interactive(track_service)
        elif choice == "4":
            delete_track_interactive(track_service)
        elif choice == "5":
            break
        else:
            print("Unknown option.\n")


def studio_menu(studio_service: StudioService):
    while True:
        print("==== STUDIO MENU ====")
        print("1) List studios")
        print("2) Create studio")
        print("3) View studio session")
        print("4) Edit studio meta")
        print("5) Create / Save query")
        print("6) Run optimizer")
        print("7) Delete studio")
        print("8) Back")
        choice = input("> ").strip()

        if choice == "1":
            print_studios(studio_service)
        elif choice == "2":
            create_studio_interactive(studio_service)
        elif choice == "3":
            view_studio_session_interactive(studio_service)
        elif choice == "4":
            edit_studio_meta_interactive(studio_service)
        elif choice == "5":
            create_query_interactive(studio_service)
        elif choice == "6":
            run_optimizer_interactive(studio_service)
        elif choice == "7":
            delete_studio_interactive(studio_service)
        elif choice == "8":
            break
        else:
            print("Unknown option.\n")


def main():
    track_service = TrackService(TRACK_STORE)
    studio_service = StudioService(STUDIO_STORE)

    while True:
        print("==== MUSIC DROP APP TEST ====")
        print("1) Track library")
        print("2) Studio")
        print("3) Exit")
        choice = input("> ").strip()

        if choice == "1":
            track_menu(track_service)
        elif choice == "2":
            studio_menu(studio_service)
        elif choice == "3":
            break
        else:
            print("Unknown option.\n")


if __name__ == "__main__":
    main()
