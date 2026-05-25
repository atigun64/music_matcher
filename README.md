# Music Alignment Optimizer

A Python project for automatically matching music tracks to a requested timeline by aligning musical drop points and beat structure.

This repository now contains three main capabilities:

- **ML-based drop detection**: a trained model (`drop_model.joblib`) scores likely drop points inside tracks.
- **Track library / app services**: upload tracks, extract features and drop candidates, store track metadata and annotations.
- **Studio query + optimizer**: define a studio query with requested drop points, then run the optimizer to align tracks to that query.

## What it does today

### Drop detection

The pipeline extracts audio features and beat information from an uploaded track, then:

- runs heuristic candidate detection on audio feature signals,
- scores candidates with a trained `drop_model.joblib`,
- filters strong drop proposals,
- returns time-aligned drop annotations for the track.

The ML model is used by `music_core/drop/drop_ml.py` and is integrated into `app/services/track_service.py` during track upload.

### Track library


Uploaded tracks are stored in `data/track_library/` with:

- `meta.json` metadata (length, BPM, signature, speed constraints),
- `annotations.json` detected drop points,
- `audio_path.txt` reference to the source audio file.

Track upload and track CRUD operations are managed by `app.services.track_service.TrackService` and `app.storage.track_store.TrackStore`.

### Studio query / alignment

The app currently supports a studio workflow that can:

- create and persist `StudioSession` objects,
- save a user query (`QuerySpec`) describing the desired timeline,
- store requested drop points and overall track signature,
- run the optimizer and save the resulting `AlignmentSpec`.

Studio session data is persisted under `data/studios/`.

### Optimization

The optimizer builds an app-level query from `QuerySpec` and converts stored tracks into optimizer track objects. It uses a beam search implementation in `optimizer/core/optimizer.py` to search for a placement of tracks that maximizes alignment score.

The optimizer considers:

- requested drop point alignment,
- track BPM and signature,
- track speed constraints,
- overlap and gap scoring.

The result contains placed tracks with start times, speed adjustments, and aligned drop points.

## Current status

✅ Core backend is implemented:

- `music_core` feature extraction and ML drop detection
- `app/services/track_service.py` for track upload and annotation extraction
- `app/services/studio_service.py` for studio session and query management
- `app/services/run_optimization.py` for converting app models into optimizer models and running the beam search
- `app/storage` persistence for track and studio data

⚠️ Remaining work:

- UI layer is not finished yet
- query creation and video input front-end is currently manual / code-driven
- better track signature computation is TODO (`TrackService` uses a placeholder signature today)
- richer query/video matching experience

## How to use it now

A simple console-driven workflow exists in `test2.py` that lets you:

1. upload a track and detect drop points,
2. list saved tracks,
3. create a studio session,
4. set a query with requested drop points,
5. run the optimizer,
6. inspect the resulting alignment.

## Project structure

- `music_core/` — audio feature extraction and drop candidate detection logic
- `music_drop/` — dataset and training-related scripts
- `app/` — higher-level services, models, persistence, and optimization integration
- `optimizer/` — search engine and scoring for matching tracks to a requested timeline

## Notes

This repo is currently best described as a backend proof-of-concept where drop detection, track library management, and alignment optimization are working. The missing piece is the user-facing UI, which remains the next development step.
