# Music Alignment Optimizer

A Python project for detecting musical drops, building a track library, and aligning tracks to a requested timeline using drop points, beat structure, BPM, and signature constraints.

The system has three major parts:

1. **Drop detection pipeline**
2. **Track/query data model**
3. **Alignment optimizer**

---

## Overview

The core idea is:

- extract audio features from a track,
- compute beat-synchronous descriptors,
- detect likely drop regions using a heuristic,
- optionally refine those candidates with a trained ML model,
- use the resulting drop annotations as input to a beam-search optimizer,
- align tracks to a requested query timeline.

This project is currently a **backend-first prototype**.  
The main logic works, while the user-facing UI is still evolving.

---

## Main components

| Component | Purpose |
|---|---|
| `music_core/` | Feature extraction, heuristic scoring, ML drop scoring, candidate selection |
| `music_drop/` | Dataset utilities, labeling tools, training scripts, active-learning helpers |
| `optimizer/` | Query/track models and beam-search alignment engine |
| `app/` | Higher-level services, persistence, and integration layer |
| `data/` | Stored track metadata, studio sessions, and related artifacts |

---

## How the system works

### 1) Feature extraction
For each audio track, the system extracts beat-synchronous features such as:

- energy
- onset envelope
- brightness / spectral centroid
- bass-related features
- beat times / beat positions

These are aggregated over beat windows to create a track-level feature sequence.

---

### 2) Heuristic drop candidate detection
A handcrafted heuristic scores each beat region based on:

- buildup behavior before the drop,
- contrast between pre-drop and drop regions,
- sudden changes in energy and other descriptors.

This heuristic is used as a **proposal stage**: it does not define the final answer, but it finds likely drop-like regions.

---

### 3) ML-based refinement
A trained model (`drop_model.joblib`) is used to refine candidate selection.

Training data was built manually using:

- heuristic-generated candidates,
- human labels,
- active-learning style iterations,
- track windows around candidate beats.

The model uses feature vectors built from:

- heuristic score
- flattened local feature window

So the final classifier learns when to trust the heuristic and when to override it.

---

### 4) Candidate filtering
The current drop pipeline typically works as:

1. detect heuristic candidates,
2. score them with the ML model,
3. suppress candidates that are too close together,
4. keep only the strongest drop proposals.

This is important because real drops are usually **far apart**, not clustered every few beats.

---

### 5) Alignment optimization
Once tracks have drop annotations, the optimizer builds:

- a `Query` describing the requested timeline,
- a `TrackLibrary` containing track metadata and drop annotations,

then runs beam search to find the best placement of tracks into the query.

The optimizer considers:

- requested drop points,
- track BPM,
- track signature,
- track speed constraints,
- overlap / gap penalties,
- alignment of placed drops to requested drops.

The output is an `Alignment` containing:

- placed tracks,
- start times,
- speed adjustments,
- matching scores.

---

## Training workflow

The ML detector was trained using a manual/active-learning loop.

### Training process
1. generate candidate drop windows with the heuristic,
2. inspect them manually,
3. label them as drop / not drop,
4. save labels to disk,
5. train a model on:
   - heuristic score
   - local feature window
6. repeat with new batches of uncertain examples.

### Feature vector
The current model input is:

```python
[hscore, flattened_feature_window]
```

So the model does **not** use raw audio directly at inference time.  
It uses the same beat-window representation that was used during training.

---

## Current drop-detection behavior

The detector is intentionally **region-based**, not exact-frame-based.

That means:

- a candidate does not need to hit the exact sample or beat center,
- a beat within a drop region is usually good enough,
- nearby beats around a drop are often acceptable.

This is deliberate because exact beat localization is usually too strict for music structure matching.

---

## Track library

Uploaded or indexed tracks are stored with metadata such as:

- length
- BPM
- signature
- speed constraints
- detected annotations / candidate points

The project supports storing track metadata separately from audio files so the library can be reloaded and reused.

---

## Studio query / alignment

A query describes the requested timeline, for example:

- total length,
- requested drop points,
- overall signature.

The optimizer takes this query and tries to place tracks so that their drop annotations line up with the request points.

---

## Current status

### Working
- feature extraction pipeline
- heuristic drop detection
- ML-based drop scoring
- manual training data collection
- track/query alignment optimizer
- track metadata persistence

### Still rough / in progress
- user-facing UI
- polished annotation editing
- faster browsing / candidate inspection
- better visualization of alignment results
- more robust training-data management

---

## How to use it

At the moment, the most useful workflows are script-driven:

### Drop detection
- run feature extraction on a track,
- detect candidates,
- inspect the timestamps,
- optionally label / save them.

### Training
- load labeled examples,
- train the model,
- save the resulting `drop_model.joblib`.

### Optimization
- create a query,
- load tracks and annotations,
- run beam search,
- inspect the resulting alignment.

---

## Project structure

```text
music_core/
  feature extraction
  heuristic drop scoring
  ML candidate scoring
  candidate selection

music_drop/
  dataset tools
  labeling helpers
  model training
  active learning utilities

optimizer/
  query / track models
  beam search
  alignment scoring

app/
  services
  persistence
  integration layer

data/
  saved tracks
  metadata
  studio sessions
```

---

## Notes

This repository is best understood as a **drop-aware music alignment system**:

- it can detect likely drops,
- use them as structured annotations,
- and optimize track placement against a requested timeline.

The architecture is already useful, but the next major step is to finish the UI and make the detection/annotation workflow easier to use interactively.

---

## Future work

- stronger calibration of the ML drop detector
- better UI for browsing tracks and drop candidates
- improving scoring function for the beam search for better optimization results
- better annotation management
- more robust feature extraction / caching
- better query editing and timeline visualization
- improved preview playback of aligned results
