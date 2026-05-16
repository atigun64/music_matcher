import essentia.standard as es
import librosa

import numpy as np
from .utils import beat_sync
from music_core.preprocessing import smooth

def extract_features(audio_path):
    # ----------------------------
    # 1) Load audio with librosa
    # ----------------------------
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # ----------------------------
    # 2) Beat tracking with Essentia
    # ----------------------------
    # Essentia loader
    loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
    y_es = loader()

    # Rhythm extraction
    rhythm = es.RhythmExtractor2013(method="multifeature")
    out = rhythm(y_es)

    # Expected: out[0] = bpm, out[1] = beat timestamps
    bpm = float(out[0])
    beat_times = np.array(out[1], dtype=float)


    if len(beat_times) < 8:
        raise RuntimeError("Not enough beats detected. Try using librosa beat tracking fallback.")

    # ----------------------------
    # 3) Librosa features
    # ----------------------------
    hop = 512
    n_fft = 2048

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    bass_band = (freqs >= 20) & (freqs <= 150)
    bass_energy = S[bass_band].mean(axis=0)
    full_energy = S.mean(axis=0) + 1e-9
    bass_ratio = bass_energy / full_energy

    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    E = beat_sync(rms, beat_times, frame_times)
    O = beat_sync(onset_env, beat_times, frame_times)
    C = beat_sync(centroid, beat_times, frame_times)
    F = beat_sync(flatness, beat_times, frame_times)
    B = beat_sync(bass_ratio, beat_times, frame_times)

    return smooth(E), smooth(O), smooth(C), smooth(F), smooth(B), bpm, beat_times, frame_times