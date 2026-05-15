from typing import List

class Annotation:
    def __init__(self, label):
        self.label = label
        self.score = None

class PointAnnotation(Annotation):
    def __init__(self, label, time=None, strength=1.0):
        super().__init__(label)
        self.time = time
        self.strength = strength

class TrackSignature:
    def __init__(self, sig: List[float]):
        self.sig = sig

class Track:
    def __init__(self):
        self.length = None
        self.BPM = None
        self.signature = None
        self.annotations = []
        self.preference = 1.0
        self.min_speed = 0.98
        self.max_speed = 1.2

    def set_length(self, length):
        self.length = length

    def set_BPM(self, bpm):
        self.BPM = bpm

    def set_signature(self, signature: TrackSignature):
        self.signature = signature

    def add_annotation(self, annotation: Annotation):
        self.annotations.append(annotation)

class TrackLibrary:
    def __init__(self):
        self.tracks = []
    
    def get_tracks(self):
        return self.tracks

    def add_track(self, track):
        self.tracks.append(track)

class AssignedTrack(Track):
    def __init__(self, length=None):
        super().__init__()
        self.length = length
        self.start_time = None
        self.speed = 1.0

class Query:
    def __init__(self):
        self.length = None
        self.signature = None
        self.annotations = []

    def set_signature(self, signature: 'TrackSignature'):
        self.signature = signature

    def set_length(self, length):
        self.length = length

    def add_annotation(self, annotation: Annotation):
        self.annotations.append(annotation)

class Alignment:
    def __init__(self):
        self.tracks = []
        self.score = None
