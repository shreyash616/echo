from __future__ import annotations
from pydantic import BaseModel


class TrackResult(BaseModel):
    id: str
    title: str
    artist: str
    album: str
    albumArtUrl: str
    previewUrl: str | None = None
    durationMs: int = 0
    matchScore: float | None = None  # similarity 0-1
    vibes: list[str] = []
    bpm: int = 120
    felt_bpm: int | None = None
    key: str = "—"
    camelotKey: str | None = None
    modeMinorProb: float | None = None
    brightness: float | None = None
    buildScore: float | None = None
    loudnessRange: float | None = None
    stereoWidth: float = 0.5
    phaseCorrelation: float = 0.95
    crestFactor: float = 10.0
    spatialStaging: str = "Balanced Stereo"
    lyricalIntent: str = "Chill & Atmospheric"
    energy: float = 0.5   # 0-1
    valence: float = 0.5  # 0-1
    languageAffinity: float | None = None
    instruments: list[str] = []
    instrumentOrder: list[str] = []
    instrumentPace: str = "Steady Mid-Tempo"
    vocalPresence: float = 0.5
    vocalPattern: str = "Dynamic Melodic Phrasing"
    vocalComplementation: str = "In-The-Pocket Groove"
    year: int | None = None
    decade: str | None = None
    matchReasons: list[str] = []


class IdentifyResponse(BaseModel):
    identified: TrackResult | None
    recommendations: list[TrackResult]


class SearchResponse(BaseModel):
    results: list[TrackResult]
    recommendations: list[TrackResult]
