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
    key: str = "—"
    energy: float = 0.5   # 0-1
    valence: float = 0.5  # 0-1
    instruments: list[str] = []
    instrumentOrder: list[str] = []
    instrumentPace: str = "Steady Mid-Tempo"
    vocalPresence: float = 0.5
    vocalPattern: str = "Dynamic Melodic Phrasing"
    vocalComplementation: str = "In-The-Pocket Groove"


class IdentifyResponse(BaseModel):
    identified: TrackResult | None
    recommendations: list[TrackResult]


class SearchResponse(BaseModel):
    results: list[TrackResult]
    recommendations: list[TrackResult]
