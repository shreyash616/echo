from __future__ import annotations
from pydantic import BaseModel


class TrackResult(BaseModel):
    id: str
    title: str
    artist: str
    album: str
    albumArtUrl: str
    previewUrl: str | None = None
    durationMs: int
    matchScore: float | None = None  # cosine similarity 0-1
    vibes: list[str]
    bpm: int
    key: str
    energy: float   # 0-1
    valence: float  # 0-1


class IdentifyResponse(BaseModel):
    identified: TrackResult | None
    recommendations: list[TrackResult]


class SearchResponse(BaseModel):
    results: list[TrackResult]
    recommendations: list[TrackResult]
