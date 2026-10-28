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
    felt_bpm: int | None = None
    key: str = "—"
    theme: str | None = None
    brightness: float | None = None
    energy: float | None = None
    languageAffinity: float | None = None
    year: int | None = None
    decade: str | None = None
    matchReasons: list[str] = []
    # 11.5 Feel descriptors
    darkness: float | None = None
    grit: float | None = None
    pulse_rate: float | None = None
    kick_class: str | None = None
    restraint_bucket: str | None = None
    intimacy: float | None = None
    crest_factor: float | None = None
    mode_polarity: float | None = None
    mode_undefined: bool | None = None
    changes_per_bar: float | None = None
    riff_flag: bool | None = None
    language: str | None = None
    language_conf: float | None = None
    language_source: str | None = None


class IdentifyResponse(BaseModel):
    identified: TrackResult | None
    recommendations: list[TrackResult]


class SearchResponse(BaseModel):
    results: list[TrackResult]
    recommendations: list[TrackResult]
