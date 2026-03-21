from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.config import settings
from app.models.schemas import IdentifyResponse, SearchResponse, TrackResult
from app.services.audio_recognition import recognizer
from app.services.spotify import enrich_track, fetch_preview_audio, search_track
from app.services.recommender import recommender


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _to_track(meta: dict) -> TrackResult:
    return TrackResult(
        id=meta.get("id", ""),
        title=meta.get("title", ""),
        artist=meta.get("artist", ""),
        album=meta.get("album", ""),
        albumArtUrl=meta.get("albumArtUrl", ""),
        previewUrl=meta.get("previewUrl"),
        durationMs=meta.get("durationMs", 0),
        matchScore=meta.get("matchScore"),
        vibes=meta.get("vibes", []),
        bpm=meta.get("bpm", 0),
        key=meta.get("key", ""),
        energy=meta.get("energy", 0.5),
        valence=meta.get("valence", 0.5),
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "recommender_loaded": recommender._loaded}


@router.post("/identify", response_model=IdentifyResponse)
async def identify(audio: UploadFile = File(...)) -> IdentifyResponse:
    """
    Receive an audio snippet, fingerprint it via AcoustID to identify the song,
    then use the Spotify 30s preview of that song as the clean audio source for
    ML recommendations. Falls back to encoding the raw mic audio if the song
    cannot be identified or has no Spotify preview.
    """
    content = await audio.read()
    size_mb = len(content) / 1_048_576

    if size_mb > settings.max_audio_size_mb:
        raise HTTPException(413, f"Audio file too large ({size_mb:.1f} MB)")

    # 1. Fingerprint with AcoustID to identify the song
    raw = await recognizer.recognize(content)
    parsed = recognizer.parse_result(raw)

    identified_track: TrackResult | None = None
    audio_for_embedding: bytes = content  # fallback: use raw mic audio

    if parsed:
        # 2. Look up on Spotify for display metadata + preview URL
        query = f"{parsed['title']} {parsed['artist']}"
        tracks = await search_track(query, limit=1)
        if tracks:
            identified_track = _to_track(tracks[0])
            # Use the clean Spotify 30s preview instead of the noisy mic clip
            preview_url = tracks[0].get("previewUrl")
            if preview_url:
                preview_bytes = await fetch_preview_audio(preview_url)
                if preview_bytes:
                    audio_for_embedding = preview_bytes

    # 3. Encode → FAISS → recommendations
    embedding = await asyncio.to_thread(recommender.encode_audio, audio_for_embedding)
    exclude_id = identified_track.id if identified_track else None
    raw_recs = recommender.recommend(
        embedding,
        exclude_id=exclude_id,
        k=settings.max_recommendations,
    )
    recommendations = [_to_track(r) for r in raw_recs]

    return IdentifyResponse(
        identified=identified_track,
        recommendations=recommendations,
    )


@router.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., min_length=1)) -> SearchResponse:
    """
    Search by song / artist name. Returns Spotify results for the user to pick from.
    No recommendations yet — user selects a track, then /recommendations is called.
    """
    tracks_data = await search_track(q, limit=10)
    results = [_to_track(t) for t in tracks_data]
    return SearchResponse(results=results, recommendations=[])


@router.get("/recommendations/{track_id}", response_model=list[TrackResult])
async def get_recommendations(track_id: str) -> list[TrackResult]:
    """
    User selected a track from search — download its Spotify 30s preview,
    encode with CnnMusicEncoder, and return similar songs from the FAISS index.
    """
    enriched = await enrich_track(track_id)
    if not enriched:
        raise HTTPException(404, f"Track {track_id} not found on Spotify")

    preview_url = enriched.get("previewUrl")

    if not preview_url:
        raise HTTPException(
            404,
            f"No 30s preview available for track {track_id} — cannot encode audio",
        )

    preview_bytes = await fetch_preview_audio(preview_url)
    if not preview_bytes:
        raise HTTPException(503, "Failed to download Spotify preview audio")

    embedding = await asyncio.to_thread(recommender.encode_audio, preview_bytes)
    raw_recs  = recommender.recommend(
        embedding,
        exclude_id=track_id,
        k=settings.max_recommendations,
    )
    return [_to_track(r) for r in raw_recs]
