from __future__ import annotations

import asyncio
import logging
import time
import unicodedata

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

try:
    from langdetect import detect as _langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # make langdetect deterministic
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

from app.config import settings
from app.models.schemas import IdentifyResponse, SearchResponse, TrackResult
from app.services.audio_recognition import recognizer
from app.services.spotify import enrich_track, fetch_preview_audio, search_track
from app.services.recommender import recommender
from app.services.deezer import fetch_deezer_preview


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


# Maps language codes to human-readable search hints for Spotify queries
_LANG_SEARCH_HINT: dict[str, str] = {
    "hi": "hindi", "ar": "arabic", "ko": "korean", "ja": "japanese",
    "zh-cn": "chinese", "zh-tw": "chinese", "ru": "russian", "th": "thai",
    "ta": "tamil", "te": "telugu", "bn": "bengali", "pa": "punjabi",
    "es": "spanish", "fr": "french", "pt": "portuguese", "de": "german",
    "it": "italian", "tr": "turkish", "nl": "dutch", "pl": "polish",
}


def _detect_language(title: str, artist: str) -> str | None:
    """
    Returns a BCP-47-style language code for the track, or None for English.
    Uses Unicode script detection for non-Latin scripts (reliable) and
    langdetect for Latin-script languages (Spanish, French, etc.).
    """
    text = f"{title} {artist}".strip()
    if not text:
        return None

    # Non-Latin scripts — check character by character (highly reliable)
    _SCRIPT_LANGS = {
        "DEVANAGARI": "hi", "ARABIC": "ar", "HANGUL": "ko",
        "HIRAGANA": "ja", "KATAKANA": "ja", "CJK UNIFIED": "zh-cn",
        "CYRILLIC": "ru", "THAI": "th", "TAMIL": "ta",
        "TELUGU": "te", "BENGALI": "bn", "GURMUKHI": "pa",
    }
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        for script, lang in _SCRIPT_LANGS.items():
            if script in name:
                logger.debug("detect_language | script=%s  lang=%s", script, lang)
                return lang

    # Latin script — use langdetect
    if _LANGDETECT_AVAILABLE:
        try:
            lang = _langdetect(text)
            return lang if lang != "en" else None
        except Exception:
            pass

    return None  # English or undetectable → no filtering


async def _resolve_to_spotify(raw_recs: list[dict], limit: int, source_lang: str | None = None) -> list[TrackResult]:
    """
    FAISS results come from the FMA dataset — most tracks aren't on Spotify.
    For each result, search Spotify by title + artist and return the real
    Spotify track (with proper ID, album art, and preview URL), preserving
    the ML match score. Runs all lookups concurrently.
    """
    lang_hint = _LANG_SEARCH_HINT.get(source_lang, source_lang) if source_lang else None

    async def lookup(meta: dict) -> TrackResult | None:
        base = f"{meta.get('title', '')} {meta.get('artist', '')}"
        # Add language hint so Spotify surfaces culturally-matching tracks
        query = f"{base} {lang_hint}" if lang_hint else base
        try:
            results = await search_track(query, limit=1)
            if results:
                spotify_meta = results[0]
                spotify_meta["matchScore"] = meta.get("matchScore")
                return _to_track(spotify_meta)
        except Exception as exc:
            logger.debug("resolve_to_spotify | lookup failed  query=%r  err=%s", query, exc)
        return None

    results = await asyncio.gather(*[lookup(r) for r in raw_recs])
    found = [r for r in results if r is not None]

    # Post-filter: keep only tracks that match the source language
    if source_lang and found:
        matching = [
            r for r in found
            if _detect_language(r.title, r.artist) == source_lang
        ]
        # Use language-filtered results if we have enough, otherwise fall back
        if len(matching) >= min(3, limit):
            logger.info(
                "resolve_to_spotify | language filter  lang=%s  before=%d  after=%d",
                source_lang, len(found), len(matching),
            )
            found = matching

    logger.info("resolve_to_spotify | %d/%d resolved to Spotify", len(found), len(raw_recs))
    return found[:limit]


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
    t0 = time.perf_counter()
    content = await audio.read()
    size_mb = len(content) / 1_048_576
    logger.info("identify | received audio  size=%.2f MB", size_mb)

    if size_mb > settings.max_audio_size_mb:
        raise HTTPException(413, f"Audio file too large ({size_mb:.1f} MB)")

    # 1. Fingerprint with AcoustID to identify the song
    raw = await recognizer.recognize(content)
    parsed = recognizer.parse_result(raw)

    identified_track: TrackResult | None = None
    audio_for_embedding: bytes = content  # fallback: use raw mic audio

    if parsed:
        logger.info(
            "identify | AcoustID match  title=%r  artist=%r  score=%.2f",
            parsed["title"], parsed["artist"], parsed.get("score", 0),
        )
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
                    logger.info("identify | using Spotify preview for embedding")
                else:
                    logger.warning("identify | Spotify preview download failed, using mic audio")
            else:
                logger.info("identify | no Spotify preview URL, using mic audio")
    else:
        logger.info("identify | AcoustID no match, using raw mic audio for embedding")

    # 3. Encode → FAISS → resolve FMA results to Spotify tracks
    embedding = await asyncio.to_thread(recommender.encode_audio, audio_for_embedding)
    exclude_id = identified_track.id if identified_track else None
    raw_recs = recommender.recommend(
        embedding,
        exclude_id=exclude_id,
        k=settings.max_recommendations * 2,  # oversample — some won't be on Spotify
    )
    source_lang = _detect_language(
        identified_track.title if identified_track else "",
        identified_track.artist if identified_track else "",
    )
    recommendations = await _resolve_to_spotify(raw_recs, limit=settings.max_recommendations, source_lang=source_lang)

    logger.info(
        "identify | done  identified=%s  recs=%d  total=%.3fs",
        identified_track.title if identified_track else "none",
        len(recommendations),
        time.perf_counter() - t0,
    )
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
    t0 = time.perf_counter()
    logger.info("search | query=%r", q)
    tracks_data = await search_track(q, limit=10)
    results = [_to_track(t) for t in tracks_data]
    logger.info("search | results=%d  total=%.3fs", len(results), time.perf_counter() - t0)
    return SearchResponse(results=results, recommendations=[])


@router.get("/recommendations/{track_id}", response_model=list[TrackResult])
async def get_recommendations(track_id: str) -> list[TrackResult]:
    """
    User selected a track from search — download its Spotify 30s preview,
    encode with CnnMusicEncoder, and return similar songs from the FAISS index.
    """
    t0 = time.perf_counter()
    logger.info("recommendations | track_id=%s", track_id)

    enriched = await enrich_track(track_id)
    if not enriched:
        raise HTTPException(404, f"Track {track_id} not found on Spotify")

    preview_url = enriched.get("previewUrl")
    if not preview_url:
        logger.info("recommendations | no Spotify preview, trying Deezer  track_id=%s", track_id)
        preview_url = await fetch_deezer_preview(
            enriched.get("title", ""), enriched.get("artist", "")
        )
    if not preview_url:
        logger.warning("recommendations | no preview on Spotify or Deezer  track_id=%s", track_id)
        return []

    preview_bytes = await fetch_preview_audio(preview_url)
    if not preview_bytes:
        logger.warning("recommendations | preview download failed  track_id=%s", track_id)
        return []

    embedding = await asyncio.to_thread(recommender.encode_audio, preview_bytes)
    raw_recs = recommender.recommend(
        embedding,
        exclude_id=track_id,
        k=settings.max_recommendations * 2,  # oversample — some won't be on Spotify
    )
    source_lang = _detect_language(enriched.get("title", ""), enriched.get("artist", ""))
    recommendations = await _resolve_to_spotify(raw_recs, limit=settings.max_recommendations, source_lang=source_lang)

    logger.info(
        "recommendations | done  title=%r  recs=%d  total=%.3fs",
        enriched.get("title"), len(recommendations), time.perf_counter() - t0,
    )
    return recommendations
