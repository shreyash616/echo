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
from app.services.deezer import search_track, get_track, fetch_preview_audio
from app.services.lastfm import get_artist_tags, get_similar_tracks
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


# Maps language codes to human-readable search hints for Spotify queries
_LANG_SEARCH_HINT: dict[str, str] = {
    "hi": "hindi", "ar": "arabic", "ko": "korean", "ja": "japanese",
    "zh-cn": "chinese", "zh-tw": "chinese", "ru": "russian", "th": "thai",
    "ta": "tamil", "te": "telugu", "bn": "bengali", "pa": "punjabi",
    "es": "spanish", "fr": "french", "pt": "portuguese", "de": "german",
    "it": "italian", "tr": "turkish", "nl": "dutch", "pl": "polish",
}

# Maps Spotify genre keyword fragments (lowercase) to language codes.
# Ordered from most-specific to least-specific so first match wins.
_GENRE_LANG_RULES: list[tuple[str, str]] = [
    # South Asian
    ("punjabi", "pa"), ("bhangra", "pa"),
    ("hindi", "hi"), ("bollywood", "hi"), ("filmi", "hi"), ("desi pop", "hi"),
    ("tamil", "ta"), ("telugu", "te"), ("bengali", "bn"), ("kannada", "kn"),
    ("malayalam", "ml"), ("marathi", "mr"),
    # East Asian
    ("k-pop", "ko"), ("k-indie", "ko"), ("k-r&b", "ko"), ("k-rap", "ko"), ("korean", "ko"),
    ("j-pop", "ja"), ("j-rock", "ja"), ("j-rap", "ja"), ("japanese", "ja"),
    ("mandopop", "zh-cn"), ("cantopop", "zh-cn"), ("c-pop", "zh-cn"), ("chinese", "zh-cn"),
    # Middle East / North Africa
    ("arabic", "ar"), ("khaleeji", "ar"), ("turkish", "tr"),
    # Other regions
    ("thai", "th"), ("russian", "ru"),
    # Latin (broad — check last so "latin trap" etc. don't override more specific)
    ("reggaeton", "es"), ("latin pop", "es"), ("latin", "es"),
]


def _detect_language_from_genres(genres: list[str]) -> str | None:
    """
    Map Spotify artist genre tags to a BCP-47-style language code.
    Returns None if no genre matches (i.e. treat as English/unknown).
    """
    joined = " ".join(genres).lower()
    for keyword, lang in _GENRE_LANG_RULES:
        if keyword in joined:
            logger.debug("detect_genre_lang | keyword=%r  lang=%s  genres=%s", keyword, lang, genres)
            return lang
    return None


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


async def _get_source_lang(track_info: dict) -> str | None:
    """
    Determine the cultural/language origin of a source track.
    Tries Spotify artist genres first (most reliable), falls back to
    Unicode script / langdetect heuristics on the title + artist text.
    """
    artist_name = track_info.get("artist", "")
    if artist_name:
        tags = await get_artist_tags(artist_name)
        lang = _detect_language_from_genres(tags)
        if lang:
            logger.info(
                "source_lang | lastfm  artist=%r  tags=%s  lang=%s",
                artist_name, tags, lang,
            )
            return lang

    # Fallback: text-based detection (works for non-Latin scripts)
    lang = _detect_language(track_info.get("title", ""), track_info.get("artist", ""))
    if lang:
        logger.info("source_lang | text  lang=%s", lang)
    return lang


async def _resolve_tracks(raw_recs: list[dict], limit: int, source_lang: str | None = None) -> list[TrackResult]:
    """
    FAISS results come from the FMA dataset — resolve each one to a real
    Deezer track (proper ID, album art, preview URL), preserving the ML
    match score. Runs all lookups concurrently.

    When source_lang is set, a language hint is appended to each search query
    and each resolved track is verified to be in the same language.
    """
    lang_hint = _LANG_SEARCH_HINT.get(source_lang, source_lang) if source_lang else None

    async def lookup(meta: dict) -> TrackResult | None:
        base = f"{meta.get('title', '')} {meta.get('artist', '')}"
        query = f"{base} {lang_hint}" if lang_hint else base
        try:
            results = await search_track(query, limit=1)
            if results:
                track_meta = results[0]
                # Language gate: verify the resolved track matches source language.
                if source_lang:
                    tags = await get_artist_tags(track_meta.get("artist", ""))
                    result_lang = _detect_language_from_genres(tags) or _detect_language(
                        track_meta.get("title", ""), track_meta.get("artist", "")
                    )
                    if result_lang != source_lang:
                        logger.debug(
                            "resolve_tracks | lang mismatch  want=%s  got=%s  track=%r",
                            source_lang, result_lang, base,
                        )
                        return None
                track_meta["matchScore"] = meta.get("matchScore")
                return _to_track(track_meta)
        except Exception as exc:
            logger.debug("resolve_tracks | lookup failed  query=%r  err=%s", query, exc)
        return None

    results = await asyncio.gather(*[lookup(r) for r in raw_recs])
    found = [r for r in results if r is not None]
    logger.info("resolve_tracks | %d/%d resolved", len(found), len(raw_recs))
    return found[:limit]


async def _lastfm_recommendations(artist: str, title: str, source_lang: str, limit: int) -> list[TrackResult]:
    """
    For non-English/Western sources, FMA/FAISS won't have matching tracks.
    Instead use Last.fm cultural similarity → resolve each result to Deezer.
    """
    similar = await get_similar_tracks(artist, title, limit=limit * 3)
    if not similar:
        return []

    seen: set[str] = set()

    async def lookup(meta: dict) -> TrackResult | None:
        key = f"{meta['artist'].lower()}:{meta['title'].lower()}"
        if key in seen:
            return None
        seen.add(key)
        try:
            results = await search_track(f"{meta['title']} {meta['artist']}", limit=1)
            if not results:
                return None
            track_meta = results[0]
            # Verify same language
            tags = await get_artist_tags(track_meta.get("artist", ""))
            result_lang = _detect_language_from_genres(tags) or _detect_language(
                track_meta.get("title", ""), track_meta.get("artist", "")
            )
            if result_lang != source_lang:
                return None
            return _to_track(track_meta)
        except Exception as exc:
            logger.debug("lastfm_recs | lookup failed  meta=%r  err=%s", meta, exc)
            return None

    results = await asyncio.gather(*[lookup(m) for m in similar])
    found = [r for r in results if r is not None]
    logger.info("lastfm_recs | %d/%d resolved  lang=%s", len(found), len(similar), source_lang)
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
    source_track_info = tracks[0] if (parsed and tracks) else {}
    source_lang = await _get_source_lang(source_track_info) if source_track_info else None

    if source_lang:
        # Non-English source: FMA index is Western music, use Last.fm cultural similarity instead
        recommendations = await _lastfm_recommendations(
            source_track_info.get("artist", ""),
            source_track_info.get("title", ""),
            source_lang,
            limit=settings.max_recommendations,
        )
    else:
        recommendations = await _resolve_tracks(raw_recs, limit=settings.max_recommendations)

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
    Search by song / artist name. Returns Deezer results for the user to pick from.
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
    User selected a track from search — download its Deezer 30s preview,
    encode with CnnMusicEncoder, and return similar songs from the FAISS index.
    """
    t0 = time.perf_counter()
    logger.info("recommendations | track_id=%s", track_id)

    track = await get_track(track_id)
    if not track:
        raise HTTPException(404, f"Track {track_id} not found on Deezer")

    preview_url = track.get("previewUrl")
    if not preview_url:
        logger.warning("recommendations | no preview available  track_id=%s", track_id)
        return []

    preview_bytes = await fetch_preview_audio(preview_url)
    if not preview_bytes:
        logger.warning("recommendations | preview download failed  track_id=%s", track_id)
        return []

    embedding = await asyncio.to_thread(recommender.encode_audio, preview_bytes)
    raw_recs = recommender.recommend(
        embedding,
        exclude_id=track_id,
        k=settings.max_recommendations * 2,
    )
    source_lang = await _get_source_lang(track)

    if source_lang:
        # Non-English source: use Last.fm cultural similarity instead of FAISS
        recommendations = await _lastfm_recommendations(
            track.get("artist", ""),
            track.get("title", ""),
            source_lang,
            limit=settings.max_recommendations,
        )
    else:
        recommendations = await _resolve_tracks(raw_recs, limit=settings.max_recommendations)

    logger.info(
        "recommendations | done  title=%r  recs=%d  total=%.3fs",
        track.get("title"), len(recommendations), time.perf_counter() - t0,
    )
    return recommendations
