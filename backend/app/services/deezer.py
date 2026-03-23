"""
Deezer public API — no authentication required.
Used for track search, metadata, and 30s preview audio.
Docs: https://developers.deezer.com/api
"""
from __future__ import annotations

import logging

import httpx
from cachetools import TTLCache

logger = logging.getLogger(__name__)

DEEZER_BASE = "https://api.deezer.com"

_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)


def _normalize(item: dict) -> dict:
    """Map a Deezer track object to the internal track dict format."""
    artist = item.get("artist", {})
    album = item.get("album", {})
    art = album.get("cover_xl") or album.get("cover_medium") or album.get("cover", "")
    return {
        "id":          str(item["id"]),
        "title":       item.get("title", ""),
        "artist":      artist.get("name", ""),
        "artistId":    str(artist.get("id", "")),
        "album":       album.get("title", ""),
        "albumArtUrl": art,
        "previewUrl":  item.get("preview") or None,
        "durationMs":  item.get("duration", 0) * 1000,
        "bpm":         0,
        "key":         "—",
        "energy":      0.5,
        "valence":     0.5,
        "vibes":       [],
    }


async def search_track(query: str, limit: int = 5) -> list[dict]:
    """Search Deezer for tracks. Returns normalized track dicts."""
    cache_key = f"search:{query}:{limit}"
    if cache_key in _cache:
        logger.info("deezer | search cache hit  query=%r", query)
        return _cache[cache_key]  # type: ignore[return-value]

    logger.info("deezer | search  query=%r  limit=%d", query, limit)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{DEEZER_BASE}/search",
                params={"q": query, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("deezer | search failed  query=%r  err=%s", query, exc)
        return []

    results = [_normalize(t) for t in data.get("data", [])]
    _cache[cache_key] = results
    return results


async def get_track(deezer_id: str) -> dict | None:
    """Fetch a single track by Deezer ID. Returns normalized dict or None."""
    cache_key = f"track:{deezer_id}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{DEEZER_BASE}/track/{deezer_id}")
            resp.raise_for_status()
            item = resp.json()
    except Exception as exc:
        logger.warning("deezer | get_track failed  id=%s  err=%s", deezer_id, exc)
        return None

    if item.get("error"):
        logger.warning("deezer | track not found  id=%s", deezer_id)
        return None

    result = _normalize(item)
    _cache[cache_key] = result
    return result


_PREVIEW_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.deezer.com/",
    "Origin": "https://www.deezer.com",
}


async def fetch_preview_audio(url: str) -> bytes | None:
    """Download a 30s preview MP3 from any URL. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=_PREVIEW_HEADERS)
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.warning("deezer | preview download failed  url=%s  err=%s", url, exc)
        return None
