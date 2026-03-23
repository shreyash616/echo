"""
Spotify Web API client for enriching track metadata.
Used to fill in album art, audio features (BPM, key, energy, valence), etc.
"""
from __future__ import annotations

import logging
import time
import httpx
from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"

_token_cache: dict = {}
_search_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)


async def _get_token() -> str:
    now = time.time()
    if _token_cache.get("expires_at", 0) > now:
        return _token_cache["token"]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(settings.spotify_client_id, settings.spotify_client_secret),
            timeout=10.0,
        )
        resp.raise_for_status()
        body = resp.json()
        _token_cache["token"] = body["access_token"]
        _token_cache["expires_at"] = now + body["expires_in"] - 60
        return _token_cache["token"]


async def _get(path: str, params: dict | None = None) -> dict:
    token = await _get_token()
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{API_BASE}{path}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()


async def enrich_track(spotify_id: str) -> dict | None:
    """Fetch track display metadata from Spotify (title, artist, album art, etc.)."""
    cache_key = f"enrich:{spotify_id}"
    if cache_key in _search_cache:
        logger.info("spotify | enrich cache hit  id=%s", spotify_id)
        return _search_cache[cache_key]  # type: ignore[return-value]

    try:
        track_data = await _get(f"/tracks/{spotify_id}")
    except Exception as exc:
        logger.warning("spotify | enrich failed  id=%s  error=%s", spotify_id, exc)
        return None

    images = track_data.get("album", {}).get("images", [])
    album_art = images[0]["url"] if images else ""

    artists = track_data.get("artists", [])
    result = {
        "id":          spotify_id,
        "title":       track_data.get("name", ""),
        "artist":      ", ".join(a["name"] for a in artists),
        "artistId":    artists[0]["id"] if artists else "",
        "album":       track_data.get("album", {}).get("name", ""),
        "albumArtUrl": album_art,
        "previewUrl":  track_data.get("preview_url"),
        "durationMs":  track_data.get("duration_ms", 0),
        "bpm":         0,
        "key":         "—",
        "energy":      0.5,
        "valence":     0.5,
        "vibes":       [],
    }
    _search_cache[cache_key] = result
    return result


async def get_artist_genres(artist_id: str) -> list[str]:
    """Fetch genre tags for a Spotify artist. Returns empty list on failure."""
    if not artist_id:
        return []
    cache_key = f"artist_genres:{artist_id}"
    if cache_key in _search_cache:
        return _search_cache[cache_key]  # type: ignore[return-value]
    try:
        data = await _get(f"/artists/{artist_id}")
        genres: list[str] = data.get("genres", [])
        _search_cache[cache_key] = genres
        logger.debug("spotify | artist genres  id=%s  genres=%s", artist_id, genres)
        return genres
    except Exception as exc:
        logger.debug("spotify | artist genres failed  id=%s  err=%s", artist_id, exc)
        return []


async def fetch_preview_audio(preview_url: str) -> bytes | None:
    """Download the 30s Spotify preview MP3. Returns None on failure."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(preview_url)
            resp.raise_for_status()
            return resp.content
    except Exception:
        return None


async def search_track(query: str, limit: int = 5) -> list[dict]:
    """Search Spotify for tracks matching the query string."""
    cache_key = f"search:{query}:{limit}"
    if cache_key in _search_cache:
        logger.info("spotify | search cache hit  query=%r", query)
        return _search_cache[cache_key]  # type: ignore[return-value]

    logger.info("spotify | search  query=%r  limit=%d", query, limit)
    data = await _get("/search", params={"q": query, "type": "track", "limit": limit})
    items = data.get("tracks", {}).get("items", [])

    results = []
    for item in items:
        sid = item.get("id", "")
        enriched = await enrich_track(sid)
        if enriched:
            results.append(enriched)

    _search_cache[cache_key] = results
    return results


