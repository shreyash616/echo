"""
Last.fm API — artist tag lookup and track/artist similarity.
Free API key required: https://www.last.fm/api/account/create
"""
from __future__ import annotations

import logging

import httpx
from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)


async def _call(params: dict) -> dict:
    params = {**params, "api_key": settings.lastfm_api_key, "format": "json"}
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(LASTFM_BASE, params=params)
        resp.raise_for_status()
        return resp.json()


async def get_artist_tags(artist_name: str) -> list[str]:
    """Return Last.fm top tags for an artist (e.g. ['punjabi pop', 'desi pop'])."""
    if not artist_name or not settings.lastfm_api_key:
        return []

    cache_key = f"tags:{artist_name.lower()}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    try:
        data = await _call({"method": "artist.getTopTags", "artist": artist_name, "autocorrect": 1})
    except Exception as exc:
        logger.warning("lastfm | tags failed  artist=%r  err=%s", artist_name, exc)
        return []

    if "error" in data:
        logger.debug("lastfm | API error  artist=%r  msg=%s", artist_name, data.get("message"))
        return []

    tags: list[str] = [t["name"].lower() for t in data.get("toptags", {}).get("tag", [])]
    logger.debug("lastfm | tags  artist=%r  tags=%s", artist_name, tags)
    _cache[cache_key] = tags
    return tags


async def get_similar_tracks(artist: str, title: str, limit: int = 30) -> list[dict]:
    """
    Return Last.fm similar tracks as list of {artist, title} dicts.
    Falls back to similar artists' top tracks if track.getSimilar returns nothing.
    """
    if not settings.lastfm_api_key:
        return []

    cache_key = f"similar:{artist.lower()}:{title.lower()}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    results: list[dict] = []
    try:
        data = await _call({
            "method": "track.getSimilar",
            "artist": artist,
            "track": title,
            "limit": limit,
            "autocorrect": 1,
        })
        for t in data.get("similartracks", {}).get("track", []):
            a = t.get("artist", {})
            results.append({
                "title": t.get("name", ""),
                "artist": a.get("name", "") if isinstance(a, dict) else str(a),
            })
    except Exception as exc:
        logger.warning("lastfm | similar tracks failed  artist=%r  err=%s", artist, exc)

    # Fallback: similar artists → their top tracks
    if not results:
        try:
            data = await _call({
                "method": "artist.getSimilar",
                "artist": artist,
                "limit": 10,
                "autocorrect": 1,
            })
            similar_artists = [
                a.get("name", "") for a in data.get("similarartists", {}).get("artist", [])
            ]
            for sim_artist in similar_artists[:8]:
                try:
                    tdata = await _call({
                        "method": "artist.getTopTracks",
                        "artist": sim_artist,
                        "limit": 5,
                    })
                    for t in tdata.get("toptracks", {}).get("track", []):
                        results.append({"title": t.get("name", ""), "artist": sim_artist})
                        if len(results) >= limit:
                            break
                except Exception:
                    continue
                if len(results) >= limit:
                    break
        except Exception as exc:
            logger.warning("lastfm | similar artists failed  artist=%r  err=%s", artist, exc)

    logger.info("lastfm | similar  artist=%r  title=%r  found=%d", artist, title, len(results))
    _cache[cache_key] = results
    return results
