"""
Last.fm API — used for artist tag lookup to detect cultural/language origin.
Free API key required: https://www.last.fm/api/account/create
Docs: https://www.last.fm/api/show/artist.getTopTags
"""
from __future__ import annotations

import logging

import httpx
from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"

_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)


async def get_artist_tags(artist_name: str) -> list[str]:
    """
    Return Last.fm top tags for an artist (e.g. ['punjabi pop', 'desi pop']).
    Returns empty list if no API key configured, artist not found, or on error.
    """
    if not artist_name or not settings.lastfm_api_key:
        return []

    cache_key = f"tags:{artist_name.lower()}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                LASTFM_BASE,
                params={
                    "method": "artist.getTopTags",
                    "artist": artist_name,
                    "api_key": settings.lastfm_api_key,
                    "format": "json",
                    "autocorrect": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
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
