"""
Deezer public API — used as a fallback audio preview source when a track
has no Spotify 30s preview (common with major label releases).

No authentication required for basic search + preview URLs.
Docs: https://developers.deezer.com/api/search
"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

DEEZER_SEARCH_URL = "https://api.deezer.com/search"


async def fetch_deezer_preview(title: str, artist: str) -> str | None:
    """
    Search Deezer for a track by title + artist and return its 30s MP3 preview URL.
    Returns None if not found or on any error.
    """
    query = f'artist:"{artist}" track:"{title}"'
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(DEEZER_SEARCH_URL, params={"q": query, "limit": 1})
            resp.raise_for_status()
            data = resp.json()

        tracks = data.get("data", [])
        if tracks and tracks[0].get("preview"):
            logger.info("deezer | preview found  title=%r  artist=%r", title, artist)
            return tracks[0]["preview"]

        logger.info("deezer | no preview  title=%r  artist=%r", title, artist)
    except Exception as exc:
        logger.warning("deezer | search failed  title=%r  error=%s", title, exc)

    return None
