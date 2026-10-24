"""
Deezer public API — no authentication required.
Used for track search, metadata, genre lookup, related-artist candidate discovery,
and 30s preview audio downloads.
Docs: https://developers.deezer.com/api
"""
from __future__ import annotations

import asyncio
import logging

import httpx
from cachetools import TTLCache
import re

logger = logging.getLogger(__name__)

DEEZER_BASE = "https://api.deezer.com"

_cache: TTLCache = TTLCache(maxsize=1024, ttl=3600)


def _extract_year_decade(date_str: str | None) -> tuple[int | None, str | None]:
    """Parse release year (e.g. 1982) and derive release decade (e.g. '1980s')."""
    if not date_str:
        return None, None
    try:
        clean = str(date_str).strip()
        if len(clean) >= 4 and clean[:4].isdigit():
            y = int(clean[:4])
            if 1920 <= y <= 2035:
                return y, f"{(y // 10) * 10}s"
    except Exception:
        pass
    return None, None


def _normalize(item: dict, genre_override: str = "") -> dict:
    """Map a Deezer track object to the internal track dict format."""
    artist = item.get("artist", {})
    album = item.get("album", {})
    art = album.get("cover_xl") or album.get("cover_medium") or album.get("cover", "")
    raw_date = item.get("release_date") or album.get("release_date") or item.get("releaseDate")
    year, decade = _extract_year_decade(raw_date)
    return {
        "id":          str(item["id"]),
        "title":       item.get("title", ""),
        "artist":      artist.get("name", ""),
        "artistId":    str(artist.get("id", "")),
        "album":       album.get("title", ""),
        "albumId":     str(album.get("id", "")),
        "albumArtUrl": art,
        "previewUrl":  item.get("preview") or None,
        "durationMs":  item.get("duration", 0) * 1000,
        "genre":       genre_override,
        "bpm":         0,
        "key":         "—",
        "energy":      0.5,
        "valence":     0.5,
        "vibes":       [genre_override.lower()] if genre_override else [],
        "year":        year,
        "decade":      decade,
    }


async def search_track(query: str, limit: int = 5) -> list[dict]:
    """Search Deezer for tracks. Returns normalized track dicts."""
    cache_key = f"search:{query}:{limit}"
    if cache_key in _cache:
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
    if not results:
        # Fall back to iTunes search if Deezer returned empty results in this region
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                it_resp = await client.get(
                    "https://itunes.apple.com/search",
                    params={"term": query, "entity": "song", "limit": limit},
                )
                if it_resp.status_code == 200:
                    for item in it_resp.json().get("results", []):
                        art = (item.get("artworkUrl100") or "").replace("100x100bb", "600x600bb")
                        y, dec = _extract_year_decade(item.get("releaseDate"))
                        results.append({
                            "id":          f"it_{item['trackId']}",
                            "title":       item.get("trackName", ""),
                            "artist":      item.get("artistName", ""),
                            "artistId":    str(item.get("artistId", "")),
                            "album":       item.get("collectionName", ""),
                            "albumId":     str(item.get("collectionId", "")),
                            "albumArtUrl": art,
                            "previewUrl":  item.get("previewUrl"),
                            "durationMs":  int(item.get("trackTimeMillis", 30000)),
                            "genre":       item.get("primaryGenreName", ""),
                            "bpm":         0,
                            "key":         "—",
                            "energy":      0.5,
                            "valence":     0.5,
                            "vibes":       [item.get("primaryGenreName", "").lower()] if item.get("primaryGenreName") else [],
                            "year":        y,
                            "decade":      dec,
                        })
        except Exception as exc:
            logger.debug("itunes search fallback failed: %s", exc)

    _cache[cache_key] = results
    return results


async def get_track(deezer_id: str) -> dict | None:
    """Fetch a single track by Deezer ID or iTunes ID, including album genre and release year."""
    cache_key = f"track:{deezer_id}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    if str(deezer_id).startswith("it_"):
        it_id = str(deezer_id)[3:]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                it_resp = await client.get(
                    "https://itunes.apple.com/lookup",
                    params={"id": it_id, "entity": "song"},
                )
                if it_resp.status_code == 200:
                    results = it_resp.json().get("results", [])
                    if results:
                        item = results[0]
                        art = (item.get("artworkUrl100") or "").replace("100x100bb", "600x600bb")
                        y, dec = _extract_year_decade(item.get("releaseDate"))
                        res = {
                            "id":          deezer_id,
                            "title":       item.get("trackName", ""),
                            "artist":      item.get("artistName", ""),
                            "artistId":    str(item.get("artistId", "")),
                            "album":       item.get("collectionName", ""),
                            "albumId":     str(item.get("collectionId", "")),
                            "albumArtUrl": art,
                            "previewUrl":  item.get("previewUrl"),
                            "durationMs":  int(item.get("trackTimeMillis", 30000)),
                            "genre":       item.get("primaryGenreName", ""),
                            "bpm":         0,
                            "key":         "—",
                            "energy":      0.5,
                            "valence":     0.5,
                            "vibes":       [item.get("primaryGenreName", "").lower()] if item.get("primaryGenreName") else [],
                            "year":        y,
                            "decade":      dec,
                        }
                        _cache[cache_key] = res
                        return res
        except Exception as exc:
            logger.debug("itunes lookup failed: %s", exc)
        return None
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
    if not result.get("year"):
        raw_date = item.get("release_date") or item.get("album", {}).get("release_date")
        y, dec = _extract_year_decade(raw_date)
        if y:
            result["year"] = y
            result["decade"] = dec
    album_id = result.get("albumId")
    if album_id:
        genres = await get_album_genres(album_id)
        if genres:
            result["genre"] = genres[0]
            result["genres"] = genres

    _cache[cache_key] = result
    return result


async def get_album_genres(album_id: str) -> list[str]:
    """
    Fetch genre names from Deezer's /album/{id} endpoint (e.g. ['Pop', 'Indie Pop']).
    Free, requires no API key.
    """
    if not album_id:
        return []
    cache_key = f"album_genres:{album_id}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{DEEZER_BASE}/album/{album_id}")
            resp.raise_for_status()
            data = resp.json()
        genres_data = data.get("genres", {}).get("data", [])
        genres = [g.get("name", "").strip() for g in genres_data if g.get("name")]
        _cache[cache_key] = genres
        return genres
    except Exception as exc:
        logger.debug("deezer | get_album_genres failed  album_id=%s  err=%s", album_id, exc)
        return []


async def get_related_artists(artist_id: str, limit: int = 12) -> list[dict]:
    """
    Fetch artists in the same genre and cultural/language scene via Deezer's
    /artist/{id}/related endpoint.
    """
    if not artist_id:
        return []
    cache_key = f"related_artists:{artist_id}:{limit}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{DEEZER_BASE}/artist/{artist_id}/related",
                params={"limit": limit},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
        artists = [
            {"id": str(a.get("id", "")), "name": a.get("name", "")}
            for a in data
            if a.get("id") and a.get("name")
        ]
        _cache[cache_key] = artists
        return artists
    except Exception as exc:
        logger.debug("deezer | get_related_artists failed  artist_id=%s  err=%s", artist_id, exc)
        return []


async def get_artist_top_tracks(artist_id: str, limit: int = 5, genre_hint: str = "") -> list[dict]:
    """Fetch top tracks for a Deezer artist ID (all include 30s preview URLs)."""
    if not artist_id:
        return []
    cache_key = f"artist_top:{artist_id}:{limit}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{DEEZER_BASE}/artist/{artist_id}/top",
                params={"limit": limit},
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
        tracks = [_normalize(t, genre_override=genre_hint) for t in data if t.get("preview")]
        _cache[cache_key] = tracks
        return tracks
    except Exception as exc:
        logger.debug("deezer | get_artist_top_tracks failed  artist_id=%s  err=%s", artist_id, exc)
        return []


async def get_genre_and_language_candidates(
    track_info: dict,
    genre_hint: str = "",
    lang_hint: str = "",
    limit: int = 35,
) -> list[dict]:
    """
    Build a rich pool of candidate tracks that inherently share the source
    track's genre, language, era, and musical scene using Deezer's artist graph,
    multi-artist decomposition, and targeted scene searches:
      1. Tracks by the source artist(s) & collaborating artists
      2. Top tracks from Deezer's `/artist/{id}/related` artists
      3. Targeted era, language, and genre queries
    """
    artist_id = str(track_info.get("artistId", ""))
    artist_name = str(track_info.get("artist", ""))
    source_id = str(track_info.get("id", ""))
    source_year = track_info.get("year")
    source_decade = track_info.get("decade")

    # Split multi-artist string (e.g. "Asha Bhosle & Kishore Kumar" -> ["Asha Bhosle", "Kishore Kumar"])
    artist_parts = [
        p.strip()
        for p in re.split(r"\s*(?:,|&|\bfeat\.?|\bft\.?|\bx\b|/)\s*", artist_name, flags=re.IGNORECASE)
        if p.strip()
    ]

    # If artistId is missing, resolve it via a quick search
    if not artist_id and artist_name:
        found = await search_track(f"{track_info.get('title', '')} {artist_name}", limit=1)
        if found:
            artist_id = str(found[0].get("artistId", ""))

    candidates: list[dict] = []
    seen_ids: set[str] = {source_id} if source_id else set()
    seen_keys: set[str] = {
        f"{artist_name.lower().strip()}:{str(track_info.get('title', '')).lower().strip()}"
    }

    def _add_candidate(t: dict, trusted: bool = False) -> None:
        tid = str(t.get("id", ""))
        key = f"{str(t.get('artist', '')).lower().strip()}:{str(t.get('title', '')).lower().strip()}"
        if not tid or tid in seen_ids or key in seen_keys or not t.get("previewUrl"):
            return
        seen_ids.add(tid)
        seen_keys.add(key)
        if genre_hint and not t.get("genre"):
            t = {**t, "genre": genre_hint}
        if trusted:
            t["_trusted_scene"] = True
        candidates.append(t)

    # 1. Deezer artist graph (for Deezer artist IDs)
    if artist_id and not artist_id.startswith("it_"):
        own_top_task = get_artist_top_tracks(artist_id, limit=3, genre_hint=genre_hint)
        related_task = get_related_artists(artist_id, limit=12)
        own_top, related_artists = await asyncio.gather(own_top_task, related_task)

        # Include at most 2 other tracks from the same artist
        for t in own_top[:2]:
            _add_candidate(t, trusted=True)

        # Fetch top 3 tracks from each related artist concurrently
        if related_artists:
            rel_lists = await asyncio.gather(
                *[
                    get_artist_top_tracks(ra["id"], limit=3, genre_hint=genre_hint)
                    for ra in related_artists[:12]
                ]
            )
            for track_list in rel_lists:
                for t in track_list:
                    _add_candidate(t, trusted=True)
                    if len(candidates) >= limit:
                        break
                if len(candidates) >= limit:
                    break

    # 2. Multi-artist and scene-targeted search (especially when iTunes or Deezer graph had few items)
    if len(candidates) < limit:
        search_tasks = []
        # Query for each named artist individually
        for part in artist_parts[:3]:
            search_tasks.append(search_track(part, limit=5))
            if source_decade:
                search_tasks.append(search_track(f"{part} {source_decade}", limit=4))

        # Query for primary artist + genre
        if artist_parts and genre_hint:
            search_tasks.append(search_track(f"{artist_parts[0]} {genre_hint}", limit=5))

        # Query for language + genre
        query_parts = [p for p in [lang_hint, genre_hint] if p]
        if query_parts:
            search_tasks.append(search_track(" ".join(query_parts), limit=10))

        extra_lists = await asyncio.gather(*search_tasks)
        for i, extra in enumerate(extra_lists):
            is_artist_specific = (i < len(artist_parts) * 2)
            for t in extra:
                _add_candidate(t, trusted=is_artist_specific)
                if len(candidates) >= limit:
                    break
            if len(candidates) >= limit:
                break

    return candidates[:limit]


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