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

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=10.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return _client


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


def _normalize(item: dict) -> dict:
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
        "genre":       "",
        "bpm":         0,
        "key":         "—",
        "energy":      0.5,
        "valence":     0.5,
        "vibes":       [],
        "year":        year,
        "decade":      decade,
        "rank":        int(item.get("rank", 0)),
    }


REMIX_PENALTY_TERMS = [
    "remix", "rmx", "re-mix", "club mix", "extended mix", "dub mix", "vip mix",
    "cover", "tribute", "karaoke", "instrumental version", "acoustic cover",
    "originally performed by", "made famous by", "in the style of", "performs",
    "slowed", "reverb", "sped up", "speed up", "nightcore", "chopped",
    "live at", "live from", "bootleg", "parody", "lullaby", "music box", "backing track"
]

COVER_FARM_ARTISTS = [
    "music box mania", "the karaoke channel", "tribute band", "tribute players",
    "workout music", "lullaby players", "piano dreamers", "rockabye baby",
    "midnite string quartet", "smooth jazz all stars", "cover pop", "karaoke tracks",
    "the hit makers", "acoustic chillout sessions", "tribute allstars"
]

BRACKET_TAG_TERMS = [
    "remix", "rmx", "live", "edit", "version", "deluxe", "bonus", "karaoke",
    "tribute", "cover", "slowed", "reverb", "sped up", "acoustic", "instrumental"
]


def score_track_originality(query: str, item: dict, target_artist: str = "") -> float:
    """Score track originality to prioritize original canonical tracks over remixes, covers, and karaoke."""
    score = 100.0
    title = str(item.get("title", "")).lower()
    artist = str(item.get("artist", "")).lower()
    album = str(item.get("album", "")).lower()
    q_lower = query.lower().strip()
    clean_q = re.sub(r"[^a-z0-9 ]", "", q_lower).strip()
    clean_t = re.sub(r"[^a-z0-9 ]", "", title).strip()
    clean_a = re.sub(r"[^a-z0-9 ]", "", artist).strip()

    # 1. Exact / Prefix / Substring Title Match
    if clean_t == clean_q:
        score += 80.0
    elif clean_t.startswith(clean_q):
        score += 35.0
    elif clean_q in clean_t:
        score += 15.0
    else:
        q_tokens = set(clean_q.split())
        t_tokens = set(clean_t.split())
        overlap = len(q_tokens & t_tokens)
        score += overlap * 5.0
        score -= 20.0

    # 2. Artist matching & cover detection
    effective_target = target_artist.lower().strip()
    if not effective_target and clean_a and clean_a in clean_q:
        effective_target = clean_a

    if effective_target:
        clean_target = re.sub(r"[^a-z0-9 ]", "", effective_target).strip()
        target_tokens = set(clean_target.split())
        # Large bonus (+60) when candidate artist matches query artist
        if clean_target in clean_a or clean_a in clean_target:
            score += 60.0
        # Cover demotion (-160): query artist tokens appear in candidate TITLE but not in candidate ARTIST
        if (clean_target in clean_t or (len(target_tokens) >= 2 and target_tokens.issubset(set(clean_t.split())))) and not (clean_target in clean_a or clean_a in clean_target):
            score -= 160.0

    # Also detect covers where bracket contents name an original artist not matching candidate artist
    bracket_matches = re.findall(r"[\(\[](.*?)[\)\]]", title)
    for bracket_content in bracket_matches:
        b_clean = re.sub(r"[^a-z0-9 ]", "", bracket_content.lower()).strip()
        if b_clean and b_clean in clean_q and b_clean not in clean_a and len(b_clean.split()) >= 2:
            score -= 160.0
            break

    # 3. Derivative / Remix / Cover terms
    for term in REMIX_PENALTY_TERMS:
        if term in title or term in album:
            if term in q_lower:
                score += 40.0  # Explicit user intent to search for remix/live
            else:
                score -= 120.0
            break

    # 4. Cover Farm Artists
    for farm in COVER_FARM_ARTISTS:
        if farm in artist or farm in album:
            score -= 160.0
            break

    # 5. Context-aware brackets check
    for bracket_content in bracket_matches:
        b_lower = bracket_content.lower().strip()
        if any(tag in b_lower for tag in BRACKET_TAG_TERMS):
            if not any(tag in q_lower for tag in BRACKET_TAG_TERMS):
                score -= 30.0

    # 6. Popularity / Rank bonus
    rank = item.get("rank", 0)
    if rank and rank > 0:
        score += min(25.0, float(rank) / 35000.0)

    # 7. Native Deezer source preference
    if not str(item.get("id", "")).startswith("it_"):
        score += 20.0

    return score


async def search_track(query: str, limit: int = 5) -> list[dict]:
    """Search Deezer and iTunes for tracks, prioritizing canonical originals over remixes and covers."""
    cache_key = f"search:{query}:{limit}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    logger.info("deezer | search  query=%r  limit=%d", query, limit)
    raw_candidates: list[dict] = []
    pool_limit = max(limit * 5, 25)

    client = _get_client()

    # Search Deezer and iTunes concurrently
    async def _fetch_deezer() -> list[dict]:
        try:
            resp = await client.get(
                f"{DEEZER_BASE}/search",
                params={"q": query, "limit": pool_limit},
            )
            if resp.status_code == 200:
                return [_normalize(t) for t in resp.json().get("data", [])]
        except Exception as exc:
            logger.debug("deezer search failed: %s", exc)
        return []

    async def _fetch_itunes() -> list[dict]:
        try:
            it_resp = await client.get(
                "https://itunes.apple.com/search",
                params={"term": query, "entity": "song", "limit": pool_limit},
            )
            if it_resp.status_code == 200:
                items = []
                for pos, item in enumerate(it_resp.json().get("results", [])):
                    art = (item.get("artworkUrl100") or "").replace("100x100bb", "600x600bb")
                    y, dec = _extract_year_decade(item.get("releaseDate"))
                    it_rank = max(10000, 850000 - pos * 30000)
                    items.append({
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
                        "rank":        it_rank,
                    })
                return items
        except Exception as exc:
            logger.debug("itunes search failed: %s", exc)
        return []

    deezer_results, itunes_results = await asyncio.gather(_fetch_deezer(), _fetch_itunes())
    raw_candidates.extend(deezer_results)
    raw_candidates.extend(itunes_results)

    # Deduplicate by clean (title, artist)
    seen = set()
    deduped = []
    for item in raw_candidates:
        c_title = re.sub(r"[^a-z0-9]", "", str(item.get("title", "")).lower())
        c_artist = re.sub(r"[^a-z0-9]", "", str(item.get("artist", "")).lower())
        key = (c_title, c_artist)
        if key not in seen and item.get("previewUrl"):
            seen.add(key)
            deduped.append(item)

    # Detect target artist from query and candidates
    clean_q = re.sub(r"[^a-z0-9 ]", "", query.lower()).strip()
    cand_artists = [
        re.sub(r"[^a-z0-9 ]", "", str(item.get("artist", "")).lower()).strip()
        for item in raw_candidates
        if len(str(item.get("artist", "")).strip()) >= 3
    ]
    matching_artists = [a for a in cand_artists if a in clean_q]
    target_artist = max(matching_artists, key=len) if matching_artists else ""

    # Sort descending by originality score
    deduped.sort(key=lambda x: score_track_originality(query, x, target_artist=target_artist), reverse=True)
    results = deduped[:limit]

    _cache[cache_key] = results
    return results


async def get_track(deezer_id: str) -> dict | None:
    """Fetch a single track by Deezer ID or iTunes ID, including album genre and release year."""
    cache_key = f"track:{deezer_id}"
    if cache_key in _cache:
        return _cache[cache_key]  # type: ignore[return-value]

    client = _get_client()

    if str(deezer_id).startswith("it_"):
        it_id = str(deezer_id)[3:]
        try:
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

    try:
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

    if not result.get("previewUrl"):
        try:
            fb = await search_track(f"{result.get('title', '')} {result.get('artist', '')}", limit=1)
            if fb and fb[0].get("previewUrl"):
                result["previewUrl"] = fb[0]["previewUrl"]
        except Exception:
            pass

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
        client = _get_client()
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
        client = _get_client()
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
        client = _get_client()
        resp = await client.get(
            f"{DEEZER_BASE}/artist/{artist_id}/top",
            params={"limit": limit},
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        tracks = [_normalize(t) for t in data if t.get("preview")]
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
    Build a candidate track pool from Deezer's artist graph, multi-artist decomposition,
    and targeted scene searches. Note: candidates carry their own Deezer genre or nothing;
    genre and trusted_scene are no longer stamped on candidates.
    """
    artist_id = str(track_info.get("artistId", ""))
    artist_name = str(track_info.get("artist", ""))
    source_id = str(track_info.get("id", ""))
    source_year = track_info.get("year")
    source_decade = track_info.get("decade")

    # Split multi-artist string
    artist_parts = [
        p.strip()
        for p in re.split(r"\s*(?:,|&|\bfeat\.?|\bft\.?|\bx\b|/)\s*", artist_name, flags=re.IGNORECASE)
        if p.strip()
    ]

    # Resolve real Deezer artist ID by artist name if missing or from iTunes
    if (not artist_id or str(source_id).startswith("it_") or str(artist_id).startswith("it_")) and artist_name:
        try:
            client = _get_client()
            resp = await client.get(f"{DEEZER_BASE}/search/artist", params={"q": artist_name, "limit": 1})
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    artist_id = str(data[0].get("id", ""))
        except Exception:
            pass

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
        # Vision alignment: candidates carry their own genre, do NOT stamp source genre or _trusted_scene
        candidates.append(t)

    # 1. Deezer artist graph (for Deezer artist IDs)
    if artist_id and not artist_id.startswith("it_"):
        own_top_task = get_artist_top_tracks(artist_id, limit=3, genre_hint=genre_hint)
        related_task = get_related_artists(artist_id, limit=12)
        own_top, related_artists = await asyncio.gather(own_top_task, related_task)

        # Include at most 2 other tracks from the same artist
        for t in own_top[:2]:
            _add_candidate(t, trusted=True)

        # Fetch top 3 tracks from each related artist concurrently (with search fallback)
        if related_artists:
            async def _fetch_rel(ra: dict) -> list[dict]:
                tracks = await get_artist_top_tracks(ra["id"], limit=3, genre_hint=genre_hint)
                if not tracks and ra.get("name"):
                    tracks = await search_track(ra["name"], limit=2)
                return tracks

            rel_lists = await asyncio.gather(*[_fetch_rel(ra) for ra in related_artists[:12]])
            for track_list in rel_lists:
                for t in track_list:
                    _add_candidate(t, trusted=True)
                    if len(candidates) >= limit:
                        break
                if len(candidates) >= limit:
                    break

    # 2. Multi-artist and scene-targeted search
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

        # Vision alignment Task 1.3: Dropped bare "{lang} {genre}" and bare "{genre}" queries

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
        client = _get_client()
        resp = await client.get(url, headers=_PREVIEW_HEADERS)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        logger.warning("deezer | preview download failed  url=%s  err=%s", url, exc)
        return None