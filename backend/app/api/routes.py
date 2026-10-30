from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
import numpy as np

from fastapi import APIRouter, UploadFile, File, Header, HTTPException, Query

from app.config import settings
from app.models.schemas import IdentifyResponse, SearchResponse, TrackResult
from app.services.audio_recognition import recognizer
from app.services.deezer import (
    search_track,
    get_track,
    get_album_genres,
    fetch_preview_audio,
    _extract_year_decade,
)
from app.services.recommender import (
    recommender,
    to_camelot_key,
    make_track_key,
)
from app.audio.features import language_from_meta
from app.services.language_affinity import language_affinity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Concurrency limiter for preview downloads
_ENCODE_SEM = asyncio.Semaphore(8)


import httpx

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=6.0, limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))
    return _http_client


def _normalize_title_for_key(title: str) -> str:
    """Normalize track title for deduplication (strip parens, feat, brackets, collapse space)."""
    t = re.sub(r"\(.*?\)", "", title)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.split(r"\s+-\s+|\s+feat\.|\s+ft\.|\s+with\s+", t, flags=re.IGNORECASE)[0]
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _to_track(meta: dict) -> TrackResult:
    raw_k = meta.get("key")
    if raw_k and raw_k != "—" and ":" in str(raw_k):
        track_key = str(raw_k)
    else:
        track_key = make_track_key(str(meta.get("artist", "")), str(meta.get("title", "")))
    return TrackResult(
        id=str(meta.get("id", "")),
        title=str(meta.get("title", "")),
        artist=str(meta.get("artist", "")),
        album=str(meta.get("album", "")),
        albumArtUrl=str(meta.get("albumArtUrl", "")),
        previewUrl=meta.get("previewUrl"),
        durationMs=int(meta.get("durationMs", 0)),
        matchScore=meta.get("matchScore"),
        vibes=list(meta.get("vibes", [])),
        theme=str(meta["theme"]) if meta.get("theme") else None,
        felt_bpm=int(meta["felt_bpm"]) if meta.get("felt_bpm") else None,
        key=str(track_key or "—"),
        brightness=float(meta["brightness"]) if meta.get("brightness") is not None else None,
        energy=float(meta["energy"]) if meta.get("energy") is not None else None,
        languageAffinity=float(meta["languageAffinity"]) if meta.get("languageAffinity") is not None else None,
        year=int(meta["year"]) if meta.get("year") else None,
        decade=str(meta["decade"]) if meta.get("decade") else None,
        matchReasons=list(meta.get("matchReasons", [])),
        darkness=float(meta["darkness"]) if meta.get("darkness") is not None else None,
        grit=float(meta["grit"]) if meta.get("grit") is not None else None,
        pulse_rate=float(meta["pulse_rate"]) if meta.get("pulse_rate") is not None else None,
        kick_class=str(meta["kick_class"]) if meta.get("kick_class") is not None else None,
        restraint_bucket=str(meta["restraint_bucket"]) if meta.get("restraint_bucket") is not None else None,
        intimacy=float(meta["intimacy"]) if meta.get("intimacy") is not None else None,
        crest_factor=float(meta["crest_factor"]) if meta.get("crest_factor") is not None else None,
        mode_polarity=float(meta["mode_polarity"]) if meta.get("mode_polarity") is not None else None,
        mode_undefined=bool(meta["mode_undefined"]) if meta.get("mode_undefined") is not None else None,
        changes_per_bar=float(meta["changes_per_bar"]) if meta.get("changes_per_bar") is not None else None,
        riff_flag=bool(meta["riff_flag"]) if meta.get("riff_flag") is not None else None,
        language=str(meta["language"]) if meta.get("language") is not None else None,
        language_conf=float(meta["language_conf"]) if meta.get("language_conf") is not None else None,
        language_source=str(meta["language_source"]) if meta.get("language_source") is not None else None,
    )


def _primary_artist_name(artist: str) -> str:
    """Extract the primary artist name from multi-artist strings ('A, B & C' -> 'A')."""
    if not artist:
        return ""
    parts = re.split(r"\s*(?:,|&|\bfeat\.?|\bft\.?|\bx\b|/)\s*", artist, flags=re.IGNORECASE)
    return parts[0].strip() if parts else artist.strip()


# Maps language codes to human-readable search hints
_LANG_SEARCH_HINT: dict[str, str] = {
    "hi": "hindi", "pa": "punjabi", "ta": "tamil", "te": "telugu",
    "bn": "bengali", "kn": "kannada", "ml": "malayalam", "mr": "marathi",
    "ko": "korean", "ja": "japanese", "zh-cn": "chinese",
    "ar": "arabic", "tr": "turkish", "ru": "russian", "th": "thai",
    "es": "spanish", "fr": "french", "pt": "portuguese", "de": "german",
    "it": "italian",
}

# Genre / tag fragments -> language code (ordered most-specific to least-specific)
_GENRE_LANG_RULES: list[tuple[str, str]] = [
    # South Asian
    ("punjabi", "pa"), ("bhangra", "pa"),
    ("hindi", "hi"), ("bollywood", "hi"), ("filmi", "hi"), ("desi", "hi"),
    ("indian pop", "hi"), ("indipop", "hi"), ("sufi", "hi"),
    ("tamil", "ta"), ("kollywood", "ta"),
    ("telugu", "te"), ("tollywood", "te"),
    ("bengali", "bn"), ("kannada", "kn"), ("malayalam", "ml"), ("marathi", "mr"),
    # East Asian
    ("k-pop", "ko"), ("kpop", "ko"), ("k-indie", "ko"), ("k-r&b", "ko"), ("k-rap", "ko"), ("korean", "ko"),
    ("j-pop", "ja"), ("jpop", "ja"), ("j-rock", "ja"), ("j-rap", "ja"), ("anime", "ja"), ("japanese", "ja"),
    ("mandopop", "zh-cn"), ("cantopop", "zh-cn"), ("c-pop", "zh-cn"), ("chinese", "zh-cn"),
    # Middle East / Europe / Latin
    ("arabic", "ar"), ("khaleeji", "ar"), ("turkish", "tr"),
    ("thai", "th"), ("russian", "ru"),
    ("reggaeton", "es"), ("latin", "es"), ("urbano latino", "es"), ("bachata", "es"), ("salsa", "es"),
    ("french", "fr"), ("chanson", "fr"),
    ("german", "de"), ("deutschrap", "de"),
    ("italian", "it"), ("sertanejo", "pt"), ("funk carioca", "pt"), ("bossa nova", "pt"),
]

# Canonical genre families -> matching tag substrings
_GENRE_FAMILIES: dict[str, list[str]] = {
    "hip-hop":    ["hip hop", "hip-hop", "rap", "trap", "drill", "boom bap", "grime", "phonk"],
    "electronic": ["electronic", "electro", "edm", "house", "techno", "trance", "dubstep", "drum and bass", "dnb", "synthwave", "dance", "club"],
    "rock":       ["rock", "alternative", "punk", "grunge", "shoegaze", "emo", "new wave", "britpop", "garage"],
    "metal":      ["metal", "hardcore", "metalcore", "deathcore", "thrash", "doom"],
    "r&b":        ["r&b", "rnb", "soul", "neo-soul", "neo soul", "funk", "motown", "contemporary r&b"],
    "indie":      ["indie", "lo-fi", "lofi", "bedroom pop", "dream pop", "chillwave"],
    "acoustic":   ["acoustic", "folk", "singer-songwriter", "country", "americana", "unplugged"],
    "jazz":       ["jazz", "blues", "bebop", "swing", "bossa"],
    "classical":  ["classical", "orchestra", "piano", "symphony", "instrumental", "soundtrack", "score", "ambient"],
    "pop":        ["pop", "mainstream", "teen pop", "electropop", "synthpop", "disco"],
}


def _detect_language_from_tags(tags: list[str]) -> str | None:
    """Map genre/artist/album tags to a language code, or None for English/unspecified."""
    joined = " ".join(tags).lower()
    for keyword, lang in _GENRE_LANG_RULES:
        if keyword in joined:
            return lang
    return None


def _detect_script_language(title: str, artist: str) -> str | None:
    """
    Detect language from non-Latin Unicode scripts ONLY.
    Does NOT run statistical `langdetect` on short Latin-script song titles,
    avoiding false positives where English titles like 'Levitating' get tagged as Italian.
    """
    text = f"{title} {artist}".strip()
    if not text:
        return None

    _SCRIPT_LANGS = {
        "DEVANAGARI": "hi", "GURMUKHI": "pa", "TAMIL": "ta",
        "TELUGU": "te", "BENGALI": "bn", "KANNADA": "kn", "MALAYALAM": "ml",
        "HANGUL": "ko", "HIRAGANA": "ja", "KATAKANA": "ja", "CJK UNIFIED": "zh-cn",
        "ARABIC": "ar", "CYRILLIC": "ru", "THAI": "th",
    }
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        for script, lang in _SCRIPT_LANGS.items():
            if script in name:
                return lang
    return None


def _detect_genre_family(tags: list[str]) -> tuple[str, list[str]]:
    """
    Map raw genre/artist tags to a canonical genre family name and its matching keywords.
    Returns (genre_family, family_keywords).
    """
    joined = " ".join(tags).lower()
    for family, keywords in _GENRE_FAMILIES.items():
        if any(kw in joined for kw in keywords):
            return family, keywords
    if tags:
        clean = tags[0].lower().strip()
        return clean, [clean]
    return "", []


async def _detect_track_context(track_info: dict) -> tuple[str | None, str, list[str]]:
    """
    Determine the source track's (language_code, genre_family, genre_keywords)
    using Deezer album genres, Last.fm artist tags, and Unicode script detection.
    """
    raw_tags: list[str] = []

    if track_info.get("genre"):
        raw_tags.append(str(track_info["genre"]))
    if track_info.get("genres"):
        raw_tags.extend(str(g) for g in track_info["genres"])

    album_id = str(track_info.get("albumId", ""))
    artist_name = _primary_artist_name(str(track_info.get("artist", "")))

    if album_id and not track_info.get("genres"):
        album_genres = await get_album_genres(album_id)
        raw_tags.extend(album_genres)

    # 1. Language detection (tags first, then non-Latin script fallback)
    lang = _detect_language_from_tags(raw_tags) or _detect_script_language(
        str(track_info.get("title", "")),
        str(track_info.get("artist", "")),
    )

    # 2. Genre family detection
    genre_family, genre_keywords = _detect_genre_family(raw_tags)

    # 3. Script-based and genre-based language detection (Section 11.5.8)
    if not lang:
        meta_dict = {"title": track_info.get("title", ""), "artist": artist_name, "genre": genre_family}
        l_code, _, _ = language_from_meta(meta_dict)
        if l_code:
            lang = l_code

    # 4. iTunes lookup fallback to disambiguate generic Deezer tags (e.g. 'Films/Games') and resolve true release era
    needs_itunes_genre = not lang or genre_family in ["classical", "general", "films/games", "soundtrack", ""]
    if needs_itunes_genre or not track_info.get("year"):
        try:
            client = _get_http_client()
            query = f"{artist_name} {track_info.get('title', '')}".strip()
            it_resp = await client.get(
                "https://itunes.apple.com/search",
                params={"term": query, "entity": "song", "limit": 1},
            )
            if it_resp.status_code == 200:
                    it_items = it_resp.json().get("results", [])
                    if it_items:
                        it_genre = str(it_items[0].get("primaryGenreName", "")).lower()
                        it_lang = _detect_language_from_tags([it_genre])
                        if it_lang and not lang:
                            lang = it_lang
                        if not genre_family or genre_family in ["classical", "general"]:
                            it_fam, it_kw = _detect_genre_family([it_genre])
                            if it_fam:
                                genre_family, genre_keywords = it_fam, it_kw
                        # Resolve original release year if missing or digital re-release date
                        it_date = it_items[0].get("releaseDate")
                        if it_date:
                            y, dec = _extract_year_decade(it_date)
                            if y and (not track_info.get("year") or y < track_info.get("year", 9999)):
                                track_info["year"] = y
                                track_info["decade"] = dec
        except Exception as exc:
            logger.debug("itunes context resolution failed: %s", exc)

    if not genre_family and lang and lang in _LANG_SEARCH_HINT:
        genre_family = _LANG_SEARCH_HINT[lang]
        genre_keywords = [genre_family]

    logger.info(
        "track_context | title=%r  artist=%r  lang=%s  genre=%r  year=%s",
        track_info.get("title"), track_info.get("artist"), lang or "en", genre_family or "general", track_info.get("year"),
    )
    return lang, genre_family, genre_keywords


async def _enrich_single_track_with_own_metrics(
    track_meta: dict,
    genre_hint: str = "",
) -> tuple[TrackResult, np.ndarray | None, dict]:
    """
    Download a track's 30s preview (if available), compute Section 11.5 descriptors + CLAP,
    and return (enriched_TrackResult, clap_vector, desc_dict).
    """
    preview_url = track_meta.get("previewUrl")
    if not preview_url:
        return _to_track(track_meta), None, {}

    preview_bytes = await fetch_preview_audio(preview_url)
    if not preview_bytes:
        return _to_track(track_meta), None, {}

    desc = await asyncio.to_thread(
        recommender.describe,
        preview_bytes,
        meta=track_meta,
    )
    enriched = {
        **track_meta,
        **desc,
        "camelotKey": to_camelot_key(str(desc.get("key", "—"))),
    }
    clap_vec = np.array(desc.get("clap_vector") or [0.0] * 512, dtype=np.float32)
    return _to_track(enriched), clap_vec, desc


async def _find_and_rank_similar_tracks(
    source_track_info: dict,
    source_embedding: np.ndarray | None = None,
    source_metrics: dict | None = None,
    limit: int = 10,
    diversity_lambda: float = 0.70,
    cross_language_max: int = 6,
) -> list[TrackResult]:
    """
    Step 11.7 unified recommendation pipeline:
    Calls recommender.recommend() with MMR and supply-scaled cross-language budget.
    """
    seed = dict(source_track_info)
    if source_metrics:
        seed.update(source_metrics)
    recs = await asyncio.to_thread(
        recommender.recommend,
        seed=seed,
        k=limit,
        cross_language_max=cross_language_max,
        diversity=diversity_lambda,
    )
    return [_to_track(r) for r in recs]


@router.get("/config")
async def get_client_config() -> dict:
    return {
        "spotify_client_id": settings.spotify_client_id,
        "spotify_redirect_uri": settings.spotify_redirect_uri,
    }


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "recommender_loaded": recommender._loaded}


# ---- Admin: artist discovery as a background task (SCALE_PLAN WP3) ----------------------------------
# Both routes exist only when the Space secret ECHO_ADMIN_TOKEN is set (404 otherwise) and require it in
# X-Admin-Token, compared in constant time. Status reports progress and whether HF_TOKEN is present and
# its role, never a value.
_DISCOVERY: dict = {"thread": None, "progress": {}, "error": None, "started_at": None, "finished_at": None}
_DISCOVERY_LOCK = threading.Lock()


def _admin_guard(token: str | None) -> None:
    secret = os.environ.get("ECHO_ADMIN_TOKEN")
    if not secret:
        raise HTTPException(status_code=404, detail="Not Found")
    if not token or not hmac.compare_digest(token.encode("utf-8"), secret.encode("utf-8")):
        raise HTTPException(status_code=403, detail="forbidden")


def _hf_token_status() -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"hf_token_present": False, "hf_token_role": None}
    try:
        from huggingface_hub import HfApi
        role = (HfApi(token=token).whoami().get("auth", {}).get("accessToken", {}) or {}).get("role")
    except Exception as exc:
        role = f"unknown (whoami failed: {type(exc).__name__})"
    return {"hf_token_present": True, "hf_token_role": role}


@router.post("/admin/discovery/start")
async def admin_discovery_start(x_admin_token: str | None = Header(None, alias="X-Admin-Token")) -> dict:
    _admin_guard(x_admin_token)
    if not os.environ.get("HF_TOKEN"):
        raise HTTPException(status_code=412, detail="HF_TOKEN is not set; discovery could not checkpoint to the dataset")
    with _DISCOVERY_LOCK:
        t = _DISCOVERY["thread"]
        if t is not None and t.is_alive():
            raise HTTPException(status_code=409, detail="discovery is already running")
        from app.services import discovery

        progress: dict = {}
        work = Path(os.environ.get("ECHO_DISCOVERY_DIR") or tempfile.gettempdir()) / "echo_discovery" / "discovery"
        disc = discovery.Discovery(work, upload=True, progress=progress)

        def run() -> None:
            try:
                disc.run()
            except Exception as exc:  # the thread's last word, readable through status
                logger.exception("discovery failed")
                _DISCOVERY["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            finally:
                _DISCOVERY["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        _DISCOVERY.update({"progress": progress, "error": None, "finished_at": None,
                           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        _DISCOVERY["thread"] = threading.Thread(target=run, name="discovery", daemon=True)
        _DISCOVERY["thread"].start()
    logger.info("admin | discovery started")
    return {"started": True, "started_at": _DISCOVERY["started_at"]}


@router.get("/admin/discovery/status")
async def admin_discovery_status(x_admin_token: str | None = Header(None, alias="X-Admin-Token")) -> dict:
    _admin_guard(x_admin_token)
    t = _DISCOVERY["thread"]
    return {
        "running": bool(t is not None and t.is_alive()),
        "started_at": _DISCOVERY["started_at"],
        "finished_at": _DISCOVERY["finished_at"],
        "error": _DISCOVERY["error"],
        "progress": dict(_DISCOVERY["progress"]),
        **(await asyncio.to_thread(_hf_token_status)),
    }


@router.post("/identify", response_model=IdentifyResponse)
async def identify(audio: UploadFile = File(...)) -> IdentifyResponse:
    """
    Receive an audio snippet, fingerprint it via AcoustID to identify the song,
    compute self-generated audio metrics + embedding from its clean preview (or
    raw mic clip as fallback), and return genre- and language-matched songs
    ranked by our own ML + audio similarity metrics.
    """
    t0 = time.perf_counter()
    content = await audio.read()
    size_mb = len(content) / 1_048_576
    logger.info("identify | received audio  size=%.2f MB", size_mb)

    if size_mb > settings.max_audio_size_mb:
        raise HTTPException(413, f"Audio file too large ({size_mb:.1f} MB)")

    parsed = None
    try:
        raw = await recognizer.recognize(content)
        parsed = recognizer.parse_result(raw)
    except Exception as exc:
        logger.warning("identify | AcoustID recognition skipped/failed: %s", exc)
        parsed = None

    identified_track: TrackResult | None = None
    source_track_info: dict = {}
    audio_for_embedding: bytes = content

    if parsed:
        logger.info(
            "identify | AcoustID match  title=%r  artist=%r  score=%.2f",
            parsed["title"], parsed["artist"], parsed.get("score", 0),
        )
        query = f"{parsed['title']} {parsed['artist']}"
        tracks = await search_track(query, limit=1)
        if tracks:
            source_track_info = tracks[0]
            preview_url = source_track_info.get("previewUrl")
            if preview_url:
                preview_bytes = await fetch_preview_audio(preview_url)
                if preview_bytes:
                    audio_for_embedding = preview_bytes
                    logger.info("identify | using clean 30s preview for embedding & metrics")

    _, genre_family, _ = await _detect_track_context(source_track_info) if source_track_info else (None, "", [])
    try:
        desc = await asyncio.to_thread(
            recommender.describe,
            audio_for_embedding,
            meta=source_track_info or {"genre": genre_family},
        )
        source_metrics = desc
        source_embedding = np.array(desc.get("clap_vector") or [0.0] * 512, dtype=np.float32)
    except Exception as exc:
        logger.error("identify | audio description failed: %s", exc)
        source_embedding = np.zeros(512, dtype=np.float32)
        source_metrics = {
            "bpm": 120, "key": "8B", "energy": 0.5, "valence": 0.5,
            "vibes": ["Balanced"], "instruments": ["Synthesizer & Keys"],
            "instrumentPace": "Steady Mid-Tempo", "stereoWidth": 0.5,
            "phaseCorrelation": 0.95, "crestFactor": 10.0, "spatialStaging": "Balanced Stereo"
        }

    if source_track_info:
        identified_track = _to_track({
            **source_track_info,
            **source_metrics,
        })
        recommendations = await _find_and_rank_similar_tracks(
            source_track_info=source_track_info,
            source_embedding=source_embedding,
            source_metrics=source_metrics,
            limit=settings.max_recommendations,
            cross_language_max=6,
        )
    else:
        # Build an identified_track representation for the raw mic sound analysis
        identified_track = _to_track({
            "id": "mic_recording",
            "title": "Acoustic Voice / Mic Input",
            "artist": "Direct Audio Analysis",
            "album": "Neural Waveform Inspection",
            "albumArtUrl": "",
            "previewUrl": None,
            "durationMs": 0,
            "matchScore": 1.0,
            **source_metrics,
            "matchReasons": ["Direct Audio Waveform Match"],
        })
        faiss_recs = await asyncio.to_thread(
            recommender.recommend,
            seed=source_metrics,
            k=settings.max_recommendations,
        )
        recommendations = [_to_track(r) for r in faiss_recs]

    recommendations.sort(key=lambda t: t.matchScore or 0.0, reverse=True)
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
async def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
    enrich: bool = Query(False),
) -> SearchResponse:
    """
    Search by song / artist name. Returns fast results for debounced autocomplete.
    """
    t0 = time.perf_counter()
    logger.info("search | query=%r  limit=%d  enrich=%s", q, limit, enrich)
    tracks_data = await search_track(q, limit=limit)
    if not tracks_data:
        return SearchResponse(results=[], recommendations=[])

    if enrich and tracks_data:
        top_enriched, _, _ = await _enrich_single_track_with_own_metrics(tracks_data[0])
        results = [top_enriched] + [_to_track(t) for t in tracks_data[1:]]
    else:
        results = [_to_track(t) for t in tracks_data]

    logger.info("search | results=%d  total=%.3fs", len(results), time.perf_counter() - t0)
    return SearchResponse(results=results, recommendations=[])


_describe_promises: dict[str, Any] = {}
_describe_lock = asyncio.Lock()


async def _resolve_and_describe_track(track_id: str, key: str | None = None) -> dict:
    """
    Resolve track metadata and descriptors. If not in catalogue or extra_records,
    downloads preview once and shares an in-flight promise so /track and /recommendations
    do not re-describe concurrently (Section 12.5).
    """
    rec = recommender.get_track_record(track_id=track_id, key=key)
    if rec is not None:
        return rec

    dedup_key = key or str(track_id)
    is_initiator = False

    async with _describe_lock:
        rec = recommender.get_track_record(track_id=track_id, key=key)
        if rec is not None:
            return rec

        if dedup_key in _describe_promises:
            promise = _describe_promises[dedup_key]
        else:
            loop = asyncio.get_running_loop()
            promise = loop.create_future()
            _describe_promises[dedup_key] = promise
            is_initiator = True

    if not is_initiator:
        return await promise

    try:
        # 1. Fetch Deezer track info
        track = await get_track(track_id)
        if not track:
            raise HTTPException(404, f"Track {track_id} not found")

        # 2. If Deezer preview is missing (geo-filtered), try iTunes US search fallback
        if not track.get("previewUrl"):
            query = f"{track.get('artist', '')} {track.get('title', '')}".strip()
            client = _get_http_client()
            try:
                it_resp = await client.get(
                    "https://itunes.apple.com/search",
                    params={"term": query, "entity": "song", "limit": 1, "country": "US"},
                )
                if it_resp.status_code == 200:
                    it_res = it_resp.json().get("results", [])
                    if it_res and it_res[0].get("previewUrl"):
                        track["previewUrl"] = it_res[0]["previewUrl"]
                        if not track.get("itunes_id"):
                            track["itunes_id"] = str(it_res[0]["trackId"])
            except Exception as exc:
                logger.debug("iTunes preview resolution failed: %s", exc)

        # 3. Audio description
        _, _, desc = await _enrich_single_track_with_own_metrics(track)
        record = {**track, **desc}
        norm_key = key or make_track_key(str(track.get("artist", "")), str(track.get("title", "")))
        record["key"] = norm_key
        record["id"] = str(track.get("id", track_id))

        # 4. Save to in-memory cache and persistent ml/data/extra_records.jsonl
        recommender._extra_records[str(track_id)] = record
        recommender._extra_records[norm_key] = record
        if record.get("clap_vector") and recommender._side_index is not None:
            cvec = np.array([record["clap_vector"]], dtype=np.float32)
            recommender._side_index.add(cvec)

        try:
            recommender._extra_records_path.parent.mkdir(parents=True, exist_ok=True)
            with open(recommender._extra_records_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("Could not persist extra record: %s", exc)

        promise.set_result(record)
        return record
    except Exception as exc:
        if not promise.done():
            promise.set_exception(exc)
        raise exc
    finally:
        async with _describe_lock:
            _describe_promises.pop(dedup_key, None)


@router.get("/track/{track_id}", response_model=TrackResult)
async def get_track_detail(
    track_id: str,
    key: str | None = Query(None, description="Canonical artist:title key"),
) -> TrackResult:
    """
    Fetch track details. Checks in-catalogue and extra_records first.
    If unknown, describes once with in-flight lock sharing (Section 12.5).
    """
    rec = recommender.get_track_record(track_id=track_id, key=key)
    if rec is not None:
        return _to_track(rec)
    rec = await _resolve_and_describe_track(track_id=track_id, key=key)
    return _to_track(rec)


@router.get("/recommendations/{track_id}")
async def get_recommendations(
    track_id: str,
    key: str | None = Query(None, description="Canonical artist:title key"),
    diversity: float = Query(0.70, ge=0.0, le=1.0, description="MMR trade-off: 1.0=pure match, 0.5=high discovery"),
    cross_language_max: int = Query(6, ge=0, le=20, description="Maximum cross-language tracks allowed in final slate"),
    debug: int = Query(0, description="Debug mode: return axes, weights, pool_source, and candidate pool"),
):
    """
    Step 11.7 / 12.5 Serving Path:
    If track_id or key is in catalogue or extra_records, recommend directly in < 300 ms.
    If cold seed, resolve/describe once with in-flight deduplication, and recommend.
    If debug=1, return axes, weights, pool_source, and candidate_pool.
    """
    t0 = time.perf_counter()
    logger.info("recommendations | track_id=%s key=%s diversity=%.2f debug=%d", track_id, key, diversity, debug)

    seed_rec = recommender.get_track_record(track_id=track_id, key=key)
    if seed_rec is None:
        seed_rec = await _resolve_and_describe_track(track_id=track_id, key=key)

    if debug == 1:
        recs, pool = await asyncio.to_thread(
            recommender.recommend,
            seed=seed_rec,
            k=settings.max_recommendations,
            cross_language_max=cross_language_max,
            diversity=diversity,
            debug=True,
            return_pool=True,
        )
        debug_results = []
        for r in recs:
            td = _to_track(r).model_dump(exclude_none=True)
            td["axes"] = r.get("axes", {})
            td["weights"] = r.get("weights", {})
            td["pool_source"] = r.get("pool_source", "clap")
            debug_results.append(td)
        logger.info("recommendations | track_id=%s key=%s debug=1 elapsed=%.3fs", track_id, key, time.perf_counter() - t0)
        return {
            "results": debug_results,
            "candidate_pool": [
                {
                    "id": str(c.get("id", "")),
                    "title": str(c.get("title", "")),
                    "artist": str(c.get("artist", "")),
                    "pool_source": str(c.get("pool_source", "")),
                }
                for c in pool
            ],
        }

    recs = await asyncio.to_thread(
        recommender.recommend,
        seed=seed_rec,
        k=settings.max_recommendations,
        cross_language_max=cross_language_max,
        diversity=diversity,
        debug=False,
        return_pool=False,
    )
    logger.info("recommendations | track_id=%s key=%s elapsed=%.3fs", track_id, key, time.perf_counter() - t0)
    return [_to_track(r) for r in recs]