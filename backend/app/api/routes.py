from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
import numpy as np

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

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
from app.services.lastfm import get_artist_tags
from app.services.recommender import (
    recommender,
    to_camelot_key,
)
from app.audio.features import language_from_meta
from app.services.language_affinity import language_affinity

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Concurrency limiter for preview downloads + ONNX encoding
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
        bpm=int(meta.get("bpm", 0)),
        felt_bpm=int(meta["felt_bpm"]) if meta.get("felt_bpm") else None,
        key=str(meta.get("key", "—")),
        camelotKey=str(meta.get("camelotKey")) if meta.get("camelotKey") else to_camelot_key(str(meta.get("key", "—"))),
        modeMinorProb=float(meta["modeMinorProb"]) if meta.get("modeMinorProb") is not None else None,
        brightness=float(meta["brightness"]) if meta.get("brightness") is not None else None,
        buildScore=float(meta["buildScore"]) if meta.get("buildScore") is not None else None,
        loudnessRange=float(meta["loudnessRange"]) if meta.get("loudnessRange") is not None else None,
        stereoWidth=float(meta.get("stereoWidth", 0.5)),
        phaseCorrelation=float(meta.get("phaseCorrelation", 0.95)),
        crestFactor=float(meta.get("crestFactor", 10.0)),
        spatialStaging=str(meta.get("spatialStaging", "Balanced Stereo")),
        lyricalIntent=str(meta.get("lyricalIntent") or meta.get("emotionalTheme", "Chill & Atmospheric")),
        energy=float(meta.get("energy", 0.5)),
        valence=float(meta.get("valence", 0.5)),
        languageAffinity=float(meta["languageAffinity"]) if meta.get("languageAffinity") is not None else None,
        instruments=list(meta.get("instruments", [])),
        instrumentOrder=list(meta.get("instrumentOrder", [])),
        instrumentPace=str(meta.get("instrumentPace", "Steady Mid-Tempo")),
        vocalPresence=float(meta.get("vocalPresence", 0.5)),
        vocalPattern=str(meta.get("vocalPattern", "Dynamic Melodic Phrasing")),
        vocalComplementation=str(meta.get("vocalComplementation", "In-The-Pocket Groove")),
        year=int(meta["year"]) if meta.get("year") else None,
        decade=str(meta["decade"]) if meta.get("decade") else None,
        matchReasons=list(meta.get("matchReasons", [])),
        darkness=float(meta["darkness"]) if meta.get("darkness") is not None else None,
        grit=float(meta["grit"]) if meta.get("grit") is not None else None,
        pulse_rate=float(meta["pulse_rate"]) if meta.get("pulse_rate") is not None else None,
        kick_class=str(meta["kick_class"]) if meta.get("kick_class") is not None else None,
        restraint_bucket=str(meta["restraint_bucket"]) if meta.get("restraint_bucket") is not None else None,
        intimacy=float(meta["intimacy"]) if meta.get("intimacy") is not None else None,
        mode_polarity=float(meta["mode_polarity"]) if meta.get("mode_polarity") is not None else None,
        mode_undefined=bool(meta["mode_undefined"]) if meta.get("mode_undefined") is not None else None,
        changes_per_bar=float(meta["changes_per_bar"]) if meta.get("changes_per_bar") is not None else None,
        loop_bars=int(meta["loop_bars"]) if meta.get("loop_bars") is not None else None,
        riff_flag=bool(meta["riff_flag"]) if meta.get("riff_flag") is not None else None,
        language=str(meta["language"]) if meta.get("language") is not None else None,
        language_conf=float(meta["language_conf"]) if meta.get("language_conf") is not None else None,
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

    album_task = get_album_genres(album_id) if album_id and not track_info.get("genres") else asyncio.sleep(0, result=[])
    lastfm_task = get_artist_tags(artist_name) if artist_name else asyncio.sleep(0, result=[])
    album_genres, artist_tags = await asyncio.gather(album_task, lastfm_task)

    raw_tags.extend(album_genres)
    raw_tags.extend(artist_tags)

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
        l_code, _ = language_from_meta(meta_dict)
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
    min_language_affinity: float = 0.50,
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
            min_language_affinity=0.50,
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


@router.get("/track/{track_id}", response_model=TrackResult)
async def get_track_detail(track_id: str) -> TrackResult:
    """
    Fetch track details and enrich with self-generated audio analytics if preview is available.
    """
    if track_id in recommender._track_id_map:
        return _to_track(recommender._metadata[recommender._track_id_map[track_id]])
    track = await get_track(track_id)
    if not track:
        raise HTTPException(404, f"Track {track_id} not found")
    enriched, _, _ = await _enrich_single_track_with_own_metrics(track)
    return enriched


@router.get("/recommendations/{track_id}", response_model=list[TrackResult])
async def get_recommendations(
    track_id: str,
    diversity: float = Query(0.70, ge=0.0, le=1.0, description="MMR trade-off: 1.0=pure match, 0.5=high discovery"),
    cross_language_max: int = Query(6, ge=0, le=20, description="Maximum cross-language tracks allowed in final slate"),
    min_language_affinity: float = Query(0.50, ge=0.0, le=1.0, description="Minimum language affinity score required"),
) -> list[TrackResult]:
    """
    Step 11.7 Serving Path:
    If track_id is in catalogue, recommend directly in < 300 ms.
    If cold seed, fetch preview, describe in ~2s, and recommend.
    """
    t0 = time.perf_counter()
    logger.info("recommendations | track_id=%s diversity=%.2f", track_id, diversity)

    # 1. Fast in-catalogue lookup
    if track_id in recommender._track_id_map:
        seed_meta = recommender._metadata[recommender._track_id_map[track_id]]
        recs = await asyncio.to_thread(
            recommender.recommend,
            seed=seed_meta,
            k=settings.max_recommendations,
            cross_language_max=cross_language_max,
            diversity=diversity,
        )
        logger.info("recommendations | in-catalogue hit track_id=%s  elapsed=%.3fs", track_id, time.perf_counter() - t0)
        return [_to_track(r) for r in recs]

    # 2. Cold seed path
    track = await get_track(track_id)
    if not track:
        raise HTTPException(404, f"Track {track_id} not found")

    recs = await asyncio.to_thread(
        recommender.recommend,
        seed=track,
        k=settings.max_recommendations,
        cross_language_max=cross_language_max,
        diversity=diversity,
    )
    logger.info("recommendations | cold seed track_id=%s  elapsed=%.3fs", track_id, time.perf_counter() - t0)
    return [_to_track(r) for r in recs]