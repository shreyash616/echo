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
    get_genre_and_language_candidates,
    fetch_preview_audio,
    _extract_year_decade,
)
from app.services.lastfm import get_artist_tags, get_similar_tracks
from app.services.recommender import (
    recommender,
    compute_composite_match_score,
    _era_similarity,
    to_camelot_key,
    compute_match_reasons,
)
from app.services.lyrics import fetch_lyrics, classify_lyrical_intent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# Concurrency limiter for preview downloads + ONNX encoding
_ENCODE_SEM = asyncio.Semaphore(8)

_CULTURAL_ARTIST_RULES: list[tuple[set[str], str, str, list[str]]] = [
    (
        {"kishore kumar", "lata mangeshkar", "asha bhosle", "mohammed rafi", "mohd rafi",
         "r.d. burman", "rd burman", "mukesh", "manna dey", "alka yagnik", "kumar sanu",
         "udit narayan", "sonu nigam", "arijit singh", "shreya ghoshal", "sunidhi chauhan",
         "bappi lahiri", "hemant kumar", "anuradha paudwal", "jagjit singh", "pankaj udhas",
         "lucky ali", "kk", "pritam", "amit trivedi", "vishal-shekhar", "neha kakkar",
         "kalyanji-anandji", "laxmikant-pyarelal", "s.d. burman", "sd burman"},
        "hi", "bollywood", ["bollywood", "hindi", "filmi"],
    ),
    (
        {"sidhu moose wala", "diljit dosanjh", "ap dhillon", "karan aujla", "shubh",
         "amrinder gill", "gurdas maan", "bohemia", "imran khan", "jazzy b", "mankirt aulakh"},
        "pa", "punjabi", ["punjabi", "bhangra", "desi"],
    ),
    (
        {"s. p. balasubrahmanyam", "spb", "ilayaraja", "anirudh ravichander", "sid sriram",
         "yuvan shankar raja", "d. imman", "hariharan", "k. s. chithra", "sujatha"},
        "ta", "tamil", ["tamil", "kollywood"],
    ),
    (
        {"bts", "blackpink", "twice", "stray kids", "exo", "newjeans", "seventeen", "iu", "red velvet"},
        "ko", "k-pop", ["k-pop", "kpop", "korean"],
    ),
]


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
        key=str(meta.get("key", "—")),
        camelotKey=str(meta.get("camelotKey")) if meta.get("camelotKey") else to_camelot_key(str(meta.get("key", "—"))),
        stereoWidth=float(meta.get("stereoWidth", 0.5)),
        phaseCorrelation=float(meta.get("phaseCorrelation", 0.95)),
        crestFactor=float(meta.get("crestFactor", 10.0)),
        spatialStaging=str(meta.get("spatialStaging", "Balanced Stereo")),
        lyricalIntent=str(meta.get("lyricalIntent") or meta.get("emotionalTheme", "Chill & Atmospheric")),
        energy=float(meta.get("energy", 0.5)),
        valence=float(meta.get("valence", 0.5)),
        instruments=list(meta.get("instruments", [])),
        instrumentOrder=list(meta.get("instrumentOrder", [])),
        instrumentPace=str(meta.get("instrumentPace", "Steady Mid-Tempo")),
        vocalPresence=float(meta.get("vocalPresence", 0.5)),
        vocalPattern=str(meta.get("vocalPattern", "Dynamic Melodic Phrasing")),
        vocalComplementation=str(meta.get("vocalComplementation", "In-The-Pocket Groove")),
        year=int(meta["year"]) if meta.get("year") else None,
        decade=str(meta["decade"]) if meta.get("decade") else None,
        matchReasons=list(meta.get("matchReasons", [])),
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

    # 3. Cultural artist heuristics for well-known artists
    if not lang or genre_family in ["classical", "general", "films/games", "soundtrack", ""]:
        artist_lower = artist_name.lower()
        for artists, rule_lang, rule_genre, rule_kw in _CULTURAL_ARTIST_RULES:
            if any(a in artist_lower for a in artists):
                lang = rule_lang
                genre_family = rule_genre
                genre_keywords = list(rule_kw)
                break

    # 4. iTunes lookup fallback to disambiguate generic Deezer tags (e.g. 'Films/Games') and resolve true release era
    needs_itunes_genre = not lang or genre_family in ["classical", "general", "films/games", "soundtrack", ""]
    is_classic_artist_post2000 = (track_info.get("year", 0) > 2000) and any(
        a in artist_name.lower() for a in ["kishore", "rafi", "mukesh", "mangeshkar", "burman", "bhosle", "manna"]
    )
    if needs_itunes_genre or not track_info.get("year") or is_classic_artist_post2000:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
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
    Download a track's 30s preview (if available), run CnnMusicEncoder +
    self-generated audio analytics (BPM, Key, Energy, Valence, Vibes), and
    return (enriched_TrackResult, embedding, metrics_dict).
    """
    preview_url = track_meta.get("previewUrl")
    if not preview_url:
        return _to_track(track_meta), None, {}

    preview_bytes = await fetch_preview_audio(preview_url)
    if not preview_bytes:
        return _to_track(track_meta), None, {}

    emb, metrics = await asyncio.to_thread(
        recommender.encode_and_analyze_audio,
        preview_bytes,
        genre_hint or str(track_meta.get("genre", "")),
    )
    if track_meta.get("year"):
        metrics["year"] = track_meta["year"]
    if track_meta.get("decade"):
        metrics["decade"] = track_meta["decade"]

    lyrics = await fetch_lyrics(str(track_meta.get("artist", "")), str(track_meta.get("title", "")))
    intent = classify_lyrical_intent(
        str(track_meta.get("title", "")),
        str(track_meta.get("artist", "")),
        lyrics=lyrics,
        genre=genre_hint or str(track_meta.get("genre", "")),
    )
    metrics["lyricalIntent"] = intent

    enriched = {
        **track_meta,
        "bpm":                  metrics["bpm"],
        "key":                  metrics["key"],
        "camelotKey":           to_camelot_key(str(metrics.get("key", "—"))),
        "energy":               metrics["energy"],
        "valence":              metrics["valence"],
        "vibes":                metrics["vibes"],
        "instruments":          metrics.get("instruments", []),
        "instrumentOrder":      metrics.get("instrumentOrder", []),
        "instrumentPace":       metrics.get("instrumentPace", "Steady Mid-Tempo"),
        "vocalPresence":        metrics.get("vocalPresence", 0.5),
        "vocalPattern":         metrics.get("vocalPattern", "Dynamic Melodic Phrasing"),
        "vocalComplementation": metrics.get("vocalComplementation", "In-The-Pocket Groove"),
        "year":                 track_meta.get("year"),
        "decade":               track_meta.get("decade"),
        "lyricalIntent":        intent,
    }
    return _to_track(enriched), emb, metrics


def _apply_mmr(
    candidates: list[tuple[TrackResult, float, np.ndarray | None]],
    limit: int = 10,
    diversity_lambda: float = 0.72,
    max_per_artist: int = 2,
) -> list[TrackResult]:
    """
    Maximal Marginal Relevance (MMR) Reranker with Artist Frequency Capping:
      Selects an ordered slate balancing acoustic query relevance against
      pairwise candidate similarity to prevent catalog redundancy:
        MMR(c) = lambda * Rel(q, c) - (1 - lambda) * max_{s in S} Sim(c, s)
      - lambda = 0.72: SOTA MIR balance between fidelity and catalog discovery
      - max_per_artist = 2: Prevents any single artist from dominating recommendations
    """
    if not candidates:
        return []

    # If lambda is pure relevance (>= 0.98), greedy rank with artist cap
    if diversity_lambda >= 0.98:
        selected_pure: list[TrackResult] = []
        artist_counts: dict[str, int] = {}
        for track, _, _ in sorted(candidates, key=lambda x: x[1], reverse=True):
            art = _primary_artist_name(track.artist).lower()
            if artist_counts.get(art, 0) < max_per_artist:
                selected_pure.append(track)
                artist_counts[art] = artist_counts.get(art, 0) + 1
                if len(selected_pure) >= limit:
                    break
        if len(selected_pure) < limit:
            for track, _, _ in sorted(candidates, key=lambda x: x[1], reverse=True):
                if track not in selected_pure:
                    selected_pure.append(track)
                    if len(selected_pure) >= limit:
                        break
        return selected_pure

    remaining = list(candidates)
    selected: list[tuple[TrackResult, float, np.ndarray | None]] = []
    artist_counts: dict[str, int] = {}

    # 1. Start with the highest-scoring candidate as anchor
    first_pick = max(remaining, key=lambda x: x[1])
    selected.append(first_pick)
    remaining.remove(first_pick)
    first_art = _primary_artist_name(first_pick[0].artist).lower()
    artist_counts[first_art] = artist_counts.get(first_art, 0) + 1

    # 2. Greedy iterative selection via MMR
    while len(selected) < limit and remaining:
        best_cand: tuple[TrackResult, float, np.ndarray | None] | None = None
        best_mmr_score = -float("inf")

        for cand in remaining:
            cand_track, cand_score, cand_emb = cand
            cand_art = _primary_artist_name(cand_track.artist).lower()
            is_capped = artist_counts.get(cand_art, 0) >= max_per_artist

            # Maximum similarity with any already-selected track
            max_sim = 0.0
            for sel_track, _, sel_emb in selected:
                sim = 0.0
                if cand_emb is not None and sel_emb is not None:
                    sim = float(np.dot(cand_emb.flatten(), sel_emb.flatten()))
                else:
                    # Acoustic distance fallback from rhythm and energy
                    bpm_diff = abs(cand_track.bpm - sel_track.bpm) / 80.0
                    e_diff = abs(cand_track.energy - sel_track.energy)
                    v_diff = abs(cand_track.valence - sel_track.valence)
                    sim = max(0.0, 1.0 - (0.4 * bpm_diff + 0.3 * e_diff + 0.3 * v_diff))

                # Same primary artist gets an acoustic overlap penalty
                if cand_art and cand_art == _primary_artist_name(sel_track.artist).lower():
                    sim = max(sim, 0.85)

                if sim > max_sim:
                    max_sim = sim

            mmr_val = (diversity_lambda * cand_score) - ((1.0 - diversity_lambda) * max_sim)
            if is_capped:
                mmr_val -= 0.35  # Discourage exceeding artist cap

            if mmr_val > best_mmr_score:
                best_mmr_score = mmr_val
                best_cand = cand

        if best_cand is None:
            break

        selected.append(best_cand)
        remaining.remove(best_cand)
        sel_art = _primary_artist_name(best_cand[0].artist).lower()
        artist_counts[sel_art] = artist_counts.get(sel_art, 0) + 1

    # Sort the final diverse MMR slate in descending order of matchScore for monotonic presentation
    sorted_tracks = sorted([item[0] for item in selected], key=lambda t: t.matchScore or 0.0, reverse=True)

    # Anti-Adjacency Window Check: prevent consecutive duplicate artists in top ranks
    for i in range(len(sorted_tracks) - 1):
        curr_art = _primary_artist_name(sorted_tracks[i].artist).lower()
        next_art = _primary_artist_name(sorted_tracks[i + 1].artist).lower()
        if curr_art and curr_art == next_art:
            for j in range(i + 2, len(sorted_tracks)):
                cand_art = _primary_artist_name(sorted_tracks[j].artist).lower()
                if cand_art != curr_art:
                    sorted_tracks[i + 1], sorted_tracks[j] = sorted_tracks[j], sorted_tracks[i + 1]
                    break

    return sorted_tracks


async def _find_and_rank_similar_tracks(
    source_track_info: dict,
    source_embedding: np.ndarray,
    source_metrics: dict,
    limit: int,
    diversity_lambda: float = 0.72,
) -> list[TrackResult]:
    """
    Unified recommendation pipeline that:
      1. Identifies the source song's language and genre family.
      2. Collects genre- and language-matched candidate songs from:
         - Deezer's artist/genre neighborhood (`get_genre_and_language_candidates`)
         - Last.fm similar tracks (`get_similar_tracks`)
         - FAISS index neighbors (`recommender.recommend`) filtered by genre
      3. Enforces genre & language matching so recommendations stay within the
         source song's language and musical style.
      4. Encodes every candidate's 30s audio preview using our own `CnnMusicEncoder`
         and computes our own self-generated metrics (`bpm`, `key`, `energy`,
         `valence`, `vibes`).
      5. Ranks all candidates by `compute_composite_match_score` with MMR
         diversity reranking (avoiding catalog monotony and capping artist repetition).
    """
    source_lang, genre_family, genre_keywords = await _detect_track_context(source_track_info)
    lang_hint = _LANG_SEARCH_HINT.get(source_lang, "") if source_lang else ""

    primary_artist = _primary_artist_name(str(source_track_info.get("artist", "")))
    source_title = str(source_track_info.get("title", ""))
    source_id = str(source_track_info.get("id", ""))

    # 1. Gather candidates concurrently from Deezer artist/genre graph, Last.fm, and FAISS
    deezer_cand_task = get_genre_and_language_candidates(
        source_track_info,
        genre_hint=genre_family,
        lang_hint=lang_hint,
        limit=max(limit * 2, 30),
    )
    lastfm_cand_task = (
        get_similar_tracks(primary_artist, source_title, limit=limit * 2)
        if primary_artist and source_title
        else asyncio.sleep(0, result=[])
    )
    faiss_raw = recommender.recommend(
        source_embedding,
        source_metrics=source_metrics,
        genre_keywords=genre_keywords + ([lang_hint] if lang_hint else []),
        source_lang=source_lang,
        exclude_id=source_id,
        k=limit * 2,
    )

    deezer_cands, lastfm_raw = await asyncio.gather(deezer_cand_task, lastfm_cand_task)

    # Mark candidates from the source artist's direct Deezer related-artist graph
    # (these inherently match the artist's language & genre scene)
    for c in deezer_cands:
        c["_trusted_scene"] = True

    # Resolve Last.fm and FAISS candidates to Deezer tracks with preview URLs concurrently
    external_queries: list[str] = []
    for m in lastfm_raw[:20]:
        q = f"{m.get('title', '')} {m.get('artist', '')}".strip()
        if q:
            external_queries.append(q)

    # Always include FAISS nearest neighbors (already from our trained commercial index)
    direct_faiss_cands: list[dict] = []
    for m in faiss_raw[:20]:
        if m.get("previewUrl"):
            direct_faiss_cands.append(m)
        else:
            q = f"{m.get('title', '')} {m.get('artist', '')}".strip()
            if q:
                external_queries.append(q)

    async def _resolve_query(q: str) -> dict | None:
        try:
            res = await search_track(q, limit=1)
            return res[0] if res and res[0].get("previewUrl") else None
        except Exception:
            return None

    resolved_external = await asyncio.gather(*[_resolve_query(q) for q in external_queries[:16]])

    # Merge and deduplicate candidate pool
    seen_ids: set[str] = {source_id} if source_id else set()
    seen_keys: set[str] = {f"{primary_artist.lower()}:{source_title.lower().strip()}"}
    merged_candidates: list[dict] = []

    for cand in direct_faiss_cands + deezer_cands + [r for r in resolved_external if r is not None]:
        cid = str(cand.get("id", ""))
        ckey = f"{_primary_artist_name(str(cand.get('artist', ''))).lower()}:{str(cand.get('title', '')).lower().strip()}"
        if not cid or cid in seen_ids or ckey in seen_keys or not cand.get("previewUrl"):
            continue
        seen_ids.add(cid)
        seen_keys.add(ckey)
        merged_candidates.append(cand)
        if len(merged_candidates) >= 30:
            break

    if not merged_candidates:
        logger.warning("recommendations | no candidates with preview URLs found")
        return [_to_track(m) for m in faiss_raw[:limit]]

    # 2. Verify genre, language compatibility, and era, then score each candidate
    async def _evaluate_candidate(cand: dict) -> tuple[TrackResult, float, bool, np.ndarray | None] | None:
        try:
            cand_artist = _primary_artist_name(str(cand.get("artist", "")))
            cand_title = str(cand.get("title", ""))

            # Step 1: Language detection (script, explicit genre tag, and artist tags)
            cand_script_lang = _detect_script_language(cand_title, cand_artist)
            cand_tags: list[str] = []
            if cand.get("genre"):
                cand_tags.append(str(cand["genre"]))

            if not cand.get("_trusted_scene"):
                cand_tags.extend(await get_artist_tags(cand_artist))
                if not cand_tags and cand.get("albumId"):
                    cand_tags.extend(await get_album_genres(str(cand["albumId"])))

            cand_tag_lang = _detect_language_from_tags(cand_tags)
            cand_lang = cand_tag_lang or cand_script_lang

            # Step 2: Strict language barrier
            # - If source is non-English, candidate MUST match source_lang
            # - If source is English, reject non-English tracks
            if source_lang:
                if cand_lang and cand_lang != source_lang:
                    return None
                if not cand.get("_trusted_scene") and cand_lang != source_lang:
                    return None
            else:
                if cand_lang is not None:
                    return None

            # Step 3: Genre compatibility
            cand_genre_family, _ = _detect_genre_family(cand_tags)
            genre_matched = bool(
                cand.get("_trusted_scene")
                or not genre_family
                or cand_genre_family == genre_family
                or any(kw in " ".join(cand_tags).lower() for kw in genre_keywords)
            )

            # Step 4: Fast path for FAISS candidates that already passed language barrier
            if cand.get("matchScore") is not None and cand.get("bpm", 0) > 0:
                score = float(cand["matchScore"])
                src_year = source_metrics.get("year") or source_track_info.get("year")
                cand_year = cand.get("year")
                s_era = _era_similarity(src_year, cand_year)
                if s_era < 0.5:
                    score = round(max(0.05, score * s_era * 1.5), 4)

                cand_copy = dict(cand)
                cand_copy["matchScore"] = score
                cand_copy["camelotKey"] = cand_copy.get("camelotKey") or to_camelot_key(str(cand_copy.get("key", "—")))
                if source_metrics:
                    cand_copy["matchReasons"] = compute_match_reasons(source_metrics, cand_copy)
                cand_emb = recommender.get_embedding(cand.get("_idx", -1))
                return _to_track(cand_copy), score, genre_matched, cand_emb

            # Step 5: External audio preview download + ML encoding + structural analysis
            async with _ENCODE_SEM:
                preview_bytes = await fetch_preview_audio(str(cand["previewUrl"]))
                if not preview_bytes:
                    return None

                cand_emb, cand_metrics = await asyncio.to_thread(
                    recommender.encode_and_analyze_audio,
                    preview_bytes,
                    genre_family or cand_genre_family,
                )
                if cand.get("year"):
                    cand_metrics["year"] = cand["year"]
                if cand.get("decade"):
                    cand_metrics["decade"] = cand["decade"]

                cand_lyrics = await fetch_lyrics(cand_artist, cand_title)
                cand_intent = classify_lyrical_intent(
                    cand_title, cand_artist, lyrics=cand_lyrics, genre=cand.get("genre", "")
                )
                cand_metrics["lyricalIntent"] = cand_intent

                score = compute_composite_match_score(
                    source_embedding,
                    source_metrics,
                    cand_emb,
                    cand_metrics,
                )
                if genre_matched:
                    score = round(min(0.99, score + 0.03), 4)

                reasons = compute_match_reasons(source_metrics, cand_metrics)

                enriched = {
                    **cand,
                    "matchScore":           score,
                    "bpm":                  cand_metrics["bpm"],
                    "key":                  cand_metrics["key"],
                    "camelotKey":           to_camelot_key(str(cand_metrics.get("key", "—"))),
                    "energy":               cand_metrics["energy"],
                    "valence":              cand_metrics["valence"],
                    "vibes":                cand_metrics["vibes"],
                    "lyricalIntent":        cand_intent,
                    "instruments":          cand_metrics.get("instruments", []),
                    "instrumentOrder":      cand_metrics.get("instrumentOrder", []),
                    "instrumentPace":       cand_metrics.get("instrumentPace", "Steady Mid-Tempo"),
                    "vocalPresence":        cand_metrics.get("vocalPresence", 0.5),
                    "vocalPattern":         cand_metrics.get("vocalPattern", "Dynamic Melodic Phrasing"),
                    "vocalComplementation": cand_metrics.get("vocalComplementation", "In-The-Pocket Groove"),
                    "year":                 cand.get("year"),
                    "decade":               cand.get("decade"),
                    "matchReasons":         reasons,
                }
                return _to_track(enriched), score, genre_matched, cand_emb
        except Exception as exc:
            logger.debug("evaluate_candidate failed for %r: %s", cand.get("title"), exc)
            return None

    evaluated = await asyncio.gather(*[_evaluate_candidate(c) for c in merged_candidates])
    valid = [item for item in evaluated if item is not None]

    # Partition candidates by genre match
    genre_strict = [(item[0], item[1], item[3]) for item in valid if item[2]]
    other_valid = [(item[0], item[1], item[3]) for item in valid if not item[2]]

    # Prioritize genre-matched candidates; supplement with other valid if needed
    candidate_pool = genre_strict if len(genre_strict) >= limit else (genre_strict + other_valid)

    ranked_tracks = _apply_mmr(
        candidate_pool,
        limit=limit,
        diversity_lambda=diversity_lambda,
        max_per_artist=2,
    )

    if not ranked_tracks and faiss_raw:
        ranked_tracks = [_to_track(m) for m in faiss_raw[:limit]]

    logger.info(
        "rank_similar | evaluated=%d/%d  genre_strict=%d  ranked=%d  diversity=%.2f",
        len(valid), len(merged_candidates), len(genre_strict), len(ranked_tracks), diversity_lambda,
    )
    return ranked_tracks[:limit]


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

    raw = await recognizer.recognize(content)
    parsed = recognizer.parse_result(raw)

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
    source_embedding, source_metrics = await asyncio.to_thread(
        recommender.encode_and_analyze_audio,
        audio_for_embedding,
        genre_family,
    )

    if source_track_info:
        lyrics = await fetch_lyrics(str(source_track_info.get("artist", "")), str(source_track_info.get("title", "")))
        intent = classify_lyrical_intent(
            str(source_track_info.get("title", "")),
            str(source_track_info.get("artist", "")),
            lyrics=lyrics,
            genre=genre_family,
        )
        source_metrics["lyricalIntent"] = intent

        identified_track = _to_track({
            **source_track_info,
            "bpm":           source_metrics["bpm"],
            "key":           source_metrics["key"],
            "energy":        source_metrics["energy"],
            "valence":       source_metrics["valence"],
            "vibes":         source_metrics["vibes"],
            "lyricalIntent": intent,
        })

    if source_track_info:
        recommendations = await _find_and_rank_similar_tracks(
            source_track_info=source_track_info,
            source_embedding=source_embedding,
            source_metrics=source_metrics,
            limit=settings.max_recommendations,
        )
    else:
        # Unidentified raw mic audio: fall back to FAISS nearest neighbors resolved on Deezer
        faiss_recs = recommender.recommend(
            source_embedding,
            source_metrics=source_metrics,
            k=settings.max_recommendations,
        )
        recommendations = [_to_track(r) for r in faiss_recs]

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
    track = await get_track(track_id)
    if not track:
        raise HTTPException(404, f"Track {track_id} not found")
    enriched, _, _ = await _enrich_single_track_with_own_metrics(track)
    return enriched


@router.get("/recommendations/{track_id}", response_model=list[TrackResult])
async def get_recommendations(
    track_id: str,
    diversity: float = Query(0.72, ge=0.0, le=1.0, description="MMR trade-off: 1.0=pure match, 0.5=high discovery"),
) -> list[TrackResult]:
    """
    Download the selected track's 30s preview, compute its embedding + self-generated
    metrics, and return genre- and language-matched songs ranked by our own ML
    and audio metrics, reranked via Maximal Marginal Relevance (MMR).
    """
    t0 = time.perf_counter()
    logger.info("recommendations | track_id=%s diversity=%.2f", track_id, diversity)

    track = await get_track(track_id)
    if not track:
        raise HTTPException(404, f"Track {track_id} not found")

    preview_url = track.get("previewUrl")
    if not preview_url:
        logger.warning("recommendations | no preview available  track_id=%s", track_id)
        return [_to_track(m) for m in recommender._metadata[:settings.max_recommendations]]

    preview_bytes = await fetch_preview_audio(preview_url)
    if not preview_bytes:
        logger.warning("recommendations | preview download failed  track_id=%s", track_id)
        return [_to_track(m) for m in recommender._metadata[:settings.max_recommendations]]

    _, genre_family, _ = await _detect_track_context(track)
    source_embedding, source_metrics = await asyncio.to_thread(
        recommender.encode_and_analyze_audio,
        preview_bytes,
        genre_family,
    )

    lyrics = await fetch_lyrics(str(track.get("artist", "")), str(track.get("title", "")))
    intent = classify_lyrical_intent(
        str(track.get("title", "")),
        str(track.get("artist", "")),
        lyrics=lyrics,
        genre=genre_family,
    )
    source_metrics["lyricalIntent"] = intent

    recommendations = await _find_and_rank_similar_tracks(
        source_track_info=track,
        source_embedding=source_embedding,
        source_metrics=source_metrics,
        limit=settings.max_recommendations,
        diversity_lambda=diversity,
    )

    if not recommendations and recommender._metadata:
        logger.info("recommendations | falling back directly to FAISS index neighbors")
        faiss_fallback = recommender.recommend(
            source_embedding,
            source_metrics=source_metrics,
            exclude_id=track_id,
            k=settings.max_recommendations,
        )
        recommendations = [_to_track(m) for m in faiss_fallback]

    logger.info(
        "recommendations | done  title=%r  recs=%d  total=%.3fs",
        track.get("title"), len(recommendations), time.perf_counter() - t0,
    )
    return recommendations