"""
Lyrics & Human Emotional Intent Service for Echo.
Fetches lyrics via open APIs (LRCLIB) and classifies songs into canonical human emotional themes.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
import httpx
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_lyrics_cache: TTLCache = TTLCache(maxsize=1024, ttl=86400)

_THEME_KEYWORDS: dict[str, list[str]] = {
    "Romantic & Devotional": [
        "love", "pyaar", "ishq", "mohabbat", "dil", "rab ne", "together", "baby", "darling",
        "sohniye", "sohna", "wife", "wedding", "jodi", "forever", "sweetheart", "romance",
        "kiss", "holding hands", "beloved", "sanam", "deewana", "tere bina", "tere layi",
        "reason", "couple", "marry", "haye", "meri jaan", "aankhon", "chahat", "deewangi",
        "humsafar", "romantic", "qismat", "kasam", "vaada"
    ],
    "Street Confrontation & Defiance": [
        "beef", "gang", "rival", "enemy", "dushman", "guns", "vairi", "pistol", "badmashi",
        "fight", "war", "anti", "threat", "hater", "goli", "muqabla", "mafia", "jail",
        "crime", "shoot", "kill", "bravado", "rap battle", "diss", "asla", "dunali", "hathyar",
        "same beef", "thok", "gangster", "shikari", "bullet", "darr", "badla", "gunda"
    ],
    "Heartbreak & Melancholy": [
        "dhokha", "bewafa", "judai", "crying", "pain", "tears", "heartbroken", "alone",
        "tanha", "dard", "sadness", "broken", "miss you", "yaad", "chhod ke", "alvida", "rona",
        "lonely", "goodbye", "hurt", "sorrow", "regret"
    ],
    "Celebration & Party": [
        "party", "dance", "nacho", "club", "daroo", "sharab", "bottle", "cheers", "daru",
        "celebrate", "weekend", "dhol", "bhangra", "disco", "dj", "thumka", "pataka",
        "celebration", "festive", "groove"
    ],
    "Triumph & Motivational": [
        "hustle", "struggle", "kamyabi", "success", "winner", "grind", "victory", "champion",
        "top", "dreams", "power", "money", "hard work", "level", "trophy", "king", "unstoppable"
    ],
    "Introspective & Spiritual": [
        "rabba", "waheguru", "khuda", "allah", "soul", "spirit", "prayer", "zindagi",
        "life", "meaning", "philosophy", "peace", "sukoon", "fakir", "dervish", "meditation"
    ],
    "Sensual & Flirtatious": [
        "flirt", "sexy", "nakhra", "hot", "body", "touch", "curves", "nasha", "tease", "sensual"
    ],
    "Chill & Escapist": [
        "chill", "relax", "breeze", "clouds", "drive", "night", "stars", "calm", "slow", "vibe"
    ]
}

_ANTAGONISTIC_INTENT_PAIRS: dict[frozenset[str], float] = {
    frozenset(["Romantic & Devotional", "Street Confrontation & Defiance"]): 0.35,
    frozenset(["Heartbreak & Melancholy", "Celebration & Party"]): 0.45,
    frozenset(["Introspective & Spiritual", "Street Confrontation & Defiance"]): 0.50,
}


async def fetch_lyrics(artist: str, title: str) -> str:
    """Fetch lyrics from LRCLIB with fast timeout, falling back gracefully."""
    cache_key = f"{artist.lower()}:{title.lower()}"
    if cache_key in _lyrics_cache:
        return _lyrics_cache[cache_key]

    clean_artist = re.split(r",|&|feat\.|ft\.", artist, flags=re.IGNORECASE)[0].strip()
    clean_title = re.sub(r"\(.*?\)|\[.*?\]", "", title).strip()

    try:
        url = f"https://lrclib.net/api/get?artist_name={urllib.parse.quote(clean_artist)}&track_name={urllib.parse.quote(clean_title)}"
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get(url, headers={"User-Agent": "Echo/1.0 (https://github.com/shreyash616/echo)"})
            if resp.status_code == 200:
                data = resp.json()
                lyrics = data.get("plainLyrics") or data.get("syncedLyrics") or ""
                _lyrics_cache[cache_key] = lyrics
                return lyrics
    except Exception as exc:
        logger.debug("Lyrics fetch skipped for %s - %s: %s", clean_artist, clean_title, exc)

    _lyrics_cache[cache_key] = ""
    return ""


def classify_lyrical_intent(title: str = "", artist: str = "", lyrics: str = "", genre: str = "") -> str:
    """Classify the primary human emotional intent & theme of a track."""
    text = f"{title} {artist} {genre} {lyrics}".lower()
    scores: dict[str, int] = {}
    for theme, keywords in _THEME_KEYWORDS.items():
        score = sum(len(re.findall(r"\b" + re.escape(kw) + r"\b", text)) for kw in keywords)
        scores[theme] = score

    best_theme = max(scores, key=scores.get)
    if scores[best_theme] > 0:
        return best_theme

    # Fallback to title keywords & genre heuristics
    if any(k in text for k in ("love", "heart", "tere", "sanware", "meri", "deewani", "baby", "darling")):
        return "Romantic & Devotional"
    if any(k in text for k in ("gang", "drill", "war", "killa", "don", "beef", "vairi")):
        return "Street Confrontation & Defiance"
    if any(k in text for k in ("sad", "tears", "blue", "cry", "lost", "broken")):
        return "Heartbreak & Melancholy"
    if any(k in text for k in ("party", "club", "dance", "remix", "bhangra", "dhol")):
        return "Celebration & Party"
    return "Chill & Atmospheric"


def lyrical_intent_similarity(theme_a: str, theme_b: str) -> tuple[float, float]:
    """
    Returns (linear_similarity, clash_multiplier):
      - Exact theme match: 1.0, 1.0
      - Neutral / compatible: 0.70 - 0.85, 1.0
      - Antagonistic emotional clash (e.g. Romance vs Street Beef): 0.10, 0.35 multiplier
    """
    if not theme_a or not theme_b or theme_a == "Chill & Atmospheric" or theme_b == "Chill & Atmospheric":
        return 0.75, 1.0
    if theme_a == theme_b:
        return 1.0, 1.0

    pair = frozenset([theme_a, theme_b])
    if pair in _ANTAGONISTIC_INTENT_PAIRS:
        return 0.10, _ANTAGONISTIC_INTENT_PAIRS[pair]

    return 0.70, 1.0
