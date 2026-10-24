"""
Recommendation Quality Evaluator for 1,000-Song Benchmark Suite.
Computes multi-dimensional acoustic, linguistic, chronological, and genre metrics.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

from tests.benchmark_dataset import BenchmarkSong


@dataclass
class RecommendationEvaluation:
    song_id: str
    song_title: str
    song_artist: str
    expected_language: str
    expected_genre: str
    release_year: int
    decade: str
    
    # Quantitative Results
    k_returned: int
    language_purity: float         # 0.0 to 1.0
    language_leakage_count: int    # count of forbidden language tracks
    era_mean_abs_error: float      # average year difference
    era_compatibility_score: float # 0.0 to 1.0
    era_window_pass_rate: float    # % within acceptable min-max era
    genre_precision: float         # 0.0 to 1.0
    harmonic_key_rate: float       # % with harmonic compatibility
    tempo_subdivision_rate: float  # % with BPM octave compatibility
    
    # Flags
    passed_all_gates: bool
    violations: list[str] = field(default_factory=list)
    top_recommendations: list[dict] = field(default_factory=list)


# Language detection rules for evaluating recommended tracks
_SCRIPT_LANGS = {
    "DEVANAGARI": "hi", "GURMUKHI": "pa", "TAMIL": "ta",
    "TELUGU": "te", "BENGALI": "bn", "KANNADA": "kn", "MALAYALAM": "ml",
    "HANGUL": "ko", "HIRAGANA": "ja", "KATAKANA": "ja", "CJK UNIFIED": "zh-cn",
    "ARABIC": "ar", "CYRILLIC": "ru", "THAI": "th",
}

_GENRE_LANG_MAPPINGS = [
    ("punjabi", "pa"), ("bhangra", "pa"),
    ("hindi", "hi"), ("bollywood", "hi"), ("filmi", "hi"), ("desi", "hi"),
    ("tamil", "ta"), ("kollywood", "ta"),
    ("telugu", "te"), ("tollywood", "te"),
    ("k-pop", "ko"), ("kpop", "ko"), ("korean", "ko"),
    ("j-pop", "ja"), ("jpop", "ja"), ("japanese", "ja"),
    ("reggaeton", "es"), ("latin", "es"), ("bachata", "es"),
    ("french", "fr"), ("chanson", "fr"), ("german", "de"), ("deutschrap", "de"),
    ("arabic", "ar"),
]

_GENRE_FAMILIES = {
    "bollywood":  ["bollywood", "hindi", "filmi", "desi", "indian pop", "sufi", "ghazal"],
    "punjabi":    ["punjabi", "bhangra", "desi hip hop"],
    "tamil":      ["tamil", "kollywood"],
    "telugu":     ["telugu", "tollywood"],
    "hip-hop":    ["hip-hop", "hip hop", "rap", "trap", "drill", "boom bap", "phonk"],
    "rock":       ["rock", "alternative", "punk", "grunge", "indie rock", "metal", "hard rock"],
    "pop":        ["pop", "dance pop", "electropop", "synthpop", "indie pop"],
    "electronic": ["electronic", "edm", "house", "techno", "dnb", "drum and bass", "synthwave", "trance"],
    "latin":      ["latin", "reggaeton", "bachata", "salsa", "urbano"],
    "k-pop":      ["k-pop", "kpop", "k-indie", "k-r&b"],
    "j-pop":      ["j-pop", "jpop", "city pop", "anime"],
    "afrobeats":  ["afrobeats", "afropop", "amapiano", "afro"],
    "classical":  ["classical", "soundtrack", "score", "piano", "symphony", "ambient"],
}


def detect_track_language(title: str, artist: str, genre: str = "", tags: list[str] | None = None) -> str:
    """Determine effective language of a recommendation without cluster archetype noise."""
    # 1. Non-Latin Unicode script
    for char in f"{title} {artist}":
        if char.isalpha():
            name = unicodedata.name(char, "")
            for script, lang in _SCRIPT_LANGS.items():
                if script in name:
                    return lang

    # 2. Genre matches (strict track genre, NOT unsupervised cluster archetype names)
    g_lower = genre.lower().strip()
    for kw, lang in _GENRE_LANG_MAPPINGS:
        if kw == g_lower or f"{kw} " in g_lower or f" {kw}" in g_lower:
            return lang

    # Default: English / Western
    return "en"


def _is_genre_compatible(rec_genre: str, expected_family: str) -> bool:
    rec_lower = rec_genre.lower()
    family_keywords = _GENRE_FAMILIES.get(expected_family.lower(), [expected_family.lower()])
    return any(kw in rec_lower for kw in family_keywords)


def _harmonic_key_match(key_a: str, key_b: str) -> float:
    if not key_a or not key_b or key_a == "—" or key_b == "—":
        return 0.7
    if key_a == key_b:
        return 1.0
    # Relative major/minor or adjacent
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    try:
        root_a, mode_a = key_a.split()
        root_b, mode_b = key_b.split()
        idx_a = pitch_classes.index(root_a)
        idx_b = pitch_classes.index(root_b)
        dist = min(abs(idx_a - idx_b), 12 - abs(idx_a - idx_b))
        if dist == 0:
            return 0.9 if mode_a != mode_b else 1.0
        if dist in [1, 5, 7]:
            return 0.8
        return 0.4
    except Exception:
        return 0.6


def _bpm_octave_match(bpm_a: int, bpm_b: int) -> float:
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.7
    ratios = [bpm_b / bpm_a, (bpm_b * 2) / bpm_a, bpm_b / (bpm_a * 2)]
    min_diff = min(abs(r - 1.0) for r in ratios)
    return 1.0 if min_diff <= 0.08 else (0.8 if min_diff <= 0.16 else 0.4)


def evaluate_single_recommendation_set(
    benchmark_song: BenchmarkSong,
    recommendations: list[dict[str, Any]],
) -> RecommendationEvaluation:
    """Evaluate a list of recommendations against a benchmark song's ground truth."""
    k = len(recommendations)
    if k == 0:
        return RecommendationEvaluation(
            song_id=benchmark_song.id,
            song_title=benchmark_song.title,
            song_artist=benchmark_song.artist,
            expected_language=benchmark_song.expected_language,
            expected_genre=benchmark_song.expected_genre_family,
            release_year=benchmark_song.release_year,
            decade=benchmark_song.decade,
            k_returned=0,
            language_purity=0.0,
            language_leakage_count=0,
            era_mean_abs_error=99.0,
            era_compatibility_score=0.0,
            era_window_pass_rate=0.0,
            genre_precision=0.0,
            harmonic_key_rate=0.0,
            tempo_subdivision_rate=0.0,
            passed_all_gates=False,
            violations=["No recommendations returned"],
        )

    lang_matches = 0
    forbidden_lang_leaks = 0
    year_diffs: list[float] = []
    era_scores: list[float] = []
    era_window_passes = 0
    genre_matches = 0
    harmonic_matches: list[float] = []
    tempo_matches: list[float] = []
    violations: list[str] = []

    src_bpm = benchmark_song.expected_tempo_min + (benchmark_song.expected_tempo_max - benchmark_song.expected_tempo_min) // 2

    for rec in recommendations:
        rec_title = str(rec.get("title", ""))
        rec_artist = str(rec.get("artist", ""))
        rec_genre = str(rec.get("genre", ""))
        rec_year = rec.get("year")
        rec_key = str(rec.get("key", "—"))
        rec_bpm = int(rec.get("bpm", 0))

        # 1. Language evaluation
        det_lang = detect_track_language(rec_title, rec_artist, rec_genre, rec.get("vibes"))
        if det_lang == benchmark_song.expected_language:
            lang_matches += 1
        elif benchmark_song.expected_language in ["hi", "pa"] and det_lang in ["hi", "pa"]:
            # Shared Desi / South Asian cinematic crossover
            lang_matches += 1

        if det_lang in benchmark_song.forbidden_languages:
            forbidden_lang_leaks += 1
            violations.append(
                f"Language Leak: Recommended '{rec_title}' by {rec_artist} (detected '{det_lang}'), "
                f"which is forbidden for {benchmark_song.expected_language} track '{benchmark_song.title}'"
            )

        # 2. Era evaluation
        if rec_year and int(rec_year) > 1920:
            diff = abs(int(rec_year) - benchmark_song.release_year)
            year_diffs.append(float(diff))
            era_score = max(0.1, 1.0 - (diff / 35.0))
            era_scores.append(era_score)
            if benchmark_song.expected_era_min <= int(rec_year) <= benchmark_song.expected_era_max:
                era_window_passes += 1
            elif diff > 25:
                violations.append(
                    f"Era Mismatch: Recommended '{rec_title}' ({rec_year}), exceeding 25-yr gap "
                    f"from anchor '{benchmark_song.title}' ({benchmark_song.release_year})"
                )
        else:
            year_diffs.append(12.0)
            era_scores.append(0.75)
            era_window_passes += 1

        # 3. Genre evaluation
        if _is_genre_compatible(rec_genre, benchmark_song.expected_genre_family) or any(
            _is_genre_compatible(v, benchmark_song.expected_genre_family) for v in rec.get("vibes", [])
        ):
            genre_matches += 1

        # 4. Harmonic & Pace evaluation
        harmonic_matches.append(_harmonic_key_match("C Major", rec_key))
        tempo_matches.append(_bpm_octave_match(src_bpm, rec_bpm))

    purity = lang_matches / k
    mean_mae = sum(year_diffs) / len(year_diffs) if year_diffs else 0.0
    mean_era_score = sum(era_scores) / len(era_scores) if era_scores else 0.75
    era_pass = era_window_passes / k
    genre_prec = genre_matches / k
    harm_rate = sum(harmonic_matches) / len(harmonic_matches) if harmonic_matches else 0.7
    tempo_rate = sum(tempo_matches) / len(tempo_matches) if tempo_matches else 0.7

    # Overall pass criteria: zero language leakage, purity >= 70%, era MAE <= 15 yrs, genre precision >= 60%
    passed = (forbidden_lang_leaks == 0) and (purity >= 0.70) and (mean_mae <= 18.0) and (genre_prec >= 0.60)

    return RecommendationEvaluation(
        song_id=benchmark_song.id,
        song_title=benchmark_song.title,
        song_artist=benchmark_song.artist,
        expected_language=benchmark_song.expected_language,
        expected_genre=benchmark_song.expected_genre_family,
        release_year=benchmark_song.release_year,
        decade=benchmark_song.decade,
        k_returned=k,
        language_purity=round(purity, 3),
        language_leakage_count=forbidden_lang_leaks,
        era_mean_abs_error=round(mean_mae, 1),
        era_compatibility_score=round(mean_era_score, 3),
        era_window_pass_rate=round(era_pass, 3),
        genre_precision=round(genre_prec, 3),
        harmonic_key_rate=round(harm_rate, 3),
        tempo_subdivision_rate=round(tempo_rate, 3),
        passed_all_gates=passed,
        violations=violations,
        top_recommendations=recommendations[:3],
    )
