"""
ml/training/learn_affinity.py: Step R7 Language Affinity from PMI (Sections 11.8.3, 12.6).

Computes Pointwise Mutual Information (PMI) between language pairs over playlist
co-membership:
  PMI(a, b) = log( P(a, b) / (P(a) * P(b)) )
Mapped to [0.50, 0.95] by rank within pairs, blended 50/50 with Section 5 table.
Outputs: ml/inference/language_affinity.json
"""
from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

# Add backend to path
sys.path.insert(0, "backend")
from app.services.recommender import make_track_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("learn_affinity")

CATALOG_FEATURES_PATH = Path("ml/data/catalog_features.jsonl")
PLAYLIST_MEMBERSHIP_PATH = Path("ml/data/playlist_membership.jsonl")
OUT_AFFINITY_PATH = Path("ml/inference/language_affinity.json")

# Baseline Section 5 table (Table v2)
BASE_TABLE_V2: dict[tuple[str, str], float] = {
    ("pa", "hi"): 0.90, ("hi", "pa"): 0.90,
    ("hi", "ur"): 0.90, ("ur", "hi"): 0.90,
    ("pa", "ur"): 0.90, ("ur", "pa"): 0.90,
    ("es", "pt"): 0.85, ("pt", "es"): 0.85,
    ("es", "it"): 0.85, ("it", "es"): 0.85,
    ("pt", "it"): 0.85, ("it", "pt"): 0.85,
    ("fr", "it"): 0.85, ("it", "fr"): 0.85,
    ("fr", "es"): 0.85, ("es", "fr"): 0.85,
    ("fr", "pt"): 0.85, ("pt", "fr"): 0.85,
    ("ta", "te"): 0.85, ("te", "ta"): 0.85,
    ("ta", "kn"): 0.85, ("kn", "ta"): 0.85,
    ("ta", "ml"): 0.85, ("ml", "ta"): 0.85,
    ("te", "kn"): 0.85, ("kn", "te"): 0.85,
    ("te", "ml"): 0.85, ("ml", "te"): 0.85,
    ("kn", "ml"): 0.85, ("ml", "kn"): 0.85,
    ("en", "hi"): 0.70, ("hi", "en"): 0.70,
    ("en", "pa"): 0.70, ("pa", "en"): 0.70,
    ("en", "ur"): 0.70, ("ur", "en"): 0.70,
    ("en", "bn"): 0.70, ("bn", "en"): 0.70,
    ("en", "es"): 0.70, ("es", "en"): 0.70,
    ("en", "fr"): 0.70, ("fr", "en"): 0.70,
    ("en", "pt"): 0.70, ("pt", "en"): 0.70,
    ("en", "it"): 0.70, ("it", "en"): 0.70,
    ("en", "ko"): 0.60, ("ko", "en"): 0.60,
    ("en", "ja"): 0.60, ("ja", "en"): 0.60,
    ("en", "zh-cn"): 0.60, ("zh-cn", "en"): 0.60,
    ("es", "ar"): 0.60, ("ar", "es"): 0.60,
    ("fr", "ar"): 0.60, ("ar", "fr"): 0.60,
    ("hi", "ta"): 0.55, ("ta", "hi"): 0.55,
    ("hi", "te"): 0.55, ("te", "hi"): 0.55,
}


def compute_pmi() -> dict[str, float]:
    logger.info("Computing PMI from catalogue and playlist membership...")

    # 1. Load catalog track languages (keyed by id, deezer_id, and make_track_key)
    track_langs: dict[str, str] = {}
    if CATALOG_FEATURES_PATH.exists():
        with open(CATALOG_FEATURES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        l = rec.get("language") or "en"
                        if l and l != "unknown":
                            if rec.get("id"):
                                track_langs[str(rec["id"])] = str(l)
                            if rec.get("deezer_id"):
                                track_langs[str(rec["deezer_id"])] = str(l)
                            k = rec.get("key") or make_track_key(str(rec.get("artist", "")), str(rec.get("title", "")))
                            track_langs[k] = str(l)
                    except Exception:
                        pass

    logger.info("Indexed languages for %d track references.", len(track_langs))

    # 2. Group playlist tracks by language
    pl_langs: dict[str, list[str]] = {}
    if PLAYLIST_MEMBERSHIP_PATH.exists():
        with open(PLAYLIST_MEMBERSHIP_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    pl_id = str(entry["playlist_id"])
                    did = str(entry.get("deezer_track_id", ""))
                    k = make_track_key(str(entry.get("artist", "")), str(entry.get("title", "")))

                    lang = track_langs.get(k) or (track_langs.get(did) if did else None)
                    if lang:
                        pl_langs.setdefault(pl_id, []).append(lang)
                except Exception:
                    continue

    lang_counts = Counter()
    pair_counts = Counter()
    n_playlists = max(1, len(pl_langs))

    for pl_id, langs in pl_langs.items():
        u_langs = list(set(langs))
        for l in u_langs:
            lang_counts[l] += 1
        for i in range(len(u_langs)):
            for j in range(i + 1, len(u_langs)):
                l1, l2 = sorted([u_langs[i], u_langs[j]])
                pair_counts[(l1, l2)] += 1

    pmi_scores: dict[tuple[str, str], float] = {}
    for (l1, l2), c12 in pair_counts.items():
        p12 = c12 / n_playlists
        p1 = lang_counts[l1] / n_playlists
        p2 = lang_counts[l2] / n_playlists
        pmi = math.log(max(1e-9, p12) / max(1e-9, p1 * p2))
        pmi_scores[(l1, l2)] = pmi

    # Rank-map to [0.50, 0.95]
    sorted_pairs = sorted(pmi_scores.items(), key=lambda x: x[1])
    n_p = len(sorted_pairs)
    pmi_norm: dict[tuple[str, str], float] = {}
    for r, ((l1, l2), _) in enumerate(sorted_pairs):
        frac = r / max(1, n_p - 1) if n_p > 1 else 0.5
        val = 0.50 + 0.45 * frac
        pmi_norm[(l1, l2)] = val
        pmi_norm[(l2, l1)] = val

    # Blend 50/50 with Base Table v2
    final_table: dict[str, float] = {}
    all_keys = set(pmi_norm.keys()) | set(BASE_TABLE_V2.keys())
    for l1, l2 in all_keys:
        v_base = BASE_TABLE_V2.get((l1, l2), 0.55)
        v_pmi = pmi_norm.get((l1, l2), v_base)
        blended = 0.50 * v_base + 0.50 * v_pmi
        final_table[f"{l1}:{l2}"] = round(blended, 4)

    OUT_AFFINITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_AFFINITY_PATH, "w", encoding="utf-8") as f:
        json.dump(final_table, f, indent=2)

    logger.info("Saved %d language pair affinity entries to %s", len(final_table), OUT_AFFINITY_PATH)
    return final_table


if __name__ == "__main__":
    compute_pmi()
