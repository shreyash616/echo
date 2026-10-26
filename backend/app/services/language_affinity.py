"""
Language Affinity Service for Echo (Section 5 Table v2).
Encodes linguistic distances as soft affinity scores in [0.0, 1.0].
Used for soft language ranking prior and diversity budgets.
"""
from __future__ import annotations

import json
import os

_GROUPS: dict[str, set[str]] = {
    "indic_north": {"hi", "pa", "ur", "bn", "mr", "gu"},
    "indic_south": {"ta", "te", "ml", "kn"},
    "romance":     {"es", "pt", "it", "fr", "ca"},
    "east_asia":   {"ko", "ja", "zh"},
    "germanic":    {"en", "de", "nl", "sv", "no", "da"},
    "arabic":      {"ar", "fa", "tr"},
}

_SPECIFIC_PAIRS: dict[frozenset[str], float] = {
    frozenset({"hi", "pa"}): 0.90,
    frozenset({"hi", "ur"}): 0.90,
    frozenset({"pa", "ur"}): 0.90,
}

_CROSS_GROUP: dict[frozenset[str], float] = {
    frozenset({"germanic", "indic_north"}):    0.70,
    frozenset({"germanic", "romance"}):        0.70,
    frozenset({"germanic", "east_asia"}):      0.60,
    frozenset({"romance", "arabic"}):          0.60,
    frozenset({"indic_north", "indic_south"}): 0.55,
}

SAME = 1.00
SAME_GROUP = 0.85
DEFAULT_CROSS = 0.55
UNKNOWN = 0.75

_LEARNED_AFFINITY: dict[str, float] | None = None
_LEARNED_PATH = os.path.join("ml", "inference", "language_affinity.json")


def _load_learned():
    global _LEARNED_AFFINITY
    if _LEARNED_AFFINITY is None and os.path.exists(_LEARNED_PATH):
        try:
            with open(_LEARNED_PATH, "r", encoding="utf-8") as f:
                _LEARNED_AFFINITY = json.load(f)
        except Exception:
            _LEARNED_AFFINITY = {}


def _group_of(lang: str) -> str | None:
    l_lower = lang.lower().strip()
    for grp, members in _GROUPS.items():
        if l_lower in members:
            return grp
    return None


def language_affinity(seed: str | None, cand: str | None) -> float:
    """
    Calculate soft affinity between two language codes (Section 5 Table v2).
    1.00 = same language
    0.90 = pa <-> hi <-> ur
    0.85 = same family/group
    0.70 = en <-> indic_north, en <-> romance
    0.60 = en <-> east_asia, romance <-> arabic
    0.55 = indic_north <-> indic_south, default cross-group
    0.75 = unknown on either side
    """
    if not seed or not cand:
        return UNKNOWN
    s_clean = seed.lower().strip()
    c_clean = cand.lower().strip()
    if s_clean == c_clean:
        return SAME

    _load_learned()
    if _LEARNED_AFFINITY:
        key = f"{s_clean}:{c_clean}"
        rev_key = f"{c_clean}:{s_clean}"
        if key in _LEARNED_AFFINITY:
            return float(_LEARNED_AFFINITY[key])
        if rev_key in _LEARNED_AFFINITY:
            return float(_LEARNED_AFFINITY[rev_key])

    pair = frozenset({s_clean, c_clean})
    if pair in _SPECIFIC_PAIRS:
        return _SPECIFIC_PAIRS[pair]

    gs = _group_of(s_clean)
    gc = _group_of(c_clean)
    if gs and gs == gc:
        return SAME_GROUP
    if gs and gc:
        return _CROSS_GROUP.get(frozenset({gs, gc}), DEFAULT_CROSS)
    return DEFAULT_CROSS
