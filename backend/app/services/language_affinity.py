"""
Language Affinity Service for Echo.
Encodes linguistic distances as soft affinity scores in [0.0, 1.0].
Used for soft language ranking prior and diversity budgets.
"""
from __future__ import annotations

# Affinity in [0, 1]. 1.0 = same language.
_GROUPS: dict[str, set[str]] = {
    "indic_north": {"hi", "pa", "ur", "bn", "mr", "gu"},
    "indic_south": {"ta", "te", "ml", "kn"},
    "romance":     {"es", "pt", "it", "fr", "ca"},
    "east_asia":   {"ko", "ja", "zh"},
    "germanic":    {"en", "de", "nl", "sv", "no", "da"},
    "arabic":      {"ar", "fa", "tr"},
}

_CROSS_GROUP: dict[frozenset[str], float] = {
    frozenset({"indic_north", "indic_south"}): 0.70,
    frozenset({"germanic", "romance"}):        0.70,   # en <-> es/fr/pt: Townes -> Manu Chao, Bang Data -> Thee Oh Sees
    frozenset({"germanic", "indic_north"}):    0.60,   # en <-> hi/pa: Punjabi trap-pop borrows from US hip-hop
    frozenset({"germanic", "east_asia"}):      0.60,
    frozenset({"romance", "arabic"}):          0.60,
}

SAME = 1.0
SAME_GROUP = 0.85
DEFAULT_CROSS = 0.50
UNKNOWN = 0.60


def _group_of(lang: str) -> str | None:
    l_lower = lang.lower().strip()
    for grp, members in _GROUPS.items():
        if l_lower in members:
            return grp
    return None


def language_affinity(seed: str | None, cand: str | None) -> float:
    """
    Calculate soft affinity between two language codes.
    1.0 = same language
    0.85 = same family/group
    0.70 / 0.60 = culturally adjacent groups
    0.50 = cross-group default
    0.60 = unknown fallback
    """
    if not seed or not cand:
        return UNKNOWN
    s_clean = seed.lower().strip()
    c_clean = cand.lower().strip()
    if s_clean == c_clean:
        return SAME
    gs = _group_of(s_clean)
    gc = _group_of(c_clean)
    if gs and gs == gc:
        return SAME_GROUP
    if gs and gc:
        return _CROSS_GROUP.get(frozenset({gs, gc}), DEFAULT_CROSS)
    return DEFAULT_CROSS
