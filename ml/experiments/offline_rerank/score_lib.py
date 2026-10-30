"""Offline replica of recommender._compute_pair_score and the union candidate pool.

Reads the served artifacts from ECHO_ARTIFACT_DIR (default ml/inference) and must reproduce the
live debug=1 axes exactly; report.py --validate-live checks that. Any change to the scoring in
recommender.py must be mirrored here, and the validation re-run, before a weight search.
"""
import json, os, re
import numpy as np
import faiss

HF = os.environ.get("ECHO_ARTIFACT_DIR", "ml/inference")
AXES = ["s_clap", "s_anchor", "s_tempo", "s_pulse", "s_groove", "s_dark", "s_grit", "s_vocal",
        "s_intimacy", "s_restraint", "s_meaning", "s_language", "s_harm", "s_mode"]
AUDIO_KEYS = AXES[:10]

# ---- language affinity (copied semantics from backend/app/services/language_affinity.py, no learned file) ----
_GROUPS = {
    "indic_north": {"hi", "pa", "ur", "bn", "mr", "gu"},
    "indic_south": {"ta", "te", "ml", "kn"},
    "romance": {"es", "pt", "it", "fr", "ca"},
    "east_asia": {"ko", "ja", "zh"},
    "germanic": {"en", "de", "nl", "sv", "no", "da"},
    "arabic": {"ar", "fa", "tr"},
}
_SPECIFIC = {frozenset({"hi", "pa"}): 0.90, frozenset({"hi", "ur"}): 0.90, frozenset({"pa", "ur"}): 0.90}
_CROSS = {
    frozenset({"germanic", "indic_north"}): 0.70, frozenset({"germanic", "romance"}): 0.70,
    frozenset({"germanic", "east_asia"}): 0.60, frozenset({"romance", "arabic"}): 0.60,
    frozenset({"indic_north", "indic_south"}): 0.55,
}

def _group_of(l):
    l = l.lower().strip()
    for g, m in _GROUPS.items():
        if l in m:
            return g
    return None

def language_affinity(s, c):
    if not s or not c:
        return 0.75
    s, c = s.lower().strip(), c.lower().strip()
    if s == c:
        return 1.0
    p = frozenset({s, c})
    if p in _SPECIFIC:
        return _SPECIFIC[p]
    gs, gc = _group_of(s), _group_of(c)
    if gs and gs == gc:
        return 0.85
    if gs and gc:
        return _CROSS.get(frozenset({gs, gc}), 0.55)
    return 0.55

def _norm_text(s):
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()

def _norm_artist(s):
    art = re.split(r",|\s+feat\.|\s+ft\.|\s+with\s+|&", str(s), flags=re.IGNORECASE)[0]
    return _norm_text(art)

def make_track_key(artist, title):
    return f"{_norm_artist(artist)}:{_norm_text(title)}"


TAU = 1.0  # T2, as recommender.TAU
MODE_SHRINK = 0.5  # as recommender.MODE_SHRINK


def _zsim(a, b, std):
    return float(np.exp(-abs(a - b) / (TAU * max(std, 1e-6))))


def _fold(x):
    if abs(x - 1.0) <= 0.25:
        return min(x, 0.4)
    if abs(x - 0.5849625) <= 0.15:
        return 0.7
    return min(1.0, x)


def features(seed, cand, clap_a, clap_b, fs=None):
    """fs: the metadata header's feature_std (T2); a missing axis falls back to std 1.0 as the Space does."""
    fs = fs or {}
    f = {}
    if len(clap_a) and len(clap_b) == len(clap_a) and np.linalg.norm(clap_a) > 0 and np.linalg.norm(clap_b) > 0:
        f["s_clap"] = float(np.clip(np.dot(clap_a, clap_b), 0.0, 1.0))
    else:
        f["s_clap"] = 0.5
    pa, pb = seed.get("anchor_profile"), cand.get("anchor_profile")
    if pa and pb and len(pa) == len(pb):
        pa = np.array(pa, dtype=np.float32); pb = np.array(pb, dtype=np.float32)
        na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
        f["s_anchor"] = float(np.clip(0.5 * (float(np.dot(pa, pb) / (na * nb)) + 1.0), 0, 1)) if na > 0 and nb > 0 else 0.5
    else:
        f["s_anchor"] = 0.5
    da, db = seed.get("felt_bpm_dist") or {}, cand.get("felt_bpm_dist") or {}
    if da and db:
        d = 0.0
        for ka, p_a in da.items():
            try: ba = float(ka)
            except ValueError: continue
            for kb, p_b in db.items():
                try: bb = float(kb)
                except ValueError: continue
                if ba > 0 and bb > 0:
                    d += float(p_a) * float(p_b) * _fold(abs(float(np.log2(ba / bb))))
        f["s_tempo"] = float(np.clip(1.0 - d, 0, 1))
    else:
        ba = float(seed.get("felt_bpm") or seed.get("bpm", 120)); bb = float(cand.get("felt_bpm") or cand.get("bpm", 120))
        f["s_tempo"] = float(np.clip(1.0 - _fold(abs(float(np.log2(ba / bb)))), 0, 1)) if ba > 0 and bb > 0 else 0.5
    g = lambda r, k, dft: float(r.get(k, dft))
    z = lambda k, dft: _zsim(g(seed, k, dft), g(cand, k, dft), fs.get(k, 1.0))
    f["s_pulse"] = z("pulse_rate", .5)
    kc = 1.0 if seed.get("kick_class") == cand.get("kick_class") else 0.0
    bb_sim = z("backbeat", .5)
    hpb = z("hits_per_bar", 4.0)
    lag = z("lag_ms", 0.0)
    f["s_groove"] = float(np.clip((0.4 * kc + 0.2 * bb_sim + 0.2 * hpb + 0.1 * lag) / 0.9, 0, 1))  # T3.5, no swing
    f["s_dark"] = z("darkness", .5)
    f["s_grit"] = z("grit", .5)
    syl = z("syllable_rate", 2.0)
    sus = z("sustain_ratio", .5)
    pit = z("pitch_range", 12.0)
    tun = 1.0 - abs(g(seed, "tuning_depth", .5) - g(cand, "tuning_depth", .5))  # no T2 std, linear as the Space
    vib = z("vibrato", 20.0)
    wa = float(seed.get("lyric_words_per_minute") or 0.0); wb = float(cand.get("lyric_words_per_minute") or 0.0)
    parts = [syl, sus, pit, tun, vib]
    if wa > 0 and wb > 0:
        parts.append(max(0.0, 1.0 - abs(wa - wb) / 100.0))
    f["s_vocal"] = float(np.clip(np.mean(parts), 0, 1))
    f["s_intimacy"] = z("intimacy", .5)
    dz_lr = abs(g(seed, "loudness_range", 0.0) - g(cand, "loudness_range", 0.0)) / max(fs.get("loudness_range", 1.0), 1e-6)
    dz_cf = abs(g(seed, "crest_factor", 0.0) - g(cand, "crest_factor", 0.0)) / max(fs.get("crest_factor", 1.0), 1e-6)
    f["s_restraint"] = float(np.exp(-(dz_lr + dz_cf) / 2.0))
    ta, tb = seed.get("theme_vec"), cand.get("theme_vec")
    both = bool(seed.get("has_lyrics") and cand.get("has_lyrics") and ta and tb and len(ta) == len(tb))
    if both:
        ta = np.array(ta, dtype=np.float32); tb = np.array(tb, dtype=np.float32)
        na, nb = np.linalg.norm(ta), np.linalg.norm(tb)
        f["s_meaning"] = float(np.clip(0.5 * (float(np.dot(ta, tb) / (na * nb)) + 1.0), 0, 1)) if na > 0 and nb > 0 else 0.5
    else:
        f["s_meaning"] = 0.5
    f["s_language"] = language_affinity(seed.get("language"), cand.get("language"))
    cpb = z("changes_per_bar", 1.0)
    lb = 1.0 if seed.get("loop_bars") == cand.get("loop_bars") else 0.0
    riff = 1.0 if bool(seed.get("riff_flag")) == bool(cand.get("riff_flag")) else 0.0
    f["s_harm"] = float(np.clip(0.4 * cpb + 0.4 * lb + 0.2 * riff, 0, 1))
    sim_pol = z("mode_polarity", 0.0)
    if seed.get("mode_undefined") or cand.get("mode_undefined"):
        f["s_mode"] = 0.7 + MODE_SHRINK * (sim_pol - 0.7)  # T3.4 as amended a0d9b43
    else:
        f["s_mode"] = sim_pol
    return f, both


def effective_weights(w, both):
    w_eff = dict(w)
    if not both and w_eff.get("s_meaning", 0.0) > 0.0:
        m = w_eff.pop("s_meaning")
        sa = sum(w_eff.get(k, 0.0) for k in AUDIO_KEYS)
        if sa > 0:
            for k in AUDIO_KEYS:
                w_eff[k] = w_eff.get(k, 0.0) + m * (w_eff.get(k, 0.0) / sa)
    return w_eff


def score_matrix(F, both_flags, w):
    """F: (n,14) feature matrix, both_flags: (n,) bool. Returns clipped scores."""
    wv = np.array([w.get(k, 0.0) for k in AXES], dtype=np.float64)
    we = effective_weights(w, False)
    wv_nolyr = np.array([we.get(k, 0.0) for k in AXES], dtype=np.float64)
    s = np.where(both_flags, F @ wv, F @ wv_nolyr)
    return np.clip(s, 0.05, 0.99)


class Catalogue:
    def __init__(self):
        raw = json.load(open(f"{HF}/track_metadata.json"))
        self.header = raw["header"]
        self.meta = raw["tracks"]
        self.index = faiss.read_index(f"{HF}/music_index.faiss")
        self.clap = self.index.reconstruct_n(0, self.index.ntotal)
        self.feel_matrix = np.load(f"{HF}/feel_matrix.npy")
        self.feel_index = faiss.read_index(f"{HF}/feel_index.faiss")
        self.id_map = {}
        for i, r in enumerate(self.meta):
            if r.get("id"):
                self.id_map[str(r["id"])] = i
            self.id_map[r.get("key") or make_track_key(r.get("artist", ""), r.get("title", ""))] = i
            if r.get("deezer_id"):
                self.id_map[str(r["deezer_id"])] = i
            if r.get("itunes_id"):
                self.id_map[str(r["itunes_id"])] = i
                self.id_map[f"it_{r['itunes_id']}"] = i
        self.weights_live = json.load(open(f"{HF}/rerank_weights.json"))
        self.feature_std = self.header.get("feature_std") or {}
        self.k_clap = 300  # as config.candidates_clap
        self.k_feel = 1500  # as config.candidates_feel

    def pool(self, seed_key, k_clap=None, k_feel=None):
        k_clap = k_clap or self.k_clap
        k_feel = k_feel or self.k_feel
        si = self.id_map[seed_key]
        _, I = self.index.search(self.clap[si:si + 1], min(k_clap, self.index.ntotal))
        clap_idx = {int(i) for i in I[0] if i >= 0}
        _, If = self.feel_index.search(self.feel_matrix[si:si + 1].astype(np.float32), min(k_feel, self.feel_index.ntotal))
        feel_idx = {int(i) for i in If[0] if i >= 0}
        out = []
        for idx in clap_idx | feel_idx:
            r = self.meta[idx]
            key = r.get("key") or make_track_key(r.get("artist", ""), r.get("title", ""))
            if key == seed_key:
                continue
            src = "both" if (idx in clap_idx and idx in feel_idx) else ("clap" if idx in clap_idx else "feel")
            out.append((idx, src))
        return si, out

    def feature_table(self, seed_key):
        si, pool = self.pool(seed_key)
        seed = self.meta[si]
        rows, flags, idxs, srcs = [], [], [], []
        for idx, src in pool:
            f, both = features(seed, self.meta[idx], self.clap[si], self.clap[idx], self.feature_std)
            rows.append([f[k] for k in AXES]); flags.append(both); idxs.append(idx); srcs.append(src)
        return si, np.array(idxs), np.array(srcs), np.array(rows, dtype=np.float64), np.array(flags, dtype=bool)
