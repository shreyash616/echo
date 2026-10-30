"""
ml/training/search_weights.py: T7 weight search on the replica (FIX_PLAN_2.md T7, P8 GO c63b103).

  ECHO_ARTIFACT_DIR=ml/inference python ml/training/search_weights.py

Every evaluation ranks, per TUNE seed, the post-MMR top 20 the Space would return: the union pool
(CLAP 300, feel 3000, the seed and copies of it dropped), score_lib.score_pool's axes,
score_lib.mmr_select (diversity 0.70, artist cap 2, cross-language budget); hits by
feel_benchmark.match_track's rule. No HTTP, no recommender.

- Objective: 0.5 x hits@20 + 0.3 x MRR + 0.2 x hits@10, each over all 81 TUNE positives (a positive
  not returned is a miss), minus 0.5 per negative in the top 10.
- Constraints on the rerank weights: on the simplex, no axis above 0.25, s_clap + s_anchor <= 0.25,
  s_language <= 0.08, s_mode <= 0.05.
- Block 1: 20 restarts (the served weights projected onto the constraints, then 19 Dirichlet draws
  inside them) with coordinate ascent from each. Block 2: the feel-pool weights by coordinate
  ascent under the best rerank weights. Block 3: the rerank weights again under the new pool.
- Every evaluation is appended to ml/training/weight_search_log.jsonl; the chosen point is a
  "chosen" record (build_index_v2.py --feel-weights-from reads its feel weights into the header);
  HOLDOUT is evaluated once, after it, as the last line.
- Writes ml/inference/rerank_weights.json and ml/inference/WEIGHTS_REPORT.md.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, os.path.abspath("ml/experiments/offline_rerank"))
sys.path.insert(0, os.path.abspath("tests"))
from score_lib import AXES, Catalogue, _norm_artist, make_track_key, mmr_select, score_matrix, score_pool  # noqa: E402
from feel_benchmark import match_track, normalize_title  # noqa: E402

BENCHMARK_PATH = Path("tests/feel_benchmark_v2.json")
WEIGHTS_PATH = Path("ml/inference/rerank_weights.json")
LOG_PATH = Path("ml/training/weight_search_log.jsonl")
REPORT_PATH = Path("ml/inference/WEIGHTS_REPORT.md")

CAPS = {a: 0.25 for a in AXES}
CAPS["s_language"] = 0.08
CAPS["s_mode"] = 0.05
PAIR_CAP = 0.25  # s_clap + s_anchor
N_RESTARTS = 20
RERANK_STEPS = [0.04, 0.02, 0.01]
FEEL_STEPS = [0.2, 0.1]
FEEL_BOUNDS = (0.1, 3.0)
MAX_PASSES = 60
EPS = 1e-9


# ---- constraints --------------------------------------------------------------------------------
def feasible(w: dict) -> bool:
    if abs(sum(w.values()) - 1.0) > 1e-6:
        return False
    if any(w[a] < -EPS or w[a] > CAPS[a] + EPS for a in AXES):
        return False
    return w["s_clap"] + w["s_anchor"] <= PAIR_CAP + EPS


def project(w: dict) -> dict:
    """Nearest-in-spirit feasible point: clip to the caps, scale s_clap + s_anchor down to the pair
    cap, and hand the freed mass to the axes still under their caps in proportion; repeat."""
    v = {a: max(0.0, float(w.get(a, 0.0))) for a in AXES}
    tot = sum(v.values()) or 1.0
    v = {a: x / tot for a, x in v.items()}
    for _ in range(200):
        fixed = set()
        for a in AXES:
            if v[a] > CAPS[a]:
                v[a] = CAPS[a]
                fixed.add(a)
        pair = v["s_clap"] + v["s_anchor"]
        if pair > PAIR_CAP:
            f = PAIR_CAP / pair
            v["s_clap"] *= f
            v["s_anchor"] *= f
            fixed |= {"s_clap", "s_anchor"}
        free = [a for a in AXES if a not in fixed and v[a] < CAPS[a] - EPS
                and not (a in ("s_clap", "s_anchor") and v["s_clap"] + v["s_anchor"] >= PAIR_CAP - EPS)]
        gap = 1.0 - sum(v.values())
        if abs(gap) < 1e-12:
            break
        base = sum(v[a] for a in free)
        for a in free:
            v[a] += gap * (v[a] / base if base > 0 else 1.0 / len(free))
    return v


def move(w: dict, axis: str, delta: float) -> dict:
    """axis += delta, the other axes rescaled in proportion so the sum stays 1."""
    new = w[axis] + delta
    if new < 0.0 or new > 1.0:
        return {}
    rest = 1.0 - w[axis]
    if rest <= 0:
        return {}
    f = (1.0 - new) / rest
    return {a: (new if a == axis else w[a] * f) for a in AXES}


# ---- match_track over the whole catalogue, from pre-normalised strings ---------------------------
class Matcher:
    """feel_benchmark.match_track(row, target) for every catalogue row at once: equal keys, or title
    containment either way and artist containment either way on normalize_title'd strings."""

    def __init__(self, meta: list[dict]):
        self.keys = [r.get("key") or make_track_key(str(r.get("artist", "")), str(r.get("title", ""))) for r in meta]
        self.titles = [normalize_title(str(r.get("title", ""))) for r in meta]
        self.artists = [normalize_title(str(r.get("artist", ""))) for r in meta]

    def mask(self, target: dict) -> np.ndarray:
        t_key = target.get("key") or make_track_key(str(target.get("artist", "")), str(target.get("title", "")))
        t_t = normalize_title(str(target.get("title", "")))
        t_a = normalize_title(str(target.get("artist", "")))
        return np.array([(k == t_key) or ((t_t in c_t or c_t in t_t) and (t_a in c_a or c_a in t_a))
                         for k, c_t, c_a in zip(self.keys, self.titles, self.artists)], dtype=bool)


# ---- per-seed cache ------------------------------------------------------------------------------
class SeedCache:
    """One seed scored against every catalogue row once; a pool is then a row subset."""

    def __init__(self, cat: Catalogue, matcher: Matcher, entry: dict):
        self.label = f'{entry["seed"]["artist"]} - {entry["seed"]["title"]}'
        self.seed_key = make_track_key(entry["seed"]["artist"], entry["seed"]["title"])
        self.si = cat.id_map[self.seed_key]
        seed = cat.meta[self.si]
        n = len(cat.meta)
        self.F, self.flags = score_pool(seed, cat.meta, cat.clap[self.si], cat.clap, cat.feature_std)
        self.cross = self.F[:, AXES.index("s_language")] < 0.99
        # Catalogue.pool's filter: the seed's id, its key, and match_track(row, seed)
        seed_id = str(seed.get("id", ""))
        ids = np.array([str(r.get("id", "")) == seed_id for r in cat.meta], dtype=bool) if seed_id else np.zeros(n, bool)
        keys = np.array([k == self.seed_key for k in matcher.keys], dtype=bool)
        self.excluded = ids | keys | matcher.mask(seed)
        _, I = cat.index.search(cat.clap[self.si:self.si + 1], min(cat.k_clap, cat.index.ntotal))
        self.clap_idx = {int(i) for i in I[0] if i >= 0}
        self.pos = entry.get("positives", [])
        self.neg = entry.get("negatives", [])
        self.pos_mask = np.stack([matcher.mask(p) for p in self.pos], axis=1) if self.pos else np.zeros((n, 0), bool)
        self.neg_mask = np.stack([matcher.mask(p) for p in self.neg], axis=1) if self.neg else np.zeros((n, 0), bool)
        self.pos_keys = [make_track_key(p["artist"], p["title"]) for p in self.pos]

    def pool(self, feel_idx: set[int]) -> np.ndarray:
        # the order Catalogue.pool iterates, so stable-sort ties break the same way
        return np.array([i for i in self.clap_idx | feel_idx if not self.excluded[i]], dtype=np.int64)


class Replica:
    def __init__(self, cat: Catalogue, entries: list[dict]):
        self.cat = cat
        self.matcher = Matcher(cat.meta)
        self.arts = np.array([_norm_artist(r.get("artist", "")) for r in cat.meta], dtype=object)
        self.seeds = [SeedCache(cat, self.matcher, e) for e in entries
                      if make_track_key(e["seed"]["artist"], e["seed"]["title"]) in cat.id_map]
        h = cat.header
        self.feel_axes = h["feel_axes"]
        self.feel_w0 = np.array([h["feel_weights"][a] for a in self.feel_axes], dtype=np.float32)
        self.z = cat.feel_matrix / self.feel_w0  # standardised feel rows
        self._pools: dict[tuple, list[np.ndarray]] = {}

    def pools(self, feel_w: np.ndarray) -> list[np.ndarray]:
        key = tuple(np.round(feel_w, 6))
        if key not in self._pools:
            if np.array_equal(feel_w.astype(np.float32), self.feel_w0):
                M, index = self.cat.feel_matrix, self.cat.feel_index  # the served index, exactly
            else:
                M = (self.z * feel_w.astype(np.float32)).astype(np.float32)
                index = faiss.IndexFlatL2(M.shape[1])
                index.add(M)
            out = []
            for s in self.seeds:
                _, If = index.search(M[s.si:s.si + 1], min(self.cat.k_feel, index.ntotal))
                out.append(s.pool({int(i) for i in If[0] if i >= 0}))
            self._pools[key] = out
        return self._pools[key]

    def evaluate(self, w: dict, feel_w: np.ndarray) -> dict:
        n_pos = h10 = h20 = negs = pooled = 0
        rr = 0.0
        for s, P in zip(self.seeds, self.pools(feel_w)):
            scores = score_matrix(s.F[P], s.flags[P], w)
            sel = mmr_select(scores, self.cat.clap[P], list(self.arts[P]), s.cross[P])
            top = P[sel]
            pm = s.pos_mask[top]
            for j in range(pm.shape[1]):
                n_pos += 1
                hit = np.flatnonzero(pm[:, j])
                if len(hit):
                    r = int(hit[0]) + 1
                    h10 += r <= 10
                    h20 += r <= 20
                    rr += 1.0 / r
            negs += int(s.neg_mask[top[:10]].any(axis=0).sum())
            pset = set(P.tolist())
            pooled += sum(1 for k in s.pos_keys if self.cat.id_map.get(k) in pset)
        n = max(1, n_pos)
        obj = 0.5 * h20 / n + 0.3 * rr / n + 0.2 * h10 / n - 0.5 * negs
        return {"objective": obj, "hits@10": h10, "hits@20": h20, "mrr": rr / n, "negatives@10": negs,
                "positives": n_pos, "pooled": pooled}


# ---- search --------------------------------------------------------------------------------------
def main():
    t_start = time.time()
    entries = json.load(open(BENCHMARK_PATH, encoding="utf-8"))
    tune = [e for e in entries if e.get("split") == "tune"]
    holdout = [e for e in entries if e.get("split") == "holdout"]
    cat = Catalogue()
    served = {a: float(cat.weights_live.get(a, 0.0)) for a in AXES}
    print(f"pools {cat.k_clap} / {cat.k_feel}; building the TUNE cache ...", flush=True)
    rep = Replica(cat, tune)

    # self-check: the vectorised match equals feel_benchmark.match_track on the first seed's pool
    s0, P0 = rep.seeds[0], rep.pools(rep.feel_w0)[0]
    for j, p in enumerate(s0.pos):
        assert all(bool(s0.pos_mask[i, j]) == match_track(cat.meta[i], p) for i in P0), "Matcher disagrees with match_track"
    print(f"cache ready for {len(rep.seeds)} seeds in {time.time() - t_start:.0f} s; Matcher agrees with match_track", flush=True)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, "w", encoding="utf-8")
    n_evals = 0

    def record(kind, w, fw, m, restart=0, step=0, extra=None):
        nonlocal n_evals
        n_evals += 1
        row = {"n": n_evals, "eval_type": kind, "split": "holdout" if kind == "holdout" else "tune",
               "restart_id": restart, "step_id": step, "weights": {a: round(w[a], 6) for a in AXES},
               "feel_weights": {a: round(float(x), 4) for a, x in zip(rep.feel_axes, fw)},
               "feasible": feasible(w), **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()},
               "t": round(time.time() - t_start, 1)}
        if extra:
            row.update(extra)
        log.write(json.dumps(row) + "\n")
        log.flush()

    def ev(kind, w, fw, restart=0, step=0):
        m = rep.evaluate(w, fw)
        record(kind, w, fw, m, restart, step)
        return m

    fw0 = rep.feel_w0.astype(np.float64)
    base = ev("baseline", served, fw0)
    print(f"BASELINE (served, {'feasible' if feasible(served) else 'outside the constraints'}): {base}", flush=True)

    def ascend_rerank(w, fw, m, restart, steps, kind):
        step_id = 0
        for step in steps:
            for _ in range(MAX_PASSES):
                improved = False
                for a in AXES:
                    for sign in (1.0, -1.0):
                        cand = move(w, a, sign * step)
                        if not cand or not feasible(cand):
                            continue
                        step_id += 1
                        cm = ev(kind, cand, fw, restart, step_id)
                        if cm["objective"] > m["objective"] + EPS:
                            w, m, improved = cand, cm, True
                            break
                    if improved:
                        break
                if not improved:
                    break
        return w, m

    rng = np.random.default_rng(42)
    best_w, best_m, best_fw = None, None, fw0
    for r in range(N_RESTARTS):
        if r == 0:
            w = project(served)
        else:
            while True:
                d = rng.dirichlet(np.ones(len(AXES)))
                w = {a: float(d[i]) for i, a in enumerate(AXES)}
                if feasible(w):
                    break
        m = ev("rerank_start", w, fw0, r, 0)
        w, m = ascend_rerank(w, fw0, m, r, RERANK_STEPS, "rerank_ascent")
        print(f"restart {r:2d}: objective {m['objective']:.5f} hits@10 {m['hits@10']} hits@20 {m['hits@20']} "
              f"MRR {m['mrr']:.4f} negatives {m['negatives@10']} (evals {n_evals})", flush=True)
        if best_m is None or m["objective"] > best_m["objective"] + EPS:
            best_w, best_m = w, m

    # Block 2: feel-pool weights, swing left out (constant, its standardised column is zero)
    fw, m = fw0.copy(), best_m
    feel_axes_search = [i for i, a in enumerate(rep.feel_axes) if a != "swing"]
    step_id = 0
    for step in FEEL_STEPS:
        for _ in range(MAX_PASSES):
            improved = False
            for i in feel_axes_search:
                for sign in (1.0, -1.0):
                    cand = fw.copy()
                    cand[i] = round(min(FEEL_BOUNDS[1], max(FEEL_BOUNDS[0], cand[i] + sign * step)), 4)
                    if cand[i] == fw[i]:
                        continue
                    step_id += 1
                    cm = ev("feel_ascent", best_w, cand, 0, step_id)
                    if cm["objective"] > m["objective"] + EPS:
                        fw, m, improved = cand, cm, True
                        break
                if improved:
                    break
            if not improved:
                break
    print(f"feel block: objective {m['objective']:.5f} (evals {n_evals})", flush=True)
    if m["objective"] > best_m["objective"] + EPS:
        best_fw, best_m = fw, m

    # Block 3: rerank weights again under the chosen pool
    w3, m3 = ascend_rerank(best_w, best_fw, best_m, 0, RERANK_STEPS[1:], "rerank_ascent_final")
    if m3["objective"] > best_m["objective"] + EPS:
        best_w, best_m = w3, m3
    print(f"final block: objective {best_m['objective']:.5f} (evals {n_evals})", flush=True)

    # chosen: rounded to 6 places as written, re-evaluated
    chosen = {a: round(best_w[a], 6) for a in AXES}
    cm = rep.evaluate(chosen, best_fw)
    record("chosen", chosen, best_fw, cm, extra={"beats_baseline": cm["objective"] > base["objective"] + EPS})
    tune_evals = n_evals

    # HOLDOUT, once, with the chosen weights and pool
    h_rep = Replica(cat, holdout)
    hm = h_rep.evaluate(chosen, best_fw)
    record("holdout", chosen, best_fw, hm)
    log.close()
    wall = time.time() - t_start

    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    write_report(served, base, chosen, cm, rep.feel_axes, fw0, best_fw, hm, len(h_rep.seeds), tune_evals, wall, cat)
    print(f"chosen: {cm}\nHOLDOUT (once): {hm}\nevaluations {tune_evals} on TUNE + 1 HOLDOUT; wall {wall:.0f} s", flush=True)


def write_report(served, base, chosen, cm, feel_axes, fw0, fw, hm, n_h, n_evals, wall, cat):
    fmt = lambda m: (f"pooled {m['pooled']} of {m['positives']}, hits@10 {m['hits@10']}, hits@20 {m['hits@20']}, "
                     f"MRR {m['mrr']:.4f}, negatives {m['negatives@10']}, objective {m['objective']:.5f}")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Weights report (T7, P8)\n\n")
        f.write(f"- Search: `ml/training/search_weights.py` on the replica (`score_lib`), pools {cat.k_clap} / {cat.k_feel}; "
                f"{n_evals} TUNE evaluations, then HOLDOUT once; wall time {wall / 60:.1f} min. Log: `ml/training/weight_search_log.jsonl`.\n")
        f.write("- Every evaluation ranks the post-MMR top 20 the Space returns (diversity 0.70, artist cap 2, cross-language budget; "
                "the seed and copies of it dropped) and counts a hit by `feel_benchmark.match_track`.\n")
        f.write("- Objective: 0.5 x hits@20 + 0.3 x MRR + 0.2 x hits@10, each as a share of all TUNE positives (a positive not "
                "returned is a miss), minus 0.5 per negative in the top 10.\n")
        f.write("- Constraints: no axis above 0.25, s_clap + s_anchor <= 0.25, s_language <= 0.08, s_mode <= 0.05, weights sum to 1. "
                f"{N_RESTARTS} restarts (the served weights projected, then Dirichlet draws), coordinate ascent at steps "
                f"{RERANK_STEPS}; feel-pool weights by coordinate ascent at steps {FEEL_STEPS} in {list(FEEL_BOUNDS)}, swing left out.\n\n")
        f.write("## TUNE, replica\n\n| state | result |\n|---|---|\n")
        f.write("| before, old definition (pre-MMR raw-score rank, exact key; P7 report) | hits@10 1, hits@20 1, MRR 0.0020 |\n")
        f.write(f"| before, aligned definition (served weights, this search's evaluation) | {fmt(base)} |\n")
        f.write(f"| after (chosen weights) | {fmt(cm)} |\n\n")
        f.write(f"## HOLDOUT, evaluated once with the chosen weights ({n_h} seeds in the catalogue)\n\n{fmt(hm)}\n\n")
        f.write("## Rerank weights\n\n| axis | served | chosen |\n|---|---|---|\n")
        for a in AXES:
            f.write(f"| `{a}` | {served[a]:.4f} | {chosen[a]:.4f} |\n")
        f.write("\n## Feel-pool weights\n\n| axis | served | chosen |\n|---|---|---|\n")
        for a, x0, x in zip(feel_axes, fw0, fw):
            f.write(f"| `{a}` | {x0:.2f} | {x:.2f} |\n")


if __name__ == "__main__":
    main()
