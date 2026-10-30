"""P7 exactness gate (reviewer 601081a): the vectorised pool scoring and MMR against the loops.

  ECHO_ARTIFACT_DIR=ml/inference python ml/experiments/offline_rerank/exactness_p7.py

For every TUNE seed in the catalogue, at the configured pools (config.candidates_clap /
candidates_feel, overridable through CANDIDATES_CLAP / CANDIDATES_FEEL):
  1. recommender._score_pool against _compute_pair_score per pair: max abs difference on every
     axis and on the score, and whether the match reasons agree;
  2. recommender.recommend's vectorised MMR against the pre-P7 loop (copied below from 2f825a6)
     run on the loop's scores: the same 20 ids in the same order, at diversity 0.70 and 1.0;
  3. score_lib.score_pool against score_lib.features per pair, axes and score.
Asserts every difference <= 1e-6 and every id list identical; prints the per-seed maxima.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath("backend"))
from score_lib import AXES, Catalogue, features, make_track_key, score_matrix, score_pool  # noqa: E402
from app.config import settings  # noqa: E402
from app.services.recommender import POOL_AXES, _norm_artist, recommender  # noqa: E402
from app.services.language_affinity import language_affinity  # noqa: E402

TOL = 1e-6


def mmr_loop(seed_record, scored_candidates, k, cross_language_max, diversity):
    """recommender.recommend's MMR as of 2f825a6, verbatim but for the inputs."""
    top_60 = scored_candidates[:60]
    same_count = sum(1 for c, _, _ in top_60 if (c.get("languageAffinity") or 0.0) >= 0.99)
    budget = cross_language_max if same_count >= 14 else k
    selected = []
    artist_counts = {}
    cross_lang_used = 0
    selected_claps = []
    remaining_records = [
        (cand, score, _norm_artist(cand.get("artist", "")), bool((cand.get("languageAffinity") or 0.0) < 0.99), recommender.get_clap_vector(cand))
        for cand, score, _ in scored_candidates
    ]
    while len(selected) < k and remaining_records:
        best_idx = -1
        best_mmr_score = -1e9
        best_same = max((rec[1] for rec in remaining_records if not rec[3]), default=0.0)
        for idx, (cand, score, art, is_cross, cand_clap) in enumerate(remaining_records):
            is_capped = bool(artist_counts.get(art, 0) >= 2)
            budget_exceeded = bool(is_cross and cross_lang_used >= budget and score < (best_same + 0.05))
            if selected_claps:
                max_sim = max(float(np.dot(cand_clap, s_clap)) for s_clap in selected_claps)
            else:
                max_sim = 0.0
            mmr_val = diversity * score - (1.0 - diversity) * max_sim
            if is_capped:
                mmr_val -= 100.0
            if budget_exceeded:
                mmr_val -= 50.0
            if mmr_val > best_mmr_score:
                best_mmr_score = mmr_val
                best_idx = idx
        if best_idx < 0:
            best_idx = 0
        cand, score, art, is_cross, cand_clap = remaining_records.pop(best_idx)
        artist_counts[art] = artist_counts.get(art, 0) + 1
        if is_cross:
            cross_lang_used += 1
        selected.append(cand)
        selected_claps.append(cand_clap)
    return selected


def ident(t):
    return t.get("key") or str(t.get("id"))


def main():
    recommender.load()
    cat = Catalogue()
    cat.k_clap, cat.k_feel = settings.candidates_clap, settings.candidates_feel  # the replica pools the same sizes
    w_live = cat.weights_live
    print(f"pools: candidates_clap {settings.candidates_clap}, candidates_feel {settings.candidates_feel}")
    seeds = [make_track_key(e["seed"]["artist"], e["seed"]["title"])
             for e in json.load(open("tests/feel_benchmark_v2.json", encoding="utf-8")) if e.get("split") == "tune"]
    failures = []
    print("| seed | pool | live max axis diff | live max score diff | reasons equal | MMR 0.70 same 20 | MMR 1.0 same 20 | replica max axis diff | replica max score diff |")
    print("|---|---|---|---|---|---|---|---|---|")
    for seed_key in seeds:
        seed = recommender.get_track_record(key=seed_key)
        if seed is None:
            print(f"| {seed_key} | not in catalogue | | | | | | | |")
            continue
        sel_07, pool_cands = recommender.recommend(seed=seed, k=20, key=seed_key, return_pool=True)
        sel_10, _ = recommender.recommend(seed=seed, k=20, key=seed_key, diversity=1.0, return_pool=True)

        # 1. live: vectorised pool against the single-pair reference
        vec = recommender._score_pool(seed, pool_cands)
        ref_axes, ref_scores, ref_reasons = [], [], []
        for c in pool_cands:
            s, r, f, _ = recommender._compute_pair_score(seed, c, return_debug=True)
            ref_axes.append([f[a] for a in POOL_AXES]); ref_scores.append(s); ref_reasons.append(r)
        ref_axes = np.array(ref_axes, dtype=np.float64)
        d_axes = float(np.abs(ref_axes - vec["axes"]).max())
        d_score = float(np.abs(np.array(ref_scores) - vec["scores"]).max())
        reasons_eq = ref_reasons == vec["reasons"]

        # 2. MMR: the loop on the loop's scores, as recommend built them before P7
        scored = []
        for c, s, r in zip(pool_cands, ref_scores, ref_reasons):
            cr = dict(c)
            cr["matchScore"] = s
            cr["matchReasons"] = r
            cr["languageAffinity"] = language_affinity(seed.get("language"), c.get("language"))
            scored.append((cr, s, r))
        scored.sort(key=lambda x: x[1], reverse=True)
        mmr_eq = {}
        for div, sel in ((0.70, sel_07), (1.0, sel_10)):
            ref_ids = [ident(t) for t in mmr_loop(seed, scored, 20, 6, div)]
            mmr_eq[div] = ref_ids == [ident(t) for t in sel]

        # 3. replica: score_lib.score_pool against score_lib.features per pair
        si, idxs, srcs, F, flags = cat.feature_table(seed_key)
        rows, rflags = [], []
        for idx in idxs:
            f, both = features(cat.meta[si], cat.meta[int(idx)], cat.clap[si], cat.clap[int(idx)], cat.feature_std)
            rows.append([f[a] for a in AXES]); rflags.append(both)
        R = np.array(rows, dtype=np.float64)
        rflags = np.array(rflags, dtype=bool)
        r_axes = float(np.abs(R - F).max())
        r_score = float(np.abs(score_matrix(R, rflags, w_live) - score_matrix(F, flags, w_live)).max())
        flags_eq = bool((rflags == flags).all())

        print(f"| {seed_key} | {len(pool_cands)} | {d_axes:.2e} | {d_score:.2e} | {'yes' if reasons_eq else 'NO'} | "
              f"{'yes' if mmr_eq[0.70] else 'NO'} | {'yes' if mmr_eq[1.0] else 'NO'} | {r_axes:.2e} | {r_score:.2e} |")
        for name, ok in (("live axes", d_axes <= TOL), ("live score", d_score <= TOL), ("reasons", reasons_eq),
                         ("MMR 0.70", mmr_eq[0.70]), ("MMR 1.0", mmr_eq[1.0]), ("replica axes", r_axes <= TOL),
                         ("replica score", r_score <= TOL), ("replica lyric flags", flags_eq)):
            if not ok:
                failures.append(f"{seed_key}: {name}")
    print("\nFAILURES: " + ("; ".join(failures) if failures else "none"))
    assert not failures, failures


if __name__ == "__main__":
    main()
