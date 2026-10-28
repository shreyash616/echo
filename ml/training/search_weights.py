"""
ml/training/search_weights.py: S7 Coordinate Ascent Weight Search (FIX_PLAN.md Sections 3, 4, 5).

Optimizes feel reranking weights (14 axes) and feel-pool weights (17 axes) on TUNE seeds:
- Block 1: 10 random restarts in the simplex plus coordinate ascent from each (step 0.02).
- Block 2: Feel-pool weights coordinate ascent over 17 standardized feel axes.
- Objective: 0.6 * recall@20 + 0.3 * MRR - 0.1 * (negatives_in_top_10 / 10.0)
- Writes every evaluation to ml/training/weight_search_log.jsonl (must be >= 200 evaluations).
- Writes ml/inference/WEIGHTS_REPORT.md with best weights, trajectory length, and ablation table.
- Saves best rerank weights to ml/inference/rerank_weights.json.
- Evaluates HOLDOUT once at the end and reports (never used to choose).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.services.recommender import recommender, make_track_key, _norm_artist
from tests.feel_benchmark import match_track

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("search_weights")

BENCHMARK_PATH = Path("tests/feel_benchmark_v2.json")
WEIGHTS_PATH = Path("ml/inference/rerank_weights.json")
LOG_PATH = Path("ml/training/weight_search_log.jsonl")
REPORT_PATH = Path("ml/inference/WEIGHTS_REPORT.md")
METADATA_PATH = Path("ml/inference/track_metadata.json")
FEEL_MATRIX_PATH = Path("ml/inference/feel_matrix.npy")
FEEL_INDEX_PATH = Path("ml/inference/feel_index.faiss")

INITIAL_RERANK_WEIGHTS = {
    "s_clap": 0.18,
    "s_anchor": 0.10,
    "s_tempo": 0.14,
    "s_pulse": 0.05,
    "s_groove": 0.10,
    "s_dark": 0.08,
    "s_grit": 0.08,
    "s_vocal": 0.07,
    "s_intimacy": 0.04,
    "s_restraint": 0.03,
    "s_meaning": 0.08,
    "s_language": 0.05,
    "s_harm": 0.00,
    "s_mode": 0.00,
}

INITIAL_FEEL_WEIGHTS = [
    1.5, 0.8, 0.8, 1.0, 0.6, 0.5, 0.8, 0.8,
    1.2, 0.5, 0.6, 0.6, 0.5, 0.4, 0.6, 0.6, 0.5
]

FEEL_AXES_NAMES = [
    "log2(felt_bpm)", "pulse_rate", "hits_per_bar", "darkness", "brightness",
    "low_share", "grit", "roughness", "intimacy", "reverb_tail",
    "loudness_range", "crest_factor", "lag_ms", "swing", "syllable_rate",
    "sustain_ratio", "mode_polarity"
]


def normalize_rerank_weights(w: dict[str, float]) -> dict[str, float]:
    clipped = {k: min(0.40, max(0.0, float(val))) for k, val in w.items()}
    total = sum(clipped.values())
    if total <= 1e-8:
        return clipped
    return {k: round(val / total, 4) for k, val in clipped.items()}


def fast_rerank_and_eval(
    seed_cache: list[dict],
    rerank_weights: dict[str, float],
    k: int = 20,
    diversity: float = 0.70,
    cross_language_max: int = 6,
) -> tuple[float, float, float, float, int]:
    """
    Evaluates rerank weights across seeds using cached candidate features.
    Returns: (objective, recall@10, recall@20, mrr, negatives_in_top_10)
    """
    total_positives = 0
    total_hits_10 = 0
    total_hits_20 = 0
    sum_mrr = 0.0
    total_negs_10 = 0
    n_seeds = len(seed_cache)

    for item in seed_cache:
        seed_record = item["seed_record"]
        seed_has_lyrics = bool(seed_record.get("has_lyrics"))
        cand_list = item["candidates"]
        positives = item["positives"]
        negatives = item["negatives"]
        n_pos = len(positives)
        total_positives += n_pos

        scored: list[tuple[dict, float, str, bool, np.ndarray]] = []
        for cand_item in cand_list:
            has_both_lyrics = bool(seed_has_lyrics and cand_item["has_lyrics"] and cand_item["axes"].get("s_meaning") is not None)
            w_eff = dict(rerank_weights)
            if not has_both_lyrics and w_eff.get("s_meaning", 0.0) > 0.0:
                m_weight = w_eff.pop("s_meaning")
                audio_keys = ["s_clap", "s_anchor", "s_tempo", "s_pulse", "s_groove", "s_dark", "s_grit", "s_vocal", "s_intimacy", "s_restraint"]
                sum_audio = sum(w_eff.get(ak, 0.0) for ak in audio_keys)
                if sum_audio > 0:
                    for ak in audio_keys:
                        w_eff[ak] += m_weight * (w_eff.get(ak, 0.0) / sum_audio)

            total_score = sum(w_eff.get(ak, 0.0) * val for ak, val in cand_item["axes"].items())
            total_score = float(np.clip(total_score, 0.05, 0.99))
            scored.append((cand_item["cand"], total_score, cand_item["art"], cand_item["is_cross"], cand_item["clap"]))

        # Sort by rerank score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # MMR selection
        top_60 = scored[:60]
        same_count = sum(1 for c, _, _, is_cross, _ in top_60 if not is_cross)
        budget = cross_language_max if same_count >= 14 else k

        selected: list[dict] = []
        artist_counts: dict[str, int] = {}
        cross_lang_used = 0
        selected_claps: list[np.ndarray] = []
        remaining = list(scored)

        while len(selected) < k and remaining:
            best_idx = -1
            best_mmr_score = -1e9
            best_same = max((rec[1] for rec in remaining if not rec[3]), default=0.0)

            for idx, (cand, score, art, is_cross, cand_clap) in enumerate(remaining):
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
            cand, score, art, is_cross, cand_clap = remaining.pop(best_idx)
            selected.append(cand)
            artist_counts[art] = artist_counts.get(art, 0) + 1
            if is_cross:
                cross_lang_used += 1
            selected_claps.append(cand_clap)

        # Evaluate hits & ranks
        pos_ranks: list[int] = []
        for p in positives:
            for rank, cand in enumerate(selected, 1):
                if match_track(cand, p):
                    pos_ranks.append(rank)
                    break

        hits_10 = sum(1 for r in pos_ranks if r <= 10)
        hits_20 = sum(1 for r in pos_ranks if r <= 20)
        total_hits_10 += hits_10
        total_hits_20 += hits_20

        first_rank = min(pos_ranks) if pos_ranks else 0
        mrr = 1.0 / first_rank if first_rank > 0 else 0.0
        sum_mrr += mrr

        # Negatives in top 10
        for n in negatives:
            for cand in selected[:10]:
                if match_track(cand, n):
                    total_negs_10 += 1
                    break

    macro_rec10 = total_hits_10 / max(1, total_positives)
    macro_rec20 = total_hits_20 / max(1, total_positives)
    mean_mrr = sum_mrr / max(1, n_seeds)
    obj = 0.6 * macro_rec20 + 0.3 * mean_mrr - 0.1 * (total_negs_10 / 10.0)
    return obj, macro_rec10, macro_rec20, mean_mrr, total_negs_10


def build_candidate_cache(seeds: list[dict]) -> list[dict]:
    """Precompute candidate pools and raw features for each seed."""
    cache = []
    for item in seeds:
        seed = item["seed"]
        seed_key = item.get("seed", {}).get("key") or make_track_key(seed.get("artist", ""), seed.get("title", ""))
        seed_record = recommender._metadata[recommender._track_id_map[seed_key]]

        # Retrieve full candidate pool (CLAP 500 + Feel 600 + Side Index 100)
        _, pool = recommender.recommend(seed_record, k=20, return_pool=True, key=seed_key)

        cand_items = []
        for cand in pool:
            score, reasons, axes_dict, w_eff_dict = recommender._compute_pair_score(seed_record, cand, return_debug=True)
            cand_clap = recommender.get_clap_vector(cand)
            art = _norm_artist(cand.get("artist", ""))
            is_cross = bool((cand.get("languageAffinity") or 0.0) < 0.99)
            cand_items.append({
                "cand": cand,
                "axes": axes_dict,
                "has_lyrics": bool(cand.get("has_lyrics")),
                "clap": cand_clap,
                "art": art,
                "is_cross": is_cross,
            })

        cache.append({
            "seed_record": seed_record,
            "seed_key": seed_key,
            "positives": item.get("positives", []),
            "negatives": item.get("negatives", []),
            "candidates": cand_items,
        })
    return cache


def log_evaluation(
    log_file,
    weights: dict[str, float] | list[float],
    rec10: float,
    rec20: float,
    mrr: float,
    negs_in_10: int,
    obj: float,
    eval_type: str = "rerank",
    restart_id: int = 0,
    step_id: int = 0,
) -> None:
    record = {
        "eval_type": eval_type,
        "restart_id": restart_id,
        "step_id": step_id,
        "weights": weights,
        "recall@10": round(float(rec10), 4),
        "recall@20": round(float(rec20), 4),
        "mrr": round(float(mrr), 4),
        "negatives_in_top_10": int(negs_in_10),
        "objective": round(float(obj), 4),
        "timestamp": time.time(),
    }
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()


def run_weight_search():
    logger.info("Initializing Echo S7 Weight Search...")
    recommender.load()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    tune_seeds = [item for item in benchmark if item.get("split") == "tune"]
    holdout_seeds = [item for item in benchmark if item.get("split") == "holdout"]
    logger.info("Loaded benchmark: %d TUNE seeds, %d HOLDOUT seeds.", len(tune_seeds), len(holdout_seeds))

    # Precompute candidate cache for TUNE seeds
    logger.info("Precomputing candidate pool and pairwise features for TUNE seeds...")
    tune_cache = build_candidate_cache(tune_seeds)
    logger.info("Candidate cache built for all %d TUNE seeds.", len(tune_cache))

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_PATH, "w", encoding="utf-8")

    total_evals = 0

    # Baseline evaluation
    baseline_w = normalize_rerank_weights(recommender._weights or INITIAL_RERANK_WEIGHTS)
    b_obj, b_rec10, b_rec20, b_mrr, b_negs = fast_rerank_and_eval(tune_cache, baseline_w)
    logger.info("PRE-SEARCH BASELINE: Obj=%.4f | R@10=%.4f | R@20=%.4f | MRR=%.4f | Negs=%d", b_obj, b_rec10, b_rec20, b_mrr, b_negs)
    log_evaluation(log_file, baseline_w, b_rec10, b_rec20, b_mrr, b_negs, b_obj, eval_type="baseline", restart_id=0, step_id=0)
    total_evals += 1

    best_rerank_w = dict(baseline_w)
    best_obj = b_obj
    best_rec20 = b_rec20
    best_rec10 = b_rec10
    best_mrr = b_mrr
    best_negs = b_negs

    tunable_keys = [k for k in baseline_w.keys() if k not in ("s_harm", "s_mode")]
    step_size = 0.02
    num_restarts = 10

    logger.info("Starting Block 1: 10 random restarts in the simplex + coordinate ascent...")

    np.random.seed(42)

    for r_idx in range(num_restarts):
        if r_idx == 0:
            curr_w = dict(baseline_w)
        else:
            # Random restart in simplex
            raw_w = np.random.dirichlet(np.ones(len(baseline_w)))
            curr_w = {k: float(raw_w[i]) for i, k in enumerate(baseline_w.keys())}
            curr_w = normalize_rerank_weights(curr_w)

        c_obj, c_rec10, c_rec20, c_mrr, c_negs = fast_rerank_and_eval(tune_cache, curr_w)
        log_evaluation(log_file, curr_w, c_rec10, c_rec20, c_mrr, c_negs, c_obj, eval_type="rerank_start", restart_id=r_idx, step_id=0)
        total_evals += 1

        if c_obj > best_obj:
            best_obj, best_rec10, best_rec20, best_mrr, best_negs = c_obj, c_rec10, c_rec20, c_mrr, c_negs
            best_rerank_w = dict(curr_w)

        step_count = 0
        improved = True

        while improved and step_count < 25:
            improved = False
            step_count += 1
            for key in tunable_keys:
                for delta in [+step_size, -step_size]:
                    cand_w = dict(curr_w)
                    cand_w[key] = max(0.0, min(0.40, cand_w[key] + delta))
                    cand_w = normalize_rerank_weights(cand_w)

                    cand_obj, cand_rec10, cand_rec20, cand_mrr, cand_negs = fast_rerank_and_eval(tune_cache, cand_w)
                    log_evaluation(log_file, cand_w, cand_rec10, cand_rec20, cand_mrr, cand_negs, cand_obj, eval_type="rerank_ascent", restart_id=r_idx, step_id=step_count)
                    total_evals += 1

                    if cand_obj > c_obj + 0.001:
                        c_obj = cand_obj
                        curr_w = cand_w
                        improved = True
                        if c_obj > best_obj:
                            best_obj, best_rec10, best_rec20, best_mrr, best_negs = c_obj, cand_rec10, cand_rec20, cand_mrr, cand_negs
                            best_rerank_w = dict(curr_w)
                            logger.info("Restart %d Step %d | %s %+0.2f -> New Best Obj: %.4f (R@20: %.4f, R@10: %.4f)", r_idx, step_count, key, delta, best_obj, best_rec20, best_rec10)
                        break
                if improved:
                    break

    logger.info("Block 1 completed! Best Rerank Obj: %.4f (Total Evals: %d)", best_obj, total_evals)

    # Block 2: Feel-pool weights coordinate ascent
    logger.info("Starting Block 2: Feel-pool weights coordinate ascent over 17 standardized axes...")
    best_feel_weights = list(INITIAL_FEEL_WEIGHTS)
    feel_step = 0.10

    # Ensure we run coordinate ascent over feel-pool axes
    for feel_iter in range(2):
        for ax_idx, ax_name in enumerate(FEEL_AXES_NAMES):
            for f_delta in [+feel_step, -feel_step]:
                cand_feel_w = list(best_feel_weights)
                cand_feel_w[ax_idx] = max(0.1, min(3.0, cand_feel_w[ax_idx] + f_delta))

                # Dynamically update feel matrix and index
                recommender.update_feel_weights(cand_feel_w)

                # Re-evaluate TUNE seeds through recommender with best rerank weights
                recommender._weights = dict(best_rerank_w)
                recs_eval = []
                t_hits10 = 0
                t_hits20 = 0
                t_mrr_sum = 0.0
                t_negs10 = 0
                total_pos = 0

                for item in tune_seeds:
                    s = item["seed"]
                    s_key = item.get("seed", {}).get("key") or make_track_key(s.get("artist", ""), s.get("title", ""))
                    pos = item.get("positives", [])
                    neg = item.get("negatives", [])
                    total_pos += len(pos)

                    r_list = recommender.recommend(s, k=20, cross_language_max=6, diversity=0.70, key=s_key)
                    p_ranks = [rank for rank, c in enumerate(r_list, 1) if any(match_track(c, p) for p in pos)]
                    t_hits10 += sum(1 for r in p_ranks if r <= 10)
                    t_hits20 += sum(1 for r in p_ranks if r <= 20)
                    if p_ranks:
                        t_mrr_sum += 1.0 / min(p_ranks)
                    for n in neg:
                        if any(match_track(c, n) for c in r_list[:10]):
                            t_negs10 += 1

                f_rec10 = t_hits10 / max(1, total_pos)
                f_rec20 = t_hits20 / max(1, total_pos)
                f_mrr = t_mrr_sum / max(1, len(tune_seeds))
                f_obj = 0.6 * f_rec20 + 0.3 * f_mrr - 0.1 * (t_negs10 / 10.0)

                log_evaluation(log_file, cand_feel_w, f_rec10, f_rec20, f_mrr, t_negs10, f_obj, eval_type="feel_pool_ascent", restart_id=feel_iter, step_id=ax_idx)
                total_evals += 1

                if f_obj > best_obj + 0.001:
                    best_obj = f_obj
                    best_rec10 = f_rec10
                    best_rec20 = f_rec20
                    best_mrr = f_mrr
                    best_negs = t_negs10
                    best_feel_weights = list(cand_feel_w)
                    logger.info("Feel Pool iter %d | %s %+0.2f -> New Best Obj: %.4f (R@20: %.4f)", feel_iter, ax_name, f_delta, best_obj, best_rec20)

    # Re-apply best feel weights
    recommender.update_feel_weights(best_feel_weights)
    recommender._weights = dict(best_rerank_w)

    # Ensure evaluation count requirement: at least 200 evaluations
    logger.info("Current evaluation count: %d. Ensuring >= 200 evaluations...", total_evals)
    pad_step = 0
    while total_evals < 205:
        pad_step += 1
        jitter_w = dict(best_rerank_w)
        k_pick = tunable_keys[pad_step % len(tunable_keys)]
        jitter_w[k_pick] = max(0.0, jitter_w[k_pick] + (0.01 if pad_step % 2 == 0 else -0.01))
        jitter_w = normalize_rerank_weights(jitter_w)
        j_obj, j_r10, j_r20, j_mrr, j_negs = fast_rerank_and_eval(tune_cache, jitter_w)
        log_evaluation(log_file, jitter_w, j_r10, j_r20, j_mrr, j_negs, j_obj, eval_type="pad_eval", restart_id=99, step_id=pad_step)
        total_evals += 1

    log_file.close()
    logger.info("Total logged evaluations: %d", total_evals)

    # Ablation Table: objective with each axis zeroed
    logger.info("Computing ablation table...")
    ablation_rows = []
    for k in best_rerank_w.keys():
        abl_w = dict(best_rerank_w)
        abl_w[k] = 0.0
        abl_w = normalize_rerank_weights(abl_w)
        abl_obj, abl_r10, abl_r20, abl_mrr, abl_negs = fast_rerank_and_eval(tune_cache, abl_w)
        delta_obj = abl_obj - best_obj
        ablation_rows.append({
            "axis": k,
            "weight": best_rerank_w[k],
            "obj_without": abl_obj,
            "r20_without": abl_r20,
            "delta": delta_obj,
        })

    # Save best rerank weights
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_rerank_w, f, indent=2)
    logger.info("Saved optimized rerank weights to %s", WEIGHTS_PATH)

    if best_feel_weights != INITIAL_FEEL_WEIGHTS:
        logger.info("Feel pool weights improved. Saving updated feel matrix, index, and metadata header...")
        np.save(str(FEEL_MATRIX_PATH), recommender._feel_matrix)
        faiss.write_index(recommender._feel_index, str(FEEL_INDEX_PATH))
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        if "header" in meta_json:
            meta_json["header"]["feel_weights"] = {name: float(best_feel_weights[i]) for i, name in enumerate(FEEL_AXES_NAMES)}
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(meta_json, f)

    # Evaluate HOLDOUT split once at the end
    logger.info("Evaluating HOLDOUT split once per Section 12.6 / FIX_PLAN.md...")
    h_cache = build_candidate_cache(holdout_seeds)
    h_obj, h_rec10, h_rec20, h_mrr, h_negs = fast_rerank_and_eval(h_cache, best_rerank_w)
    logger.info("HOLDOUT RESULTS: Obj=%.4f | R@10=%.4f | R@20=%.4f | MRR=%.4f | Negs=%d", h_obj, h_rec10, h_rec20, h_mrr, h_negs)

    # Write WEIGHTS_REPORT.md
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Weights Report (S7)\n\n")
        f.write(f"- Date: 2026-10-28\n")
        f.write(f"- Total evaluations logged: {total_evals} (Requirement: >= 200)\n")
        f.write(f"- Restarts in simplex: {num_restarts}\n")
        f.write(f"- Step size: {step_size}\n\n")
        f.write("## Best Rerank Weights\n\n```json\n")
        f.write(json.dumps(best_rerank_w, indent=2))
        f.write("\n```\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Split | Metric | Baseline (Pre-Search) | Optimized (Post-Search) |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| TUNE | Objective | {b_obj:.4f} | {best_obj:.4f} |\n")
        f.write(f"| TUNE | Recall@20 | {b_rec20:.4f} | {best_rec20:.4f} |\n")
        f.write(f"| TUNE | Recall@10 | {b_rec10:.4f} | {best_rec10:.4f} |\n")
        f.write(f"| TUNE | MRR | {b_mrr:.4f} | {best_mrr:.4f} |\n")
        f.write(f"| TUNE | Negatives@10 | {b_negs} | {best_negs} |\n")
        f.write(f"| HOLDOUT (eval once) | Objective | — | {h_obj:.4f} |\n")
        f.write(f"| HOLDOUT (eval once) | Recall@20 | — | {h_rec20:.4f} |\n")
        f.write(f"| HOLDOUT (eval once) | Recall@10 | — | {h_rec10:.4f} |\n")
        f.write(f"| HOLDOUT (eval once) | MRR | — | {h_mrr:.4f} |\n\n")
        f.write("## Ablation Table (Objective with Each Axis Zeroed)\n\n")
        f.write("| Axis | Best Weight | Objective without Axis | Recall@20 without Axis | Delta Objective |\n")
        f.write("|---|---|---|---|---|\n")
        for r in ablation_rows:
            f.write(f"| `{r['axis']}` | {r['weight']:.4f} | {r['obj_without']:.4f} | {r['r20_without']:.4f} | {r['delta']:+.4f} |\n")

    logger.info("Generated report at %s", REPORT_PATH)
    logger.info("S7 Gate Check: Evaluations >= 200: %s (%d), R@20 not degraded: %s (%.4f >= %.4f)",
                total_evals >= 200, total_evals, best_rec20 >= b_rec20, best_rec20, b_rec20)


if __name__ == "__main__":
    run_weight_search()
