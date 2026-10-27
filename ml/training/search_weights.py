"""
ml/training/search_weights.py: Step R7 Weight Search (Sections 11.8.2, 12.6).

Performs coordinate ascent over the 12 feel reranking weights on TUNE seeds:
- Step: 0.02, weights constrained to [0.0, 0.40], renormalised to sum 1.0.
- Objective: mean over TUNE seeds of:
    0.6 * recall@20 + 0.3 * MRR - 0.1 * (negatives_in_top_10 / 10.0)
- Writes: ml/inference/rerank_weights.json
- Reads HOLDOUT split only if --eval-holdout is explicitly passed (Gate G4 only).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.services.recommender import recommender, make_track_key
from tests.feel_benchmark import match_track

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("search_weights")

BENCHMARK_PATH = Path("tests/feel_benchmark_v2.json")
WEIGHTS_PATH = Path("ml/inference/rerank_weights.json")

INITIAL_SECTION_12_WEIGHTS = {
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


def evaluate_split(
    seeds_data: list[dict],
    weights: dict[str, float],
    k: int = 20,
) -> tuple[float, float, float, float]:
    """
    Evaluates objective on given seeds:
    objective = 0.6 * recall@20 + 0.3 * MRR - 0.1 * (negatives_in_top_10 / 10.0)
    """
    orig_weights = copy.deepcopy(recommender._weights)
    recommender._weights = dict(weights)

    recalls: list[float] = []
    mrrs: list[float] = []
    neg_fractions: list[float] = []

    for item in seeds_data:
        seed = item["seed"]
        seed_key = seed.get("key") or make_track_key(seed.get("artist", ""), seed.get("title", ""))
        positives = item.get("positives", [])
        negatives = item.get("negatives", [])

        # Recommender call offline against catalogue
        recs = recommender.recommend(seed, k=k, cross_language_max=6, diversity=0.70, key=seed_key)

        # Hits
        hits_found = 0
        first_rank = 0
        for rank, cand in enumerate(recs, 1):
            for p in positives:
                if match_track(cand, p):
                    hits_found += 1
                    if first_rank == 0:
                        first_rank = rank
                    break

        rec_val = hits_found / max(1, len(positives))
        mrr_val = 1.0 / first_rank if first_rank > 0 else 0.0

        # Negatives in top 10
        top_10 = recs[:10]
        neg_count = 0
        for cand in top_10:
            for n in negatives:
                if match_track(cand, n):
                    neg_count += 1
                    break

        recalls.append(rec_val)
        mrrs.append(mrr_val)
        neg_fractions.append(neg_count / 10.0)

    recommender._weights = orig_weights

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
    mean_neg = float(np.mean(neg_fractions)) if neg_fractions else 0.0
    obj = 0.6 * mean_recall + 0.3 * mean_mrr - 0.1 * mean_neg
    return obj, mean_recall, mean_mrr, mean_neg


def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    clipped = {k: min(0.40, max(0.0, val)) for k, val in w.items()}
    total = sum(clipped.values())
    if total <= 1e-8:
        return clipped
    return {k: round(val / total, 4) for k, val in clipped.items()}


def coordinate_ascent(tune_seeds: list[dict], initial_weights: dict[str, float], step: float = 0.02) -> dict[str, float]:
    current_w = normalize_weights(initial_weights)
    best_obj, b_rec, b_mrr, b_neg = evaluate_split(tune_seeds, current_w)
    logger.info("Initial TUNE objective: %.4f (Recall@20: %.4f, MRR: %.4f, Neg@10: %.4f)", best_obj, b_rec, b_mrr, b_neg)

    tunable_keys = [k for k in current_w.keys() if k not in ("s_harm", "s_mode")]
    improved = True
    iteration = 0

    while improved and iteration < 30:
        improved = False
        iteration += 1
        for key in tunable_keys:
            for delta in [+step, -step]:
                cand_w = dict(current_w)
                cand_w[key] = max(0.0, min(0.40, cand_w[key] + delta))
                cand_w = normalize_weights(cand_w)

                cand_obj, c_rec, c_mrr, c_neg = evaluate_split(tune_seeds, cand_w)
                if cand_obj >= best_obj + 0.002:
                    best_obj = cand_obj
                    current_w = cand_w
                    improved = True
                    logger.info(
                        "Iter %d | %s %s%.2f -> Obj: %.4f (Recall: %.4f, MRR: %.4f)",
                        iteration, key, "+" if delta > 0 else "", delta, best_obj, c_rec, c_mrr
                    )
                    break
            if improved:
                break

    return current_w


def main():
    parser = argparse.ArgumentParser(description="Echo Step 11.8.2 Weight Search")
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--eval-holdout", action="store_true", help="Read HOLDOUT split (Gate G4 only)")
    args = parser.parse_args()

    recommender.load()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    tune_seeds = [item for item in benchmark if item.get("split") == "tune"]
    holdout_seeds = [item for item in benchmark if item.get("split") == "holdout"]
    logger.info("Loaded benchmark: %d TUNE seeds, %d HOLDOUT seeds.", len(tune_seeds), len(holdout_seeds))

    # Run coordinate ascent on TUNE seeds ONLY (HOLDOUT is untouched per 12.6)
    best_weights = coordinate_ascent(tune_seeds, INITIAL_SECTION_12_WEIGHTS, step=args.step)

    # Save best weights
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_weights, f, indent=2)
    logger.info("Saved optimized rerank weights to %s", WEIGHTS_PATH)

    # Final TUNE evaluation
    t_obj, t_rec, t_mrr, t_neg = evaluate_split(tune_seeds, best_weights)
    print("\n" + "=" * 60)
    print("STEP 11.8.2 WEIGHT SEARCH RESULTS")
    print(f"TUNE Recall@20: {t_rec:.4f} | MRR: {t_mrr:.4f} | Negatives@10: {t_neg:.4f} | Obj: {t_obj:.4f}")

    if args.eval_holdout:
        h_obj, h_rec, h_mrr, h_neg = evaluate_split(holdout_seeds, best_weights)
        print(f"HOLDOUT Recall@20: {h_rec:.4f} | MRR: {h_mrr:.4f} | Negatives@10: {h_neg:.4f} | Obj: {h_obj:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
