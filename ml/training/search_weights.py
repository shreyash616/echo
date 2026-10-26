"""
ml/training/search_weights.py: Step 3.2 Weight Search (Section 11.8.2).

Performs coordinate ascent over the 12 feel reranking weights on TUNE seeds:
- Step: 0.02, weights constrained to [0.0, 0.4], renormalised to sum 1.0.
- Objective: mean over TUNE seeds of:
    0.6 * recall@20 + 0.3 * MRR - 0.1 * (negatives_in_top_10 / 10.0)
- Writes: ml/inference/rerank_weights.json
- Reads HOLDOUT split only at the end (Gate G4) and prints it for recording.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

from app.services.recommender import recommender
from app.services.language_affinity import language_affinity

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("search_weights")

BENCHMARK_PATH = Path("tests/feel_benchmark_v2.json")
WEIGHTS_PATH = Path("ml/inference/rerank_weights.json")


def _norm_str(s: str) -> str:
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def _is_match(c_title: str, c_artist: str, t_title: str, t_artist: str) -> bool:
    nc_t = _norm_str(c_title)
    nc_a = _norm_str(c_artist)
    nt_t = _norm_str(t_title)
    nt_a = _norm_str(t_artist)
    return bool((nt_t in nc_t or nc_t in nt_t) and (nt_a in nc_a or nc_a in nt_a))


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
        positives = item.get("positives", [])
        negatives = item.get("negatives", [])

        # Recommender call offline against catalogue
        recs = recommender.recommend(seed, k=k, cross_language_max=6, diversity=0.70)

        # Hits
        hits_found = 0
        first_rank = 0
        for rank, r in enumerate(recs, 1):
            r_t = r.get("title", "")
            r_a = r.get("artist", "")
            for p in positives:
                if _is_match(r_t, r_a, p["title"], p["artist"]):
                    hits_found += 1
                    if first_rank == 0:
                        first_rank = rank
                    break

        rec_val = hits_found / max(1, len(positives))
        mrr_val = 1.0 / first_rank if first_rank > 0 else 0.0

        # Negatives in top 10
        top_10 = recs[:10]
        neg_count = 0
        for r in top_10:
            r_t = r.get("title", "")
            r_a = r.get("artist", "")
            for n in negatives:
                if _is_match(r_t, r_a, n["title"], n["artist"]):
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
    logger.info(f"Initial TUNE objective: {best_obj:.4f} (Recall@20: {b_rec:.4f}, MRR: {b_mrr:.4f}, Neg@10: {b_neg:.4f})")

    tunable_keys = [k for k in current_w.keys() if k != "clap_cosine"]
    improved = True
    iteration = 0

    while improved and iteration < 50:
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
                        f"Iter {iteration} | {key} {'+' if delta > 0 else ''}{delta:.2f} -> "
                        f"Obj: {best_obj:.4f} (Recall: {c_rec:.4f}, MRR: {c_mrr:.4f})"
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
    logger.info(f"Benchmark: {len(tune_seeds)} TUNE seeds, {len(holdout_seeds)} HOLDOUT seeds.")

    # Load initial weights
    if WEIGHTS_PATH.exists():
        with open(WEIGHTS_PATH, "r", encoding="utf-8") as f:
            init_weights = json.load(f)
    else:
        init_weights = recommender._weights

    # Run coordinate ascent on TUNE seeds ONLY
    best_weights = coordinate_ascent(tune_seeds, init_weights, step=args.step)

    # Save best weights
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_weights, f, indent=2)
    logger.info(f"Saved optimized rerank weights to {WEIGHTS_PATH}")

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
