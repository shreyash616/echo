"""
tests/feel_benchmark.py: Feel Benchmark Runner v2 (Sections 11.9, 12.6).

Features:
- Run offline against catalogue (default) or live with --base-url.
- Resolve each title with Section 11.3 / make_track_key rule; fails loudly if any missing.
- Separate TUNE and HOLDOUT splits; never mix them.
- Per-seed and aggregate metrics: Recall@10, Recall@20, MRR, nDCG@20, Negatives@10.
- Boyfriend scene test: Mean rank of outside positives vs inside negatives (outside < inside).
- Wilson 95% confidence intervals on Recall@20.
- Records seed feel descriptors (felt_bpm, darkness, kick_class, restraint_bucket) as info.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.recommender import recommender, make_track_key


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate Wilson score interval (default 95% confidence, z=1.96)."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95%
    p = float(k) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    spread = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return lower, upper


def dcg_at_k(hits_ranks: list[int], k: int = 20) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for r in hits_ranks:
        if 1 <= r <= k:
            dcg += 1.0 / math.log2(r + 1)
    return dcg


def idcg_at_k(n_positives: int, k: int = 20) -> float:
    """Ideal DCG at k for binary relevance."""
    ideal_hits = min(n_positives, k)
    return sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))


def normalize_title(title: str) -> str:
    """Normalize title for matching."""
    t = re.sub(r"\(.*?\)", "", title)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.sub(r"[^\w\s]", "", t).strip().lower()
    return re.sub(r"\s+", " ", t)


def match_track(cand: dict, target: dict) -> bool:
    """Match candidate against target using canonical key or normalized title+artist."""
    c_key = cand.get("key") or make_track_key(str(cand.get("artist", "")), str(cand.get("title", "")))
    t_key = target.get("key") or make_track_key(str(target.get("artist", "")), str(target.get("title", "")))
    if c_key and t_key and c_key == t_key:
        return True

    # Title containment fallback
    c_title = normalize_title(str(cand.get("title", "")))
    t_title = normalize_title(str(target.get("title", "")))
    c_art = normalize_title(str(cand.get("artist", "")))
    t_art = normalize_title(str(target.get("artist", "")))

    title_match = (t_title in c_title) or (c_title in t_title)
    art_match = (t_art in c_art) or (c_art in t_art)
    return bool(title_match and art_match)


def run_benchmark(
    split_target: str = "tune",
    base_url: str | None = None,
    benchmark_path: str = "tests/feel_benchmark_v2.json",
) -> dict:
    print("=" * 80)
    print(f"ECHO FEEL BENCHMARK RUNNER (Split: {split_target.upper()})")
    print(f"Benchmark File: {benchmark_path}")
    print(f"Mode          : {'Remote API (' + base_url + ')' if base_url else 'Offline / In-Process Recommender'}")
    print("=" * 80)

    # 1. Load benchmark file
    b_path = Path(benchmark_path)
    if not b_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {b_path}")

    with open(b_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Filter by split
    if split_target != "all":
        items = [item for item in benchmark_data if item.get("split", "").lower() == split_target.lower()]
    else:
        items = benchmark_data

    if not items:
        raise ValueError(f"No benchmark items found for split: '{split_target}'")

    # 2. Recommender / Client initialization
    client = None
    if base_url:
        import httpx
        client = httpx.Client(base_url=base_url, timeout=30.0)
    else:
        if not recommender._loaded:
            recommender.load()

    # 3. Validation: Verify EVERY benchmark title is resolvable in catalogue (Section 11.9)
    missing_titles: list[dict] = []
    for item in items:
        entities = [item.get("seed")] + item.get("positives", []) + item.get("negatives", []) + item.get("near_misses", [])
        st = item.get("scene_test")
        if st:
            entities.extend(st.get("inside_negatives", []) + st.get("outside_positives", []))

        for ent in entities:
            if not ent:
                continue
            t = ent.get("title", "")
            a = ent.get("artist", "")
            k = ent.get("key") or make_track_key(a, t)

            # Check presence in catalogue
            if base_url:
                # In live mode check /api/search
                resp = client.get("/api/search", params={"q": f"{t} {a}", "limit": 1})
                if resp.status_code != 200 or not resp.json().get("results"):
                    missing_titles.append({"title": t, "artist": a, "key": k})
            else:
                if k not in recommender._track_id_map:
                    missing_titles.append({"title": t, "artist": a, "key": k})

    if missing_titles:
        print("\n" + "!" * 80)
        print(f"FATAL ERROR: {len(missing_titles)} benchmark titles missing from catalogue!")
        for m in missing_titles:
            print(f"  Missing: {m['artist']} - {m['title']} (key: {m['key']})")
        print("!" * 80 + "\n")
        raise RuntimeError(f"Section 11.9 Loud Fail: {len(missing_titles)} benchmark titles absent from catalogue.")

    print(f"[OK] All benchmark titles in split '{split_target}' verified present in catalogue.")

    # 4. Evaluation loop
    results: list[dict] = []
    total_positives = 0
    total_recall_10 = 0
    total_recall_20 = 0
    sum_mrr = 0.0
    sum_ndcg20 = 0.0
    total_negatives_10 = 0

    scene_test_result: dict | None = None

    for item in items:
        seed = item["seed"]
        seed_key = seed.get("key") or make_track_key(seed.get("artist", ""), seed.get("title", ""))
        positives = item.get("positives", [])
        negatives = item.get("negatives", [])
        near_misses = item.get("near_misses", [])
        n_pos = len(positives)

        # Retrieve recommendations
        if base_url:
            resp = client.get(f"/api/recommendations/{seed_key}", params={"key": seed_key, "k": 20})
            if resp.status_code != 200:
                raise RuntimeError(f"API failed for seed {seed_key}: {resp.status_code} {resp.text}")
            data = resp.json()
            recs = data if isinstance(data, list) else data.get("recommendations", [])
            seed_record = client.get(f"/api/track/{seed_key}", params={"key": seed_key}).json()
        else:
            seed_record = recommender._metadata[recommender._track_id_map[seed_key]]
            recs = recommender.recommend(seed_record, k=20, key=seed_key)

        # Informational descriptors (Section 11.9)
        info_bpm = seed_record.get("felt_bpm") or seed_record.get("bpm")
        info_dark = seed_record.get("darkness")
        info_kick = seed_record.get("kick_class")
        info_restraint = seed_record.get("restraint_bucket")

        # Evaluate Positives
        pos_ranks: list[int] = []
        for pos in positives:
            matched_rank = None
            for r_idx, cand in enumerate(recs, 1):
                if match_track(cand, pos):
                    matched_rank = r_idx
                    break
            if matched_rank is not None:
                pos_ranks.append(matched_rank)

        hits_10 = sum(1 for r in pos_ranks if r <= 10)
        hits_20 = sum(1 for r in pos_ranks if r <= 20)
        rec10 = hits_10 / max(n_pos, 1)
        rec20 = hits_20 / max(n_pos, 1)

        mrr = 1.0 / min(pos_ranks) if pos_ranks else 0.0
        dcg = dcg_at_k(pos_ranks, k=20)
        idcg = idcg_at_k(n_pos, k=20)
        ndcg = dcg / idcg if idcg > 0.0 else 0.0

        # Evaluate Negatives in top 10
        neg_ranks: list[int] = []
        for neg in negatives:
            for r_idx, cand in enumerate(recs[:10], 1):
                if match_track(cand, neg):
                    neg_ranks.append(r_idx)
        negs_in_10 = len(neg_ranks)

        # Evaluate Near Misses
        nm_ranks: list[int] = []
        for nm in near_misses:
            for r_idx, cand in enumerate(recs, 1):
                if match_track(cand, nm):
                    nm_ranks.append(r_idx)

        # Boyfriend Scene Test (Section 11.9, 12.6)
        st = item.get("scene_test")
        if st and seed.get("title", "").lower() == "boyfriend":
            inside_negs = st.get("inside_negatives", [])
            outside_pos = st.get("outside_positives", [])

            inside_ranks: list[int] = []
            for t in inside_negs:
                found_rank = 50
                for r_idx, cand in enumerate(recs, 1):
                    if match_track(cand, t):
                        found_rank = r_idx
                        break
                inside_ranks.append(found_rank)

            outside_ranks: list[int] = []
            for t in outside_pos:
                found_rank = 50
                for r_idx, cand in enumerate(recs, 1):
                    if match_track(cand, t):
                        found_rank = r_idx
                        break
                outside_ranks.append(found_rank)

            mean_inside = sum(inside_ranks) / max(len(inside_ranks), 1)
            mean_outside = sum(outside_ranks) / max(len(outside_ranks), 1)
            scene_passed = bool(mean_outside < mean_inside)
            scene_test_result = {
                "mean_outside_positives_rank": mean_outside,
                "mean_inside_negatives_rank": mean_inside,
                "passed": scene_passed,
            }

        total_positives += n_pos
        total_recall_10 += hits_10
        total_recall_20 += hits_20
        sum_mrr += mrr
        sum_ndcg20 += ndcg
        total_negatives_10 += negs_in_10

        item_result = {
            "seed": f"{seed.get('artist')} - {seed.get('title')}",
            "split": item.get("split"),
            "info": {
                "felt_bpm": info_bpm,
                "darkness": round(info_dark, 3) if info_dark is not None else None,
                "kick_class": info_kick,
                "restraint_bucket": info_restraint,
            },
            "n_positives": n_pos,
            "hits_at_10": hits_10,
            "hits_at_20": hits_20,
            "recall_at_10": rec10,
            "recall_at_20": rec20,
            "mrr": mrr,
            "ndcg_at_20": ndcg,
            "negatives_at_10": negs_in_10,
            "pos_ranks": pos_ranks,
            "near_miss_ranks": nm_ranks,
        }
        results.append(item_result)

    # 5. Summary metrics
    n_seeds = len(items)
    macro_recall_10 = total_recall_10 / max(total_positives, 1)
    macro_recall_20 = total_recall_20 / max(total_positives, 1)
    mean_mrr = sum_mrr / max(n_seeds, 1)
    mean_ndcg = sum_ndcg20 / max(n_seeds, 1)

    wilson_lower, wilson_upper = wilson_interval(total_recall_20, total_positives)

    print("\n" + "-" * 80)
    print(f"BENCHMARK RESULTS SUMMARY ({split_target.upper()} SPLIT)")
    print("-" * 80)
    print(f"Evaluated Seeds          : {n_seeds}")
    print(f"Total Target Positives   : {total_positives}")
    print(f"Hits @ 10 / Recall@10    : {total_recall_10} / {total_positives} ({macro_recall_10*100:.2f}%)")
    print(f"Hits @ 20 / Recall@20    : {total_recall_20} / {total_positives} ({macro_recall_20*100:.2f}%)")
    print(f"Wilson 95% CI (Recall@20): [{wilson_lower*100:.2f}%, {wilson_upper*100:.2f}%] (lower bound: {wilson_lower:.3f})")
    print(f"Mean MRR                 : {mean_mrr:.4f}")
    print(f"Mean nDCG@20             : {mean_ndcg:.4f}")
    print(f"Negatives in Top 10      : {total_negatives_10} (Target: 0)")

    if scene_test_result:
        print("\nBoyfriend Scene Test:")
        print(f"  Mean Outside Positives Rank: {scene_test_result['mean_outside_positives_rank']:.2f}")
        print(f"  Mean Inside Negatives Rank : {scene_test_result['mean_inside_negatives_rank']:.2f}")
        print(f"  Scene Test Status          : {'PASS (outside < inside)' if scene_test_result['passed'] else 'FAIL'}")

    print("\nPer-Seed Breakdown:")
    for res in results:
        inf = res["info"]
        print(
            f"  {res['seed']:<38} | R@10: {res['recall_at_10']*100:>5.1f}% | R@20: {res['recall_at_20']*100:>5.1f}% | "
            f"MRR: {res['mrr']:>5.3f} | Neg@10: {res['negatives_at_10']} | "
            f"[BPM: {inf['felt_bpm']}, kick: {inf['kick_class']}, rest: {inf['restraint_bucket']}]"
        )
    print("-" * 80 + "\n")

    return {
        "split": split_target,
        "n_seeds": n_seeds,
        "total_positives": total_positives,
        "recall_at_10": macro_recall_10,
        "recall_at_20": macro_recall_20,
        "wilson_95_ci": [wilson_lower, wilson_upper],
        "wilson_lower": wilson_lower,
        "mean_mrr": mean_mrr,
        "mean_ndcg_at_20": mean_ndcg,
        "negatives_at_10": total_negatives_10,
        "scene_test": scene_test_result,
        "per_seed": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feel Benchmark Runner v2")
    parser.add_argument("--split", type=str, default="tune", choices=["tune", "holdout", "all"], help="Evaluation split")
    parser.add_argument("--base-url", type=str, default=None, help="Remote Echo backend base URL")
    parser.add_argument("--benchmark", type=str, default="tests/feel_benchmark_v2.json", help="Benchmark JSON path")
    args = parser.parse_args()

    run_benchmark(split_target=args.split, base_url=args.base_url, benchmark_path=args.benchmark)
