"""
tests/test_live_overlap.py: Overlap Test across Four Benchmark Worlds (Section 12.6, Rule 9).

Evaluates four seeds from four distinct musical worlds:
  1. Lynyrd Skynyrd - Simple Man (Southern rock / heavy blues)
  2. Karan Aujla - Boyfriend (contemporary pop / Punjabi trap bounce)
  3. Bang Data - Bang Data (Latin / garage punk)
  4. Townes Van Zandt - Waiting Around to Die (sparse folk / acoustic)

Requirements (Section 12.6):
  - Pairwise top-20 overlap <= 6 for all pairs
  - Pairwise top-10 overlap <= 3 for all pairs
  - Fails loudly on violation
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.recommender import recommender, make_track_key

SEEDS = [
    {
        "name": "Simple Man",
        "artist": "Lynyrd Skynyrd",
        "title": "Simple Man",
        "world": "Southern Rock / Heavy Blues",
    },
    {
        "name": "Boyfriend",
        "artist": "Karan Aujla",
        "title": "Boyfriend",
        "world": "Contemporary Pop / Punjabi Trap Bounce",
    },
    {
        "name": "Bang Data",
        "artist": "Bang Data",
        "title": "Bang Data",
        "world": "Latin / Garage Punk",
    },
    {
        "name": "Waiting Around to Die",
        "artist": "Townes Van Zandt",
        "title": "Waiting Around to Die",
        "world": "Sparse Folk / Acoustic",
    },
]


def run_overlap_test(base_url: str | None = None) -> dict:
    print("=" * 70)
    print("TEST LIVE OVERLAP: 4 Worlds Diversity Test (Section 12.6 / Rule 9)")
    print("=" * 70)

    results_by_seed: dict[str, list[dict]] = {}

    if base_url:
        import httpx
        client = httpx.Client(base_url=base_url, timeout=30.0)
        print(f"Running against remote URL: {base_url}\n")
        for s in SEEDS:
            k = make_track_key(s["artist"], s["title"])
            resp = client.get(f"/api/recommendations/{k}", params={"key": k, "k": 20})
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch recommendations for {s['name']}: {resp.status_code} {resp.text}")
            recs = resp.json().get("recommendations", [])
            results_by_seed[s["name"]] = recs
    else:
        print("Running in-process using MusicRecommender...\n")
        if not recommender._loaded:
            recommender.load()

        for s in SEEDS:
            k = make_track_key(s["artist"], s["title"])
            seed_dict = {"artist": s["artist"], "title": s["title"], "key": k}
            recs = recommender.recommend(seed_dict, k=20, key=k)
            results_by_seed[s["name"]] = recs

    # Print top 5 recommendations for each seed
    for s in SEEDS:
        name = s["name"]
        recs = results_by_seed[name]
        print(f"--- Seed: {s['artist']} - {s['title']} ({s['world']}) ---")
        print(f"Total returned: {len(recs)}")
        for idx, r in enumerate(recs[:5], 1):
            key = r.get("key") or make_track_key(r.get("artist", ""), r.get("title", ""))
            vibes = r.get("vibes", [])
            vibe_str = vibes[0] if vibes else "none"
            score = r.get("matchScore", 0.0)
            print(f"  {idx:2d}. {r.get('artist')} - {r.get('title')} | vibe: {vibe_str} | score: {score:.3f} | key: {key}")
        print()

    # Compute pairwise overlaps
    seed_keys: dict[str, list[str]] = {}
    for s in SEEDS:
        name = s["name"]
        seed_keys[name] = [
            r.get("key") or make_track_key(r.get("artist", ""), r.get("title", ""))
            for r in results_by_seed[name]
        ]

    pair_stats: list[dict] = []
    failed_pairs: list[str] = []

    print("-" * 70)
    print(f"{'Pair':<45} | {'Top-10 Overlap':<14} | {'Top-20 Overlap':<14} | {'Status'}")
    print("-" * 70)

    for s1, s2 in combinations(SEEDS, 2):
        name1, name2 = s1["name"], s2["name"]
        pair_label = f"{name1} vs {name2}"

        list1 = seed_keys[name1]
        list2 = seed_keys[name2]

        top10_1 = set(list1[:10])
        top10_2 = set(list2[:10])
        overlap_10 = len(top10_1.intersection(top10_2))

        top20_1 = set(list1[:20])
        top20_2 = set(list2[:20])
        overlap_20 = len(top20_1.intersection(top20_2))

        # Constraints: top-10 <= 3, top-20 <= 6
        ok = (overlap_10 <= 3) and (overlap_20 <= 6)
        status = "PASS" if ok else "FAIL"

        if not ok:
            failed_pairs.append(pair_label)

        pair_stats.append({
            "pair": pair_label,
            "overlap_10": overlap_10,
            "overlap_20": overlap_20,
            "passed": ok,
        })

        print(f"{pair_label:<45} | {overlap_10:>2d} / 10 (max 3) | {overlap_20:>2d} / 20 (max 6) | {status}")

    print("-" * 70)

    if failed_pairs:
        msg = f"OVERLAP TEST FAILED: The following pairs exceeded overlap thresholds: {failed_pairs}"
        print(f"\n[FAIL] {msg}\n")
        raise AssertionError(msg)

    print("\n[SUCCESS] ALL PAIRS PASSED OVERLAP THRESHOLDS (Rule 9 Satisfied)!\n")
    return {"seeds": SEEDS, "pair_stats": pair_stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Live Overlap across 4 worlds")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL of live Echo backend")
    args = parser.parse_args()

    run_overlap_test(args.base_url)
