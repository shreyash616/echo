"""
tests/test_candidate_pool.py: Measures candidate pool coverage on TUNE seeds (S1).
Asserts:
- Reports, for each of the 4 TUNE seeds, which benchmark positives are in the union pool.
- At least 6 of the 11 owner positives are in the union pool across the four seeds.
- Server-side p95 of /api/recommendations for an in-catalogue seed stays under 300 ms (via X-Process-Time).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import httpx
import numpy as np
import pytest

BENCHMARK_PATH = os.path.abspath("tests/feel_benchmark_v2.json")


def _norm(s: str) -> str:
    import unicodedata
    import re
    clean = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", "", clean.lower()).strip()


OWNER_POSITIVES_MAP = {
    "Simple Man": ["When the Levee Breaks", "Stairway to Heaven", "Rooster"],
    "Waiting Around to Die": ["No Surprises", "Me Gustas Tu", "Black"],
    "Bang Data": ["Tidal Wave"],
    "Boyfriend": ["Antidote", "For A Reason", "Baller", "Roll With Me"],
}


def run_candidate_pool_test(base_url: str | None = None):
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    tune_seeds = [item for item in benchmark if item.get("split") == "tune"]
    # Filter to the 4 operative tune seeds if benchmark has more
    tune_seeds = [item for item in tune_seeds if item["seed"]["title"] in OWNER_POSITIVES_MAP]
    assert len(tune_seeds) == 4, f"Expected 4 tune seeds, found {len(tune_seeds)}"

    client = httpx.Client(base_url=base_url or "http://localhost:8000", timeout=30.0)

    total_owner_positives = 0
    positives_in_pool = 0
    all_benchmark_positives = 0
    all_in_pool = 0
    process_times = []

    print("\n" + "=" * 95)
    print("CANDIDATE POOL MEASUREMENT (S1)")
    print("=" * 95)

    for item in tune_seeds:
        seed = item["seed"]
        seed_title = seed["title"]
        seed_artist = seed["artist"]
        positives = item.get("positives", [])
        owner_targets = [_norm(t) for t in OWNER_POSITIVES_MAP.get(seed_title, [])]

        # First resolve seed ID
        search_res = client.get("/api/search", params={"q": f"{seed_title} {seed_artist}"})
        assert search_res.status_code == 200, f"Search failed for {seed_title}: {search_res.text}"
        results = search_res.json()
        if isinstance(results, dict):
            results = results.get("results", [])
        assert results, f"No track found for {seed_title}"
        track_id = results[0]["id"]
        track_key = results[0].get("key") or f"{_norm(seed_artist)}:{_norm(seed_title)}"

        rec_res = client.get(f"/api/recommendations/{track_id}?key={track_key}&debug=1")
        assert rec_res.status_code == 200, f"Rec failed for {track_id}: {rec_res.text}"

        pt = rec_res.headers.get("X-Process-Time")
        if pt:
            process_times.append(float(pt.replace("ms", "").strip()))

        rec_data = rec_res.json()
        pool_tracks = (
            rec_data.get("candidate_pool")
            if (isinstance(rec_data, dict) and "candidate_pool" in rec_data)
            else (rec_data if isinstance(rec_data, list) else rec_data.get("results", []))
        )

        pool_titles_artists = [(_norm(t.get("title", "")), _norm(t.get("artist", "")), t.get("pool_source", "")) for t in pool_tracks]

        print(f"\nSeed: {seed_title} by {seed_artist} (Pool size: {len(pool_tracks)})")
        print("  Positives:")
        for pos in positives:
            all_benchmark_positives += 1
            pos_t = _norm(pos["title"])
            pos_a = _norm(pos["artist"])
            is_owner = any(target in pos_t or pos_t in target for target in owner_targets)
            if is_owner:
                total_owner_positives += 1

            matched_source = None
            for pt, pa, psrc in pool_titles_artists:
                if (pos_t in pt or pt in pos_t) and (pos_a in pa or pa in pos_a):
                    matched_source = psrc or "in_pool"
                    break

            if matched_source:
                all_in_pool += 1
                if is_owner:
                    positives_in_pool += 1
                mark = "[X] (OWNER)" if is_owner else "[X]"
                print(f"    {mark} {pos['title']} - {pos['artist']} -> {matched_source.upper()}")
            else:
                mark = "[ ] (OWNER)" if is_owner else "[ ]"
                print(f"    {mark} {pos['title']} - {pos['artist']} -> NOT IN POOL")

    print("\n" + "=" * 95)
    print("Summary:")
    print(f"  Owner Positives in Union Pool: {positives_in_pool} / {total_owner_positives} (Required >= 6 of 11)")
    print(f"  All Positives in Union Pool:   {all_in_pool} / {all_benchmark_positives}")
    if process_times:
        p95 = np.percentile(process_times, 95)
        print(f"  Server-side p95 Process Time:  {p95:.2f} ms (Required < 300 ms)")
        assert p95 < 300.0, f"Server p95 {p95:.2f} ms exceeds 300 ms limit"
    print("=" * 95)

    assert positives_in_pool >= 6, (
        f"Candidate pool test failed: only {positives_in_pool}/{total_owner_positives} positives in pool (required >= 6)"
    )


# Reviewer decision 2026-10-30 (f864957): 4 of 11 on the P4 index is a known state fixed by T6 (P7).
# strict=True turns an unexpected pass into a failure, so P7 must delete this marker.
@pytest.mark.xfail(strict=True, reason="4 of 11 owner positives on the P4 index; T6 pools (FIX_PLAN_2.md P7) must delete this marker")
def test_candidate_pool():
    # Pytest entry point (runs against local or default if available)
    base_url = os.environ.get("ECHO_BASE_URL") or os.environ.get("TEST_BASE_URL", "https://shreyash616-echo-backend.hf.space")
    try:
        run_candidate_pool_test(base_url=base_url)
    except httpx.ConnectError:
        pytest.skip("Remote/local backend not reachable; run manually with --base-url")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://shreyash616-echo-backend.hf.space")
    args = parser.parse_args()
    run_candidate_pool_test(base_url=args.base_url)
