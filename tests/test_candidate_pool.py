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
    import re
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def run_candidate_pool_test(base_url: str | None = None):
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    tune_seeds = [item for item in benchmark if item.get("split") == "tune"]
    assert len(tune_seeds) == 4, f"Expected 4 tune seeds, found {len(tune_seeds)}"

    client = httpx.Client(base_url=base_url or "http://localhost:8000", timeout=30.0)

    total_owner_positives = 0
    positives_in_pool = 0
    process_times = []

    print("\n" + "=" * 95)
    print(f"CANDIDATE POOL MEASUREMENT (S1)")
    print("=" * 95)

    for item in tune_seeds:
        seed = item["seed"]
        seed_title = seed["title"]
        seed_artist = seed["artist"]
        positives = item.get("positives", [])

        # Fetch recommendations with debug=1 to inspect candidate pool
        # First resolve seed ID
        search_res = client.get("/api/search", params={"q": f"{seed_title} {seed_artist}"})
        assert search_res.status_code == 200, f"Search failed for {seed_title}: {search_res.text}"
        results = search_res.json()
        assert results, f"No track found for {seed_title}"
        track_id = results[0]["id"]
        track_key = results[0].get("key") or f"{_norm(seed_artist)}:{_norm(seed_title)}"

        rec_res = client.get(f"/api/recommendations/{track_id}?key={track_key}&debug=1")
        assert rec_res.status_code == 200, f"Rec failed for {track_id}: {rec_res.text}"

        pt = rec_res.headers.get("X-Process-Time")
        if pt:
            process_times.append(float(pt.replace("ms", "").strip()))

        rec_data = rec_res.json()
        pool_tracks = rec_data if isinstance(rec_data, list) else rec_data.get("results", [])

        pool_titles_artists = [(_norm(t.get("title", "")), _norm(t.get("artist", ""))) for t in pool_tracks]

        print(f"\nSeed: {seed_title} by {seed_artist}")
        print(f"  Positives:")
        for pos in positives:
            total_owner_positives += 1
            pos_t = _norm(pos["title"])
            pos_a = _norm(pos["artist"])
            found = any(pos_t in pt and (pos_a in pa or pa in pos_a) for pt, pa in pool_titles_artists)
            if found:
                positives_in_pool += 1
                print(f"    [X] {pos['title']} - {pos['artist']} (IN POOL)")
            else:
                print(f"    [ ] {pos['title']} - {pos['artist']} (not in pool)")

    print("\n" + "=" * 95)
    print(f"Summary:")
    print(f"  Owner Positives in Union Pool: {positives_in_pool} / {total_owner_positives} (Required >= 6)")
    if process_times:
        p95 = np.percentile(process_times, 95)
        print(f"  Server-side p95 Process Time:  {p95:.2f} ms (Required < 300 ms)")
        assert p95 < 300.0, f"Server p95 {p95:.2f} ms exceeds 300 ms limit"
    print("=" * 95)

    assert positives_in_pool >= 6, (
        f"Candidate pool test failed: only {positives_in_pool}/{total_owner_positives} positives in pool (required >= 6)"
    )


def test_candidate_pool():
    # Pytest entry point (runs against local or default if available)
    base_url = os.environ.get("TEST_BASE_URL", "https://shreyash616-echo-backend.hf.space")
    try:
        run_candidate_pool_test(base_url=base_url)
    except httpx.ConnectError:
        pytest.skip("Remote/local backend not reachable; run manually with --base-url")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://shreyash616-echo-backend.hf.space")
    args = parser.parse_args()
    run_candidate_pool_test(base_url=args.base_url)
