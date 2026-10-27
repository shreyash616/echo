"""
tests/test_live_api.py: Step R2 verification.
Verifies:
1. In-catalogue recommendations latency p95 < 300 ms over 20 calls.
2. Key-based resolution (?key=...).
3. /api/track/{id} returns cached record immediately without re-describing.
"""
from __future__ import annotations

import argparse
import sys
import time
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from fastapi.testclient import TestClient
from main import app
from app.services.recommender import recommender, make_track_key


def test_api_latency_and_keys(base_url: str | None = None):
    print("=" * 60)
    print("Step R2 API & Key Mapping Verification")
    print("=" * 60)

    if base_url:
        import httpx
        client = httpx.Client(base_url=base_url, timeout=30.0)
        print(f"Testing against remote server: {base_url}")
    else:
        # Load recommender for in-process test client
        print("Initializing in-process FastAPI TestClient...")
        recommender.load()
        client = TestClient(app)
        print("In-process TestClient ready.")

    # 1. Health check
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print(f"[OK] Health check: {health_data}")

    # 2. Key search test
    search_resp = client.get("/api/search", params={"q": "Simple Man Lynyrd Skynyrd", "limit": 3})
    assert search_resp.status_code == 200, f"Search failed: {search_resp.text}"
    search_data = search_resp.json()
    results = search_data.get("results", [])
    assert len(results) > 0, "No search results returned"
    first_res = results[0]
    res_key = first_res.get("key")
    print(f"[OK] Search returned track: '{first_res.get('title')}' by '{first_res.get('artist')}' with key='{res_key}'")
    assert res_key, "Search result missing 'key' field!"

    # 3. In-catalogue recommendations benchmark (20 calls)
    cat_id = "it_1440838314"  # Simple Man in catalogue
    norm_key = make_track_key("Lynyrd Skynyrd", "Simple Man")

    print(f"\nBenchmarking in-catalogue recommendations over 20 calls for {cat_id} (?key={norm_key})...")
    latencies: list[float] = []

    for i in range(20):
        t0 = time.perf_counter()
        r = client.get(f"/api/recommendations/{cat_id}", params={"key": norm_key, "diversity": 0.70})
        if "x-process-time" in r.headers:
            dt_ms = float(r.headers["x-process-time"]) * 1000
        else:
            dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)
        assert r.status_code == 200, f"Recommendation request {i} failed: {r.text}"
        recs = r.json()
        assert len(recs) > 0, f"No recommendations returned on call {i}"

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    mean_lat = float(np.mean(latencies))

    print(f"Timing Results over 20 calls:")
    print(f"  Min:    {min_lat:.2f} ms")
    print(f"  Mean:   {mean_lat:.2f} ms")
    print(f"  Median: {p50:.2f} ms")
    print(f"  P95:    {p95:.2f} ms  (Target: < 300 ms)")
    print(f"  Max:    {max_lat:.2f} ms")

    assert p95 < 300.0, f"In-catalogue p95 latency {p95:.2f} ms exceeds 300 ms target!"
    print(f"[PASS] In-catalogue p95 latency {p95:.2f} ms < 300 ms target!")

    # 4. Test /api/track/{track_id} fast path (does not re-describe)
    t0 = time.perf_counter()
    track_resp = client.get(f"/api/track/{cat_id}", params={"key": norm_key})
    if "x-process-time" in track_resp.headers:
        track_dt_ms = float(track_resp.headers["x-process-time"]) * 1000
    else:
        track_dt_ms = (time.perf_counter() - t0) * 1000
    assert track_resp.status_code == 200, f"/track failed: {track_resp.text}"
    track_data = track_resp.json()
    print(f"\n[OK] /api/track/{cat_id} returned '{track_data.get('title')}' in {track_dt_ms:.2f} ms")
    assert track_dt_ms < 100.0, f"/track took {track_dt_ms:.2f} ms, indicating unexpected re-description!"

    print("\n" + "=" * 60)
    print("ALL STEP R2 VERIFICATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default=None, help="Remote server base URL")
    args = parser.parse_args()
    test_api_latency_and_keys(base_url=args.base_url)
