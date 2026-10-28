"""
tests/test_no_constant_fields.py: Asserts no constant placeholder fields in API response (S6).
Over 50 random in-catalogue tracks, asserts that no response field has the exact same value on all 50 tracks.
Asserts debug payload for Simple Man contains s_tempo for each result.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import httpx
import pytest

PROHIBITED_FIELDS = {
    "valence", "stereoWidth", "phaseCorrelation", "crestFactor",
    "vocalPresence", "vocalPattern", "vocalComplementation",
    "lyricalIntent", "spatialStaging", "instrumentPace", "instruments",
    "instrumentOrder", "camelotKey"
}


def run_no_constant_fields_test(base_url: str):
    client = httpx.Client(base_url=base_url, timeout=30.0)

    # 1. Load catalogue to pick 50 random track IDs, or discover via search if running in CI without artifacts
    meta_path = os.path.abspath("ml/inference/track_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tracks = data.get("tracks", data) if isinstance(data, dict) else data
        random.seed(42)
        sample_tracks = random.sample(tracks, 50)
    else:
        sample_tracks = []
        seen_ids = set()
        for q in ["the", "love", "night", "man", "time", "one", "heart", "world", "you", "fire"]:
            r = client.get("/api/search", params={"q": q})
            if r.status_code == 200:
                res = r.json()
                items = res.get("results", []) if isinstance(res, dict) else res
                for it in items:
                    tid = it.get("id")
                    if tid and tid not in seen_ids:
                        seen_ids.add(tid)
                        sample_tracks.append(it)
                    if len(sample_tracks) >= 50:
                        break
            if len(sample_tracks) >= 50:
                break
        if len(sample_tracks) < 50:
            pytest.skip(f"Could not gather 50 tracks via search (found {len(sample_tracks)})")

    rec_field_values: dict[str, set] = {}
    track_field_values: dict[str, set] = {}

    print(f"\nQuerying /api/track and /api/recommendations for 50 tracks from {base_url}...")
    for track in sample_tracks:
        tid = track.get("id")
        key = track.get("key") or f"{track.get('artist')}:{track.get('title')}"

        # 1a. Test /api/track/{tid}
        t_resp = client.get(f"/api/track/{tid}?key={key}")
        assert t_resp.status_code == 200, f"/api/track failed for {tid}: {t_resp.text}"
        t_data = t_resp.json()
        for bad_field in PROHIBITED_FIELDS:
            assert bad_field not in t_data, f"Prohibited field '{bad_field}' found in /api/track response!"
        assert t_data.get("bpm", 1) != 0, "Field 'bpm' was 0 in /api/track; felt_bpm must be the tempo field!"

        for k, v in t_data.items():
            if k not in track_field_values:
                track_field_values[k] = set()
            try:
                track_field_values[k].add(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
            except Exception:
                track_field_values[k].add(str(v))

        # 1b. Test /api/recommendations/{tid}
        resp = client.get(f"/api/recommendations/{tid}?key={key}")
        assert resp.status_code == 200, f"Rec failed for {tid}: {resp.text}"
        rec_data = resp.json()
        results = rec_data if isinstance(rec_data, list) else rec_data.get("results", [])
        if results:
            hero = results[0]
            # Ensure deleted fields are not in the response
            for bad_field in PROHIBITED_FIELDS:
                assert bad_field not in hero, f"Prohibited placeholder field '{bad_field}' found in response!"
            assert hero.get("bpm", 1) != 0, f"Field 'bpm' was 0; felt_bpm must be the tempo field!"

            for k, v in hero.items():
                if k not in rec_field_values:
                    rec_field_values[k] = set()
                try:
                    rec_field_values[k].add(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
                except Exception:
                    rec_field_values[k].add(str(v))

    print("\n" + "=" * 70)
    print("TRACK FIELD UNIQUENESS OVER 50 /api/track RESPONSES:")
    for k, vals in track_field_values.items():
        print(f"  {k:25s}: {len(vals):3d} distinct values")
        # matchReasons, matchScore, languageAffinity may naturally be empty/None for single tracks
        if k not in ["matchScore", "languageAffinity", "matchReasons"]:
            assert len(vals) > 1, f"Field '{k}' in /api/track is constant across all 50 tracks! Value: {list(vals)[0]}"
    print("=" * 70)

    print("\n" + "=" * 70)
    print("RECOMMENDATION FIELD UNIQUENESS OVER 50 RESPONSES:")
    for k, vals in rec_field_values.items():
        print(f"  {k:25s}: {len(vals):3d} distinct values")
        # Assert no field has only 1 distinct value (constant)
        if k not in ["pool_source"]:  # pool_source may have limited categories
            assert len(vals) > 1, f"Field '{k}' in recs is constant across all 50 tracks! Value: {list(vals)[0]}"
    print("=" * 70)

    # 2. Check debug payload on Simple Man
    s_resp = client.get("/api/search?q=Simple Man Lynyrd Skynyrd")
    assert s_resp.status_code == 200
    s_data = s_resp.json()
    if isinstance(s_data, dict):
        s_data = s_data.get("results", [])
    assert s_data, "Simple Man not found in search"
    sm_id = s_data[0]["id"]
    sm_key = s_data[0].get("key")

    debug_resp = client.get(f"/api/recommendations/{sm_id}?key={sm_key}&debug=1")
    assert debug_resp.status_code == 200
    debug_data = debug_resp.json()
    debug_results = debug_data if isinstance(debug_data, list) else debug_data.get("results", [])
    assert debug_results, "No debug results for Simple Man"

    for r in debug_results:
        assert "axes" in r, "debug=1 missing 'axes' field"
        assert "s_tempo" in r["axes"], "'axes' missing 's_tempo' in debug=1 payload"
        assert "weights" in r, "debug=1 missing 'weights' field"
        assert "pool_source" in r, "debug=1 missing 'pool_source' field"

    print("Simple Man debug payload verified: axes, weights, pool_source, and s_tempo present.")
    print("ALL FIELD PARITY AND NON-CONSTANT CHECKS PASSED!")


def test_no_constant_fields():
    base_url = os.environ.get("ECHO_BASE_URL") or os.environ.get("TEST_BASE_URL", "https://shreyash616-echo-backend.hf.space")
    try:
        run_no_constant_fields_test(base_url)
    except httpx.ConnectError:
        pytest.skip("Base URL not reachable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://shreyash616-echo-backend.hf.space")
    args = parser.parse_args()
    run_no_constant_fields_test(args.base_url)
