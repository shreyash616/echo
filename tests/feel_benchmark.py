"""
Feel Benchmark Runner for Echo Recommender Engine.
Evaluates feel-based recommendation recall against the four canonical anchor seeds.
See VISION_ALIGNMENT.md for specification and targets.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx


def normalize_text(text: str) -> str:
    """Normalize track title or artist for comparison."""
    if not text:
        return ""
    # Strip parentheses and brackets content
    t = re.sub(r"\(.*?\)", "", text)
    t = re.sub(r"\[.*?\]", "", t)
    # Strip feat / ft / with
    t = re.split(r"\s+-\s+|\s+feat\.|\s+ft\.|\s+with\s+", t, flags=re.IGNORECASE)[0]
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def is_match(cand_title: str, cand_artist: str, target_title: str, target_artist: str) -> bool:
    """Check if candidate matches target song using relaxed substring containment."""
    norm_c_title = normalize_text(cand_title)
    norm_c_artist = normalize_text(cand_artist)
    norm_t_title = normalize_text(target_title)
    norm_t_artist = normalize_text(target_artist)

    title_match = (norm_t_title in norm_c_title) or (norm_c_title in norm_t_title)
    artist_match = (norm_t_artist in norm_c_artist) or (norm_c_artist in norm_t_artist)
    return bool(title_match and artist_match)


def run_benchmark(base_url: str, benchmark_path: Path, out_path: Path | None = None) -> dict:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    results = []
    total_targets = 0
    total_hits = 0
    total_self_recs = 0

    print("=" * 78)
    print(f"ECHO FEEL BENCHMARK EVALUATION")
    print(f"Target Base URL: {base_url}")
    print(f"Benchmark File : {benchmark_path}")
    print("=" * 78)

    if base_url == "local":
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
        from main import app
        from app.services.recommender import recommender
        recommender.load()
        client_ctx = TestClient(app)
    else:
        client_ctx = httpx.Client(base_url=base_url, timeout=45.0)

    with client_ctx as client:
        for item in benchmarks:
            seed = item["seed"]
            targets = item["targets"]
            total_targets += len(targets)

            query = f"{seed['title']} {seed['artist']}"
            t0 = time.perf_counter()
            search_resp = client.get("/api/search", params={"q": query, "limit": 10})
            search_data = search_resp.json()
            search_results = search_data.get("results", [])

            # Pick matching seed
            chosen_seed_track = None
            seed_artist_norm = normalize_text(seed["artist"])

            # Sort candidate seeds: prefer matching artist, no '(' or 'cover', prefer numeric Deezer ID
            candidates = []
            for r in search_results:
                r_artist_norm = normalize_text(r.get("artist", ""))
                r_title_raw = r.get("title", "").lower()
                is_cover = any(w in r_title_raw for w in ["cover", "karaoke", "tribute"])
                has_paren = "(" in r_title_raw

                is_artist_match = seed_artist_norm in r_artist_norm or r_artist_norm in seed_artist_norm
                is_numeric_id = not str(r.get("id", "")).startswith("it_")

                # Priority: artist match > not cover > no paren > numeric id
                score = 0
                if is_artist_match:
                    score += 100
                if not is_cover:
                    score += 50
                if not has_paren:
                    score += 20
                if is_numeric_id:
                    score += 10
                candidates.append((score, r))

            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates:
                chosen_seed_track = candidates[0][1]

            if not chosen_seed_track:
                print(f"\n[FAIL] Could not resolve seed track for {query}")
                continue

            seed_id = chosen_seed_track.get("id")
            resolved_title = chosen_seed_track.get("title")
            resolved_artist = chosen_seed_track.get("artist")

            # Fetch track perception metrics
            track_resp = client.get(f"/api/track/{seed_id}")
            track_metrics = track_resp.json() if track_resp.status_code == 200 else chosen_seed_track

            # Fetch recommendations
            rec_t0 = time.perf_counter()
            rec_resp = client.get(f"/api/recommendations/{seed_id}")
            rec_duration = time.perf_counter() - rec_t0
            recs = rec_resp.json() if rec_resp.status_code == 200 else []

            # Evaluate hits and self-recommendation
            hits = []
            self_recommended = False
            for rank, r in enumerate(recs, 1):
                # Check self-recommendation
                if is_match(r.get("title", ""), r.get("artist", ""), seed["title"], seed["artist"]):
                    self_recommended = True
                    total_self_recs += 1

                for target in targets:
                    if is_match(r.get("title", ""), r.get("artist", ""), target["title"], target["artist"]):
                        hits.append({
                            "target": f"{target['title']} - {target['artist']}",
                            "rank": rank,
                            "score": r.get("matchScore"),
                            "actual_title": r.get("title"),
                            "actual_artist": r.get("artist"),
                        })

            total_hits += len(hits)

            top_5 = [
                f"{r.get('title')} - {r.get('artist')} ({r.get('matchScore')})"
                for r in recs[:5]
            ]

            seed_report = {
                "seed": seed,
                "resolved_id": seed_id,
                "resolved_track": f"{resolved_title} by {resolved_artist}",
                "perception": {
                    "bpm": track_metrics.get("bpm"),
                    "felt_bpm": track_metrics.get("felt_bpm", track_metrics.get("bpm")),
                    "key": track_metrics.get("key"),
                    "energy": track_metrics.get("energy"),
                    "valence": track_metrics.get("valence"),
                    "instrumentPace": track_metrics.get("instrumentPace"),
                    "modeMinorProb": track_metrics.get("modeMinorProb"),
                    "buildScore": track_metrics.get("buildScore"),
                    "vibes": track_metrics.get("vibes"),
                },
                "round_trip_seconds": round(rec_duration, 2),
                "k_returned": len(recs),
                "targets_count": len(targets),
                "hits_count": len(hits),
                "hits": hits,
                "self_recommended": self_recommended,
                "top_5": top_5,
            }
            results.append(seed_report)

            print(f"\nSeed: {seed['title']} - {seed['artist']} [ID: {seed_id}] ({round(rec_duration, 1)}s)")
            print(f"  Perception: BPM={track_metrics.get('bpm')} (felt={track_metrics.get('felt_bpm', track_metrics.get('bpm'))}) | "
                  f"Energy={track_metrics.get('energy')} | Valence={track_metrics.get('valence')} | "
                  f"Pace={track_metrics.get('instrumentPace')}")
            print(f"  Recall@20: {len(hits)} / {len(targets)} hits")
            if hits:
                for h in hits:
                    print(f"    * HIT: {h['target']} at rank #{h['rank']} (score {h['score']})")
            else:
                print("    (No targets found in top 20)")
            if self_recommended:
                print("    [WARNING] Seed recommended itself!")
            print(f"  Top 5 returned:")
            for idx, t5 in enumerate(top_5, 1):
                print(f"    {idx}. {t5}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "total_seeds": len(results),
        "total_targets": total_targets,
        "total_hits": total_hits,
        "recall_at_20": f"{total_hits}/{total_targets}",
        "recall_ratio": round(total_hits / max(1, total_targets), 4),
        "self_recommended_count": total_self_recs,
        "seeds": results,
    }

    print("\n" + "=" * 78)
    print(f"OVERALL RECALL@20: {total_hits} / {total_targets} ({summary['recall_ratio']*100:.1f}%)")
    print(f"SELF-RECOMMENDED : {total_self_recs}")
    print("=" * 78)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Results written to: {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Echo Feel Benchmark Runner")
    parser.add_argument("--base-url", default="https://shreyash616-echo-backend.hf.space", help="Backend base URL")
    parser.add_argument("--seeds", default="tests/feel_benchmark.json", help="Path to feel benchmark seeds JSON")
    parser.add_argument("--out", default="tests/feel_results_baseline.json", help="Path to write results JSON")
    args = parser.parse_args()

    run_benchmark(args.base_url, Path(args.seeds), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
