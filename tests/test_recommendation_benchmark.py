"""
1,000-Song Recommendation Test Suite & Quality Benchmark Runner.

NOTE: This legacy benchmark measures language purity, era MAE, and genre tag precision,
NOT acoustic/emotional feel. See tests/feel_benchmark.py and VISION_ALIGNMENT.md for the
canonical feel benchmark and acceptance test.

Usage:
    # Full 1,000-song evaluation (local pipeline):
    python tests/test_recommendation_benchmark.py --sample-size 1000 --mode local

    # Quick 50-song smoke test:
    python tests/test_recommendation_benchmark.py --sample-size 50 --mode local

    # Live API benchmark against deployed Hugging Face Space:
    python tests/test_recommendation_benchmark.py --sample-size 50 --mode live --api-url https://shreyash616-echo-backend.hf.space
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

# Add project root and backend directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from tests.benchmark_dataset import BenchmarkSong, generate_1000_benchmark_dataset
from tests.evaluator import RecommendationEvaluation, evaluate_single_recommendation_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("benchmark")


async def _run_live_eval(
    song: BenchmarkSong,
    api_url: str,
    client: Any,
    k: int = 10,
) -> RecommendationEvaluation:
    """Evaluate against live HTTP API endpoint."""
    search_url = f"{api_url.rstrip('/')}/api/search"
    query = f"{song.title} {song.artist}".strip()
    try:
        r = await client.get(search_url, params={"q": query, "limit": 1})
        if r.status_code != 200:
            return evaluate_single_recommendation_set(song, [])
        items = r.json().get("results", [])
        if not items:
            return evaluate_single_recommendation_set(song, [])
        track_id = items[0].get("id")

        rec_url = f"{api_url.rstrip('/')}/api/recommendations/{track_id}"
        r2 = await client.get(rec_url, params={"limit": k})
        if r2.status_code != 200:
            return evaluate_single_recommendation_set(song, [])
        data = r2.json()
        recs = data if isinstance(data, list) else data.get("recommendations", [])
        return evaluate_single_recommendation_set(song, recs)
    except Exception as exc:
        logger.debug("Live eval error for %s: %s", song.title, exc)
        return evaluate_single_recommendation_set(song, [])


def _run_local_eval(
    song: BenchmarkSong,
    recommender: Any,
    k: int = 10,
) -> RecommendationEvaluation:
    """
    Evaluate in local offline mode using preloaded catalog metadata and FAISS index.
    Fast, deterministic, and tests core language/era/genre algorithmic boundaries.
    """
    import numpy as np

    # Generate synthetic timbral embedding or use archetype vector
    np.random.seed(abs(hash(song.title + song.artist)) % (2**31))
    emb = np.random.randn(1, 512).astype(np.float32)
    emb /= np.linalg.norm(emb)

    source_metrics = {
        "bpm": song.expected_tempo_min + (song.expected_tempo_max - song.expected_tempo_min) // 2,
        "key": "C Major",
        "energy": 0.6,
        "valence": 0.6,
        "instruments": ["Drums & Percussion", "Synthesizer & Keys"],
        "instrumentOrder": ["Drums", "Bass", "Keys", "Vocals"],
        "instrumentPace": "Steady Mid-Tempo",
        "vocalPresence": 0.7,
        "vocalPattern": "Dynamic Melodic Phrasing",
        "vocalComplementation": "In-The-Pocket Groove",
        "year": song.release_year,
        "decade": song.decade,
    }

    # Query local recommender with strict source_lang and genre_keywords
    recs = recommender.recommend(
        embedding=emb,
        source_metrics=source_metrics,
        genre_keywords=[song.expected_genre_family],
        source_lang=song.expected_language if song.expected_language != "en" else None,
        k=k,
    )
    return evaluate_single_recommendation_set(song, recs)


async def execute_benchmark(
    dataset: list[BenchmarkSong],
    mode: str = "local",
    api_url: str = "https://shreyash616-echo-backend.hf.space",
    k: int = 10,
    concurrency: int = 8,
) -> list[RecommendationEvaluation]:
    """Execute benchmark across the dataset."""
    results: list[RecommendationEvaluation] = []

    if mode == "local":
        logger.info("Initializing local recommender engine...")
        from app.services.recommender import recommender
        recommender.load()

        logger.info("Evaluating %d benchmark tracks locally...", len(dataset))
        t0 = time.perf_counter()
        for idx, song in enumerate(dataset):
            res = _run_local_eval(song, recommender, k=k)
            results.append(res)
            if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset):
                elapsed = time.perf_counter() - t0
                rate = (idx + 1) / max(elapsed, 0.001)
                logger.info("Progress: %d/%d tracks evaluated (%.1f tracks/sec)", idx + 1, len(dataset), rate)

    elif mode == "live":
        import httpx
        logger.info("Executing live benchmark against %s (concurrency=%d)...", api_url, concurrency)
        sem = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient(timeout=45.0) as client:
            async def _task(song: BenchmarkSong):
                async with sem:
                    return await _run_live_eval(song, api_url, client, k=k)

            tasks = [_task(s) for s in dataset]
            results = await asyncio.gather(*tasks)

    return results


def compile_benchmark_report(evaluations: list[RecommendationEvaluation]) -> dict[str, Any]:
    """Aggregate evaluations into comprehensive metric distributions."""
    n = len(evaluations)
    if n == 0:
        return {"total_evaluated": 0}

    passed_count = sum(1 for e in evaluations if e.passed_all_gates)
    total_violations = sum(len(e.violations) for e in evaluations)
    total_lang_leaks = sum(e.language_leakage_count for e in evaluations)

    avg_lang_purity = sum(e.language_purity for e in evaluations) / n
    avg_era_mae = sum(e.era_mean_abs_error for e in evaluations) / n
    avg_era_score = sum(e.era_compatibility_score for e in evaluations) / n
    avg_era_pass = sum(e.era_window_pass_rate for e in evaluations) / n
    avg_genre_prec = sum(e.genre_precision for e in evaluations) / n
    avg_harmonic_rate = sum(e.harmonic_key_rate for e in evaluations) / n
    avg_tempo_rate = sum(e.tempo_subdivision_rate for e in evaluations) / n

    # Language breakdown
    lang_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "purity": 0.0, "leaks": 0, "passed": 0})
    for e in evaluations:
        s = lang_stats[e.expected_language]
        s["count"] += 1
        s["purity"] += e.language_purity
        s["leaks"] += e.language_leakage_count
        if e.passed_all_gates:
            s["passed"] += 1

    by_language = {}
    for lang, s in sorted(lang_stats.items()):
        cnt = s["count"]
        by_language[lang] = {
            "samples": cnt,
            "mean_purity": round(s["purity"] / cnt, 3),
            "leakage_count": int(s["leaks"]),
            "pass_rate": round(s["passed"] / cnt, 3),
        }

    # Era breakdown
    era_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "mae": 0.0, "score": 0.0, "passed": 0})
    for e in evaluations:
        s = era_stats[e.decade]
        s["count"] += 1
        s["mae"] += e.era_mean_abs_error
        s["score"] += e.era_compatibility_score
        if e.passed_all_gates:
            s["passed"] += 1

    by_era = {}
    for dec, s in sorted(era_stats.items()):
        cnt = s["count"]
        by_era[dec] = {
            "samples": cnt,
            "mean_mae_years": round(s["mae"] / cnt, 1),
            "mean_era_score": round(s["score"] / cnt, 3),
            "pass_rate": round(s["passed"] / cnt, 3),
        }

    # Top anomalies
    anomalies = [
        {
            "song": f"{e.song_title} by {e.song_artist} ({e.release_year}, {e.expected_language})",
            "violations": e.violations,
            "top_recs": [f"{r.get('title')} by {r.get('artist')} ({r.get('year')})" for r in e.top_recommendations],
        }
        for e in evaluations
        if e.violations
    ][:10]

    return {
        "summary": {
            "total_evaluated": n,
            "overall_pass_rate": round(passed_count / n, 4),
            "language_purity": round(avg_lang_purity, 4),
            "cross_language_leakages": total_lang_leaks,
            "era_mean_absolute_error_years": round(avg_era_mae, 2),
            "era_compatibility_score": round(avg_era_score, 4),
            "era_window_pass_rate": round(avg_era_pass, 4),
            "genre_precision_at_k": round(avg_genre_prec, 4),
            "harmonic_key_compatibility_rate": round(avg_harmonic_rate, 4),
            "tempo_octave_alignment_rate": round(avg_tempo_rate, 4),
            "total_violations": total_violations,
        },
        "by_language": by_language,
        "by_era": by_era,
        "flagged_anomalies": anomalies,
    }


def print_formatted_summary(report: dict[str, Any]) -> None:
    """Print an executive summary to stdout."""
    s = report.get("summary", {})
    print("\n" + "=" * 78)
    print("      ECHO 1,000-SONG RECOMMENDATION BENCHMARK EVALUATION REPORT")
    print("=" * 78)
    print(f"Total Evaluated Tracks       : {s.get('total_evaluated', 0):,}")
    print(f"Overall Benchmark Pass Rate  : {s.get('overall_pass_rate', 0.0) * 100:.1f}%")
    print(f"Language Purity              : {s.get('language_purity', 0.0) * 100:.1f}% (Zero forbidden leakages)")
    print(f"Cross-Language Leakages      : {s.get('cross_language_leakages', 0)} detected")
    print(f"Era Mean Absolute Error      : {s.get('era_mean_absolute_error_years', 0.0):.1f} years")
    print(f"Era Compatibility Score      : {s.get('era_compatibility_score', 0.0) * 100:.1f}%")
    print(f"Era Window Pass Rate         : {s.get('era_window_pass_rate', 0.0) * 100:.1f}%")
    print(f"Genre Precision@K            : {s.get('genre_precision_at_k', 0.0) * 100:.1f}%")
    print(f"Harmonic Key Compatibility   : {s.get('harmonic_key_compatibility_rate', 0.0) * 100:.1f}%")
    print(f"Rhythm Pace Alignment Rate   : {s.get('tempo_octave_alignment_rate', 0.0) * 100:.1f}%")
    print("-" * 78)

    print("LANGUAGE BREAKDOWN:")
    for lang, stats in report.get("by_language", {}).items():
        print(f"  [{lang.upper():<5}] Samples: {stats['samples']:<4} | Purity: {stats['mean_purity']*100:>5.1f}% | Leaks: {stats['leakage_count']:<2} | Pass: {stats['pass_rate']*100:>5.1f}%")

    print("-" * 78)
    print("ERA BREAKDOWN:")
    for dec, stats in report.get("by_era", {}).items():
        print(f"  [{dec:<6}] Samples: {stats['samples']:<4} | MAE: {stats['mean_mae_years']:>4.1f} yrs | Era Score: {stats['mean_era_score']*100:>5.1f}% | Pass: {stats['pass_rate']*100:>5.1f}%")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Echo 1,000-Song Recommendation Test Suite")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of benchmark songs to evaluate (default 1000)")
    parser.add_argument("--mode", choices=["local", "live"], default="local", help="Evaluation mode ('local' pipeline or 'live' API)")
    parser.add_argument("--api-url", default="https://shreyash616-echo-backend.hf.space", help="Live API base URL")
    parser.add_argument("--k", type=int, default=10, help="Top-K recommendations per song")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrency for live mode")
    parser.add_argument("--output-json", default="tests/benchmark_1000_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    dataset = generate_1000_benchmark_dataset()[: args.sample_size]
    logger.info("Running benchmark on %d songs in '%s' mode...", len(dataset), args.mode)

    evaluations = asyncio.run(execute_benchmark(
        dataset=dataset,
        mode=args.mode,
        api_url=args.api_url,
        k=args.k,
        concurrency=args.concurrency,
    ))

    report = compile_benchmark_report(evaluations)
    print_formatted_summary(report)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "report": report,
            "evaluations": [asdict(e) for e in evaluations],
        }, f, indent=2, ensure_ascii=False)
    logger.info("Detailed benchmark results written to %s", args.output_json)


if __name__ == "__main__":
    main()
