"""
ml/data/make_discovery_seeds.py: the discovery seeds file (SCALE_PLAN WP3), generated on the box.

  python ml/data/make_discovery_seeds.py

Reads the artist names of ml/data/harvest_seed_artists.json and of tests/feel_benchmark_v2.json (seeds,
positives, negatives; the lead artist of each credit), resolves each to a Deezer artist id with
/search/artist (an exact case-insensitive name match first, else the top hit), and writes
backend/app/data/discovery_seeds.json: the ids only, in source order, with counts. No names and no
titles enter the backend, so nothing of the benchmark reaches the serving container and
tests/test_no_benchmark_leakage.py has nothing to match. Rerun when either source changes.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
HARVEST_SEEDS = ROOT / "ml" / "data" / "harvest_seed_artists.json"
BENCHMARK = ROOT / "tests" / "feel_benchmark_v2.json"
OUT = ROOT / "backend" / "app" / "data" / "discovery_seeds.json"


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", str(s).casefold()).strip()


def _lead_artist(s: str) -> str:
    return re.split(r",|\s+feat\.|\s+ft\.|\s+with\s+|&|\s+x\s+", str(s), flags=re.IGNORECASE)[0].strip()


def names() -> tuple[list[str], list[str]]:
    harvest = [str(n) for n in json.loads(HARVEST_SEEDS.read_text(encoding="utf-8"))]
    bench = []
    for e in json.loads(BENCHMARK.read_text(encoding="utf-8")):
        for ent in [e.get("seed")] + e.get("positives", []) + e.get("negatives", []):
            if ent and ent.get("artist"):
                bench.append(_lead_artist(ent["artist"]))
    return list(dict.fromkeys(harvest)), list(dict.fromkeys(bench))


def resolve(client: httpx.Client, name: str) -> tuple[int | None, bool]:
    for attempt in range(5):
        try:
            body = client.get("https://api.deezer.com/search/artist", params={"q": name, "limit": 5}).json()
        except (httpx.HTTPError, ValueError):
            time.sleep(2 + 2 * attempt)
            continue
        if (body.get("error") or {}).get("code") == 4:  # quota
            time.sleep(5)
            continue
        hits = body.get("data") or []
        exact = [h for h in hits if _norm(h.get("name")) == _norm(name)]
        if exact:
            return int(exact[0]["id"]), True
        return (int(hits[0]["id"]), False) if hits else (None, False)
    return None, False


def main() -> None:
    harvest, bench = names()
    client = httpx.Client(timeout=20.0, headers={"User-Agent": "echo-discovery-seeds/1.0"})
    ids: list[int] = []
    counts = {"harvest_seed_names": len(harvest), "benchmark_artist_names": len(bench),
              "exact": 0, "top_hit": 0, "unresolved": 0}
    for name in list(dict.fromkeys(harvest + bench)):
        aid, exact = resolve(client, name)
        time.sleep(0.15)
        if aid is None:
            counts["unresolved"] += 1
            continue
        counts["exact" if exact else "top_hit"] += 1
        if aid not in ids:
            ids.append(aid)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_by": "ml/data/make_discovery_seeds.py",
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "sources": ["ml/data/harvest_seed_artists.json", "tests/feel_benchmark_v2.json (artist names only)"],
               "counts": {**counts, "unique_ids": len(ids)}, "deezer_ids": ids}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"]), "->", OUT.relative_to(ROOT), file=sys.stdout)


if __name__ == "__main__":
    main()
