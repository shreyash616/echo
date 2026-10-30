"""
ml/data/discover_artists.py: the artist universe (SCALE_PLAN section 4, WP3).

  python ml/data/discover_artists.py [--out-dir DIR] [--max-artists N] [--no-upload]

Run on the Space as a background task (POST /api/admin/discovery/start) or from a shell. Builds
discovery/artists.jsonl, one artist a line: deezer_id, name, nb_fan, source, depth (and country for
chart artists), in discovery order:

1. Seeds, depth 0: every artist of tests/feel_benchmark_v2.json (seeds, positives, negatives) and the
   names in ml/data/harvest_seed_artists.json, resolved with /search/artist; the top 100 artists of
   each country chart in CHARTS (Deezer serves country charts as the "Deezer Charts" account's
   playlists, not through /chart/{country}); every artist of /genre/{id}/artists.
2. A breadth-first walk over /artist/{id}/related, in file order, until MAX_ARTISTS unique artists or
   a depth with nothing new.
3. Deezer allows about 50 requests per 5 s: the client keeps that window, and sleeps and retries on
   a quota error, a 4xx/5xx or a network error, with backoff.

Every CHECKPOINT_EVERY artists, artists.jsonl and progress.json (rows expanded so far) go to the
dataset; a restart downloads both and resumes: the queue is the file's rows from `expanded` on, since
a breadth-first walk expands rows in the order it writes them. At the end MANIFEST.json (counts per
source, chart countries, depths, SHA256 of artists.jsonl) and shards.json (250 artists a shard, row
ranges covering every artist exactly once) are written and uploaded.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import httpx

logger = logging.getLogger("discover_artists")

API = "https://api.deezer.com"
REPO = "shreyash616/echo-ml-artifacts"
MAX_ARTISTS = 120_000
CHECKPOINT_EVERY = 5_000
SHARD_SIZE = 250
CHART_TOP = 100
WINDOW_REQUESTS, WINDOW_S = 50, 5.0
HERE = Path(__file__).resolve().parent
SEED_FILES = (HERE / "harvest_seed_artists.json", HERE.parent.parent / "tests" / "feel_benchmark_v2.json")

# Country charts (section 4 step 1) as Deezer serves them: the "Deezer Charts" account's playlists.
# country -> (playlist id, kind). The ids are resolved from the account's playlist list at run time.
CHART_COUNTRIES = {
    "SG": "Top Singapore", "US": "Top USA", "GB": "Top UK", "ES": "Top Spain", "MX": "Top Mexico",
    "BR": "Top Brazil", "AR": "Top Argentina", "KR": "Top South Korea", "JP": "Top Japan", "FR": "Top France",
    "DE": "Top Germany", "IT": "Top Italy", "ID": "Top Indonesia", "PH": "Top Philippines", "TR": "Top Turkey",
    "NG": "Top Nigeria", "ZA": "Top South Africa", "SA": "Top Saudi Arabia", "EG": "Top Egypt",
    "AE": "Top United Arab Emirates", "TH": "Top Thailand", "MY": "Top Malaysia",
}
DEEZER_CHARTS_USER = 637006841


class Deezer:
    """Deezer API client keeping WINDOW_REQUESTS per WINDOW_S, retrying quota errors and failures."""

    def __init__(self, stop: threading.Event | None = None):
        self.client = httpx.Client(timeout=20.0, headers={"User-Agent": "echo-discovery/1.0"})
        self.sent: collections.deque[float] = collections.deque()
        self.requests = 0
        self.failures = 0
        self.stop = stop or threading.Event()

    def get(self, path: str, **params) -> dict | None:
        for attempt in range(6):
            now = time.monotonic()
            while self.sent and now - self.sent[0] > WINDOW_S:
                self.sent.popleft()
            if len(self.sent) >= WINDOW_REQUESTS:
                time.sleep(WINDOW_S - (now - self.sent[0]) + 0.01)
            self.sent.append(time.monotonic())
            self.requests += 1
            try:
                r = self.client.get(API + path, params=params)
                if r.status_code >= 400:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                body = r.json()
            except (httpx.HTTPError, ValueError) as exc:
                wait = min(60.0, 2.0 * 2 ** attempt)
                logger.info("deezer %s failed (%s), retry in %.0f s", path, str(exc)[:80], wait)
                time.sleep(wait)
                continue
            err = body.get("error") if isinstance(body, dict) else None
            if err:
                if err.get("code") == 4:  # quota exceeded
                    time.sleep(WINDOW_S)
                    continue
                if err.get("code") == 800:  # no data
                    return None
                logger.info("deezer %s error %s", path, err)
                time.sleep(min(60.0, 2.0 * 2 ** attempt))
                continue
            return body
        self.failures += 1
        return None

    def pages(self, path: str, limit: int | None = None, **params) -> list[dict]:
        out, index = [], 0
        while True:
            body = self.get(path, index=index, limit=100, **params)
            if not body:
                break
            data = body.get("data") or []
            out += data
            if not data or not body.get("next") or (limit and len(out) >= limit):
                break
            index += len(data)
        return out[:limit] if limit else out


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", str(s).casefold()).strip()


def _lead_artist(s: str) -> str:
    return re.split(r",|\s+feat\.|\s+ft\.|\s+with\s+|&|\s+x\s+", str(s), flags=re.IGNORECASE)[0].strip()


def seed_names() -> list[str]:
    """Artist names from the seed files, read at run time (the script itself names none)."""
    names: list[str] = []
    seeds_path, bench_path = SEED_FILES
    if seeds_path.exists():
        names += [str(n) for n in json.loads(seeds_path.read_text(encoding="utf-8"))]
    if bench_path.exists():
        for e in json.loads(bench_path.read_text(encoding="utf-8")):
            for ent in [e.get("seed")] + e.get("positives", []) + e.get("negatives", []):
                if ent and ent.get("artist"):
                    names.append(_lead_artist(ent["artist"]))
    return list(dict.fromkeys(n for n in names if n))


class Discovery:
    def __init__(self, out_dir: Path, upload: bool = True, max_artists: int = MAX_ARTISTS,
                 progress: dict | None = None, stop: threading.Event | None = None):
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.upload = upload
        self.max_artists = max_artists
        self.progress = progress if progress is not None else {}
        self.stop = stop or threading.Event()
        self.dz = Deezer(self.stop)
        self.rows: list[dict] = []
        self.seen: set[int] = set()
        self.expanded = 0
        self.charts: dict[str, dict] = {}
        self.unresolved: list[str] = []
        self.started = time.time()
        self._last_checkpoint = 0
        self._api = None

    # ---- dataset I/O -------------------------------------------------------------------------
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        return self._api

    def push(self, name: str) -> None:
        if self.upload:
            self.api().upload_file(path_or_fileobj=str(self.out / name), path_in_repo=f"discovery/{name}",
                                   repo_id=REPO, repo_type="dataset", commit_message=f"discovery: {name}")

    def resume(self) -> bool:
        """Download artists.jsonl and progress.json from the dataset if both exist; True when resumed."""
        if not self.upload:
            return False
        from huggingface_hub import hf_hub_download
        try:
            for name in ("progress.json", "artists.jsonl"):
                hf_hub_download(REPO, f"discovery/{name}", repo_type="dataset", local_dir=str(self.out.parent),
                                token=os.environ.get("HF_TOKEN") or None)
        except Exception as exc:  # no checkpoint yet: start fresh
            logger.info("no checkpoint to resume (%s)", str(exc)[:120])
            return False
        marker = json.loads((self.out / "progress.json").read_text(encoding="utf-8"))
        if marker.get("finished"):
            return False
        self.rows = [json.loads(l) for l in (self.out / "artists.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        self.seen = {r["deezer_id"] for r in self.rows}
        self.expanded = int(marker["expanded"])
        self.charts = marker.get("charts", {})
        self.unresolved = marker.get("unresolved", [])
        self._last_checkpoint = len(self.rows)
        logger.info("resumed: %d artists, %d expanded", len(self.rows), self.expanded)
        return True

    def write_rows(self) -> None:
        with open(self.out / "artists.jsonl", "w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def checkpoint(self, finished: bool = False) -> None:
        self.write_rows()
        marker = {"artists": len(self.rows), "expanded": self.expanded, "charts": self.charts,
                  "unresolved": self.unresolved, "requests": self.dz.requests, "failures": self.dz.failures,
                  "finished": finished, "t": round(time.time() - self.started, 1)}
        (self.out / "progress.json").write_text(json.dumps(marker, ensure_ascii=False), encoding="utf-8")
        self.push("artists.jsonl")
        self.push("progress.json")
        self._last_checkpoint = len(self.rows)
        self.progress.update({"checkpoint_artists": len(self.rows), "checkpoint_expanded": self.expanded,
                              "checkpoint_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        logger.info("checkpoint: %d artists, %d expanded", len(self.rows), self.expanded)

    # ---- discovery ---------------------------------------------------------------------------
    def add(self, a: dict, source: str, depth: int, **extra) -> None:
        aid = int(a["id"])
        if aid in self.seen or len(self.rows) >= self.max_artists:
            return
        self.seen.add(aid)
        self.rows.append({"deezer_id": aid, "name": a.get("name"), "nb_fan": a.get("nb_fan"),
                          "source": source, "depth": depth, **extra})

    def tick(self) -> None:
        self.progress.update({"artists": len(self.rows), "expanded": self.expanded, "requests": self.dz.requests,
                              "failures": self.dz.failures, "elapsed_s": round(time.time() - self.started)})
        if len(self.rows) - self._last_checkpoint >= CHECKPOINT_EVERY:
            self.checkpoint()

    def seeds(self) -> None:
        names = seed_names()
        self.progress["phase"] = f"seeds ({len(names)} names)"
        for name in names:
            if self.stop.is_set():
                return
            body = self.dz.get("/search/artist", q=name, limit=5) or {}
            hits = body.get("data") or []
            exact = [h for h in hits if _norm(h.get("name")) == _norm(name)]
            if exact or hits:
                self.add((exact or hits)[0], "seed", 0)
            else:
                self.unresolved.append(name)
            self.tick()

    def chart_artists(self) -> None:
        self.progress["phase"] = "charts"
        playlists = {p["title"]: p["id"] for p in self.dz.pages(f"/user/{DEEZER_CHARTS_USER}/playlists")}
        for cc, title in CHART_COUNTRIES.items():
            pid = playlists.get(title)
            if pid is None:
                self.charts[cc] = {"title": title, "playlist_id": None, "kind": "official", "artists": 0}
                continue
            tracks = self.dz.pages(f"/playlist/{pid}/tracks", limit=500)
            artists = list({t["artist"]["id"]: t["artist"] for t in tracks if t.get("artist")}.values())[:CHART_TOP]
            before = len(self.rows)
            for a in artists:
                self.add(a, "chart", 0, country=cc)
            self.charts[cc] = {"title": title, "playlist_id": pid, "kind": "official", "artists": len(artists),
                               "new": len(self.rows) - before}
            self.tick()

    def genre_artists(self) -> None:
        self.progress["phase"] = "genres"
        for g in (self.dz.get("/genre") or {}).get("data", []):
            if g.get("id") == 0:
                continue
            for a in self.dz.pages(f"/genre/{g['id']}/artists"):
                self.add(a, "genre", 0, genre=g.get("name"))
            self.tick()

    def walk(self) -> None:
        self.progress["phase"] = "related walk"
        while self.expanded < len(self.rows) and len(self.rows) < self.max_artists and not self.stop.is_set():
            r = self.rows[self.expanded]
            body = self.dz.get(f"/artist/{r['deezer_id']}/related", limit=20) or {}
            for a in body.get("data") or []:
                self.add(a, "related", r["depth"] + 1)
            self.expanded += 1
            self.tick()

    def fill_fans(self) -> None:
        """nb_fan for the seed and chart rows, which /search and playlists may not carry."""
        self.progress["phase"] = "nb_fan for seeds and charts"
        for r in self.rows:
            if r.get("nb_fan") is None and not self.stop.is_set():
                body = self.dz.get(f"/artist/{r['deezer_id']}") or {}
                r["nb_fan"] = body.get("nb_fan")

    def finish(self) -> dict:
        self.progress["phase"] = "manifest"
        self.checkpoint(finished=True)
        sha = hashlib.sha256((self.out / "artists.jsonl").read_bytes()).hexdigest()
        n = len(self.rows)
        shards = [{"shard": f"a{i // SHARD_SIZE:05d}", "start": i, "end": min(i + SHARD_SIZE, n),
                   "first_deezer_id": self.rows[i]["deezer_id"], "last_deezer_id": self.rows[min(i + SHARD_SIZE, n) - 1]["deezer_id"]}
                  for i in range(0, n, SHARD_SIZE)]
        (self.out / "shards.json").write_text(json.dumps({"shard_size": SHARD_SIZE, "rows": n, "file": "discovery/artists.jsonl",
                                                          "range": "row index [start, end) in artists.jsonl",
                                                          "shards": shards}, indent=1), encoding="utf-8")
        chart_countries = sorted(cc for cc, c in self.charts.items() if c.get("artists"))
        manifest = {
            "rows": n, "sha256_artists_jsonl": sha,
            "counts_per_source": dict(collections.Counter(r["source"] for r in self.rows)),
            "chart_countries": chart_countries, "chart_country_count": len(chart_countries), "charts": self.charts,
            "depths": dict(sorted(collections.Counter(r["depth"] for r in self.rows).items())),
            "expanded": self.expanded, "stopped_by": "cap" if n >= self.max_artists else "no new artists",
            "unresolved_seed_names": len(self.unresolved), "requests": self.dz.requests, "failed_requests": self.dz.failures,
            "shards": len(shards), "shard_size": SHARD_SIZE, "elapsed_s": round(time.time() - self.started),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (self.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        self.push("shards.json")
        self.push("MANIFEST.json")
        self.progress.update({"phase": "done", "manifest": manifest})
        return manifest

    def run(self) -> dict | None:
        self.progress["phase"] = "starting"
        if not self.resume():
            self.seeds()
            self.chart_artists()
            self.genre_artists()
            self.checkpoint()
        self.walk()
        if self.stop.is_set():
            self.checkpoint()
            self.progress["phase"] = "stopped"
            return None
        self.fill_fans()
        return self.finish()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="ml/data/discovery")
    ap.add_argument("--max-artists", type=int, default=MAX_ARTISTS)
    ap.add_argument("--no-upload", action="store_true", help="write locally only; no dataset reads or writes")
    args = ap.parse_args()
    m = Discovery(Path(args.out_dir), upload=not args.no_upload, max_artists=args.max_artists).run()
    print(json.dumps(m, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
