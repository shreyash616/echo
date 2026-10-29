"""
LRCLIB lyric pass run from a machine whose IP LRCLIB serves, for rows the describing box could
not fetch (its exit IP gets 503 on 41% of requests whatever the rate). Same requests, acceptance
and file naming as redescribe_catalog.fetch_lyrics, so the output drops into ml/data/lyrics as is.

  python ml/data/fetch_lyrics_remote.py --tracks track_metadata.json --out lyrics_out \
      [--skip-ids ids.txt] [--spacing 0.75] [--limit N]

Writes <safe_id>.txt per fetched row, lyrics_none_ids.txt (LRCLIB has no lyrics), and
lyrics_error_ids.txt (failed after the retries). --skip-ids is re-read every 200 rows, so ids the
box reports as already present can be added while the pass runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from pathlib import Path

import httpx

USER_AGENT = "echo-redescribe/1.0 (feel-based music discovery)"
LRCLIB_STATUS: Counter = Counter()
_LAST = [0.0]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lyrics")
logging.getLogger("httpx").setLevel(logging.WARNING)


def safe_id(row: dict) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", row.get("id", ""))


def fetch_lyrics(row: dict, client: httpx.Client, out_dir: Path, spacing: float) -> str | None:
    def _get(endpoint: str, params: dict):
        retried_503 = False
        while True:
            wait = spacing - (time.time() - _LAST[0])
            if wait > 0:
                time.sleep(wait)
            _LAST[0] = time.time()
            resp = client.get(f"https://lrclib.net/api/{endpoint}", params=params, timeout=10.0)
            LRCLIB_STATUS[(endpoint, resp.status_code)] += 1
            if resp.status_code == 429:
                try:
                    after = float(resp.headers.get("Retry-After", "10"))
                except ValueError:
                    after = 10.0
                time.sleep(after * (1.0 + random.uniform(0.0, 0.3)))
                continue
            if resp.status_code == 503 and not retried_503:
                retried_503 = True
                time.sleep(10.0)
                continue
            return resp

    title, artist = row.get("title", ""), row.get("artist", "")
    params = {"track_name": title, "artist_name": artist}
    if row.get("durationMs"):
        params["duration"] = round(row["durationMs"] / 1000)
    text = None
    r = _get("get", {**params, "album_name": row["album"]} if row.get("album") else params)
    if r.status_code == 404 and row.get("album"):
        r = _get("get", params)
    if r.status_code == 200:
        text = r.json().get("plainLyrics")
    elif r.status_code == 404:
        r2 = _get("search", {"track_name": title, "artist_name": artist})
        r2.raise_for_status()
        norm_art = re.sub(r"[^a-z0-9]", "", artist.lower())
        for h in r2.json():
            if norm_art and norm_art in re.sub(r"[^a-z0-9]", "", (h.get("artistName") or "").lower()):
                text = h.get("plainLyrics")
                if text:
                    break
    else:
        r.raise_for_status()
    if not text or len(text.strip()) <= 20:
        return None
    (out_dir / f"{safe_id(row)}.txt").write_text(text.strip(), encoding="utf-8")
    return text.strip()


def read_ids(path: Path | None) -> set[str]:
    return set(path.read_text(encoding="utf-8").split()) if path and path.exists() else set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="track_metadata.json or catalog_features.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-ids", help="ids that already have a lyric file elsewhere; re-read every 200 rows")
    ap.add_argument("--spacing", type=float, default=0.75)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.tracks)
    if src.suffix == ".jsonl":
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        d = json.load(open(src, encoding="utf-8"))
        rows = d["tracks"] if isinstance(d, dict) else d
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    none_path, err_path = out / "lyrics_none_ids.txt", out / "lyrics_error_ids.txt"
    skip_path = Path(args.skip_ids) if args.skip_ids else None
    skip = read_ids(skip_path) | read_ids(none_path) | {p.stem for p in out.glob("*.txt")
                                                           if not p.name.startswith("lyrics_")}
    todo, seen = [], set()
    for r in rows:
        sid = safe_id(r)
        if sid in skip or sid in seen or not r.get("title") or not r.get("artist"):
            continue
        seen.add(sid)
        todo.append(r)
    if args.limit:
        todo = todo[:args.limit]
    log.info("%d rows to fetch (%d skipped), spacing %.2f s", len(todo), len(rows) - len(todo), args.spacing)

    counts: Counter = Counter()
    failed = []
    t0 = time.time()
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client, \
            open(none_path, "a", encoding="utf-8") as none_f:
        for pass_no, batch in enumerate((todo, None)):
            if batch is None:
                batch, failed = failed, []
                log.info("retrying %d errors once", len(batch))
            for i, r in enumerate(batch):
                if pass_no == 0 and i % 200 == 0 and skip_path:
                    fresh = read_ids(skip_path)
                    if safe_id(r) in fresh:
                        counts["skipped_late"] += 1
                        continue
                try:
                    text = fetch_lyrics(r, client, out, args.spacing)
                except httpx.HTTPError as e:
                    failed.append(r)
                    counts["error"] += 1
                    log.info("error on %s: %s", r.get("id"), str(e)[:100])
                    continue
                if text:
                    counts["fetched"] += 1
                else:
                    counts["none"] += 1
                    none_f.write(safe_id(r) + "\n")
                    none_f.flush()
                n = i + 1
                if n == 300 or n % 1000 == 0:
                    log.info("status after %d rows: %s", n,
                             {f"{e} {s}": c for (e, s), c in sorted(LRCLIB_STATUS.items())})
                if n % 500 == 0:
                    rate = n / max(time.time() - t0, 1) * 60
                    log.info("%d/%d %s | %.0f rows/min, ETA %.1f h", n, len(batch), dict(counts), rate,
                             (len(batch) - n) / max(rate, 0.1) / 60)
    err_path.write_text("\n".join(safe_id(r) for r in failed) + ("\n" if failed else ""), encoding="utf-8")
    counts["error_after_retry"] = len(failed)
    log.info("done %s in %.0f min; status %s", dict(counts), (time.time() - t0) / 60,
             {f"{e} {s}": c for (e, s), c in sorted(LRCLIB_STATUS.items())})


if __name__ == "__main__":
    main()
