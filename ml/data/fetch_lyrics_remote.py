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

import os

USER_AGENT = "echo-redescribe/1.0 (feel-based music discovery)"
MAX_503_RETRIES = 5   # 503 here carries no Retry-After and does not depend on our rate; short sleeps
SLEEP_503_S = 1.0


def _dotenv() -> dict:
    """KEY=VALUE lines of the repo-root .env and backend/.env (both gitignored); quotes are stripped."""
    root = Path(__file__).resolve().parents[2]
    out = {}
    for f in (root / ".env", root / "backend" / ".env"):
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def registered_app() -> dict:
    """Lrclib-App-Id and Lrclib-Api-Key headers for the owner's registered server-side application,
    from LRCLIB_APP_ID / LRCLIB_API_KEY (or LRCLIB_APPLICATION_ID / LRCLIB_SECRET_KEY) in the environment,
    the repo-root .env, backend/.env, or ~/.config/echo/lrclib_api_key."""
    env = {**_dotenv(), **os.environ}
    app_id = env.get("LRCLIB_APP_ID") or env.get("LRCLIB_APPLICATION_ID")
    key = env.get("LRCLIB_API_KEY") or env.get("LRCLIB_SECRET_KEY")
    if not key:
        f = Path.home() / ".config" / "echo" / "lrclib_api_key"
        key = f.read_text(encoding="utf-8").strip() if f.exists() else None
    headers = {}
    if app_id:
        headers["Lrclib-App-Id"] = app_id
    if key:
        headers["Lrclib-Api-Key"] = key
    return headers


LRCLIB_STATUS: Counter = Counter()
_LAST = [0.0]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lyrics")
logging.getLogger("httpx").setLevel(logging.WARNING)


_SUFFIX = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]|\s+-\s+(\d{4}\s+)?(remaster|remastered|mono|stereo|live|demo|single|album|radio|edit|version|mix|acoustic|deluxe|bonus)\b.*$", re.I)


def clean_title(title: str) -> str:
    """Title without bracketed or dashed edition suffixes: 'Moon River (Demo Version) [Original]' -> 'Moon River'."""
    out = _SUFFIX.sub("", title).strip(" -")
    return out or title


def safe_id(row: dict) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", row.get("id", ""))


def fetch_lyrics(row: dict, client: httpx.Client, out_dir: Path, spacing: float) -> str | None:
    def _get(endpoint: str, params: dict):
        tries_503 = 0
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
            if resp.status_code == 503 and tries_503 < MAX_503_RETRIES:
                tries_503 += 1
                time.sleep(SLEEP_503_S)
                continue
            return resp

    title, artist = row.get("title", ""), row.get("artist", "")
    title_clean = clean_title(title)
    params = {"track_name": title, "artist_name": artist}
    if row.get("durationMs"):
        params["duration"] = round(row["durationMs"] / 1000)
    text = None
    # 1. exact signature; 2. cleaned title, no album (covers album-string and edition-suffix mismatches)
    r = _get("get", {**params, "album_name": row["album"]} if row.get("album") else params)
    if r.status_code == 404 and (row.get("album") or title_clean != title):
        r = _get("get", {**params, "track_name": title_clean})
    if r.status_code == 200:
        text = r.json().get("plainLyrics")
    elif r.status_code == 404:
        # 3. structured search, then 4. full-text search (what harvest_v2 used); artist-substring acceptance
        norm_art = re.sub(r"[^a-z0-9]", "", artist.lower())
        for search_params in ({"track_name": title_clean, "artist_name": artist}, {"q": f"{title_clean} {artist}"}):
            r2 = _get("search", search_params)
            r2.raise_for_status()
            for h in r2.json():
                if norm_art and norm_art in re.sub(r"[^a-z0-9]", "", (h.get("artistName") or "").lower()):
                    text = h.get("plainLyrics")
                    if text:
                        break
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
    ap.add_argument("--ids-file", help="restrict the pass to these safe_ids (one per line)")
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
    only = read_ids(Path(args.ids_file)) if args.ids_file else None
    todo, seen = [], set()
    for r in rows:
        sid = safe_id(r)
        if only is not None and sid not in only:
            continue
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
    headers = {"User-Agent": USER_AGENT}
    headers.update(registered_app())
    log.info("registered app: %s", "yes" if "Lrclib-Api-Key" in headers else "no")
    with httpx.Client(headers=headers) as client, \
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
