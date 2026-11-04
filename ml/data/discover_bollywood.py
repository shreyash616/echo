"""
ml/data/discover_bollywood.py: Hindi film soundtracks, 1970 to today, as a second discovery source (SCALE_PLAN WP6-IN).

  python ml/data/discover_bollywood.py [--store dataset|<dir>] [--from-year 1970] [--to-year 2026]

1. Films     Wikipedia's "List of Hindi films of <year>" through the parse API; every italic wikilink inside a
             table is a film, its "(<year> film)" disambiguator stripped, deduplicated within the year.
2. Albums    per film, an iTunes India album search (entity=album, attribute=albumTerm, limit=10); an album is
             accepted when its normalised name is the film title or starts with it, its release year is within
             one of the film's, and its genre is Bollywood or its credit names a seed singer or composer; then a
             lookup of the accepted album's songs. Films with no accepted album are counted, not retried.
3. Depth     every name credited on an accepted album or its songs (split on commas and ampersands) after the
             seed list, each searched once (entity=song, limit=200), Bollywood-genre songs only.
4. Rows      a row needs a previewUrl and a clean track name (harvest_v2), and its track key and iTunes id must
             be in neither catalog/v3/keys.txt nor the rows already kept; at most MAX_TRACKS rows. Stage A's song
             shape plus film, film_year, collection_id and discovery_source ("film" or "artist").

Output to the store: discovery/bollywood/tracks.jsonl, discovery/bollywood/shards.json (b00000, b00001, ... of at
most SHARD_ROWS rows, "source": "tracks", read by harvest_v3 --spec) and discovery/bollywood/MANIFEST.json (rows by
decade and source, films matched and unmatched a year, requests by status), each read back and hashed.

  python ml/data/discover_bollywood.py --round 2 [--names-only]

Round 2 (README 8756c5f): no film pass. Step 3 over every name credited in the artist field of round 1's rows, in
order of first appearance, without the names round 1 searched; those are rebuilt from round 1's cached film pass,
offline, and must number MANIFEST.json's artist_names. Rows also miss round 1's; at most ROUND2_MAX_TRACKS. Output
tracks2.jsonl, shards2.json (ids after round 1's last shard, so b00010 on) and MANIFEST2.json, which adds
names_searched and names_skipped_as_round_1. --names-only logs the name counts and sends no request.

Pacing: iTunes India (country=IN) at 8 requests a minute, since WP6's stage A spends 20 a minute from the same
address; 403 / 429 back off exponentially; nothing is sent while ml/data/bollywood_work/PAUSE exists. Every
answer is cached under ml/data/bollywood_work/cache, so a rerun after a crash repeats no request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml" / "data"))

WORK = ROOT / "ml" / "data" / "bollywood_work"
PAUSE = WORK / "PAUSE"
OUT = "discovery/bollywood"
KEYS = "catalog/v3/keys.txt"
WIKI_API = "https://en.wikipedia.org/w/api.php"
ITUNES = "https://itunes.apple.com"
ITUNES_SPACING_S = 7.5          # 8 requests a minute
WIKI_SPACING_S = 1.0
MAX_TRACKS = 100_000
ROUND2_MAX_TRACKS = 50_000      # README 8756c5f
SHARD_ROWS = 4_800
# Wikimedia answers 403 to a client without a contact in its agent string (robot policy, https://w.wiki/4wJS)
USER_AGENT = "echo-discover-bollywood/1.0 (https://github.com/shreyash616/echo; feel-based music discovery)"

# the owner's singers and composers (README 2ba543e); the album credits extend it
SEED_ARTISTS = [
    "Lata Mangeshkar", "Kishore Kumar", "Mohammed Rafi", "Asha Bhosle", "Mukesh", "R.D. Burman", "Laxmikant-Pyarelal",
    "Kalyanji-Anandji", "Bappi Lahiri", "Kumar Sanu", "Alka Yagnik", "Udit Narayan", "Sonu Nigam", "Shreya Ghoshal",
    "Sunidhi Chauhan", "Arijit Singh", "A.R. Rahman", "Pritam", "Shankar-Ehsaan-Loy", "Vishal-Shekhar", "Amit Trivedi",
    "KK", "Shaan", "Mohit Chauhan", "Atif Aslam", "Neha Kakkar", "Jubin Nautiyal", "Badshah", "Anu Malik",
    "Nadeem-Shravan", "Jatin-Lalit", "Himesh Reshammiya", "Kavita Krishnamurthy", "S.P. Balasubrahmanyam", "Hariharan",
    "Sukhwinder Singh", "Rahat Fateh Ali Khan", "Armaan Malik", "Shilpa Rao", "Sachin-Jigar", "Tanishk Bagchi",
    "Anuradha Paudwal", "Sadhana Sargam", "Abhijeet", "Suresh Wadkar", "Alisha Chinai", "Monali Thakur",
    "Shalmali Kholgade", "Mika Singh", "Jagjit Singh",
]

_TABLE = re.compile(r"^\{\|.*?^\|\}", re.S | re.M)
_ITALIC_LINK = re.compile(r"''+\s*\[\[([^\]|#]+)(?:\|([^\]]+))?\]\]\s*''")
_REF = re.compile(r"<ref\b[^>]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_NOTE_OPEN = re.compile(r"\{\{\s*(?:efn|refn|cite|sfn|notelist|reflist)", re.I)
_DISAMBIG = re.compile(r"\s*\((?:\d{4}\s+)?(?:hindi\s+|indian\s+)?film\)\s*$", re.I)

log = logging.getLogger("discover_bollywood")


def _norm_str(s: str) -> str:
    from harvest_v2 import _norm_str as norm
    return norm(s)


def _name_key(name: str) -> str:
    """A singer or composer's name for matching: normalised, spaces removed, so "A.R. Rahman" is "A. R. Rahman"."""
    return _norm_str(name).replace(" ", "")


def _strip_notes(wikitext: str, label: object = "") -> str:
    """The text without <ref> tags and footnote or citation templates, whose italic links name sources, not films.
    A template never closed is cut to the end of its line only, and logged once for the page."""
    text = _REF.sub("", wikitext)
    out, i, unclosed = [], 0, 0
    while True:
        m = _NOTE_OPEN.search(text, i)
        if not m:
            if unclosed:
                log.warning("%s: %d unclosed note template(s), each cut to the end of its line", label, unclosed)
            return "".join(out) + text[i:]
        out.append(text[i:m.start()])
        depth, j = 0, m.start()
        while j < len(text):
            if text.startswith("{{", j):
                depth, j = depth + 1, j + 2
            elif text.startswith("}}", j):
                depth, j = depth - 1, j + 2
                if depth == 0:
                    break
            else:
                j += 1
        if depth:
            unclosed += 1
            j = text.find("\n", m.start())
            j = len(text) if j < 0 else j
        i = j


def parse_films(wikitext: str, year: int | None = None) -> list[str]:
    """The film titles of one year's list: italic wikilinks inside its tables, label first, disambiguator stripped."""
    out, seen = [], set()
    for table in _TABLE.findall(_strip_notes(wikitext, year)):
        for target, label in _ITALIC_LINK.findall(table):
            title = _DISAMBIG.sub("", (label or target).replace("''", "").strip())
            if title and title.casefold() not in seen:
                seen.add(title.casefold())
                out.append(title)
    return out


def split_credit(name: str) -> list[str]:
    return [p for p in (x.strip() for x in re.split(r",|&", name or "")) if p]


def accept_album(album: dict, title: str, year: int, seeds: set[str]) -> bool:
    name, t = _norm_str(album.get("collectionName", "")), _norm_str(title)
    if not t or not (name == t or name.startswith(t + " ")):
        return False
    rel = (album.get("releaseDate") or "")[:4]
    if not rel.isdigit() or abs(int(rel) - year) > 1:
        return False
    if album.get("primaryGenreName") == "Bollywood":
        return True
    return any(_name_key(n) in seeds for n in split_credit(album.get("artistName", "")))


def track_row(it: dict, source: str, film: str | None = None, film_year: int | None = None) -> dict | None:
    """Stage A's song (harvest_v3 Harvester.resolve) plus the discovery fields, or None for a row stage A would skip."""
    from harvest_v2 import _is_clean_track_name
    from app.services.recommender import make_track_key
    if it.get("wrapperType") != "track" or not it.get("previewUrl"):
        return None
    tname = it.get("trackName", "")
    if not _is_clean_track_name(tname):
        return None
    tid = str(it["trackId"])
    release = it.get("releaseDate", "")
    year = int(release[:4]) if len(release) >= 4 and release[:4].isdigit() else 0
    return {
        "id": f"it_{tid}", "itunes_id": tid, "deezer_id": None, "key": make_track_key(it.get("artistName", ""), tname),
        "title": tname, "artist": it.get("artistName", ""), "album": it.get("collectionName", ""),
        "albumArtUrl": it.get("artworkUrl100", "").replace("100x100bb", "600x600bb"),
        "previewUrl": it.get("previewUrl"), "durationMs": int(it.get("trackTimeMillis", 30000)),
        "genre": it.get("primaryGenreName", ""), "year": year, "decade": f"{(year // 10) * 10}s" if year else "",
        "film": film, "film_year": film_year, "collection_id": it.get("collectionId"), "discovery_source": source,
    }


class Exhausted(Exception):
    """Eight attempts at one request, each a 403, a 429, a transport error or a 200 without JSON."""


class Fetcher:
    """GET with a per-host pacer, backoff on 403 / 429, the pause file, and a response cache on disk. An offline
    Fetcher answers from the cache only: an answer not there reads as None, as a non-200 did when it was asked
    (only 200s are cached), and is counted as "<host> not cached"."""

    def __init__(self, cache: Path, client: httpx.Client | None = None, offline: bool = False):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
        self.offline = offline
        self.last: dict[str, float] = {}
        self.counts: Counter = Counter()

    def get(self, host: str, url: str, params: dict, spacing: float) -> dict | None:
        """The JSON answer, None for any other status; Exhausted after eight failed attempts."""
        key = hashlib.sha1(json.dumps([url, sorted(params.items())]).encode("utf-8")).hexdigest()
        hit = self.cache / f"{key}.json"
        if hit.exists():
            self.counts[f"{host} cached"] += 1
            return json.loads(hit.read_text(encoding="utf-8"))
        if self.offline:
            self.counts[f"{host} not cached"] += 1
            return None
        for attempt in range(8):
            while PAUSE.exists():
                time.sleep(30)
            wait = spacing - (time.time() - self.last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            self.last[host] = time.time()
            try:
                r = self.client.get(url, params=params)
            except httpx.HTTPError as e:
                self.counts[f"{host} {type(e).__name__}"] += 1
                time.sleep(5 * 2 ** attempt)
                continue
            self.counts[f"{host} {r.status_code}"] += 1
            if r.status_code in (403, 429):
                back = min(600, 30 * 2 ** attempt)
                log.info("%s HTTP %d, backing off %d s", host, r.status_code, back)
                time.sleep(back)
                continue
            if r.status_code != 200:
                return None
            try:
                data = r.json()
            except ValueError:
                self.counts[f"{host} 200 without JSON"] += 1
                time.sleep(5 * 2 ** attempt)
                continue
            hit.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        self.counts[f"{host} exhausted"] += 1
        raise Exhausted(f"{host} {url} {params}")

    def wiki(self, year: int) -> str | None:
        """None for a missing or unreachable page; the MANIFEST's "page": false records it."""
        try:
            d = self.get("wikipedia", WIKI_API, {"action": "parse", "page": f"List of Hindi films of {year}",
                                                 "prop": "wikitext", "format": "json", "formatversion": "2",
                                                 "redirects": "1"}, WIKI_SPACING_S)
        except Exhausted:
            return None
        return ((d or {}).get("parse") or {}).get("wikitext")

    def itunes(self, path: str, params: dict) -> list[dict]:
        """The results; an exhausted request stops the run rather than read as a miss (README 871d37d). The cache
        keeps every answer already fetched, so the rerun repeats none of them."""
        try:
            d = self.get("itunes", f"{ITUNES}/{path}", {**params, "country": "IN"}, ITUNES_SPACING_S)
        except Exhausted as e:
            raise SystemExit(f"stopped: 8 attempts failed on {e}; requests so far {dict(sorted(self.counts.items()))}")
        return (d or {}).get("results") or []


def _keeper(rows: list[dict], seen_keys: set[str], seen_ids: set[str], max_tracks: int):
    def keep(row: dict | None) -> None:
        if row is None or len(rows) >= max_tracks or row["key"] in seen_keys or row["itunes_id"] in seen_ids:
            return
        seen_keys.add(row["key"])
        seen_ids.add(row["itunes_id"])
        rows.append(row)
    return keep


def name_list(credits: list[str]) -> list[str]:
    """The names in order of first appearance, one per _name_key."""
    names, done = [], set()
    for n in credits:
        k = _name_key(n)
        if k and k not in done:
            done.add(k)
            names.append(n)
    return names


def film_pass(fetch: Fetcher, years: range, seeds: list[str], keep, rows: list[dict], max_tracks: int,
              log_years: bool = True) -> tuple[list[str], dict, int]:
    """Steps 1 and 2: each year's films, their accepted albums and the albums' songs. Returns the names credited on
    the albums and songs in order, the films a year, and the albums accepted."""
    seed_norm = {_name_key(s) for s in seeds}
    credits: list[str] = []
    films_by_year: dict[str, dict] = {}
    albums_accepted = 0
    for year in years:
        text = fetch.wiki(year)
        films = parse_films(text, year) if text else []
        matched = 0
        for title in films:
            if len(rows) >= max_tracks:
                break
            albums = [a for a in fetch.itunes("search", {"term": title, "entity": "album", "attribute": "albumTerm", "limit": 10})
                      if a.get("wrapperType") == "collection" and accept_album(a, title, year, seed_norm)]
            matched += bool(albums)
            for a in albums:
                albums_accepted += 1
                credits += split_credit(a.get("artistName", ""))
                for it in fetch.itunes("lookup", {"id": str(a["collectionId"]), "entity": "song"}):
                    if it.get("wrapperType") == "track":
                        credits += split_credit(it.get("artistName", ""))
                        keep(track_row(it, "film", title, year))
        films_by_year[str(year)] = {"listed": len(films), "matched": matched, "page": text is not None}
        if log_years:
            log.info("%d: %d films, %d with a soundtrack, %d rows so far", year, len(films), matched, len(rows))
    return credits, films_by_year, albums_accepted


def search_names(fetch: Fetcher, names: list[str], keep, rows: list[dict], max_tracks: int) -> int:
    """Step 3: each name searched once, Bollywood-genre songs kept. Returns the names searched."""
    searched = 0
    for name in names:
        if len(rows) >= max_tracks:
            break
        for it in fetch.itunes("search", {"term": name, "entity": "song", "limit": 200}):
            if it.get("primaryGenreName") == "Bollywood":
                keep(track_row(it, "artist"))
        searched += 1
        if searched % 50 == 0 or searched == len(names):
            log.info("artists: %d of %d searched, %d rows so far", searched, len(names), len(rows))
    return searched


def _manifest(fetch: Fetcher, rows: list[dict], max_tracks: int, films_by_year: dict, albums_accepted: int,
              searched: int, names: int, seeds: int) -> dict:
    decades = Counter(f"{((r['film_year'] or r['year']) // 10) * 10}s" if (r["film_year"] or r["year"]) else "unknown"
                      for r in rows)
    listed = sum(v["listed"] for v in films_by_year.values())
    matched = sum(v["matched"] for v in films_by_year.values())
    return {
        "rows": len(rows), "max_tracks": max_tracks, "by_source": dict(Counter(r["discovery_source"] for r in rows)),
        "by_decade": dict(sorted(decades.items())),
        "films": {"listed": listed, "matched": matched, "unmatched": listed - matched, "by_year": films_by_year},
        "albums_accepted": albums_accepted, "artists_searched": searched,
        "artist_names": names, "seeds": seeds,
        "film_rows_by_genre": dict(Counter(r["genre"] for r in rows if r["discovery_source"] == "film").most_common(10)),
        "requests": dict(sorted(fetch.counts.items())),
    }


def discover(fetch: Fetcher, seen_keys: set[str], seen_ids: set[str], years: range,
             seeds: list[str] = SEED_ARTISTS, max_tracks: int = MAX_TRACKS) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    keep = _keeper(rows, seen_keys, seen_ids, max_tracks)
    credits, films_by_year, albums_accepted = film_pass(fetch, years, seeds, keep, rows, max_tracks)
    names = name_list(seeds + credits)
    searched = search_names(fetch, names, keep, rows, max_tracks)
    return rows, _manifest(fetch, rows, max_tracks, films_by_year, albums_accepted, searched, len(names), len(seeds))


def round1_names(fetch: Fetcher, years: range, seeds: list[str] = SEED_ARTISTS) -> list[str]:
    """The names round 1 searched, rebuilt from its cached film pass: the seeds, then the credits of its accepted
    albums and their songs. The Fetcher should be offline, so nothing is sent."""
    credits, _, _ = film_pass(fetch, years, seeds, lambda row: None, [], MAX_TRACKS, log_years=False)
    return name_list(seeds + credits)


def round2_names(round1_rows: list[dict], searched: list[str]) -> tuple[list[str], int]:
    """README 8756c5f: every name split_credit finds in the artist field of round 1's rows, in order of first
    appearance, without the names round 1 searched. Returns the names and how many were skipped as round 1's."""
    done = {_name_key(n) for n in searched}
    names = name_list([p for r in round1_rows for p in split_credit(r.get("artist", ""))])
    keep = [n for n in names if _name_key(n) not in done]
    return keep, len(names) - len(keep)


def discover_round2(fetch: Fetcher, seen_keys: set[str], seen_ids: set[str], round1_rows: list[dict],
                    searched: list[str], max_tracks: int = ROUND2_MAX_TRACKS) -> tuple[list[dict], dict]:
    """Step 3 alone over round2_names; rows miss keys.txt and round 1's rows."""
    for r in round1_rows:
        seen_keys.add(r["key"])
        seen_ids.add(r["itunes_id"])
    names, skipped = round2_names(round1_rows, searched)
    rows: list[dict] = []
    done = search_names(fetch, names, _keeper(rows, seen_keys, seen_ids, max_tracks), rows, max_tracks)
    manifest = _manifest(fetch, rows, max_tracks, {}, 0, done, len(names), 0)
    manifest.update({"names_searched": done, "names_skipped_as_round_1": skipped})
    return rows, manifest


def shard_spec(rows: list[dict], first: int = 0, tracks: str = f"{OUT}/tracks.jsonl") -> dict:
    shards = [{"shard": f"b{first + i // SHARD_ROWS:05d}", "start": i, "end": min(i + SHARD_ROWS, len(rows)),
               "source": "tracks"} for i in range(0, len(rows), SHARD_ROWS)]
    return {"tracks": tracks, "shard_size": SHARD_ROWS, "shards": shards}


def write(store, rows: list[dict], manifest: dict, suffix: str = "", first_shard: int = 0) -> None:
    """tracks<suffix>.jsonl, shards<suffix>.json and MANIFEST<suffix>.json to the store, each read back and hashed."""
    from harvest_v3 import sha256
    WORK.mkdir(parents=True, exist_ok=True)
    files = {f"tracks{suffix}.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
             f"shards{suffix}.json": json.dumps(shard_spec(rows, first_shard, f"{OUT}/tracks{suffix}.jsonl"), indent=1),
             f"MANIFEST{suffix}.json": json.dumps(manifest, indent=1, ensure_ascii=False)}
    for name, text in files.items():
        (WORK / name).write_text(text, encoding="utf-8")
    for name in files:
        store.put(WORK / name, f"{OUT}/{name}")
    check = WORK / "_readback"
    check.mkdir(parents=True, exist_ok=True)
    for name in files:
        got = sha256(store.fetch(f"{OUT}/{name}", check))
        if got != sha256(WORK / name):
            raise RuntimeError(f"{OUT}/{name}: uploaded SHA256 {got} does not match")
    log.info("uploaded and read back %s, SHA256s match", ", ".join(f"{OUT}/{n}" for n in files))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="dataset", help="'dataset', or a local directory")
    ap.add_argument("--from-year", type=int, default=1970)
    ap.add_argument("--to-year", type=int, default=datetime.now().year)
    ap.add_argument("--round", type=int, choices=(1, 2), default=1)
    ap.add_argument("--names-only", action="store_true", help="round 2: log the name counts, send no request")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    logname = "discover.log" if args.round == 1 else "discover2.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(WORK / logname, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    from harvest_v3 import DatasetStore, LocalStore, global_seen
    store = DatasetStore() if args.store == "dataset" else LocalStore(Path(args.store))
    if args.round == 2:
        main_round2(store, args.names_only)
        return
    keys, ids = global_seen(store)
    log.info("start: years %d to %d, %d keys in %s", args.from_year, args.to_year, len(keys), KEYS)
    rows, manifest = discover(Fetcher(WORK / "cache"), keys, ids, range(args.from_year, args.to_year + 1))
    manifest.update({"years": [args.from_year, args.to_year], "keys_at_start": len(keys),
                     "date": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    write(store, rows, manifest)
    log.info("done: %d rows, by decade %s, films %d of %d matched", manifest["rows"], manifest["by_decade"],
             manifest["films"]["matched"], manifest["films"]["listed"])


def main_round2(store, names_only: bool) -> None:
    from harvest_v3 import global_seen
    got = WORK / "_round1"
    got.mkdir(parents=True, exist_ok=True)
    m1 = json.loads(store.fetch(f"{OUT}/MANIFEST.json", got).read_text(encoding="utf-8"))
    spec1 = json.loads(store.fetch(f"{OUT}/shards.json", got).read_text(encoding="utf-8"))
    rows1 = [json.loads(l) for l in store.fetch(f"{OUT}/tracks.jsonl", got).read_text(encoding="utf-8").splitlines() if l]
    if len(rows1) != m1["rows"]:
        raise SystemExit(f"stopped: tracks.jsonl has {len(rows1)} rows, MANIFEST.json says {m1['rows']}")
    y0, y1 = m1["years"]
    replay = Fetcher(WORK / "cache", offline=True)
    searched = round1_names(replay, range(y0, y1 + 1))
    if len(searched) != m1["artist_names"]:
        raise SystemExit(f"stopped: {len(searched)} round 1 names rebuilt from the cache, MANIFEST.json says "
                         f"{m1['artist_names']}; replay {dict(replay.counts)}")
    names, skipped = round2_names(rows1, searched)
    log.info("round 2: %d names to search, %d skipped as round 1's (%d round 1 names rebuilt, replay %s; %d round 1 rows)",
             len(names), skipped, len(searched), dict(sorted(replay.counts.items())), len(rows1))
    if names_only:
        return
    keys, ids = global_seen(store)
    log.info("start: round 2, %d keys in %s, shards from b%05d", len(keys), KEYS, len(spec1["shards"]))
    rows, manifest = discover_round2(Fetcher(WORK / "cache"), keys, ids, rows1, searched)
    manifest.update({"round": 2, "round1_rows": len(rows1), "keys_at_start": len(keys),
                     "date": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    write(store, rows, manifest, suffix="2", first_shard=len(spec1["shards"]))
    log.info("done: round 2, %d rows, by decade %s, %d names searched", manifest["rows"], manifest["by_decade"],
             manifest["names_searched"])


if __name__ == "__main__":
    main()
