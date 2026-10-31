"""
ml/data/harvest_v3.py: the shard harvester (SCALE_PLAN section 5, WP4).

  python ml/data/harvest_v3.py --shard-file tests/fixtures/harvest_fixture_shard.json   # one named shard
  python ml/data/harvest_v3.py --next [--runner box] [--shards N]                        # claim and run the next N shards

Stages A to F a shard (README c2f8090: E overlapped with B to D, the next shard's A with this one's B to F):
  A resolve   iTunes artist search, then lookup of up to 200 songs; songs with a previewUrl in iTunes'
              order, at most 20 an artist, deduplicated by artist:title key and iTunes id against the
              global set and within the shard; 20 requests a minute through one module-wide pacer, so two
              shards' A never run at once; exponential backoff on 403 / 429. With --shards N > 1, when a
              shard's A finishes the next shard is claimed and its A runs in a thread under this shard's B
              to F, deduplicated against the global set and this shard's A songs.
  B fetch     previews to ml/data/previews/<itunes_id>.m4a, 8 connections, one retry; failures to the
              shard's dropped list with the reason.
  C describe  describe_audio over a pool of 10 workers (redescribe_catalog._describe_one, the P4 path),
              with the current catalogue's header cut points.
  D CLAP      one process, CUDA: each preview decoded and resampled to 48 kHz as P4 did, the 10 s windows
              of recommender._window_audio_frames, 64 windows a batch through the LAION music checkpoint,
              the song's windows averaged and L2-normalised, stored float16. One INFO line per batch.
  E text      LRCLIB lyrics through fetch_lyrics_remote (the registered application's headers from
              backend/.env), 24 fetchers over disjoint row ranges, in a thread started with B over stage A's
              songs; passes over the rows in error repeat while they exceed 2% of those songs, at most four
              passes; a breaker inside each pass stops every fetcher once, past 300 calls, LRCLIB has answered
              502 / 503 / 504 to half of them (lyrics.stopped_by "lrclib_503"; rows not tried join the error list); each
              pass records LRCLIB's status counts. --no-lyrics skips the fetch: every song without a lyric file
              goes to the error list (lyrics.stopped_by "disabled"). E's seconds are the fetchers' wall time. After D the thread is
              joined, then the P4 theme (apply_lyric_theme), words per minute and language (artist-majority
              vote, then language_from_meta).
  F write     shard_<id>.features.jsonl.gz, shard_<id>.clap.f16.npy, shard_<id>.dropped.jsonl,
              shard_<id>.lyrics_error.txt (keys still in lyric error, one a line) and
              shard_<id>.manifest.json (artists, songs, dropped, SHA256s, describe commit, seconds per
              stage) to catalog/v3/; the uploaded files are downloaded back and their SHA256s checked;
              then, except for the fixture, the shard joins catalog/v3/MANIFEST.json and its keys
              catalog/v3/keys.txt; the shard's previews are deleted.

Resume: --next reads catalog/v3/MANIFEST.json and discovery/shards.json, takes the lowest shard that is
neither finished nor claimed by another runner within 6 hours (catalog/v3/claims/<runner>.<shard>.json), and
runs it from stage A. A runner holds at most two claims (the running shard and the one whose A runs under
it); a crash leaves both, and a restart takes its own claims first, the lower one first, each from stage A.

Logs: ml/data/harvest_v3_work/<shard>.describe.log (stages A to C, E, F) and <shard>.clap.log (stage D),
and current.describe.log / current.clap.log for the owner's two terminal windows, opened once a process,
each line tagged with its shard (two shards log at once while the next shard's A runs).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # describe workers single-threaded, as P4

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml" / "data"))

from app.audio.features import language_from_meta, lead_artist  # noqa: E402
import redescribe_catalog as rc  # noqa: E402  (the P4 describe, CLAP loader and theme)
import fetch_lyrics_remote as flr  # noqa: E402  (the LRCLIB flow and the application headers)
# harvest_v2 (artist match, track filter) and the recommender (track key, CLAP windows) import torch at
# module level; they are imported inside the stages, so the spawned describe workers do not load it.

REPO = "shreyash616/echo-ml-artifacts"
PREFIX = "catalog/v3"
WORK = ROOT / "ml" / "data" / "harvest_v3_work"
PREVIEWS = ROOT / "ml" / "data" / "previews"
LYRICS = ROOT / "ml" / "data" / "lyrics"
ANCHORS = ROOT / "ml" / "inference" / "anchors.json"
HEADER = ROOT / "ml" / "inference" / "header.json"
THEME_PROMPTS = ROOT / "ml" / "inference" / "theme_prompts.json"
MAX_PER_ARTIST = 20
ITUNES_SPACING_S = 3.0          # 20 requests a minute
FETCH_WORKERS = 8
DESCRIBE_WORKERS = 10
CLAP_BATCH = 32                 # 64 stalls 3 to 5 minutes on its first call (WP4; reviewer ce741ca)
LYRIC_WORKERS = 24
LYRIC_SPACING_S = 0.25          # shared across the fetchers under _LAST_LOCK: 4 requests a second app-wide (README 045f348)
LYRIC_MAX_PASSES = 4            # passes in all, the first included (README c2f8090)
LYRIC_ERROR_SHARE = 0.02        # another pass while the rows in error exceed this share of the shard's songs
LYRIC_503_BREAKER = 0.5         # the passes stop once LRCLIB answers 502 / 503 / 504 to this share of a pass's calls
SHED_STATUS = [f"{e} {c}" for e in ("get", "search") for c in (502, 503, 504)]  # README ba79bed
LYRIC_BREAKER_MIN_CALLS = 300   # ... counted from this many calls into the pass (README 045f348)
STALE_CLAIM_S = 6 * 3600
MAX_CLAIMS = 2                  # the running shard and the next shard's stage A
USER_AGENT = "echo-harvest-v3/1.0 (feel-based music discovery)"


# ---- storage: the dataset, or a local directory for tests ---------------------------------------
class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def read_json(self, path: str):
        p = self.root / path
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def read_text(self, path: str) -> str | None:
        p = self.root / path
        return p.read_text(encoding="utf-8") if p.exists() else None

    def put(self, local: Path, path: str) -> None:
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local).read_bytes())

    def fetch(self, path: str, dest_dir: Path) -> Path:
        dest = Path(dest_dir) / Path(path).name
        dest.write_bytes((self.root / path).read_bytes())
        return dest

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        return sorted(str(p.relative_to(self.root)).replace("\\", "/") for p in base.glob("*")) if base.exists() else []

    def delete(self, path: str) -> None:
        p = self.root / path
        if p.exists():
            p.unlink()


class DatasetStore:
    def __init__(self, repo: str = REPO):
        from huggingface_hub import HfApi
        self.repo = repo
        self.api = HfApi(token=os.environ.get("HF_TOKEN") or None)

    def _download(self, path: str, dest_dir: Path) -> Path | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            return Path(hf_hub_download(self.repo, path, repo_type="dataset", local_dir=str(dest_dir), force_download=True))
        except EntryNotFoundError:
            return None

    def read_json(self, path: str):
        p = self._download(path, WORK / "_store")
        return json.loads(p.read_text(encoding="utf-8")) if p else None

    def read_text(self, path: str) -> str | None:
        p = self._download(path, WORK / "_store")
        return p.read_text(encoding="utf-8") if p else None

    def put(self, local: Path, path: str) -> None:
        self.api.upload_file(path_or_fileobj=str(local), path_in_repo=path, repo_id=self.repo, repo_type="dataset",
                             commit_message=f"harvest_v3: {path}")

    def fetch(self, path: str, dest_dir: Path) -> Path:
        p = self._download(path, dest_dir)
        if p is None:
            raise FileNotFoundError(path)
        return p

    def list(self, prefix: str) -> list[str]:
        return sorted(f for f in self.api.list_repo_files(self.repo, repo_type="dataset") if f.startswith(prefix + "/"))

    def delete(self, path: str) -> None:
        self.api.delete_file(path, repo_id=self.repo, repo_type="dataset", commit_message=f"harvest_v3: release {path}")


# ---- helpers -------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_commit() -> str:
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", "backend/app/audio/features.py"],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()


_CURRENT: dict[tuple[str, str], logging.Handler] = {}
_LOG_FMT = logging.Formatter("%(asctime)s | %(levelname)s | %(shard)s | %(message)s")


class _ShardTag(logging.Filter):
    def __init__(self, shard: str):
        super().__init__()
        self.shard = shard

    def filter(self, record: logging.LogRecord) -> bool:
        record.shard = self.shard
        return True


def shard_logger(shard: str, kind: str) -> logging.Logger:
    """The shard's own <shard>.<kind>.log, plus current.<kind>.log, truncated once a process and shared by
    the shards it runs, so the owner's two Get-Content -Wait windows show both shards while they overlap."""
    WORK.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"harvest_v3.{shard}.{kind}")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        lg.addFilter(_ShardTag(shard))
        fh = logging.FileHandler(WORK / f"{shard}.{kind}.log", encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        cur = (str(WORK), kind)
        if cur not in _CURRENT:
            _CURRENT[cur] = logging.FileHandler(WORK / f"current.{kind}.log", mode="w", encoding="utf-8")
            _CURRENT[cur].setFormatter(_LOG_FMT)
        for h in (fh, sh):
            h.setFormatter(_LOG_FMT)
        for h in (fh, sh, _CURRENT[cur]):
            lg.addHandler(h)
        lg.propagate = False  # root's basicConfig handler would print each line twice
    return lg


def _describe_only(args):
    """redescribe_catalog._describe_one without the decoded audio (stage D decodes again), with its length."""
    res = rc._describe_one(args)
    y = res.pop("y", None)
    if y is not None:
        res["n22"] = int(len(y))
    return res


_ITUNES_LAST = [0.0]            # one pacer for every stage A in the process: 20 requests a minute in all
_ITUNES_LOCK = threading.Lock()


def _window_count(n22: int) -> int:
    """recommender._window_audio_frames' window count for a 22.05 kHz decode resampled to 48 kHz (for the log's total)."""
    n = int(np.ceil(n22 * 48000 / 22050))
    win = 480000
    k = n // win
    k += (n - k * win) >= win // 2
    return max(k, 1)


class Harvester:
    def __init__(self, shard: str, artists: list[dict], store, global_keys: set[str] | None = None,
                 global_itunes: set[str] | None = None, fixture: bool = False, runner: str | None = None):
        self.shard, self.artists, self.store = shard, artists, store
        self.global_keys = global_keys or set()
        self.global_itunes = global_itunes or set()
        self.fixture = fixture
        self.runner = runner or socket.gethostname()
        self.log = shard_logger(shard, "describe")
        self.clap_log = shard_logger(shard, "clap")
        self.seconds: dict[str, float] = {}
        self.seconds_other: dict[str, float] = {}
        self.a_overlapped = False
        self.no_lyrics = False
        self.songs: list[dict] = []
        self.dropped: list[dict] = []
        self.unresolved: list[dict] = []
        self.out = WORK / shard
        self.out.mkdir(parents=True, exist_ok=True)

    def _timed(self, stage: str, fn):
        t0 = time.time()
        self.log.info("stage %s start", stage)
        out = fn()
        self.seconds[stage] = round(time.time() - t0, 1)
        self.log.info("stage %s done in %.1f s", stage, self.seconds[stage])
        return out

    def drop(self, song: dict, stage: str, reason: str) -> None:
        self.dropped.append({"key": song.get("key"), "id": song.get("id"), "title": song.get("title"),
                             "artist": song.get("artist"), "stage": stage, "reason": reason})

    # ---- A ---------------------------------------------------------------------------------------
    def resolve(self) -> None:
        def get(url: str, params: dict) -> dict | None:
            for attempt in range(8):
                with _ITUNES_LOCK:
                    wait = ITUNES_SPACING_S - (time.time() - _ITUNES_LAST[0])
                    if wait > 0:
                        time.sleep(wait)
                    _ITUNES_LAST[0] = time.time()
                try:
                    r = http.get(url, params=params, timeout=20.0)
                except httpx.HTTPError as e:
                    self.log.info("iTunes %s: %s, retry", url.rsplit("/", 1)[-1], type(e).__name__)
                    time.sleep(5 * 2 ** attempt)
                    continue
                if r.status_code in (403, 429):
                    back = min(600, 30 * 2 ** attempt)
                    self.log.info("iTunes HTTP %d, backing off %d s", r.status_code, back)
                    time.sleep(back)
                    continue
                if r.status_code != 200:
                    return None
                try:
                    return r.json()
                except ValueError:  # a 200 with a non-JSON body: retried like a transient failure
                    self.log.info("iTunes %s: 200 without JSON, retry", url.rsplit("/", 1)[-1])
                    time.sleep(5 * 2 ** attempt)
            return None

        from harvest_v2 import _is_clean_track_name, _norm_str
        from app.services.recommender import make_track_key

        seen_keys, seen_ids = set(), set()
        with httpx.Client(headers={"User-Agent": USER_AGENT}) as http:
            for a in self.artists:
                name = a["name"]
                found = get("https://itunes.apple.com/search", {"term": name, "entity": "musicArtist", "limit": 10})
                results = (found or {}).get("results") or []
                match = next((x for x in results if _norm_str(x.get("artistName", "")) == _norm_str(name)), None)
                if not match:
                    self.unresolved.append({"deezer_id": a.get("deezer_id"), "name": name, "reason": "no iTunes artist match"})
                    continue
                look = get("https://itunes.apple.com/lookup", {"id": str(match["artistId"]), "entity": "song", "limit": 200, "country": "US"})
                kept = 0
                for it in (look or {}).get("results") or []:
                    if kept >= MAX_PER_ARTIST:
                        break
                    if it.get("wrapperType") != "track" or not it.get("previewUrl"):
                        continue
                    tname = it.get("trackName", "")
                    if not _is_clean_track_name(tname):
                        continue
                    key = make_track_key(it.get("artistName", name), tname)
                    tid = str(it["trackId"])
                    if key in self.global_keys or key in seen_keys or tid in self.global_itunes or tid in seen_ids:
                        continue
                    seen_keys.add(key)
                    seen_ids.add(tid)
                    release = it.get("releaseDate", "")
                    year = int(release[:4]) if len(release) >= 4 and release[:4].isdigit() else 0
                    # the record fields harvest_v2 wrote for every catalogue row
                    self.songs.append({
                        "id": f"it_{tid}", "itunes_id": tid, "deezer_id": None, "key": key, "title": tname,
                        "artist": it.get("artistName", name), "album": it.get("collectionName", ""),
                        "albumArtUrl": it.get("artworkUrl100", "").replace("100x100bb", "600x600bb"),
                        "previewUrl": it.get("previewUrl"), "durationMs": int(it.get("trackTimeMillis", 30000)),
                        "genre": it.get("primaryGenreName", ""), "year": year, "decade": f"{(year // 10) * 10}s" if year else "",
                        "_artist_deezer_id": a.get("deezer_id"),
                    })
                    kept += 1
                self.log.info("A %s: iTunes artist %s, %d songs", name, match["artistId"], kept)
        self.log.info("A: %d songs from %d artists, %d unresolved", len(self.songs), len(self.artists), len(self.unresolved))

    # ---- B ---------------------------------------------------------------------------------------
    def fetch(self) -> None:
        PREVIEWS.mkdir(parents=True, exist_ok=True)

        def one(song: dict) -> tuple[dict, str]:
            dest = PREVIEWS / f"{song['itunes_id']}.m4a"
            last = ""
            for _ in range(2):
                try:
                    r = http.get(song["previewUrl"], timeout=20.0)
                except httpx.HTTPError as e:
                    last = type(e).__name__
                    continue
                if r.status_code == 200 and len(r.content) > 5000:
                    dest.write_bytes(r.content)
                    return song, ""
                last = f"HTTP {r.status_code}, {len(r.content)} bytes"
            return song, f"preview fetch failed ({last})"

        with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as http, \
                ThreadPoolExecutor(FETCH_WORKERS) as pool:
            results = list(pool.map(one, self.songs))
        kept = []
        for song, reason in results:
            if reason:
                self.drop(song, "B", reason)
            else:
                kept.append(song)
        self.songs = kept
        self.log.info("B: %d previews, %d dropped", len(kept), len(self.dropped))

    # ---- C ---------------------------------------------------------------------------------------
    def describe(self) -> None:
        header = json.loads(HEADER.read_text(encoding="utf-8"))
        cut_points = header["cut_points"]
        jobs = [(s["key"], str(PREVIEWS / f"{s['itunes_id']}.m4a"), cut_points) for s in self.songs]
        by_key = {s["key"]: s for s in self.songs}
        kept = []
        t0 = time.time()
        with mp.Pool(DESCRIBE_WORKERS, initializer=rc._worker_init) as pool:
            for i, res in enumerate(pool.imap(_describe_only, jobs, chunksize=2)):
                song = by_key[res["key"]]
                if "drop" in res:
                    self.drop(song, "C", res["drop"])
                else:
                    song.update(res["dsp"])
                    song["_n22"] = res["n22"]
                    kept.append(song)
                if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                    self.log.info("C: %d/%d described | %.2f songs/s", i + 1, len(jobs), (i + 1) / max(time.time() - t0, 0.1))
        self.songs = kept

    # ---- D ---------------------------------------------------------------------------------------
    def clap(self) -> np.ndarray:
        import librosa
        import torch
        from app.audio.decode import decode_preview
        from app.services.recommender import _window_audio_frames

        model = rc.load_clap()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # streamed: songs are decoded and windowed only as the next batch needs them, and a batch's windows
        # are dropped once embedded, so memory holds one batch of audio, not the shard's (reviewer ce741ca)
        n_windows = sum(_window_count(s["_n22"]) for s in self.songs)
        n_batches = (n_windows + CLAP_BATCH - 1) // CLAP_BATCH
        per_song: list[list[np.ndarray]] = [[] for _ in self.songs]
        pending: list[tuple[int, np.ndarray]] = []
        state = {"b": 0, "windows": 0, "songs_done": 0}
        t0 = time.time()

        def embed(batch: list[tuple[int, np.ndarray]]) -> None:
            tb = time.time()
            with torch.no_grad():
                e = model.get_audio_embedding_from_data(x=np.stack([w for _, w in batch]), use_tensor=False)
            for (j, _), row in zip(batch, e):
                per_song[j].append(np.asarray(row, dtype=np.float32))
            state["b"] += 1
            state["windows"] += len(batch)
            mem = torch.cuda.memory_allocated() / 2 ** 20 if device == "cuda" else 0.0
            self.clap_log.info("CLAP batch %d/%d | windows %d | device %s | %.2f s | %.2f songs/s | cuda %.0f MB",
                               state["b"], n_batches, len(batch), device, time.time() - tb,
                               state["songs_done"] / max(time.time() - t0, 1e-6), mem)

        for j, s in enumerate(self.songs):
            y = decode_preview((PREVIEWS / f"{s['itunes_id']}.m4a").read_bytes())
            y48 = librosa.resample(y, orig_sr=22050, target_sr=48000, res_type="soxr_hq")
            pending += [(j, w.astype(np.float32)) for w in _window_audio_frames(y48, 48000, window_sec=10.0)]
            del y, y48
            state["songs_done"] = j
            while len(pending) >= CLAP_BATCH:
                embed(pending[:CLAP_BATCH])
                pending = pending[CLAP_BATCH:]
        state["songs_done"] = len(self.songs)
        if pending:
            embed(pending)
            pending = []
        out, kept = [], []
        anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
        prompts = anchors.get("prompts", anchors)
        anchor_keys = list(prompts)
        anchor_matrix = np.array([prompts[k] for k in anchor_keys], dtype=np.float32)
        for j, s in enumerate(self.songs):
            mean = np.stack(per_song[j]).mean(axis=0) if per_song[j] else np.full(512, np.nan, dtype=np.float32)
            norm = float(np.linalg.norm(mean))
            if not np.isfinite(mean).all() or norm < 1e-4:
                self.drop(s, "D", "CLAP embedding empty")
                continue
            emb = (mean / norm).astype(np.float32)
            cos = (anchor_matrix @ emb).tolist()
            s["anchor_cosines"] = cos
            s["vibes"] = [anchor_keys[k] for k in np.argsort(cos)[::-1][:2]]
            out.append(emb)
            kept.append(s)
        self.songs = kept
        self.clap_log.info("CLAP done: %d songs, %d windows, %.1f s", len(kept), state["windows"], time.time() - t0)
        return np.array(out, dtype=np.float32).reshape(len(out), 512)

    # ---- E ---------------------------------------------------------------------------------------
    @staticmethod
    def lyric_headers() -> dict:
        app = flr.registered_app()
        if "Lrclib-Api-Key" not in app:  # WP1: 24 fetchers only as the owner's registered application
            raise RuntimeError("LRCLIB application headers absent (backend/.env); not running 24 anonymous fetchers")
        return {"User-Agent": flr.USER_AGENT, **app}

    def lyrics(self, songs: list[dict], headers: dict) -> None:
        """Stage E's fetch over stage A's songs, run in a thread beside B to D: passes over the rows in error
        repeat while they exceed LYRIC_ERROR_SHARE of the songs, at most LYRIC_MAX_PASSES passes."""
        LYRICS.mkdir(parents=True, exist_ok=True)
        todo = [s for s in songs if not rc.lyric_path(s).exists()]
        if self.no_lyrics:  # README 8f54dec: LRCLIB is shedding the box, so the rows wait for the backfill
            self.lyric_error_keys = [r["key"] for r in todo]
            self.lyric_counts = {"songs": len(songs), "had_file": len(songs) - len(todo), "without_file": len(todo),
                                 "passes": [], "error_after_retry": len(todo), "stopped_by": "disabled"}
            self.log.info("E: --no-lyrics: nothing fetched, %d of %d songs without a file to the error list", len(todo), len(songs))
            return
        lock = threading.Lock()

        def status_now() -> Counter:
            with flr._STATUS_LOCK:
                return Counter({f"{e} {s}": c for (e, s), c in flr.LRCLIB_STATUS.items()})

        def run_pool(rows: list[dict]) -> dict:
            failed: list[dict] = []
            untried: list[dict] = []
            counts, types = Counter(), Counter()
            before = status_now()
            tripped = threading.Event()

            def check_breaker() -> None:
                # README 045f348: past LYRIC_BREAKER_MIN_CALLS calls in this pass, a 503 share at or above
                # LYRIC_503_BREAKER stops every worker before its next row
                d = status_now() - before
                calls = sum(d.values())
                if calls >= LYRIC_BREAKER_MIN_CALLS and sum(d[k] for k in SHED_STATUS) >= LYRIC_503_BREAKER * calls:
                    tripped.set()

            def worker(mine: list[dict]) -> None:
                with httpx.Client(headers=headers) as client:
                    for i, r in enumerate(mine):
                        if tripped.is_set():
                            with lock:
                                untried.extend(mine[i:])
                            return
                        try:
                            got = flr.fetch_lyrics(r, client, LYRICS, LYRIC_SPACING_S)
                        except Exception as e:  # any row's failure is that row's, never the thread's (reviewer ce741ca)
                            got, kind = None, "error"
                            with lock:
                                failed.append(r)
                                types[type(e).__name__] += 1
                        else:
                            kind = "fetched" if got else "none"
                        with lock:
                            counts[kind] += 1
                        check_breaker()

            n = max(1, min(LYRIC_WORKERS, len(rows)))
            threads = [threading.Thread(target=worker, args=(rows[i::n],), daemon=True) for i in range(n)] if rows else []
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # LRCLIB's answers in this pass, every request including _get's own 429 / 503 retries (README 0b03ff7)
            status = dict(sorted((status_now() - before).items()))
            # a pass is none + fetched + error (+ untried when the breaker fired); error_types is a breakdown of error
            return {"rows": len(rows), "none": counts["none"], "fetched": counts["fetched"], "error": counts["error"],
                    "untried": len(untried), "error_types": dict(types), "status": status,
                    "_failed": failed + untried, "_tripped": tripped.is_set()}

        passes, rows, stopped_by = [], todo, "no_rows" if not todo else "max_passes"
        while rows and len(passes) < LYRIC_MAX_PASSES:
            p = run_pool(rows)
            rows, tripped = p.pop("_failed"), p.pop("_tripped")
            passes.append(p)
            calls = sum(p["status"].values())
            shed = sum(p["status"].get(k, 0) for k in SHED_STATUS)
            self.log.info("E: pass %d over %d rows: %d fetched, %d none, %d error %s, %d not tried; status %s; 502/503/504 %d of %d calls",
                          len(passes), p["rows"], p["fetched"], p["none"], p["error"], p["error_types"], p["untried"],
                          p["status"], shed, calls)
            if tripped:
                stopped_by = "lrclib_503"
                self.log.info("E: breaker: LRCLIB answered 502/503/504 to %d of %d calls; passes stopped, %d rows never tried, "
                              "%d rows to the error list", shed, calls, p["untried"], len(rows))
                break
            if len(rows) <= LYRIC_ERROR_SHARE * len(songs):
                stopped_by = "error_share"
                break
        self.lyric_error_keys = [r["key"] for r in rows]
        self.lyric_counts = {"songs": len(songs), "had_file": len(songs) - len(todo), "without_file": len(todo),
                             "passes": passes, "error_after_retry": len(rows), "stopped_by": stopped_by}
        self.log.info("E: lyrics for %d of %d songs without a file, %d passes, %d still in error (%.1f%% of the songs)",
                      len(todo), len(songs), len(passes), len(rows), 100.0 * len(rows) / max(len(songs), 1))

    def text(self) -> None:
        """Stage E after the fetch: theme, words per minute and language, over the songs D kept."""
        from sentence_transformers import SentenceTransformer
        minilm = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        tp = json.loads(THEME_PROMPTS.read_text(encoding="utf-8"))
        prompts = tp.get("prompts", tp)
        theme_embs = minilm.encode(list(prompts.values()), normalize_embeddings=True)
        for s in self.songs:
            rc.apply_lyric_theme(s, minilm, theme_embs)
            if s["has_lyrics"]:
                words = len(rc.lyric_path(s).read_text(encoding="utf-8").split())
                s["lyric_words_per_minute"] = float(words / max(0.5, (s.get("durationMs") or 30000) / 60000.0))
            else:
                s["lyric_words_per_minute"] = 0.0
        # language: redescribe_catalog's resolution, the artist majority over the shard first
        artist_langs: dict[str, list[str]] = {}
        for s in self.songs:
            code, conf, src = language_from_meta(s)
            if code and conf >= 0.7 and src in ("lyrics", "script", "genre"):
                artist_langs.setdefault(lead_artist(s.get("artist")), []).append(code)
        for s in self.songs:
            row = dict(s)
            votes = artist_langs.get(lead_artist(s.get("artist")))
            if votes:
                row["_artist_majority_lang"] = Counter(votes).most_common(1)[0][0]
            p = rc.lyric_path(s)
            code, conf, src = language_from_meta(row, lyrics=p.read_text(encoding="utf-8") if p.exists() else None)
            s["language"], s["language_conf"], s["language_source"] = code, float(conf), src

    # ---- F ---------------------------------------------------------------------------------------
    def write(self, clap: np.ndarray) -> dict:
        stem = f"shard_{self.shard}"
        feats, clapf, dropf, manf = (self.out / f"{stem}.features.jsonl.gz", self.out / f"{stem}.clap.f16.npy",
                                     self.out / f"{stem}.dropped.jsonl", self.out / f"{stem}.manifest.json")
        lyrerr = self.out / f"{stem}.lyrics_error.txt"
        # the shard's songs whose lyric fetch still failed, one key a line, for a refetch once LRCLIB recovers
        kept = {s["key"] for s in self.songs}
        lyrerr.write_text("".join(f"{k}\n" for k in getattr(self, "lyric_error_keys", []) if k in kept), encoding="utf-8")
        with gzip.open(feats, "wt", encoding="utf-8") as f:
            for s in self.songs:
                f.write(json.dumps({k: v for k, v in s.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
        np.save(clapf, clap.astype(np.float16))
        with open(dropf, "w", encoding="utf-8") as f:
            for d in self.dropped:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        manifest = {
            "shard": self.shard, "fixture": self.fixture, "runner": self.runner,
            "artists": len(self.artists), "first_deezer_id": self.artists[0].get("deezer_id") if self.artists else None,
            "last_deezer_id": self.artists[-1].get("deezer_id") if self.artists else None,
            "artists_unresolved": len(self.unresolved), "songs": len(self.songs), "dropped": len(self.dropped),
            "dropped_by_stage": dict(Counter(d["stage"] for d in self.dropped)),
            "lyrics": getattr(self, "lyric_counts", {}),
            "files": {"features": f"{PREFIX}/{feats.name}", "clap": f"{PREFIX}/{clapf.name}", "dropped": f"{PREFIX}/{dropf.name}",
                      "lyrics_error": f"{PREFIX}/{lyrerr.name}"},
            "sha256": {"features": sha256(feats), "clap": sha256(clapf), "dropped": sha256(dropf), "lyrics_error": sha256(lyrerr)},
            "clap_shape": list(clap.shape), "describe_commit": describe_commit(),
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds_per_stage": dict(self.seconds),
            # stages overlap (E under B to D, this shard's A under the previous shard): the wall time is not their sum
            "seconds_other": dict(self.seconds_other), "a_overlapped": self.a_overlapped,
            "free_gb_at_start": round(getattr(self, "free_bytes_at_start", 0) / 1024 ** 3, 1),
            "keys": [s["key"] for s in self.songs],
            "itunes_ids": [s["itunes_id"] for s in self.songs],
        }
        manf.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        for p in (feats, clapf, dropf, lyrerr, manf):
            self.store.put(p, f"{PREFIX}/{p.name}")
        # read back: the manifest, then each file it names, hashed against the manifest
        check = WORK / "_readback" / self.shard
        check.mkdir(parents=True, exist_ok=True)
        remote = json.loads(self.store.fetch(f"{PREFIX}/{manf.name}", check).read_text(encoding="utf-8"))
        for kind, path in remote["files"].items():
            got = sha256(self.store.fetch(path, check))
            if got != remote["sha256"][kind] or got != manifest["sha256"][kind]:
                raise RuntimeError(f"{path}: uploaded SHA256 {got} does not match the manifest")
        self.log.info("F: uploaded and read back %s, SHA256s match", ", ".join(remote["files"].values()))
        return manifest

    def run(self) -> dict:
        self.stage_a()
        return self.run_rest()

    def stage_a(self) -> None:
        """The free-space check and stage A; run_shards runs it for the next shard in a thread."""
        # rule 23: under 8 GB free on the drive that holds the previews, stop before the shard starts
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(PREVIEWS).free
        self.free_bytes_at_start = free
        if free < rc.MIN_FREE_BYTES:
            self.log.info("shard %s not started: %.1f GB free on the previews drive, under 8 GB", self.shard, free / 1024 ** 3)
            raise SystemExit(f"{free / 1024 ** 3:.1f} GB free < 8 GB (rule 23)")
        self.log.info("shard %s: %d artists, runner %s, %.1f GB free", self.shard, len(self.artists), self.runner, free / 1024 ** 3)
        self._timed("A", self.resolve)

    def run_rest(self) -> dict:
        """B to F, with E's fetch in a thread from the start of B (README c2f8090)."""
        t_all = time.time()
        headers = {} if self.no_lyrics else self.lyric_headers()  # raises before any stage starts
        lyr: dict = {}

        songs_a = list(self.songs)  # stage A's songs, before B to D drop any

        def fetch_lyrics() -> None:
            t0 = time.time()
            self.log.info("stage E start (lyric fetch, beside B to D)")
            try:
                self.lyrics(songs_a, headers)
            except BaseException as e:  # re-raised in the main thread at the join
                lyr["error"] = e
            self.seconds["E"] = round(time.time() - t0, 1)
            self.log.info("stage E lyric fetch done in %.1f s", self.seconds["E"])

        th = threading.Thread(target=fetch_lyrics, name=f"lyrics-{self.shard}", daemon=True)
        th.start()
        self._timed("B", self.fetch)
        self._timed("C", self.describe)
        clap = self._timed("D", self.clap)
        t0 = time.time()
        th.join()
        self.seconds_other["E_wait_after_D"] = round(time.time() - t0, 1)
        if "error" in lyr:
            raise lyr["error"]
        t0 = time.time()
        self.text()
        self.seconds_other["E_theme_language"] = round(time.time() - t0, 1)
        self.log.info("stage E theme, words per minute and language done in %.1f s", self.seconds_other["E_theme_language"])
        t0 = time.time()
        manifest = self.write(clap)
        self.seconds["F"] = round(time.time() - t0, 1)
        # wall time of the shard: B to F, plus A when A was not hidden under the previous shard
        self.seconds_other["wall"] = round(time.time() - t_all + (0.0 if self.a_overlapped else self.seconds.get("A", 0.0)), 1)
        manifest["seconds_per_stage"] = dict(self.seconds)
        manifest["seconds_other"] = dict(self.seconds_other)
        # the timings now include F: rewrite and re-upload the manifest, then check it once more
        manf = self.out / f"shard_{self.shard}.manifest.json"
        manf.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        self.store.put(manf, f"{PREFIX}/{manf.name}")
        if self.store.read_json(f"{PREFIX}/{manf.name}")["seconds_per_stage"] != manifest["seconds_per_stage"]:
            raise RuntimeError("re-uploaded manifest does not read back")
        deleted = 0
        for s in self.songs + [d for d in self.dropped]:
            p = PREVIEWS / f"{s.get('id', '')[3:]}.m4a" if s.get("id", "").startswith("it_") else None
            if p and p.exists():
                p.unlink()
                deleted += 1
        self.log.info("shard %s done in %.0f s: %d songs, %d dropped, %d previews deleted",
                      self.shard, time.time() - t_all, len(self.songs), len(self.dropped), deleted)
        return manifest


# ---- claims and resume -----------------------------------------------------------------------------
def finished_shards(store) -> dict:
    return (store.read_json(f"{PREFIX}/MANIFEST.json") or {}).get("shards", {})


def claims(store) -> list[dict]:
    """Every claim on the store, each with the path it was read from."""
    out = []
    for path in store.list(f"{PREFIX}/claims"):
        c = store.read_json(path)
        if c:
            out.append({**c, "_path": path})
    return out


def claim_path(runner: str, shard: str) -> str:
    return f"{PREFIX}/claims/{runner}.{shard}.json"


def next_shard(store, shard_ids: list[str], runner: str, now: float | None = None,
               exclude: set[str] | frozenset = frozenset()) -> str | None:
    """The lowest shard neither finished nor claimed by another runner within STALE_CLAIM_S; a runner's
    own claims (a crash mid-shard, up to two) are resumed first, the lower first. `exclude`: shards this
    process already holds."""
    now = time.time() if now is None else now
    done = finished_shards(store)
    active = claims(store)
    mine = sorted({c["shard"] for c in active if c["runner"] == runner and c["shard"] not in done} - set(exclude))
    if mine:
        return mine[0]
    taken = {c["shard"] for c in active if c["runner"] != runner and now - c["t"] < STALE_CLAIM_S}
    return next((s for s in shard_ids if s not in done and s not in taken and s not in exclude), None)


def claim(store, shard: str, runner: str, now: float | None = None) -> None:
    p = WORK / f"claim_{runner}.{shard}.json"
    WORK.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"runner": runner, "shard": shard, "t": time.time() if now is None else now}), encoding="utf-8")
    store.put(p, claim_path(runner, shard))


def finish(store, manifest: dict, runner: str) -> None:
    """The shard joins catalog/v3/MANIFEST.json and its keys catalog/v3/keys.txt; the claim is released."""
    m = store.read_json(f"{PREFIX}/MANIFEST.json") or {"shards": {}}
    m["shards"][manifest["shard"]] = {k: manifest[k] for k in ("songs", "dropped", "sha256", "describe_commit", "date", "runner")}
    m["songs"] = sum(v["songs"] for v in m["shards"].values())
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / "MANIFEST.json"
    p.write_text(json.dumps(m, indent=1), encoding="utf-8")
    store.put(p, f"{PREFIX}/MANIFEST.json")
    keys = (store.read_text(f"{PREFIX}/keys.txt") or "") + "".join(f"{k}\t{i}\n" for k, i in zip(manifest["keys"], manifest["itunes_ids"]))
    kp = WORK / "keys.txt"
    kp.write_text(keys, encoding="utf-8")
    store.put(kp, f"{PREFIX}/keys.txt")
    # this runner's claims on this shard, whatever their file name (claims/<runner>.json before the overlap)
    for c in claims(store):
        if c["runner"] == runner and c["shard"] == manifest["shard"]:
            store.delete(c["_path"])


def global_seen(store) -> tuple[set[str], set[str]]:
    keys, ids = set(), set()
    for line in (store.read_text(f"{PREFIX}/keys.txt") or "").splitlines():
        if "\t" in line:
            k, i = line.split("\t", 1)
            keys.add(k)
            ids.add(i)
    return keys, ids


def discovery_shard_artists(store, shard: str) -> list[dict]:
    spec = store.read_json("discovery/shards.json")
    rows = [json.loads(l) for l in (store.read_text("discovery/artists.jsonl") or "").splitlines() if l.strip()]
    s = next(x for x in spec["shards"] if x["shard"] == shard)
    return rows[s["start"]:s["end"]]


def run_shards(store, runner: str, count: int = 1, harvester_cls=Harvester, no_lyrics: bool = False) -> list[dict]:
    """Up to `count` shards in this process. When a shard's A finishes and more are wanted, the next shard is
    claimed and its A runs in a thread under this shard's B to F (README c2f8090), deduplicated against the
    global keys and this shard's A songs, which reach keys.txt only when this shard finishes."""
    spec = store.read_json("discovery/shards.json")
    if not spec:
        raise RuntimeError("discovery/shards.json not on the dataset yet")
    ids = [s["shard"] for s in spec["shards"]]

    def start(shard: str, extra_keys: set[str], extra_ids: set[str]):
        claim(store, shard, runner)
        keys, itunes = global_seen(store)
        h = harvester_cls(shard, discovery_shard_artists(store, shard), store, keys | extra_keys, itunes | extra_ids,
                          runner=runner)
        h.no_lyrics = no_lyrics
        return h

    shard = next_shard(store, ids, runner)
    if shard is None or count < 1:
        return []
    h = start(shard, set(), set())
    h.stage_a()
    done: list[dict] = []
    while True:
        pre, pre_err, th = None, {}, None
        if len(done) + 1 < count:
            nxt = next_shard(store, ids, runner, exclude={h.shard})
            if nxt is not None:
                pre = start(nxt, {s["key"] for s in h.songs}, {s["itunes_id"] for s in h.songs})
                pre.a_overlapped = True

                def run_a(p=pre) -> None:
                    try:
                        p.stage_a()
                    except BaseException as e:  # re-raised in the main thread at the join
                        pre_err["error"] = e

                th = threading.Thread(target=run_a, name=f"stage-a-{nxt}", daemon=True)
                th.start()
        manifest = h.run_rest()
        finish(store, manifest, runner)
        done.append(manifest)
        if pre is None:
            return done
        th.join()
        if "error" in pre_err:
            raise pre_err["error"]
        h = pre


def run_next(store, runner: str, harvester_cls=Harvester) -> dict | None:
    done = run_shards(store, runner, 1, harvester_cls)
    return done[0] if done else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shard-file", help="a shard file: {'shard': id, 'artists': [...]}, run as given (the fixture)")
    g.add_argument("--next", action="store_true", help="claim and run the next unfinished discovery shard")
    ap.add_argument("--shards", type=int, default=1, help="with --next: shards to run, each next A under the current B to F")
    ap.add_argument("--runner", default=socket.gethostname())
    ap.add_argument("--store", default="dataset", help="'dataset', or a local directory")
    ap.add_argument("--max-artists", type=int, default=0, help="first N artists of the shard file only")
    ap.add_argument("--no-lyrics", action="store_true", help="skip E's fetch; songs without a file go to the error list")
    args = ap.parse_args()
    store = DatasetStore() if args.store == "dataset" else LocalStore(Path(args.store))
    if args.shard_file:
        spec = json.loads(Path(args.shard_file).read_text(encoding="utf-8"))
        artists = spec["artists"][:args.max_artists] if args.max_artists else spec["artists"]
        # a named shard is harvested and uploaded under its own name only: no global manifest, no keys
        h = Harvester(spec["shard"], artists, store, fixture=True, runner=args.runner)
        h.no_lyrics = args.no_lyrics
        manifest = h.run()
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("keys", "itunes_ids")}, indent=1, ensure_ascii=False))
        return
    for manifest in run_shards(store, args.runner, args.shards, no_lyrics=args.no_lyrics):
        print(json.dumps({k: v for k, v in manifest.items() if k not in ("keys", "itunes_ids")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
