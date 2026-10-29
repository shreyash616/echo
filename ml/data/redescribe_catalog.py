"""
ml/data/redescribe_catalog.py: T1, the full data pass over the catalogue (FIX_PLAN_2.md section 4).

For every row: reuse the preview in ml/data/previews (fetch it once from previewUrl, re-resolving
through iTunes when the URL is dead, only when absent), decode it with the pinned decoder, run
describe_audio with the header cut points, compute the 10 s CLAP windows, then the T8 theme from the
lyric file (fetched from LRCLIB when absent) and the T4 language. Rows without a decoded preview are
dropped to DROPPED.jsonl with the reason (rule 17).

Disk: previews are described and deleted in batches of at most 5,000; free disk is checked before
each batch (stop under 8 GB). Each batch is checkpointed in ml/data/redescribe_work/ before its
previews are deleted, so a rerun resumes. The final batch's previews are kept for the describe
parity test until `--delete-remaining-previews`.

Outputs: catalog_features.jsonl, clap.npy (row-aligned), DROPPED.jsonl, HARVEST_MANIFEST.json.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import asyncio
import hashlib
import json
import logging
import multiprocessing as mp
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.abspath("backend"))

from app.audio.decode import decode_preview
from app.audio.features import describe_audio, language_from_meta, lead_artist

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("redescribe_catalog")

CATALOG_PATH = Path("ml/data/catalog_features.jsonl")
PREVIEWS_DIR = Path("ml/data/previews")
LYRICS_DIR = Path("ml/data/lyrics")
THEME_PROMPTS_PATH = Path("ml/inference/theme_prompts.json")
ANCHORS_PATH = Path("ml/inference/anchors.json")
TRACK_METADATA_PATH = Path("ml/inference/track_metadata.json")
CLAP_NPY_PATH = Path("ml/data/clap.npy")
DROPPED_PATH = Path("ml/data/DROPPED.jsonl")
MANIFEST_PATH = Path("ml/data/HARVEST_MANIFEST.json")
WORK_DIR = Path("ml/data/redescribe_work")
CLAP_CKPT = Path(os.path.expanduser("~/.cache/clap/music_audioset_epoch_15_esc_90.14.pt"))

BATCH_SIZE = 5000
MIN_FREE_BYTES = 8 * 1024**3
MIN_CREST_DB = 6.0  # silent or corrupt previews, as at harvest (section 11.4)
USER_AGENT = "echo-redescribe/1.0 (feel-based music discovery)"


def _chunk_lyrics_on_lines(text: str, max_words: int = 40) -> list[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    chunks = []
    curr = []
    curr_w = 0
    for l in lines:
        w = len(l.split())
        if curr_w + w > max_words and curr:
            chunks.append("\n".join(curr))
            curr = [l]
            curr_w = w
        else:
            curr.append(l)
            curr_w += w
    if curr:
        chunks.append("\n".join(curr))
    return chunks or [text]


def lyric_path(row: dict) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", row.get("id", ""))
    return LYRICS_DIR / f"{safe_id}.txt"


_LRCLIB_LAST = [0.0]
LRCLIB_SPACING_S = 0.70  # at most 85 requests a minute, under LRCLIB's ~90 for an identified UA
LRCLIB_STATUS: Counter = Counter()  # (endpoint, status) for the pass log


def fetch_lyrics(row: dict, client) -> str | None:
    """
    Fetch plain lyrics from LRCLIB and write them to the row's lyric file. /api/get with duration
    and album, then /api/get without album, then /api/search by track and artist (artist-substring
    acceptance as harvest_v2.fetch_lrclib_lyrics). One request every 0.70 s; 429 waits Retry-After,
    503 waits 10 s and retries once. Returns the text, or None when LRCLIB has no lyrics. HTTP
    errors other than 404 raise.
    """
    def _get(endpoint: str, params: dict):
        retried_503 = False
        while True:
            wait = LRCLIB_SPACING_S - (time.time() - _LRCLIB_LAST[0])
            if wait > 0:
                time.sleep(wait)
            _LRCLIB_LAST[0] = time.time()
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
    lyric_path(row).write_text(text.strip(), encoding="utf-8")
    return text.strip()


def apply_lyric_theme(row: dict, minilm, theme_embs: np.ndarray) -> None:
    """T8 and rule 17: has_lyrics is true only when a lyric file was read; theme_vec only then."""
    path = lyric_path(row)
    row["theme"] = None
    if not path.exists():
        row["has_lyrics"] = False
        row.pop("theme_vec", None)
        return
    txt = path.read_text(encoding="utf-8")
    c_embs = minilm.encode(_chunk_lyrics_on_lines(txt, max_words=40), normalize_embeddings=True)
    mean_emb = np.mean(c_embs, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    row["theme_vec"] = [float(c) for c in (theme_embs @ mean_emb).tolist()]
    row["has_lyrics"] = True


def preview_candidates(row: dict) -> list[Path]:
    return [PREVIEWS_DIR / f"{row.get('id')}.m4a", PREVIEWS_DIR / f"{row.get('itunes_id')}.m4a",
            PREVIEWS_DIR / f"it_{row.get('itunes_id')}.m4a"]


def existing_preview(row: dict) -> Path | None:
    return next((p for p in preview_candidates(row) if p.exists()), None)


def fetch_preview(row: dict, client: httpx.Client) -> tuple[Path | None, str]:
    """
    A preview missing from disk is fetched once (one retry) from previewUrl; a dead URL is
    re-resolved through iTunes US. Returns (path, "") or (None, reason).
    """
    dest = PREVIEWS_DIR / f"{re.sub(r'[^a-zA-Z0-9_-]', '_', row.get('id', ''))}.m4a"

    def _download(url: str) -> str:
        last = ""
        for _ in range(2):
            try:
                r = client.get(url, timeout=12.0)
            except httpx.HTTPError as e:
                last = type(e).__name__
                continue
            if r.status_code == 200 and len(r.content) > 5000:
                dest.write_bytes(r.content)
                return ""
            last = f"HTTP {r.status_code}, {len(r.content)} bytes"
        return last

    reason = _download(row["previewUrl"]) if row.get("previewUrl") else "no previewUrl"
    if not reason:
        return dest, ""
    from harvest_v2 import resolve_itunes_preview

    async def _resolve():
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as ac:
            return await resolve_itunes_preview(ac, row.get("title", ""), row.get("artist", ""),
                                                expected_key=row.get("key"))

    rec = asyncio.run(_resolve())
    if not rec or not rec.get("previewUrl"):
        return None, f"preview fetch failed ({reason}); iTunes re-resolve found nothing"
    reason2 = _download(rec["previewUrl"])
    if reason2:
        return None, f"preview fetch failed ({reason}); re-resolved URL failed ({reason2})"
    row["previewUrl"] = rec["previewUrl"]
    return dest, ""


def _worker_init():
    import scipy.fft
    scipy.fft.set_global_backend("scipy")


def _describe_one(args: tuple[str, str, dict]) -> dict:
    """Decode with the pinned decoder and describe. Failures come back as drops with the reason."""
    key, path, cut_points = args
    try:
        y = decode_preview(Path(path).read_bytes())
    except (OSError, ValueError) as e:
        return {"key": key, "drop": f"decode failed: {type(e).__name__}: {e}"}
    if float(np.max(np.abs(y))) < 1e-6:
        return {"key": key, "drop": "decode failed: ffmpeg returned no audio"}
    try:
        dsp = describe_audio(y, meta={}, sr=22050, cut_points=cut_points)
    except (ValueError, FloatingPointError, IndexError, ZeroDivisionError) as e:
        return {"key": key, "drop": f"describe failed: {type(e).__name__}: {e}"}
    if dsp.get("crest_factor", 0.0) < MIN_CREST_DB:
        return {"key": key, "drop": f"silent or corrupt preview: crest_factor "
                                    f"{dsp.get('crest_factor', 0.0):.2f} < {MIN_CREST_DB}"}
    for k in ("language", "language_conf", "language_source"):
        dsp.pop(k, None)
    return {"key": key, "dsp": dsp, "y": y}


def load_clap():
    import laion_clap
    import torch
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(CLAP_CKPT))
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def clap_windows(model, y_22k: np.ndarray) -> np.ndarray:
    """10 s windows at 48 kHz, embedded, averaged, L2-normalised (as harvest_v2.embed_clap_windows)."""
    import librosa
    import torch
    from app.services.recommender import _window_audio_frames
    y_48k = librosa.resample(y_22k, orig_sr=22050, target_sr=48000, res_type="soxr_hq")
    embs = []
    with torch.no_grad():
        for win in _window_audio_frames(y_48k, 48000, window_sec=10.0):
            embs.append(model.get_audio_embedding_from_data(x=win[None, :], use_tensor=False)[0])
    mean_emb = np.mean(embs, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / norm if norm > 1e-8 else mean_emb


class LyricFetcher(threading.Thread):
    """
    LRCLIB pass for rows without a lyric file, alongside the audio batches. Rows LRCLIB has no
    lyrics for are remembered in the work dir; rows that still fail after fetch_lyrics' retries are
    retried once at the end of the pass, then counted as no lyrics.
    """

    def __init__(self, rows: list[dict], work_dir: Path):
        super().__init__(daemon=True)
        self.rows = rows
        self.none_path = work_dir / "lyrics_none.txt"
        self.counts: Counter = Counter()
        self.failure: BaseException | None = None

    def _pass(self, todo, client, none_f):
        failed = []
        for i, r in enumerate(todo):
            try:
                text = fetch_lyrics(r, client)
            except httpx.HTTPError as e:
                failed.append(r)
                self.counts["error"] += 1
                logger.info("LRCLIB error on %s: %s", r["key"], str(e)[:100])
                continue
            if text:
                self.counts["fetched"] += 1
            else:
                self.counts["none"] += 1
                none_f.write(r["key"] + "\n")
                none_f.flush()
            if i + 1 == 300:
                logger.info("LRCLIB status after 300 rows: %s",
                            {f"{e} {s}": n for (e, s), n in sorted(LRCLIB_STATUS.items())})
            if (i + 1) % 500 == 0:
                logger.info("Lyrics: %d/%d %s", i + 1, len(todo), dict(self.counts))
        return failed

    def run(self):
        try:
            self._run()
        except BaseException as e:  # surfaced by main() after join
            self.failure = e
            raise

    def _run(self):
        none_keys = set(self.none_path.read_text(encoding="utf-8").split()) if self.none_path.exists() else set()
        seen: set[Path] = set()
        todo = []
        for r in self.rows:
            lp = lyric_path(r)
            if r["key"] in none_keys or lp.exists() or lp in seen:
                continue
            seen.add(lp)
            todo.append(r)
        self.counts["to_fetch"] = len(todo)
        logger.info("Lyrics: %d rows without a lyric file to fetch from LRCLIB", len(todo))
        with httpx.Client(headers={"User-Agent": USER_AGENT}) as client, \
                open(self.none_path, "a", encoding="utf-8") as none_f:
            failed = self._pass(todo, client, none_f)
            logger.info("Lyrics: first pass %s; retrying %d errors once", dict(self.counts), len(failed))
            self.counts["retried"] = len(failed)
            self.counts["first_pass_errors"] = self.counts.pop("error", 0)
            still = self._pass(failed, client, none_f)
        self.counts["error_after_retry"] = len(still)
        self.counts.pop("error", None)
        logger.info("Lyrics: done %s", dict(self.counts))
        logger.info("LRCLIB status, whole pass: %s",
                    {f"{e} {s}": n for (e, s), n in sorted(LRCLIB_STATUS.items())})


def run_batch(b: int, batch: list[dict], cut_points: dict, clap_model, anchor_keys, anchor_matrix,
              pool, http: httpx.Client, work_dir: Path) -> None:
    """Describe one batch and checkpoint it: batch_<b>_clap.npy, then batch_<b>.jsonl as the marker."""
    free = shutil.disk_usage(PREVIEWS_DIR).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"Stopping before batch {b}: {free / 1024**3:.1f} GB free < 8 GB")
    logger.info("Batch %d: %d rows, %.1f GB free", b, len(batch), free / 1024**3)

    by_key: dict[str, dict] = {}
    jobs = []
    for r in batch:
        rec = by_key.setdefault(r["key"], {"key": r["key"]})
        p = existing_preview(r)
        if p is None:
            p, reason = fetch_preview(r, http)
            if p is None:
                rec["drop"] = reason
                continue
            rec.update(fetched=True, previewUrl=r["previewUrl"])
        jobs.append((r["key"], str(p), cut_points))

    embs = []
    t0 = time.time()
    for i, res in enumerate(pool.imap(_describe_one, jobs, chunksize=4)):
        rec = by_key[res["key"]]
        if "drop" in res:
            rec["drop"] = res["drop"]
        else:
            emb = clap_windows(clap_model, res["y"])
            if not np.isfinite(emb).all() or np.linalg.norm(emb) < 1e-4:
                rec["drop"] = "CLAP embedding empty"
            else:
                cos = (anchor_matrix @ emb).tolist()
                rec.update(dsp=res["dsp"], anchor_cosines=cos, clap_idx=len(embs),
                           vibes=[anchor_keys[j] for j in np.argsort(cos)[::-1][:2]])
                embs.append(emb)
        if (i + 1) % 500 == 0 or i + 1 == len(jobs):
            logger.info("Batch %d: %d/%d described | %.2f rows/s", b, i + 1, len(jobs),
                        (i + 1) / max(time.time() - t0, 0.1))

    np.save(work_dir / f"batch_{b}_clap.npy", np.array(embs, dtype=np.float32).reshape(len(embs), -1))
    tmp = work_dir / f"batch_{b}.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in batch:
            f.write(json.dumps(by_key[r["key"]]) + "\n")
    tmp.replace(work_dir / f"batch_{b}.jsonl")


def delete_previews(paths) -> int:
    n = 0
    for p in paths:
        if p.exists():
            p.unlink()
            n += 1
    return n


def theme_labels(rows: list[dict], theme_keys: list[str]) -> list[str | None]:
    """The build_index_v2.py rule (std floor 0.08, argmax z when max z >= 0.5), for the manifest."""
    tv = [r["theme_vec"] for r in rows if r.get("has_lyrics")]
    if not tv:
        return [None] * len(rows)
    mat = np.array(tv, dtype=np.float32)
    means, stds = mat.mean(0), np.maximum(mat.std(0), 0.08)
    out: list[str | None] = []
    for r in rows:
        if not r.get("has_lyrics"):
            out.append(None)
            continue
        z = (np.array(r["theme_vec"]) - means) / stds
        out.append(theme_keys[int(np.argmax(z))] if z.max() >= 0.5 else None)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dist(values) -> dict:
    c = Counter(str(v) for v in values)
    n = max(1, sum(c.values()))
    return {k: {"count": v, "share": round(v / n, 4)} for k, v in c.most_common()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="dry run on the first N rows: own work dir, no deletes, no catalogue write")
    ap.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--delete-remaining-previews", action="store_true",
                    help="after the parity test: delete every file left in ml/data/previews and exit")
    ap.add_argument("--lrclib-spacing", type=float, default=LRCLIB_SPACING_S,
                    help="seconds between LRCLIB requests (2.0 when the exit IP is rejected at 0.70)")
    args = ap.parse_args()
    globals()["LRCLIB_SPACING_S"] = args.lrclib_spacing

    if args.delete_remaining_previews:
        n = delete_previews(list(PREVIEWS_DIR.iterdir()))
        logger.info("Deleted %d remaining previews; %d files left", n, len(list(PREVIEWS_DIR.iterdir())))
        return

    t0 = time.time()
    dry = args.limit > 0
    work_dir = WORK_DIR / "dry_run" if dry else WORK_DIR
    work_dir.mkdir(parents=True, exist_ok=True)
    # the pass reads the catalogue as it was when the pass started, so a resumed run matches its checkpoints
    source = work_dir / "catalog_features.source.jsonl"
    if not source.exists():
        shutil.copyfile(CATALOG_PATH, source)
        shutil.copyfile(CLAP_NPY_PATH, work_dir / "clap.source.npy")
    rows = [json.loads(l) for l in open(source, encoding="utf-8") if l.strip()]
    if dry:
        rows = rows[: args.limit]
    logger.info("Loaded %d rows (%s), %d workers, LRCLIB every %.2f s", len(rows), "dry run" if dry else "full pass",
                args.workers, LRCLIB_SPACING_S)

    with open(TRACK_METADATA_PATH, encoding="utf-8") as f:
        cut_points = json.load(f)["header"]["cut_points"]
    with open(ANCHORS_PATH, encoding="utf-8") as f:
        anchors_raw = json.load(f)
    anchors = anchors_raw["prompts"] if "prompts" in anchors_raw else anchors_raw  # bundle, as build_index_v2
    anchor_keys = list(anchors)
    anchor_matrix = np.array([anchors[k] for k in anchor_keys], dtype=np.float32)

    lyrics = LyricFetcher(rows, work_dir)
    lyrics.start()

    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    # a preview shared by several rows (duplicate ids) is deleted after the last batch that reads it
    last_use: dict[Path, int] = {}
    for b, batch in enumerate(batches):
        for r in batch:
            for p in preview_candidates(r):
                last_use[p] = b

    clap_model = load_clap()
    with mp.Pool(args.workers, initializer=_worker_init) as pool, \
            httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as http:
        for b, batch in enumerate(batches):
            if (work_dir / f"batch_{b}.jsonl").exists():
                logger.info("Batch %d: checkpoint present, skipped", b)
            else:
                run_batch(b, batch, cut_points, clap_model, anchor_keys, anchor_matrix, pool, http, work_dir)
            if not dry and b < len(batches) - 1:
                n = delete_previews(p for p, lb in last_use.items() if lb == b)
                logger.info("Batch %d: deleted %d previews", b, n)

    logger.info("Audio batches done in %.0fs; waiting for the lyric pass", time.time() - t0)
    lyrics.join()
    if lyrics.failure is not None:
        raise RuntimeError("lyric pass failed") from lyrics.failure

    # Assemble: drop rows without a described preview, attach DSP, CLAP, theme and language
    from sentence_transformers import SentenceTransformer
    minilm = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    with open(THEME_PROMPTS_PATH, encoding="utf-8") as f:
        tp_raw = json.load(f)
    theme_prompts = tp_raw.get("prompts", tp_raw)
    theme_keys = list(theme_prompts)
    theme_embs = minilm.encode(list(theme_prompts.values()), normalize_embeddings=True)

    kept, dropped, embs = [], [], []
    fetched_previews = 0
    for b, batch in enumerate(batches):
        clap_b = np.load(work_dir / f"batch_{b}_clap.npy")
        recs = {}
        for line in open(work_dir / f"batch_{b}.jsonl", encoding="utf-8"):
            rec = json.loads(line)
            recs[rec["key"]] = rec
        for r in batch:
            rec = recs[r["key"]]
            fetched_previews += bool(rec.get("fetched"))
            if "drop" in rec:
                dropped.append({"key": r["key"], "id": r.get("id"), "title": r.get("title"),
                                "artist": r.get("artist"), "reason": rec["drop"]})
                continue
            if rec.get("previewUrl"):
                r["previewUrl"] = rec["previewUrl"]
            r.update(rec["dsp"])
            r["anchor_cosines"] = rec["anchor_cosines"]
            r["vibes"] = rec["vibes"]
            embs.append(clap_b[rec["clap_idx"]])
            kept.append(r)
    logger.info("Assembled %d rows, dropped %d", len(kept), len(dropped))

    for r in kept:
        apply_lyric_theme(r, minilm, theme_embs)
        if r["has_lyrics"]:
            words = len(lyric_path(r).read_text(encoding="utf-8").split())
            r["lyric_words_per_minute"] = float(words / max(0.5, (r.get("durationMs") or 30000) / 60000.0))
        else:
            r["lyric_words_per_minute"] = 0.0

    artist_langs: dict[str, list[str]] = {}
    for r in kept:
        code, conf, src = language_from_meta(r)
        if code and conf >= 0.7 and src in ("lyrics", "script", "genre"):
            artist_langs.setdefault(lead_artist(r.get("artist")), []).append(code)
    for r in kept:
        row = dict(r)
        votes = artist_langs.get(lead_artist(r.get("artist")))
        if votes:
            row["_artist_majority_lang"] = Counter(votes).most_common(1)[0][0]
        p = lyric_path(r)
        text = p.read_text(encoding="utf-8") if p.exists() else None
        code, conf, src = language_from_meta(row, lyrics=text)
        r["language"], r["language_conf"], r["language_source"] = code, float(conf), src

    write_outputs(kept, dropped, np.array(embs, dtype=np.float32), lyrics, fetched_previews, theme_keys, dry, t0)


def write_outputs(kept, dropped, embs, lyrics, fetched_previews, theme_keys, dry, t0):
    N = len(kept)
    bpm = np.array([r["felt_bpm"] for r in kept], dtype=float)
    obr = np.array([r["off_beat_ratio"] for r in kept], dtype=float)
    sustain = np.array([r["sustain_ratio"] for r in kept], dtype=float)
    themes = theme_labels(kept, theme_keys)
    with_file = [r for r in kept if lyric_path(r).exists()]
    lyr_src = sum(1 for r in with_file if r["language_source"] == "lyrics")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "ml/data/redescribe_catalog.py",
                                     "backend/app/audio"], text=True).strip()

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command_line": "py " + " ".join(sys.argv),
        "describe_commit": commit + (" (dirty)" if dirty else ""),
        "rows": N,
        "previews_decoded": N,
        "rows_described_from_audio": N,
        "rows_with_lyric_file": len(with_file),
        "rows_dropped": len(dropped),
        "previews_fetched": fetched_previews,
        "lyrics_pass": dict(lyrics.counts),
        "felt_bpm": {"share_gt_150": round(float(np.mean(bpm > 150)), 4),
                     "share_lt_60": round(float(np.mean(bpm < 60)), 4),
                     "median": float(np.median(bpm)),
                     "histogram_20bpm": {f"{lo}-{lo + 19}": int(np.sum((bpm >= lo) & (bpm < lo + 20)))
                                         for lo in range(40, 220, 20)}},
        "kick_class": _dist(r["kick_class"] for r in kept),
        "off_beat_ratio": {f"p{q}": round(float(np.percentile(obr, q)), 4) for q in (5, 50, 95)},
        "restraint_bucket": _dist(r["restraint_bucket"] for r in kept),
        "language": _dist(r["language"] for r in kept),
        "language_source": _dist(r["language_source"] for r in kept),
        "language_source_lyrics_share_of_lyric_files": round(lyr_src / max(1, len(with_file)), 4),
        "theme": _dist(t for t in themes if t is not None),
        "theme_null": sum(1 for t in themes if t is None),
        "feature_std": {k: round(float(np.std([r[k] for r in kept])), 4)
                        for k in ("saturation", "sustain_ratio", "darkness", "grit", "intimacy")},
        "sustain_ratio_share_at_1": round(float(np.mean(sustain >= 0.9999)), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }

    if dry:
        print(json.dumps(manifest, indent=2))
        print("dry run drops:", json.dumps(dropped[:10], indent=2))
        return

    tmp = CATALOG_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    tmp.replace(CATALOG_PATH)
    np.save(CLAP_NPY_PATH, embs)
    with open(DROPPED_PATH, "w", encoding="utf-8") as f:
        for d in dropped:
            f.write(json.dumps(d) + "\n")
    manifest["catalog_features_file"] = CATALOG_PATH.name
    manifest["catalog_features_sha256"] = sha256(CATALOG_PATH)
    manifest["clap_npy_sha256"] = sha256(CLAP_NPY_PATH)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    logger.info("Wrote %d rows, clap.npy %s, %d dropped", N, embs.shape, len(dropped))


if __name__ == "__main__":
    main()
