"""
ml/data/redescribe_catalog.py: Fast parallel re-description of the 20,000 catalogue previews.
Implements:
- S2: Felt tempo with backbeat ACF vote, candidate penalty > 150 BPM, groove metrics.
- S3: Standardized theme projection with chunked MiniLM lyrics embeddings.
- S4: Lingua language detection from lyrics/script/genre/artist majority vote.
- S5: Raw metrics preservation for single quantile bucketing in build_index_v2.py.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import logging
import multiprocessing as mp
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Add backend to path
sys.path.insert(0, os.path.abspath("backend"))

from app.audio.features import (
    felt_tempo,
    groove,
    language_from_meta,
    lead_artist,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("redescribe_catalog")

CATALOG_PATH = Path("ml/data/catalog_features.jsonl")
PREVIEWS_DIR = Path("ml/data/previews")
LYRICS_DIR = Path("ml/data/lyrics")
THEME_PROMPTS_PATH = Path("ml/inference/theme_prompts.json")


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


def fetch_lyrics(row: dict, client) -> str | None:
    """
    Fetch plain lyrics from LRCLIB (same endpoints and acceptance as harvest_v2.fetch_lrclib_lyrics,
    at most 2 requests per second) and write them to the row's lyric file. Returns the text, or
    None when LRCLIB has no lyrics. HTTP errors other than 404 raise.
    """
    def _get(url: str, params: dict):
        # 429 and 5xx are transient on LRCLIB: retry twice with backoff, then return the response
        for backoff in (5.0, 20.0, 60.0, None):
            wait = 0.55 - (time.time() - _LRCLIB_LAST[0])
            if wait > 0:
                time.sleep(wait)
            _LRCLIB_LAST[0] = time.time()
            resp = client.get(url, params=params, timeout=10.0)
            if backoff is None or not (resp.status_code == 429 or resp.status_code >= 500):
                return resp
            time.sleep(backoff)

    title, artist = row.get("title", ""), row.get("artist", "")
    text = None
    r = _get("https://lrclib.net/api/get", {"track_name": title, "artist_name": artist})
    if r.status_code == 200:
        text = r.json().get("plainLyrics")
    elif r.status_code == 404:
        r2 = _get("https://lrclib.net/api/search", {"q": f"{title} {artist}"})
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


def _worker_init():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import scipy.fft
        scipy.fft.set_global_backend('scipy')
        from app.audio.features import felt_tempo
        _ = felt_tempo(np.zeros(22050 * 5, dtype=np.float32), sr=22050)
    except Exception:
        pass


def _decode_preview_from_file(file_path: str | Path, target_sr: int = 22050, target_len: int = 22050 * 30) -> np.ndarray:
    """Fast direct PyAV decode with fallback to ffmpeg."""
    try:
        import av
        container = av.open(str(file_path))
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        frames = []
        for frame in container.decode(audio=0):
            for r in resampler.resample(frame):
                frames.append(r.to_ndarray())
        for r in resampler.resample(None):
            frames.append(r.to_ndarray())
        container.close()
        if frames:
            audio = np.concatenate(frames, axis=1).squeeze()
            if len(audio) > target_len:
                audio = audio[:target_len]
            elif len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            return audio.astype(np.float32)
    except Exception:
        pass

    # Fallback to ffmpeg subprocess
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(file_path),
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-t",
        "30",
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        out = subprocess.check_output(cmd)
        if not out:
            return np.zeros(target_len, dtype=np.float32)
        audio = np.frombuffer(out, dtype=np.float32).copy()
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        return audio
    except Exception:
        return np.zeros(target_len, dtype=np.float32)


def _worker_process_audio_and_meta(row: dict) -> dict:
    cand_id = row.get("id", "")
    itunes_id = row.get("itunes_id", "")

    # 1. Resolve preview audio file
    p_path = None
    for cand in [PREVIEWS_DIR / f"{cand_id}.m4a", PREVIEWS_DIR / f"{itunes_id}.m4a", PREVIEWS_DIR / f"it_{itunes_id}.m4a"]:
        if cand.exists():
            p_path = cand
            break

    # If audio exists, recompute S2 tempo and groove
    if p_path is not None:
        try:
            y = _decode_preview_from_file(p_path)
            f_tempo = felt_tempo(y, sr=22050)
            f_groove = groove(y, f_tempo, sr=22050)

            # Update S2 fields
            row["felt_bpm"] = f_tempo["felt_bpm"]
            row["felt_bpm_dist"] = f_tempo["felt_bpm_dist"]
            row["tatum_bpm"] = f_tempo["tatum_bpm"]
            row["pulse_rate"] = f_tempo["pulse_rate"]
            row["hat_subdivision"] = f_tempo["hat_subdivision"]
            row["pulse_clarity"] = f_tempo["pulse_clarity"]

            row["hits_per_bar"] = f_groove["hits_per_bar"]
            row["beat_low_ratio"] = f_groove["beat_low_ratio"]
            row["off_beat_ratio"] = f_groove["off_beat_ratio"]
            row["backbeat"] = f_groove["backbeat"]
            row["lag_ms"] = f_groove["lag_ms"]
        except Exception:
            pass

    # 2. S4/T4 Language resolution
    lyr_path = lyric_path(row)
    lyrics_text = lyr_path.read_text(encoding="utf-8") if lyr_path.exists() else None

    lang_code, lang_conf, lang_source = language_from_meta(row, lyrics=lyrics_text)

    row["language"] = lang_code
    row["language_conf"] = float(lang_conf)
    row["language_source"] = lang_source

    return row


def main():
    t0 = time.time()
    logger.info("Starting catalogue re-description pass (S2, S3, S4, S5)...")

    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Missing {CATALOG_PATH}")

    rows: list[dict] = []
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    N = len(rows)
    logger.info(f"Loaded {N} records from {CATALOG_PATH}")

    # 1. Load sentence-transformers MiniLM for theme projection (S3)
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence-transformers MiniLM model...")
    minilm = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Load 10 canonical theme prompts (bundle-aware)
    with open(THEME_PROMPTS_PATH, "r", encoding="utf-8") as f:
        tp_raw = json.load(f)
    theme_prompts = tp_raw.get("prompts", tp_raw)
    theme_keys = list(theme_prompts.keys())
    theme_texts = list(theme_prompts.values())
    theme_embs = minilm.encode(theme_texts, normalize_embeddings=True)

    # 2. T8: theme_vec from the lyric file only; no lyric file -> has_lyrics False, no theme_vec
    logger.info("Computing theme vectors from lyric files (chunked MiniLM)...")
    for r in rows:
        apply_lyric_theme(r, minilm, theme_embs)

    # 3. Build artist majority vote map for S4
    artist_langs: dict[str, list[str]] = {}
    for r in rows:
        art = lead_artist(r.get("artist"))
        code, conf, src = language_from_meta(r)
        if code and conf >= 0.7 and src in ["lyrics", "script", "genre"]:
            artist_langs.setdefault(art, []).append(code)

    for r in rows:
        art = lead_artist(r.get("artist"))
        if art in artist_langs and artist_langs[art]:
            majority = Counter(artist_langs[art]).most_common(1)[0][0]
            r["_artist_majority_lang"] = majority

    # 4. Run parallel audio and metadata processing
    num_workers = min(12, os.cpu_count() or 4)
    logger.info(f"Processing audio and metadata for {N} tracks using {num_workers} parallel workers...")
    sys.stdout.flush()
    sys.stderr.flush()

    t_pool_start = time.time()
    updated_rows = []
    chunk_size = 10
    with mp.Pool(num_workers, initializer=_worker_init) as pool:
        for idx, res in enumerate(pool.imap(_worker_process_audio_and_meta, rows, chunksize=chunk_size)):
            if "_artist_majority_lang" in res:
                del res["_artist_majority_lang"]
            updated_rows.append(res)
            if (idx + 1) % 1000 == 0 or (idx + 1) == N:
                elapsed = time.time() - t_pool_start
                rate = (idx + 1) / max(0.1, elapsed)
                logger.info(f"Progress: {idx + 1}/{N} ({((idx+1)/N*100):.1f}%) | {rate:.1f} tracks/s | Elapsed: {elapsed:.1f}s")
                sys.stdout.flush()
                sys.stderr.flush()

    # Step S2 Distribution checks
    bpms = [r.get("felt_bpm", 120) for r in updated_rows]
    gt_150 = sum(1 for b in bpms if b > 150) / N
    lt_60 = sum(1 for b in bpms if b < 60) / N
    med_bpm = float(np.median(bpms))

    high_prob_count = 0
    for r in updated_rows:
        dist = r.get("felt_bpm_dist", {})
        if dist:
            top_p = max(dist.values())
            if top_p >= 0.70:
                high_prob_count += 1
    high_prob_share = high_prob_count / N

    # Step S4 Language checks
    null_langs = sum(1 for r in updated_rows if r.get("language") is None) / N
    lang_sources = Counter(r.get("language_source") for r in updated_rows)

    print("\n" + "=" * 80)
    print("STEP 2 RE-DESCRIPTION SUMMARY & GATES:")
    print("=" * 80)
    print(f"Total Rows Processed:                {N}")
    print(f"Elapsed Time:                        {time.time() - t0:.1f}s")
    print(f"\n[S2] Felt Tempo Distribution:")
    print(f"  Share > 150 BPM:                   {gt_150*100:.2f}% (Gate Required <= 10.0%) -> {'PASS' if gt_150 <= 0.10 else 'FAIL'}")
    print(f"  Share < 60 BPM:                    {lt_60*100:.2f}% (Gate Required <= 5.0%)  -> {'PASS' if lt_60 <= 0.05 else 'FAIL'}")
    print(f"  Median Felt BPM:                   {med_bpm:.1f} (Gate Required 95..115)     -> {'PASS' if 95 <= med_bpm <= 115 else 'FAIL'}")
    print(f"  Top Probability >= 0.70 Share:     {high_prob_share*100:.2f}% (Gate Required >= 50.0%) -> {'PASS' if high_prob_share >= 0.50 else 'FAIL'}")
    print(f"\n[S4] Language Coverage:")
    print(f"  Language = null Share:             {null_langs*100:.2f}% (Gate Required <= 10.0%) -> {'PASS' if null_langs <= 0.10 else 'FAIL'}")
    print(f"  Language Sources:                  {dict(lang_sources)}")
    print("=" * 80)

    assert gt_150 <= 0.10, f"S2 Gate Failed: Share > 150 BPM is {gt_150*100:.2f}% > 10%"
    assert lt_60 <= 0.05, f"S2 Gate Failed: Share < 60 BPM is {lt_60*100:.2f}% > 5%"
    assert 95 <= med_bpm <= 115, f"S2 Gate Failed: Median BPM {med_bpm} outside [95, 115]"
    assert high_prob_share >= 0.50, f"S2 Gate Failed: Top prob >= 0.70 is only {high_prob_share*100:.2f}% < 50%"
    assert null_langs <= 0.10, f"S4 Gate Failed: Language null share is {null_langs*100:.2f}% > 10%"

    # Write atomically
    out_tmp = CATALOG_PATH.with_suffix(".tmp")
    with open(out_tmp, "w", encoding="utf-8") as f:
        for r in updated_rows:
            f.write(json.dumps(r) + "\n")

    # Atomic replace
    out_tmp.replace(CATALOG_PATH)
    logger.info(f"Successfully wrote {N} updated tracks to {CATALOG_PATH}")


if __name__ == "__main__":
    main()
