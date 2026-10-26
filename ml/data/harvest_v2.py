"""
ml/data/harvest_v2.py: Step 2 Catalogue Harvester (Section 11.4).

Harvests 20k to 25k mainstream and discovery tracks from Deezer mood playlists,
charts, and discovery artist seeds with iTunes fallback. Computes 11.5 DSP
descriptors and CLAP embeddings, writes catalog_features.jsonl and playlist_membership.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import httpx
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

sys.path.insert(0, os.path.abspath("backend"))

from app.audio.decode import decode_preview
from app.audio.features import describe_audio
from app.services.deezer import search_track

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("harvest_v2")

PREVIEWS_DIR = Path("ml/data/previews")
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_FEATURES_PATH = Path("ml/data/catalog_features.jsonl")
PLAYLIST_MEMBERSHIP_PATH = Path("ml/data/playlist_membership.jsonl")
CLAP_NPY_PATH = Path("ml/data/clap.npy")
ANCHORS_PATH = Path("ml/inference/anchor_prompts.json")
HARVEST_SEEDS_PATH = Path("ml/data/harvest_seed_artists.json")
BENCHMARK_PATH = Path("tests/feel_benchmark_v2.json")

DEEZER_BASE = "https://api.deezer.com"

# Mood words by language (Section 11.4.1)
MOOD_WORDS: dict[str, list[str]] = {
    "en": [
        "sad", "melancholy", "heartbreak", "chill", "mellow", "calm", "acoustic",
        "late night", "rainy day", "slow", "intimate", "lonely", "nostalgic",
        "dark", "heavy", "brooding", "epic", "driving", "workout", "hype",
        "party", "garage", "punk", "psych", "fuzz", "lo-fi", "dreamy", "ambient",
        "folk", "blues", "soul", "groove", "bounce", "trap", "drill", "afrobeats",
        "reggaeton", "cumbia", "indie", "shoegaze", "post punk", "classic rock",
        "southern rock", "grunge", "metal ballad", "power ballad", "road trip",
        "sunset", "morning coffee", "study", "sleep"
    ],
    "es": [
        "triste", "melancolia", "desamor", "relajado", "tranquilo", "acustico",
        "noche", "lento", "oscuro", "pesado", "fiesta", "garage", "punk",
        "psicodelico", "cumbia", "reggaeton", "trap", "indie", "rock en espanol",
        "balada", "nostalgia", "carretera"
    ],
    "hi": [
        "udaas", "dard", "ishq", "romantic", "sufi", "ghazal", "chill", "lofi",
        "bollywood sad", "bollywood romantic", "old hindi", "retro", "night drive",
        "breakup", "slow", "soulful"
    ],
    "pa": [
        "sad punjabi", "punjabi romantic", "punjabi trap", "punjabi hip hop",
        "punjabi chill", "bhangra", "punjabi folk", "punjabi lofi", "late night punjabi"
    ],
    "ta_te": [
        "tamil melody", "tamil sad", "tamil lofi", "tamil love",
        "telugu melody", "telugu sad", "ilaiyaraaja", "ar rahman melody"
    ],
    "ko_ja_pt_fr": [
        "k-indie", "korean rnb", "korean hip hop", "city pop", "j-rock",
        "japanese garage", "mpb", "bossa nova", "tropicalia", "chanson",
        "french pop", "french indie"
    ]
}


class RateLimiter:
    """At most 50 requests per 5s (Section 11.4)."""
    def __init__(self, max_reqs: int = 45, window_sec: float = 5.0):
        self.max_reqs = max_reqs
        self.window = window_sec
        self.timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < self.window]
            if len(self.timestamps) >= self.max_reqs:
                sleep_needed = self.window - (now - self.timestamps[0]) + 0.1
                if sleep_needed > 0:
                    await asyncio.sleep(sleep_needed)
            self.timestamps.append(time.time())


rate_limiter = RateLimiter()


async def api_get(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict | None:
    for attempt in range(3):
        await rate_limiter.acquire()
        try:
            resp = await client.get(url, params=params, timeout=12.0)
            if resp.status_code == 200:
                data = resp.json()
                if "error" in data:
                    err = data["error"]
                    if err.get("code") == 4 or "quota" in str(err).lower():
                        logger.warning("Deezer rate limit hit. Backing off 10s...")
                        await asyncio.sleep(10.0)
                        continue
                return data
            elif resp.status_code in (429, 403):
                logger.warning(f"Status {resp.status_code}. Backing off 10s...")
                await asyncio.sleep(10.0)
        except Exception as exc:
            await asyncio.sleep(1.0)
    return None


def _extract_tensor(out) -> torch.Tensor:
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "audio_embeds") and out.audio_embeds is not None:
        return out.audio_embeds
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if isinstance(out, (tuple, list)) and len(out) > 1:
        return out[1]
    return out


async def download_track_preview(client: httpx.AsyncClient, url: str, track_id: str) -> bytes | None:
    cache_path = PREVIEWS_DIR / f"{track_id}.mp3"
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        try:
            with open(cache_path, "rb") as f:
                return f.read()
        except Exception:
            pass

    for attempt in range(2):
        try:
            resp = await client.get(url, timeout=12.0)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(cache_path, "wb") as f:
                    f.write(resp.content)
                return resp.content
        except Exception:
            await asyncio.sleep(1.0)
    return None


def build_track_record(
    track_item: dict,
    audio_bytes: bytes,
    clap_processor: ClapProcessor,
    clap_model: ClapModel,
    anchor_keys: list[str],
    anchor_embs: np.ndarray,
) -> tuple[dict, np.ndarray] | None:
    try:
        # 1. Pinned decode: mono 22050Hz, -23 LUFS, 30s
        y = decode_preview(audio_bytes)
        if len(y) < 22050 * 5:
            return None

        # 2. 11.5 DSP feel descriptors
        meta_dict = {
            "title": track_item.get("title", ""),
            "artist": track_item.get("artist", ""),
            "album": track_item.get("album", ""),
            "genre": track_item.get("genre", ""),
        }
        dsp = describe_audio(y, meta=meta_dict, sr=22050)

        # Filtering rules from Section 11.4:
        # duration >= 25s, crest_factor >= 6.0 (drop silence and corrupted previews)
        if dsp["crest_factor"] < 6.0:
            return None

        # 3. CLAP embedding (48 kHz resampled, L2-normalised)
        y_48k = librosa.resample(y, orig_sr=22050, target_sr=48000, res_type="soxr_hq")
        inputs = clap_processor(audio=y_48k, sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            out = clap_model.get_audio_features(**inputs)
            emb_tensor = _extract_tensor(out)
        clap_emb = emb_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(clap_emb)
        if norm > 1e-8:
            clap_emb = clap_emb / norm

        # 4. Vibes = top two anchor prompts by CLAP cosine
        cos_sims = anchor_embs @ clap_emb
        top2_idx = np.argsort(cos_sims)[::-1][:2]
        vibes = [anchor_keys[i] for i in top2_idx]

        record = {
            "id": str(track_item["id"]),
            "title": str(track_item.get("title", "")),
            "artist": str(track_item.get("artist", "")),
            "album": str(track_item.get("album", "")),
            "albumArtUrl": str(track_item.get("albumArtUrl", "")),
            "previewUrl": str(track_item.get("previewUrl", "")),
            "durationMs": int(track_item.get("durationMs", 30000)),
            "year": int(track_item.get("year", 0)),
            "decade": str(track_item.get("decade", "")),
            "genre": str(track_item.get("genre", "")),
            "vibes": vibes,
            **dsp,
        }
        return record, clap_emb

    except Exception as exc:
        logger.debug(f"Error processing track {track_item.get('id')}: {exc}")
        return None


async def run_harvest(target_tracks: int = 25000):
    logger.info(f"Starting catalogue harvest target={target_tracks} tracks...")

    # Load anchor prompts
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        anchor_prompts = json.load(f)
    anchor_keys = list(anchor_prompts.keys())
    anchor_texts = [anchor_prompts[k] for k in anchor_keys]

    logger.info(f"Loading CLAP model 'laion/larger_clap_music'...")
    torch.set_num_threads(4)
    clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music", local_files_only=True)
    clap_model = ClapModel.from_pretrained("laion/larger_clap_music", local_files_only=True)
    clap_model.eval()

    text_inputs = clap_processor(text=anchor_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        t_out = clap_model.get_text_features(**text_inputs)
        anchor_embs = _extract_tensor(t_out).cpu().numpy().astype(np.float32)
    anchor_embs = anchor_embs / np.maximum(np.linalg.norm(anchor_embs, axis=1, keepdims=True), 1e-8)

    # Resume existing catalogue if present
    seen_ids: set[str] = set()
    existing_records = []
    if CATALOG_FEATURES_PATH.exists():
        with open(CATALOG_FEATURES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        seen_ids.add(rec["id"])
                        existing_records.append(rec)
                    except Exception:
                        pass
        logger.info(f"Resuming harvest: {len(seen_ids)} existing tracks loaded from {CATALOG_FEATURES_PATH}")

    # Track candidates pool to harvest
    track_pool: dict[str, dict] = {}

    http_client = httpx.AsyncClient(timeout=15.0, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    try:
        # Step 0: Guarantee 100% resolution of all benchmark targets & seeds
        logger.info("Ensuring all benchmark tracks are queued for harvest...")
        if BENCHMARK_PATH.exists():
            with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
                benchmark = json.load(f)
            for item in benchmark:
                all_entities = [item.get("seed")] + item.get("positives", []) + item.get("negatives", []) + item.get("near_misses", [])
                st = item.get("scene_test")
                if st:
                    all_entities += st.get("inside_negatives", []) + st.get("outside_positives", [])
                for ent in all_entities:
                    if not ent or not ent.get("title") or not ent.get("artist"):
                        continue
                    q = f"{ent['title']} {ent['artist']}"
                    try:
                        res = await search_track(q, limit=1)
                        if res and res[0].get("previewUrl"):
                            t = res[0]
                            tid = str(t["id"])
                            if tid not in seen_ids and tid not in track_pool:
                                track_pool[tid] = t
                    except Exception as exc:
                        logger.debug(f"Could not resolve benchmark track '{q}': {exc}")

        # Step 0b: Existing catalogue tracks from ml/inference/track_metadata.json
        existing_meta_path = Path("ml/inference/track_metadata.json")
        if existing_meta_path.exists():
            try:
                with open(existing_meta_path, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
                for t in existing_meta:
                    tid = str(t.get("id", ""))
                    if tid and t.get("previewUrl") and tid not in seen_ids and tid not in track_pool:
                        track_pool[tid] = t
                logger.info(f"Queued {len(track_pool)} benchmark and existing catalogue tracks.")
            except Exception as exc:
                logger.warning(f"Could not load existing track_metadata.json: {exc}")

        # Step 1: Mood playlists (all languages) & recording playlist membership
        logger.info("Harvesting Deezer mood playlists and playlist co-occurrences...")
        all_mood_queries = []
        for lang, words in MOOD_WORDS.items():
            for word in words:
                all_mood_queries.append((word, lang))

        with open(PLAYLIST_MEMBERSHIP_PATH, "a", encoding="utf-8") as pf:
            for word, lang in tqdm(all_mood_queries, desc="Mood playlists"):
                if len(seen_ids) + len(track_pool) >= target_tracks * 1.2:
                    break
                pl_resp = await api_get(http_client, f"{DEEZER_BASE}/search/playlist", {"q": word, "limit": 25})
                if not pl_resp or "data" not in pl_resp:
                    continue
                for pl in pl_resp["data"]:
                    pl_id = str(pl["id"])
                    tracks_resp = await api_get(http_client, f"{DEEZER_BASE}/playlist/{pl_id}/tracks", {"limit": 100})
                    if not tracks_resp or "data" not in tracks_resp:
                        continue
                    for t in tracks_resp["data"]:
                        tid = str(t["id"])
                        pf.write(json.dumps({"playlist_id": pl_id, "track_id": tid}) + "\n")
                        if t.get("preview") and tid not in seen_ids and tid not in track_pool:
                            track_pool[tid] = {
                                "id": tid,
                                "title": t.get("title", ""),
                                "artist": t.get("artist", {}).get("name", ""),
                                "album": t.get("album", {}).get("title", ""),
                                "albumArtUrl": t.get("album", {}).get("cover_xl", ""),
                                "previewUrl": t.get("preview"),
                                "durationMs": t.get("duration", 30) * 1000,
                                "year": 0,
                                "genre": "",
                            }
                    pf.flush()

        # Step 2: Discovery artist seeds & iTunes tracks
        logger.info("Harvesting discovery artist seeds & iTunes tracks...")
        if HARVEST_SEEDS_PATH.exists() and len(seen_ids) + len(track_pool) < target_tracks * 1.2:
            with open(HARVEST_SEEDS_PATH, "r", encoding="utf-8") as f:
                seed_artists = json.load(f)
            for artist in tqdm(seed_artists, desc="Artist seeds"):
                if len(seen_ids) + len(track_pool) >= target_tracks * 1.2:
                    break
                try:
                    it_resp = await http_client.get(
                        "https://itunes.apple.com/search",
                        params={"term": artist, "entity": "song", "limit": 200},
                        timeout=10.0,
                    )
                    if it_resp.status_code == 200:
                        for item in it_resp.json().get("results", []):
                            if item.get("previewUrl"):
                                tid = f"it_{item['trackId']}"
                                if tid not in seen_ids and tid not in track_pool:
                                    art = (item.get("artworkUrl100") or "").replace("100x100bb", "600x600bb")
                                    track_pool[tid] = {
                                        "id": tid,
                                        "title": item.get("trackName", ""),
                                        "artist": item.get("artistName", ""),
                                        "album": item.get("collectionName", ""),
                                        "albumArtUrl": art,
                                        "previewUrl": item.get("previewUrl"),
                                        "durationMs": int(item.get("trackTimeMillis", 30000)),
                                        "genre": item.get("primaryGenreName", ""),
                                        "year": 0,
                                    }
                    await asyncio.sleep(0.15)
                except Exception as exc:
                    logger.debug(f"iTunes artist harvest failed for {artist}: {exc}")

        # Step 3: Process tracks concurrently with semaphore
        logger.info(f"Ready to process {len(track_pool)} unique candidate tracks...")
        candidate_items = [c for c in track_pool.values() if c["id"] not in seen_ids]

        cat_file = open(CATALOG_FEATURES_PATH, "a", encoding="utf-8")
        all_embs = []
        if CLAP_NPY_PATH.exists():
            try:
                all_embs.append(np.load(CLAP_NPY_PATH))
            except Exception:
                pass

        sem = asyncio.Semaphore(8)
        lock = asyncio.Lock()
        saved_count = len(seen_ids)
        pbar = tqdm(total=min(target_tracks, len(candidate_items) + saved_count), initial=saved_count, desc="Processing audio")

        async def _worker(cand: dict):
            nonlocal saved_count
            if saved_count >= target_tracks:
                return
            tid = cand["id"]
            if tid in seen_ids:
                return

            async with sem:
                audio_bytes = await download_track_preview(http_client, cand["previewUrl"], tid)
                if not audio_bytes:
                    return

                res = build_track_record(
                    cand,
                    audio_bytes,
                    clap_processor,
                    clap_model,
                    anchor_keys,
                    anchor_embs,
                )
                if res is None:
                    return

                record, clap_emb = res
                async with lock:
                    if saved_count < target_tracks and tid not in seen_ids:
                        cat_file.write(json.dumps(record) + "\n")
                        cat_file.flush()
                        all_embs.append(clap_emb[np.newaxis, :])
                        seen_ids.add(tid)
                        saved_count += 1
                        pbar.update(1)

        # Process in chunks of 50 to avoid runaway memory and checkpoint embeddings
        chunk_size = 50
        for i in range(0, len(candidate_items), chunk_size):
            if saved_count >= target_tracks:
                break
            chunk = candidate_items[i : i + chunk_size]
            await asyncio.gather(*[_worker(c) for c in chunk])
            if all_embs:
                curr_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
                np.save(CLAP_NPY_PATH, curr_embs)
                logger.debug(f"Checkpoint saved: {curr_embs.shape[0]} embeddings in {CLAP_NPY_PATH}")

        pbar.close()
        cat_file.close()

        # Save merged embeddings
        if all_embs:
            final_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
            np.save(CLAP_NPY_PATH, final_embs)
            logger.info(f"Saved CLAP embeddings matrix {final_embs.shape} to {CLAP_NPY_PATH}")

        logger.info(f"Harvest complete! Catalogue features: {saved_count} tracks saved.")

    finally:
        await http_client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo Step 11.4 Catalogue Harvester")
    parser.add_argument("--target", type=int, default=25000, help="Target number of unique tracks")
    args = parser.parse_args()

    asyncio.run(run_harvest(target_tracks=args.target))
