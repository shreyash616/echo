"""
ml/data/build_rich_catalog.py: High-speed parallel catalogue builder.
Combines:
1. Archive catalog (1,656 songs across rock, pop, hip-hop, latin, punjabi, hindi, tamil, telugu, etc.)
2. Feel benchmark seeds & targets (all 14 sets)
3. Existing catalog tracks

Features:
- Fast async preview download with disk caching (ml/data/previews/)
- Multiprocess DSP feature extraction across 8 CPU cores
- Batch CLAP inference on NVIDIA RTX 3080 GPU (CUDA)
- Builds full FAISS index and metadata
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
logger = logging.getLogger("build_rich_catalog")

PREVIEWS_DIR = Path("ml/data/previews")
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_FEATURES_PATH = Path("ml/data/catalog_features.jsonl")
CLAP_NPY_PATH = Path("ml/data/clap.npy")
ANCHORS_PROMPTS_PATH = Path("ml/inference/anchor_prompts.json")
OUT_ANCHORS_PATH = Path("ml/inference/anchors.json")
OUT_FAISS_PATH = Path("ml/inference/music_index.faiss")
OUT_METADATA_PATH = Path("ml/inference/track_metadata.json")
OUT_ID_MAP_PATH = Path("ml/inference/track_id_map.json")


def _dsp_worker(item_tuple: tuple[dict, bytes]) -> tuple[dict, np.ndarray] | None:
    """Worker function executed in child process: decode preview and compute 11.5 DSP features."""
    track_item, audio_bytes = item_tuple
    try:
        y = decode_preview(audio_bytes)
        if len(y) < 22050 * 5:
            return None

        meta_dict = {
            "title": track_item.get("title", ""),
            "artist": track_item.get("artist", ""),
            "album": track_item.get("album", ""),
            "genre": track_item.get("genre", ""),
        }
        dsp = describe_audio(y, meta=meta_dict, sr=22050)

        # Filter out corrupted / silent audio
        if dsp.get("crest_factor", 0.0) < 5.0:
            return None

        # Resample to 48kHz for CLAP
        y_48k = librosa.resample(y, orig_sr=22050, target_sr=48000, res_type="soxr_hq")
        return {**track_item, **dsp}, y_48k
    except Exception:
        return None


async def download_preview(client: httpx.AsyncClient, sem: asyncio.Semaphore, track: dict) -> tuple[dict, bytes] | None:
    tid = str(track["id"])
    url = track.get("previewUrl")
    if not url:
        return None

    ext = "m4a" if "aac" in url or "m4a" in url else "mp3"
    cache_file = PREVIEWS_DIR / f"{tid}.{ext}"
    if cache_file.exists() and cache_file.stat().st_size > 1000:
        try:
            return track, cache_file.read_bytes()
        except Exception:
            pass

    async with sem:
        for _ in range(2):
            try:
                resp = await client.get(url, timeout=12.0)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    try:
                        cache_file.write_bytes(resp.content)
                    except Exception:
                        pass
                    return track, resp.content
            except Exception:
                await asyncio.sleep(0.5)
    return None


async def main():
    logger.info("=== Starting Rich Catalogue Build ===")

    # 1. Collect all candidates
    candidates: dict[str, dict] = {}

    # Source A: Archive catalog (1,656 tracks)
    archive_cat_path = Path("ml/archive/deezer_catalog.json")
    if archive_cat_path.exists():
        with open(archive_cat_path, "r", encoding="utf-8") as f:
            arc = json.load(f)
        for t in arc:
            tid = str(t.get("id", ""))
            if tid and t.get("previewUrl"):
                candidates[tid] = t
        logger.info(f"Loaded {len(candidates)} tracks from archive catalogue.")

    # Source B: Benchmark seeds and targets
    bench_path = Path("tests/feel_benchmark_v2.json")
    if bench_path.exists():
        with open(bench_path, "r", encoding="utf-8") as f:
            bench = json.load(f)
        bench_queries = []
        for item in bench:
            entities = [item.get("seed")] + item.get("positives", []) + item.get("negatives", []) + item.get("near_misses", [])
            st = item.get("scene_test")
            if st:
                entities += st.get("inside_negatives", []) + st.get("outside_positives", [])
            for e in entities:
                if e and e.get("title") and e.get("artist"):
                    bench_queries.append(f"{e['title']} {e['artist']}")

        logger.info(f"Resolving {len(bench_queries)} benchmark tracks...")
        for q in bench_queries:
            try:
                res = await search_track(q, limit=1)
                if res and res[0].get("previewUrl"):
                    t = res[0]
                    tid = str(t["id"])
                    if tid not in candidates:
                        candidates[tid] = t
            except Exception:
                pass
        logger.info(f"Candidate pool now at {len(candidates)} unique tracks.")

    # Source C: Existing metadata tracks
    meta_path = Path("ml/inference/track_metadata.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for t in meta:
            tid = str(t.get("id", ""))
            if tid and t.get("previewUrl") and tid not in candidates:
                candidates[tid] = t
        logger.info(f"Candidate pool with existing metadata: {len(candidates)} unique tracks.")

    cand_list = list(candidates.values())

    # 2. Download audio previews concurrently
    logger.info("Downloading audio previews...")
    sem = asyncio.Semaphore(24)
    async with httpx.AsyncClient(timeout=15.0) as client:
        download_tasks = [download_preview(client, sem, t) for t in cand_list]
        downloaded = await asyncio.gather(*download_tasks)

    valid_downloads = [item for item in downloaded if item is not None]
    logger.info(f"Successfully downloaded/cached {len(valid_downloads)} audio previews.")

    # 3. Multiprocess DSP extraction (12 CPU cores)
    logger.info(f"Computing 11.5 DSP feel descriptors using ProcessPoolExecutor...")
    t0_dsp = time.perf_counter()
    num_workers = min(12, os.cpu_count() or 8)
    dsp_results: list[tuple[dict, np.ndarray]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(_dsp_worker, valid_downloads, chunksize=16), total=len(valid_downloads), desc="DSP features"):
            if res is not None:
                dsp_results.append(res)

    logger.info(f"DSP extraction complete for {len(dsp_results)} tracks in {time.perf_counter()-t0_dsp:.1f}s.")

    # 4. Batch CLAP embeddings on GPU (RTX 3080)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Computing CLAP embeddings and vibes on {device}...")

    # Load anchor prompts
    with open(ANCHORS_PROMPTS_PATH, "r", encoding="utf-8") as f:
        anchor_prompts = json.load(f)
    anchor_keys = list(anchor_prompts.keys())
    anchor_texts = [anchor_prompts[k] for k in anchor_keys]

    clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music", local_files_only=True)
    clap_model = ClapModel.from_pretrained("laion/larger_clap_music", local_files_only=True).to(device).eval()

    # Anchor embeddings
    text_inputs = clap_processor(text=anchor_texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        t_out = clap_model.get_text_features(**text_inputs)
        t_emb = t_out.pooler_output if hasattr(t_out, "pooler_output") else (t_out.text_embeds if hasattr(t_out, "text_embeds") else t_out)
        anchor_embs = t_emb.cpu().numpy().astype(np.float32)
    anchor_embs = anchor_embs / np.maximum(np.linalg.norm(anchor_embs, axis=1, keepdims=True), 1e-8)

    # Save anchors.json
    anchors_dict = {anchor_keys[i]: anchor_embs[i].tolist() for i in range(len(anchor_keys))}
    with open(OUT_ANCHORS_PATH, "w", encoding="utf-8") as f:
        json.dump(anchors_dict, f)
    logger.info(f"Saved {len(anchors_dict)} anchor embeddings to {OUT_ANCHORS_PATH}")

    # Process audio in batches
    final_records: list[dict] = []
    final_clap_embs: list[np.ndarray] = []
    batch_size = 32

    for b in tqdm(range(0, len(dsp_results), batch_size), desc="CLAP inference"):
        batch = dsp_results[b : b + batch_size]
        tracks_batch = [item[0] for item in batch]
        audios_batch = [item[1] for item in batch]

        inputs = clap_processor(audio=audios_batch, sampling_rate=48000, return_tensors="pt").to(device)
        with torch.no_grad():
            out = clap_model.get_audio_features(**inputs)
            emb_tensor = out.pooler_output if hasattr(out, "pooler_output") else (out.audio_embeds if hasattr(out, "audio_embeds") else out)
        batch_embs = emb_tensor.cpu().numpy().astype(np.float32)
        batch_norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
        batch_embs = batch_embs / np.maximum(batch_norms, 1e-8)

        # Assign vibes
        for i, rec in enumerate(tracks_batch):
            emb = batch_embs[i]
            sims = anchor_embs @ emb
            top2_idx = np.argsort(sims)[::-1][:2]
            rec["vibes"] = [anchor_keys[idx] for idx in top2_idx]
            final_records.append(rec)
            final_clap_embs.append(emb)

    # 5. Save catalog_features.jsonl and clap.npy
    with open(CATALOG_FEATURES_PATH, "w", encoding="utf-8") as f:
        for rec in final_records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Saved {len(final_records)} records to {CATALOG_FEATURES_PATH}")

    clap_matrix = np.stack(final_clap_embs, axis=0).astype(np.float32)
    np.save(CLAP_NPY_PATH, clap_matrix)
    logger.info(f"Saved CLAP matrix {clap_matrix.shape} to {CLAP_NPY_PATH}")

    # 6. Build FAISS index and metadata bundle
    import faiss
    logger.info("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(512)
    index.add(clap_matrix)
    faiss.write_index(index, str(OUT_FAISS_PATH))
    logger.info(f"Wrote FAISS index to {OUT_FAISS_PATH} with {index.ntotal} entries.")

    with open(OUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2)
    logger.info(f"Wrote metadata to {OUT_METADATA_PATH}")

    id_map = {str(rec.get("id")): i for i, rec in enumerate(final_records)}
    with open(OUT_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)
    logger.info(f"Wrote ID map with {len(id_map)} entries to {OUT_ID_MAP_PATH}")

    logger.info(f"=== Successfully built rich catalogue with {len(final_records)} tracks! ===")


if __name__ == "__main__":
    asyncio.run(main())
