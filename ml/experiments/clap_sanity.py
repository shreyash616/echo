"""
ml/experiments/clap_sanity.py
Step 1: The CLAP decision experiment (Section 11.3).

Evaluates whether the frozen CLAP space already orders benchmark positives
above negatives before building the 25k catalogue.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import httpx
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

# Add backend to sys.path so decode_preview and services can be imported
sys.path.insert(0, os.path.abspath("backend"))

from app.audio.decode import decode_preview
from app.services.deezer import search_track

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("clap_sanity")

CACHE_DIR = Path("ml/experiments/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "laion/larger_clap_music"
ANCHORS_PATH = Path("ml/inference/anchor_prompts.json")
BENCHMARK_PATH = Path("tests/feel_benchmark_v2.json")
RESULTS_PATH = Path("ml/experiments/clap_sanity_results.json")


def _norm_str(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


async def resolve_track_preview(
    title: str,
    artist: str,
    client: httpx.AsyncClient,
) -> dict | None:
    """
    11.3.1: Resolve preview by calling backend's search_track with "{title} {artist}".
    Pick first result whose normalised artist contains expected artist and whose title
    has no "cover", "karaoke", "tribute", "live", "remix", "(".
    """
    query = f"{title} {artist}".strip()
    await asyncio.sleep(0.25)
    try:
        results = await search_track(query, limit=5)
    except Exception as exc:
        logger.warning(f"search_track failed for {query}: {exc}")
        return None

    norm_exp_artist = _norm_str(artist)
    disallowed = ["cover", "karaoke", "tribute", "live", "remix", "("]

    for r in results:
        r_title = r.get("title", "")
        r_artist = r.get("artist", "")
        norm_r_artist = _norm_str(r_artist)

        # Check normalised artist match (either contains the other)
        if norm_exp_artist in norm_r_artist or norm_r_artist in norm_exp_artist:
            # Check title for disallowed variants
            r_title_lower = r_title.lower()
            if not any(d in r_title_lower for d in disallowed):
                if r.get("previewUrl"):
                    return r

    logger.info(f"Could not resolve preview for '{title}' by '{artist}'")
    return None


async def download_preview(url: str, client: httpx.AsyncClient) -> bytes | None:
    for attempt in range(2):
        try:
            resp = await client.get(url, timeout=12.0)
            if resp.status_code == 200 and len(resp.content) > 1000:
                return resp.content
        except Exception:
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


def embed_audio(
    audio_bytes: bytes,
    processor: ClapProcessor,
    model: ClapModel,
) -> np.ndarray:
    """
    11.3.2: decode_preview -> resample to 48 kHz -> embed with ClapModel -> L2-normalise.
    """
    y_22k = decode_preview(audio_bytes)  # mono float32 at 22050 Hz, -23 LUFS, 30s
    y_48k = librosa.resample(y_22k, orig_sr=22050, target_sr=48000, res_type="soxr_hq")
    inputs = processor(audio=y_48k, sampling_rate=48000, return_tensors="pt")
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
        emb = _extract_tensor(out)
    emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb_np)
    if norm > 1e-8:
        emb_np = emb_np / norm
    return emb_np


def get_cached_or_embed(
    track: dict,
    audio_bytes: bytes,
    processor: ClapProcessor,
    model: ClapModel,
) -> np.ndarray:
    track_id = str(track.get("id", ""))
    cache_path = CACHE_DIR / f"clap_{track_id}.npy"
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            pass
    emb = embed_audio(audio_bytes, processor, model)
    np.save(cache_path, emb)
    return emb


async def main():
    print("==================================================================")
    print("Step 1: CLAP Decision Experiment (Section 11.3)")
    print("==================================================================")

    # 1. Load benchmark v2
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} benchmark test sets from {BENCHMARK_PATH}")

    # 2. Load CLAP processor and model
    print(f"Loading CLAP model '{MODEL_NAME}' on CPU...")
    processor = ClapProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    model = ClapModel.from_pretrained(MODEL_NAME, local_files_only=True)
    model.eval()
    print("CLAP model ready.")

    # 3. Embed anchor prompts
    with open(ANCHORS_PATH, "r", encoding="utf-8") as f:
        anchor_prompts = json.load(f)
    anchor_keys = list(anchor_prompts.keys())
    anchor_texts = [anchor_prompts[k] for k in anchor_keys]

    print(f"Embedding {len(anchor_texts)} anchor prompts with text encoder...")
    text_inputs = processor(text=anchor_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        t_out = model.get_text_features(**text_inputs)
        text_tensor = _extract_tensor(t_out)
        text_features = text_tensor.cpu().numpy().astype(np.float32)
    text_norms = np.linalg.norm(text_features, axis=1, keepdims=True)
    text_features = text_features / np.maximum(text_norms, 1e-8)

    def top_3_anchors(audio_emb: np.ndarray) -> list[str]:
        sims = text_features @ audio_emb
        top_idx = np.argsort(sims)[::-1][:3]
        return [anchor_keys[i] for i in top_idx]

    # 4. Resolve previews and compute embeddings
    http_client = httpx.AsyncClient(timeout=15.0, headers={"User-Agent": "Echo-CLAP-Sanity/1.0"})

    total_wins = 0
    total_pairs = 0
    per_seed_results = []
    unresolved_items = []

    try:
        for item_idx, item in enumerate(benchmark):
            seed = item["seed"]
            seed_title = seed["title"]
            seed_artist = seed["artist"]
            split = item.get("split", "tune")

            print(f"\n[{item_idx + 1}/{len(benchmark)}] Seed: '{seed_title}' by {seed_artist} ({split})")

            # Resolve seed
            seed_track = await resolve_track_preview(seed_title, seed_artist, http_client)
            if not seed_track:
                unresolved_items.append({"role": "seed", "title": seed_title, "artist": seed_artist})
                print(f"  -> Seed '{seed_title}' could not be resolved! Skipping seed.")
                continue

            seed_cache_file = CACHE_DIR / f"clap_{seed_track['id']}.npy"
            if seed_cache_file.exists():
                seed_emb = np.load(seed_cache_file)
            else:
                raw_bytes = await download_preview(seed_track["previewUrl"], http_client)
                if not raw_bytes:
                    print(f"  -> Failed to download audio for seed '{seed_title}'")
                    continue
                seed_emb = get_cached_or_embed(seed_track, raw_bytes, processor, model)

            seed_anchors = top_3_anchors(seed_emb)

            # Resolve positives
            pos_records = []
            for pos in item.get("positives", []):
                p_title, p_artist = pos["title"], pos["artist"]
                track = await resolve_track_preview(p_title, p_artist, http_client)
                if not track:
                    unresolved_items.append({"role": "positive", "title": p_title, "artist": p_artist, "seed": seed_title})
                    continue
                c_file = CACHE_DIR / f"clap_{track['id']}.npy"
                if c_file.exists():
                    p_emb = np.load(c_file)
                else:
                    audio_b = await download_preview(track["previewUrl"], http_client)
                    if not audio_b:
                        continue
                    p_emb = get_cached_or_embed(track, audio_b, processor, model)
                cos_sim = float(np.dot(seed_emb, p_emb))
                p_anchors = top_3_anchors(p_emb)
                pos_records.append({
                    "title": p_title,
                    "artist": p_artist,
                    "id": track["id"],
                    "cosine": cos_sim,
                    "anchors": p_anchors,
                    "source": pos.get("source", "panel"),
                })

            # Resolve negatives
            neg_records = []
            for neg in item.get("negatives", []):
                n_title, n_artist = neg["title"], neg["artist"]
                track = await resolve_track_preview(n_title, n_artist, http_client)
                if not track:
                    unresolved_items.append({"role": "negative", "title": n_title, "artist": n_artist, "seed": seed_title})
                    continue
                c_file = CACHE_DIR / f"clap_{track['id']}.npy"
                if c_file.exists():
                    n_emb = np.load(c_file)
                else:
                    audio_b = await download_preview(track["previewUrl"], http_client)
                    if not audio_b:
                        continue
                    n_emb = get_cached_or_embed(track, audio_b, processor, model)
                cos_sim = float(np.dot(seed_emb, n_emb))
                neg_records.append({
                    "title": n_title,
                    "artist": n_artist,
                    "id": track["id"],
                    "cosine": cos_sim,
                })

            # Compute pairwise win rate for this seed
            seed_wins = 0
            seed_pairs = 0
            for p in pos_records:
                for n in neg_records:
                    seed_pairs += 1
                    if p["cosine"] > n["cosine"]:
                        seed_wins += 1

            seed_win_rate = (seed_wins / seed_pairs) if seed_pairs > 0 else 0.0
            mean_pos_cos = float(np.mean([p["cosine"] for p in pos_records])) if pos_records else 0.0
            mean_neg_cos = float(np.mean([n["cosine"] for n in neg_records])) if neg_records else 0.0

            total_wins += seed_wins
            total_pairs += seed_pairs

            print(f"  Positives resolved: {len(pos_records)}, Negatives resolved: {len(neg_records)}")
            print(f"  Mean Pos Cosine: {mean_pos_cos:.4f} | Mean Neg Cosine: {mean_neg_cos:.4f}")
            print(f"  Pairwise Win Rate: {seed_win_rate:.3f} ({seed_wins}/{seed_pairs})")
            print(f"  Seed Anchors: {', '.join(seed_anchors)}")

            per_seed_results.append({
                "seed": {"title": seed_title, "artist": seed_artist, "split": split},
                "seed_id": seed_track["id"],
                "seed_anchors": seed_anchors,
                "mean_pos_cosine": round(mean_pos_cos, 4),
                "mean_neg_cosine": round(mean_neg_cos, 4),
                "pairwise_win_rate": round(seed_win_rate, 4),
                "seed_wins": seed_wins,
                "seed_pairs": seed_pairs,
                "positives": pos_records,
                "negatives": neg_records,
            })

    finally:
        await http_client.aclose()

    overall_win_rate = (total_wins / total_pairs) if total_pairs > 0 else 0.0
    min_seed_win_rate = min([s["pairwise_win_rate"] for s in per_seed_results]) if per_seed_results else 0.0

    # Decision rule from Section 11.3:
    if overall_win_rate >= 0.70 and min_seed_win_rate >= 0.55:
        decision = "CLAP is the backbone. Initial clap_cosine weight 0.22. Proceed to 11.4."
        initial_clap_weight = 0.22
    elif overall_win_rate >= 0.55:
        decision = (
            "Proceed, but set initial clap_cosine weight in 11.7 to 0.12 instead of 0.22 "
            "and move difference onto groove and grit."
        )
        initial_clap_weight = 0.12
    else:
        decision = (
            "Overall win rate < 0.55: embedding must be replaced; DSP-only rerank "
            "shipped in 11.7 with clap_cosine weight 0."
        )
        initial_clap_weight = 0.0

    results_data = {
        "overall_pairwise_win_rate": round(overall_win_rate, 4),
        "total_wins": total_wins,
        "total_pairs": total_pairs,
        "min_seed_win_rate": round(min_seed_win_rate, 4),
        "decision": decision,
        "initial_clap_weight": initial_clap_weight,
        "unresolved_count": len(unresolved_items),
        "unresolved": unresolved_items,
        "seeds": per_seed_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Seed':<28} | {'Split':<7} | {'Pos Cos':<8} | {'Neg Cos':<8} | {'Win Rate':<10} | {'Pairs':<6}")
    print("-" * 80)
    for s in per_seed_results:
        stitle = s["seed"]["title"][:27]
        split = s["seed"]["split"]
        pcos = s["mean_pos_cosine"]
        ncos = s["mean_neg_cosine"]
        wr = s["pairwise_win_rate"]
        pairs = s["seed_pairs"]
        print(f"{stitle:<28} | {split:<7} | {pcos:<8.3f} | {ncos:<8.3f} | {wr:<10.3f} | {pairs:<6}")
    print("=" * 80)
    print(f"OVERALL PAIRWISE WIN RATE: {overall_win_rate:.4f} ({total_wins}/{total_pairs})")
    print(f"MINIMUM SEED WIN RATE:     {min_seed_win_rate:.4f}")
    print(f"DECISION RULE:             {decision}")
    print(f"INITIAL CLAP WEIGHT:       {initial_clap_weight}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
