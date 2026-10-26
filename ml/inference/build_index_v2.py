"""
ml/inference/build_index_v2.py: Step 2b Index & Space Bundle Builder (Section 11.6).

Builds:
- music_index.faiss (IndexFlatIP over 512-d normalized CLAP vectors)
- track_metadata.json (metadata list matching index order)
- anchors.json (anchor prompt embeddings dictionary)
- track_id_map.json (id -> index position)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_index_v2")

CATALOG_FEATURES_PATH = Path("ml/data/catalog_features.jsonl")
CLAP_NPY_PATH = Path("ml/data/clap.npy")
ANCHORS_PROMPTS_PATH = Path("ml/inference/anchor_prompts.json")

OUT_FAISS_PATH = Path("ml/inference/music_index.faiss")
OUT_METADATA_PATH = Path("ml/inference/track_metadata.json")
OUT_ANCHORS_PATH = Path("ml/inference/anchors.json")
OUT_ID_MAP_PATH = Path("ml/inference/track_id_map.json")


def _extract_tensor(out) -> torch.Tensor:
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if isinstance(out, (tuple, list)) and len(out) > 1:
        return out[1]
    return out


def main():
    logger.info("Starting index build (Section 11.6)...")

    # 1. Load catalog features
    if not CATALOG_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing {CATALOG_FEATURES_PATH}. Run harvest_v2.py first.")

    metadata_list: list[dict] = []
    with open(CATALOG_FEATURES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata_list.append(json.loads(line))

    N = len(metadata_list)
    logger.info(f"Loaded {N} track records from {CATALOG_FEATURES_PATH}")

    # 2. Load CLAP embeddings
    if not CLAP_NPY_PATH.exists():
        raise FileNotFoundError(f"Missing {CLAP_NPY_PATH}. Run harvest_v2.py first.")

    embs = np.load(CLAP_NPY_PATH)
    if embs.ndim == 3:
        embs = embs.squeeze(1)
    logger.info(f"Loaded embeddings matrix {embs.shape}")

    if len(embs) != N:
        logger.warning(f"Length mismatch: {len(embs)} vectors vs {N} metadata entries. Truncating to min.")
        min_len = min(len(embs), N)
        embs = embs[:min_len]
        metadata_list = metadata_list[:min_len]
        N = min_len

    # Normalize vectors
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.maximum(norms, 1e-8)
    embs = np.ascontiguousarray(embs, dtype=np.float32)

    # 3. Build FAISS IndexFlatIP
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    logger.info(f"Built FAISS IndexFlatIP with {index.ntotal} vectors of dimension {dim}")

    faiss.write_index(index, str(OUT_FAISS_PATH))
    logger.info(f"Saved FAISS index to {OUT_FAISS_PATH} ({OUT_FAISS_PATH.stat().st_size / 1024 / 1024:.2f} MB)")

    # 4. Save track_metadata.json
    with open(OUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=1)
    logger.info(f"Saved track metadata to {OUT_METADATA_PATH} ({OUT_METADATA_PATH.stat().st_size / 1024 / 1024:.2f} MB)")

    # 5. Save track_id_map.json
    id_map = {rec["id"]: idx for idx, rec in enumerate(metadata_list)}
    with open(OUT_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f)
    logger.info(f"Saved track ID map to {OUT_ID_MAP_PATH}")

    # 6. Embed anchors
    with open(ANCHORS_PROMPTS_PATH, "r", encoding="utf-8") as f:
        anchor_prompts = json.load(f)
    anchor_keys = list(anchor_prompts.keys())
    anchor_texts = [anchor_prompts[k] for k in anchor_keys]

    logger.info(f"Embedding {len(anchor_texts)} anchor prompts with CLAP text encoder...")
    clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music", local_files_only=True)
    clap_model = ClapModel.from_pretrained("laion/larger_clap_music", local_files_only=True)
    clap_model.eval()

    text_inputs = clap_processor(text=anchor_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        t_out = clap_model.get_text_features(**text_inputs)
        anchor_features = _extract_tensor(t_out).cpu().numpy().astype(np.float32)

    anchor_norms = np.linalg.norm(anchor_features, axis=1, keepdims=True)
    anchor_features = anchor_features / np.maximum(anchor_norms, 1e-8)

    anchors_dict = {
        anchor_keys[i]: anchor_features[i].tolist()
        for i in range(len(anchor_keys))
    }
    with open(OUT_ANCHORS_PATH, "w", encoding="utf-8") as f:
        json.dump(anchors_dict, f, indent=1)
    logger.info(f"Saved anchors dictionary to {OUT_ANCHORS_PATH}")

    logger.info("Index build complete!")


if __name__ == "__main__":
    main()
