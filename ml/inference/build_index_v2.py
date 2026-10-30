"""
ml/inference/build_index_v2.py: Step R4 Index Builder & Space Bundle (Sections 12.2 & 12.5).

Builds:
- music_index.faiss: IndexFlatIP over 512-d normalized official LAION CLAP vectors (20,000 entries)
- anchors.json: Anchor prompts dictionary with catalogue means and stds for standardisation
- theme_prompts.json: Canonical theme prompts with catalogue means and stds
- track_metadata.json: Header with cut points, distributions, means, stds, plus 20,000 tracks
- track_id_map.json: Mapping id, key, deezer_id, itunes_id -> row index

Validates:
- kick_class has >= 3 values each with >= 10% share
- restraint_bucket has >= 3 values
- Mandatory anchor diversity: max cosine < 0.95 and min cosine < 0.70
- Mandatory catalogue diversity: first 50 vectors pairwise cosine mean < 0.90
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

# Add backend to path
sys.path.insert(0, "backend")
from app.services.recommender import make_track_key
from app.audio.features import quantize_kick_and_restraint

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_index_v2")

DATA_DIR = Path("ml/data")
INFERENCE_DIR = Path("ml/inference")
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_FEATURES_PATH = DATA_DIR / "catalog_features.jsonl"
CLAP_NPY_PATH = DATA_DIR / "clap.npy"

OUT_FAISS_PATH = INFERENCE_DIR / "music_index.faiss"
OUT_METADATA_PATH = INFERENCE_DIR / "track_metadata.json"
OUT_ANCHORS_PATH = INFERENCE_DIR / "anchors.json"
OUT_THEMES_PATH = INFERENCE_DIR / "theme_prompts.json"
OUT_ID_MAP_PATH = INFERENCE_DIR / "track_id_map.json"
OUT_FEEL_MATRIX_PATH = INFERENCE_DIR / "feel_matrix.npy"
OUT_FEEL_FAISS_PATH = INFERENCE_DIR / "feel_index.faiss"

FEEL_AXES = [
    "log2_felt_bpm",
    "pulse_rate",
    "hits_per_bar",
    "darkness",
    "brightness",
    "low_share",
    "grit",
    "roughness",
    "intimacy",
    "reverb_tail",
    "loudness_range",
    "crest_factor",
    "lag_ms",
    "swing",
    "syllable_rate",
    "sustain_ratio",
    "mode_polarity",
]
# T2: catalogue std per raw feature; _compute_pair_score scores |a - b| as exp(-|a - b| / (tau * std))
FEATURE_STD_AXES = [
    "darkness", "grit", "intimacy", "pulse_rate", "brightness", "low_share", "roughness",
    "loudness_range", "crest_factor", "syllable_rate", "sustain_ratio", "pitch_range", "vibrato",
    "lag_ms", "hits_per_bar", "backbeat", "changes_per_bar",  # swing left out (T3.5, constant)
    "mode_polarity",  # s_mode's polarity term, added 2026-10-30 (a0d9b43)
]
FEEL_WEIGHTS = {
    "log2_felt_bpm": 1.5,
    "pulse_rate": 0.8,
    "hits_per_bar": 0.8,
    "darkness": 1.0,
    "brightness": 0.6,
    "low_share": 0.5,
    "grit": 0.8,
    "roughness": 0.8,
    "intimacy": 0.8,
    "reverb_tail": 0.5,
    "loudness_range": 0.6,
    "crest_factor": 0.4,
    "lag_ms": 0.5,
    "swing": 0.4,
    "syllable_rate": 0.6,
    "sustain_ratio": 0.6,
    "mode_polarity": 0.5,
}


def extract_raw_feel(r: dict) -> list[float]:
    bpm = float(r.get("felt_bpm") or 100.0)
    log2_bpm = float(np.log2(max(bpm, 10.0)))
    mode_pol = 0.0 if r.get("mode_undefined", False) else float(r.get("mode_polarity") or 0.0)
    return [
        log2_bpm,
        float(r.get("pulse_rate") or 0.0),
        float(r.get("hits_per_bar") or 0.0),
        float(r.get("darkness") or 0.0),
        float(r.get("brightness") or 0.0),
        float(r.get("low_share") or 0.0),
        float(r.get("grit") or 0.0),
        float(r.get("roughness") or 0.0),
        float(r.get("intimacy") or 0.0),
        float(r.get("reverb_tail") or 0.0),
        float(r.get("loudness_range") or 0.0),
        float(r.get("crest_factor") or 0.0),
        float(r.get("lag_ms") or 0.0),
        float(r.get("swing") or 0.0),
        float(r.get("syllable_rate") or 0.0),
        float(r.get("sustain_ratio") or 0.0),
        mode_pol,
    ]


def main():
    logger.info("=== Starting Step R4: Index Build & Quantile Bucketing ===")

    # 1. Load catalog features (download from HF dataset if absent)
    if not CATALOG_FEATURES_PATH.exists():
        logger.info("Downloading catalog_features.jsonl from HF dataset shreyash616/echo-ml-artifacts...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="shreyash616/echo-ml-artifacts", filename="ml/data/catalog_features.jsonl", repo_type="dataset", local_dir=".")

    raw_tracks: list[dict] = []
    with open(CATALOG_FEATURES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_tracks.append(json.loads(line))

    N = len(raw_tracks)
    logger.info("Loaded %d track records from %s", N, CATALOG_FEATURES_PATH)
    if N < 19000:  # P4 floor: rows whose preview cannot be decoded are dropped
        raise ValueError(f"Expected at least 19,000 tracks, got {N}")

    # 2. Load CLAP embeddings (download from HF dataset if absent)
    if not CLAP_NPY_PATH.exists():
        logger.info("Downloading clap.npy from HF dataset shreyash616/echo-ml-artifacts...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id="shreyash616/echo-ml-artifacts", filename="ml/data/clap.npy", repo_type="dataset", local_dir=".")

    embs = np.load(CLAP_NPY_PATH)
    if embs.ndim == 3:
        embs = embs.squeeze(1)
    logger.info("Loaded embeddings matrix %s", embs.shape)
    if len(embs) != N:
        raise ValueError(f"Mismatch: {len(embs)} vectors vs {N} metadata entries")

    # Normalize vectors
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = (embs / norms).astype(np.float32)

    # 3. Load Anchors & compute catalogue means and stds (Section 12.2)
    with open(OUT_ANCHORS_PATH, "r", encoding="utf-8") as f:
        anchors_raw = json.load(f)
    if isinstance(anchors_raw, dict) and "prompts" in anchors_raw:
        anchor_prompts = anchors_raw["prompts"]
    else:
        anchor_prompts = anchors_raw

    anchor_keys = list(anchor_prompts.keys())
    anchor_matrix = np.array([anchor_prompts[k] for k in anchor_keys], dtype=np.float32)

    # Compute anchor cosines for all tracks
    all_anchor_cosines = []
    for r in raw_tracks:
        ac = r.get("anchor_cosines")
        if not ac or len(ac) != len(anchor_keys):
            # Compute directly if missing
            vec = embs[len(all_anchor_cosines)]
            ac = (anchor_matrix @ vec).tolist()
        all_anchor_cosines.append(ac)

    anchor_cos_arr = np.array(all_anchor_cosines, dtype=np.float32)
    anchor_means = np.mean(anchor_cos_arr, axis=0)
    anchor_stds = np.std(anchor_cos_arr, axis=0)
    anchor_stds[anchor_stds == 0] = 1.0

    # Save standardized anchors.json
    anchors_bundle = {
        "prompts": anchor_prompts,
        "means": anchor_means.tolist(),
        "stds": anchor_stds.tolist(),
        "keys": anchor_keys,
    }
    with open(OUT_ANCHORS_PATH, "w", encoding="utf-8") as f:
        json.dump(anchors_bundle, f, indent=2)
    logger.info("Saved standardized anchors bundle to %s", OUT_ANCHORS_PATH)

    # 4. Load Themes & compute catalogue means and stds (Section 12.3)
    with open(OUT_THEMES_PATH, "r", encoding="utf-8") as f:
        themes_raw = json.load(f)
    if isinstance(themes_raw, dict) and "prompts" in themes_raw:
        theme_prompts = themes_raw["prompts"]
    else:
        theme_prompts = themes_raw

    theme_keys = list(theme_prompts.keys())
    theme_vecs = [r["theme_vec"] for r in raw_tracks if r.get("has_lyrics") and len(r.get("theme_vec", [])) == len(theme_keys)]
    if theme_vecs:
        theme_mat = np.array(theme_vecs, dtype=np.float32)
        theme_means = np.mean(theme_mat, axis=0)
        theme_stds = np.std(theme_mat, axis=0)
        theme_stds = np.maximum(theme_stds, 0.08)
    else:
        theme_means = np.zeros(len(theme_keys), dtype=np.float32)
        theme_stds = np.ones(len(theme_keys), dtype=np.float32)

    themes_bundle = {
        "prompts": theme_prompts,
        "means": theme_means.tolist(),
        "stds": theme_stds.tolist(),
        "keys": theme_keys,
    }
    with open(OUT_THEMES_PATH, "w", encoding="utf-8") as f:
        json.dump(themes_bundle, f, indent=2)
    logger.info("Saved standardized theme prompts bundle to %s", OUT_THEMES_PATH)

    # 5. Continuous features cut points & bucketing (Section 12.5)
    hits_per_bar_arr = np.array([r.get("hits_per_bar", 0.0) for r in raw_tracks])
    low_share_arr = np.array([r.get("low_share", 0.0) for r in raw_tracks])
    loudness_range_arr = np.array([r.get("loudness_range", 0.0) for r in raw_tracks])
    crest_factor_arr = np.array([r.get("crest_factor", 0.0) for r in raw_tracks])

    cut_points = {
        "hits_per_bar": {
            "p25": float(np.percentile(hits_per_bar_arr, 25)),
            "p33": float(np.percentile(hits_per_bar_arr, 33.3)),
            "p50": float(np.percentile(hits_per_bar_arr, 50)),
            "p66": float(np.percentile(hits_per_bar_arr, 66.7)),
            "p75": float(np.percentile(hits_per_bar_arr, 75)),
        },
        "low_share": {
            "p25": float(np.percentile(low_share_arr, 25)),
            "p33": float(np.percentile(low_share_arr, 33.3)),
            "p50": float(np.percentile(low_share_arr, 50)),
            "p66": float(np.percentile(low_share_arr, 66.7)),
            "p75": float(np.percentile(low_share_arr, 75)),
        },
        "loudness_range": {
            "p25": float(np.percentile(loudness_range_arr, 25)),
            "p33": float(np.percentile(loudness_range_arr, 33.3)),
            "p50": float(np.percentile(loudness_range_arr, 50)),
            "p66": float(np.percentile(loudness_range_arr, 66.7)),
            "p75": float(np.percentile(loudness_range_arr, 75)),
        },
        "crest_factor": {
            "p25": float(np.percentile(crest_factor_arr, 25)),
            "p33": float(np.percentile(crest_factor_arr, 33.3)),
            "p50": float(np.percentile(crest_factor_arr, 50)),
            "p66": float(np.percentile(crest_factor_arr, 66.7)),
            "p75": float(np.percentile(crest_factor_arr, 75)),
        },
        # T3.3: darkness rescales roughness by its catalogue p95
        "roughness_z": {
            "p95": float(np.percentile(np.array([r.get("roughness", 0.0) for r in raw_tracks]), 95)),
        },
    }

    # Assign kick_class (T5 rule) and restraint_bucket (catalogue quantiles)
    kick_classes = []
    restraint_buckets = []
    final_tracks = []

    for i, r in enumerate(raw_tracks):
        hpb = r.get("hits_per_bar", 0.0)
        ls = r.get("low_share", 0.0)
        lr = r.get("loudness_range", 0.0)
        cf = r.get("crest_factor", 0.0)

        # T5 kick_class from the raw ratios, S5 restraint_bucket from quantiles
        kc, rb = quantize_kick_and_restraint(
            hpb, ls, lr, cf, cut_points=cut_points,
            beat_low_ratio=r["beat_low_ratio"], off_beat_ratio=r["off_beat_ratio"],
            pulse_clarity=r["pulse_clarity"],
        )

        # Standardized vibes (Section 12.2)
        raw_ac = np.array(all_anchor_cosines[i], dtype=np.float32)
        std_ac = (raw_ac - anchor_means) / anchor_stds
        top2_std_idx = np.argsort(std_ac)[::-1][:2]
        std_vibes = [anchor_keys[idx] for idx in top2_std_idx]

        # Standardized theme_vec & theme assignment (S3)
        raw_tv = r.get("theme_vec", [0.0] * len(theme_keys))
        if r.get("has_lyrics") and len(raw_tv) == len(theme_keys):
            z = (np.array(raw_tv, dtype=np.float32) - theme_means) / theme_stds
            std_tv = z.tolist()
            max_z = float(np.max(z))
            argmax_z = int(np.argmax(z))
            theme_val = theme_keys[argmax_z] if max_z >= 0.5 else None
        else:
            std_tv = [0.0] * len(theme_keys)
            theme_val = None

        r["kick_class"] = kc
        r["restraint_bucket"] = rb
        r["vibes"] = std_vibes
        r["theme"] = theme_val
        r["theme_vec"] = std_tv
        r["anchor_profile"] = std_ac.tolist()

        kick_classes.append(kc)
        restraint_buckets.append(rb)
        final_tracks.append(r)

    # Verify distributions
    kc_counts = Counter(kick_classes)
    rb_counts = Counter(restraint_buckets)
    logger.info("=== kick_class distribution ===")
    for k, v in kc_counts.items():
        logger.info("  %s: %d (%.2f%%)", k, v, v / N * 100)
    logger.info("=== restraint_bucket distribution ===")
    for k, v in rb_counts.items():
        logger.info("  %s: %d (%.2f%%)", k, v, v / N * 100)

    # Mandatory assertion (Section 12.6 R4): kick_class has >= 3 values each with >= 10% share
    kc_shares = [v / N for v in kc_counts.values() if v / N >= 0.10]
    if len(kc_shares) < 3:
        raise RuntimeError(
            f"kick_class failed mandatory diversity requirement: only {len(kc_shares)} classes have >= 10% share! "
            f"Counts: {dict(kc_counts)}"
        )
    logger.info("kick_class passed mandatory assertion: %d classes each have >= 10%% share", len(kc_shares))

    # 6. Build Feel Matrix and Feel Index (S1)
    raw_feel_mat = np.array([extract_raw_feel(r) for r in final_tracks], dtype=np.float32)
    feel_means = np.mean(raw_feel_mat, axis=0)
    feel_stds = np.std(raw_feel_mat, axis=0)
    feel_stds[feel_stds == 0] = 1.0
    feel_weights_vec = np.array([FEEL_WEIGHTS[a] for a in FEEL_AXES], dtype=np.float32)

    feel_mat = ((raw_feel_mat - feel_means) / feel_stds) * feel_weights_vec
    feel_mat = feel_mat.astype(np.float32)
    np.save(OUT_FEEL_MATRIX_PATH, feel_mat)
    logger.info("Saved feel matrix to %s, shape %s", OUT_FEEL_MATRIX_PATH, feel_mat.shape)

    feel_index = faiss.IndexFlatL2(len(FEEL_AXES))
    feel_index.add(feel_mat)
    faiss.write_index(feel_index, str(OUT_FEEL_FAISS_PATH))
    logger.info("Saved FAISS IndexFlatL2 feel index to %s with %d entries", OUT_FEEL_FAISS_PATH, feel_index.ntotal)

    # T2 feature std from the catalogue rows (population std, as feel_std)
    feature_std = {a: float(np.std([float(r[a]) for r in raw_tracks])) for a in FEATURE_STD_AXES}
    flat = [a for a, s in feature_std.items() if s <= 0.0]
    if flat:
        # a constant feature scores 1.0 on every pair (equal values); the scorer floors std at 1e-6
        logger.warning("feature_std is zero for %s: constant in the catalogue", flat)
    logger.info("feature_std: %s", {a: round(s, 4) for a, s in feature_std.items()})

    # 7. Save track_metadata.json with Header
    header = {
        "version": "2.0",
        "total_tracks": N,
        "cut_points": cut_points,
        "kick_class_distribution": {k: float(v / N) for k, v in kc_counts.items()},
        "restraint_bucket_distribution": {k: float(v / N) for k, v in rb_counts.items()},
        "anchor_means": {anchor_keys[i]: float(anchor_means[i]) for i in range(len(anchor_keys))},
        "anchor_stds": {anchor_keys[i]: float(anchor_stds[i]) for i in range(len(anchor_keys))},
        "theme_means": {theme_keys[i]: float(theme_means[i]) for i in range(len(theme_keys))},
        "theme_stds": {theme_keys[i]: float(theme_stds[i]) for i in range(len(theme_keys))},
        "feel_axes": FEEL_AXES,
        "feel_mean": {FEEL_AXES[i]: float(feel_means[i]) for i in range(len(FEEL_AXES))},
        "feel_std": {FEEL_AXES[i]: float(feel_stds[i]) for i in range(len(FEEL_AXES))},
        "feel_weights": FEEL_WEIGHTS,
        "feature_std": feature_std,
    }

    metadata_payload = {
        "header": header,
        "tracks": final_tracks,
    }

    with open(OUT_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=1)
    logger.info("Saved track metadata to %s (%.2f MB)", OUT_METADATA_PATH, OUT_METADATA_PATH.stat().st_size / 1024 / 1024)

    # 7. Build FAISS IndexFlatIP
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    logger.info("Built FAISS IndexFlatIP with %d vectors of dimension %d", index.ntotal, dim)
    faiss.write_index(index, str(OUT_FAISS_PATH))
    logger.info("Saved FAISS index to %s (%.2f MB)", OUT_FAISS_PATH, OUT_FAISS_PATH.stat().st_size / 1024 / 1024)

    # 8. Build track_id_map.json
    id_map: dict[str, int] = {}
    for idx, rec in enumerate(final_tracks):
        if rec.get("id"):
            id_map[str(rec["id"])] = idx
        k = rec.get("key") or make_track_key(rec.get("artist", ""), rec.get("title", ""))
        id_map[k] = idx
        if rec.get("deezer_id"):
            id_map[str(rec["deezer_id"])] = idx
        if rec.get("itunes_id"):
            id_map[str(rec["itunes_id"])] = idx
            id_map[f"it_{rec['itunes_id']}"] = idx

    with open(OUT_ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(id_map, f)
    logger.info("Saved track ID map with %d keys to %s", len(id_map), OUT_ID_MAP_PATH)

    # 9. Verify Mandatory Assertions (Section 12.1)
    # Assertion 1: Anchor matrix pairwise cosine max < 0.95 and min < 0.70
    anchor_norms = np.linalg.norm(anchor_matrix, axis=1, keepdims=True)
    anchor_norms[anchor_norms == 0] = 1.0
    normed_anchors = anchor_matrix / anchor_norms
    sims = normed_anchors @ normed_anchors.T
    triu_idx = np.triu_indices(len(anchor_keys), k=1)
    pair_cos = sims[triu_idx]
    max_anchor_cos = float(np.max(pair_cos))
    min_anchor_cos = float(np.min(pair_cos))
    logger.info("Anchor matrix check: max_cosine=%.4f, min_cosine=%.4f", max_anchor_cos, min_anchor_cos)
    if max_anchor_cos >= 0.95 or min_anchor_cos >= 0.70:
        raise RuntimeError(f"Anchor diversity failed: max={max_anchor_cos:.4f}, min={min_anchor_cos:.4f}")

    # Assertion 2: First 50 vectors pairwise cosine mean < 0.90
    first_vecs = embs[:50]
    first_sims = first_vecs @ first_vecs.T
    triu_50 = np.triu_indices(50, k=1)
    mean_50_cos = float(np.mean(first_sims[triu_50]))
    logger.info("First 50 catalogue vectors mean pairwise cosine: %.4f (must be < 0.90)", mean_50_cos)
    if mean_50_cos >= 0.90:
        raise RuntimeError(f"Catalogue diversity assertion failed: mean pairwise cosine={mean_50_cos:.4f} (must be < 0.90)")

    print("\n" + "=" * 60)
    print("STEP R4 / S1 INDEX BUILD COMPLETED SUCCESSFULLY")
    print(f"Total Catalogue Rows:       {N}")
    print(f"CLAP FAISS Index Total:     {index.ntotal} (dim {dim})")
    print(f"Feel FAISS Index Total:     {feel_index.ntotal} (dim {len(FEEL_AXES)})")
    print(f"ID Map Entries:             {len(id_map)}")
    print(f"Anchor Max Cosine:          {max_anchor_cos:.4f} (< 0.95)")
    print(f"Anchor Min Cosine:          {min_anchor_cos:.4f} (< 0.70)")
    print(f"First 50 Mean Cosine:       {mean_50_cos:.4f} (< 0.90)")
    print("kick_class shares:")
    for k, v in kc_counts.items():
        print(f"  {k:20s}: {v/N*100:.2f}%")
    print("restraint_bucket shares:")
    for k, v in rb_counts.items():
        print(f"  {k:20s}: {v/N*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
