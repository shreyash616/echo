"""
Echo Recommendation Engine (Section 11.7 rewrite).

Core functions:
- load() -> None: loads FAISS index, metadata, anchors, rerank weights, CLAP model (CPU)
- describe(audio_bytes) -> dict: decode_preview -> 11.5 descriptors + CLAP + anchors
- recommend(seed, k, cross_language_max, diversity) -> list[dict]
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import faiss
import httpx
import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from app.audio.decode import decode_preview
from app.audio.features import describe_audio
from app.config import settings
from app.services.language_affinity import language_affinity

logger = logging.getLogger("recommender")

# Default weights from Section 11.7 (adjusted for 11.3 decision rule: clap_cosine weight 0.0)
DEFAULT_WEIGHTS: dict[str, float] = {
    "clap_cosine": 0.00,
    "s_tempo":     0.20,
    "s_pulse":     0.07,
    "s_groove":    0.15,
    "s_dark":      0.13,
    "s_grit":      0.13,
    "s_vocal":     0.12,
    "s_intimacy":  0.07,
    "s_restraint": 0.04,
    "s_harm":      0.03,
    "s_mode":      0.03,
    "s_language":  0.03,
}

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "s_tempo":     "Felt tempo",
    "s_pulse":     "Pulse rate",
    "s_groove":    "Groove & backbeat",
    "s_dark":      "Darkness & timbre",
    "s_grit":      "Grit & drive",
    "s_vocal":     "Vocal presence",
    "s_intimacy":  "Intimacy",
    "s_restraint": "Restraint & headroom",
    "s_harm":      "Harmonic pacing",
    "s_mode":      "Tonal mode",
    "s_language":  "Language affinity",
    "clap_cosine": "Acoustic embedding",
}

_CAMELOT_KEY_MAP: dict[str, str] = {
    "C Major": "8B", "A Minor": "8A",
    "G Major": "9B", "E Minor": "9A",
    "D Major": "10B", "B Minor": "10A",
    "A Major": "11B", "F# Minor": "11A",
    "E Major": "12B", "C# Minor": "12A",
    "B Major": "1B", "G# Minor": "1A",
    "F# Major": "2B", "D# Minor": "2A",
    "C# Major": "3B", "A# Minor": "3A",
    "G# Major": "4B", "F Minor": "4A",
    "D# Major": "5B", "C Minor": "5A",
    "A# Major": "6B", "G Minor": "6A",
    "F Major": "7B", "D Minor": "7A",
}


def to_camelot_key(key: str) -> str:
    """Return Camelot notation (e.g. '8A' for A Minor) or '—'."""
    clean = key.strip() if key else ""
    return _CAMELOT_KEY_MAP.get(clean, "—")


def _norm_text(s: str) -> str:
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def _norm_artist(s: str) -> str:
    art = re.split(r",|\s+feat\.|\s+ft\.|\s+with\s+|&", str(s), flags=re.IGNORECASE)[0]
    return _norm_text(art)


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


class MusicRecommender:
    """
    Online Serving Path for Echo (Section 11.7).
    """
    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []
        self._track_id_map: dict[str, int] = {}
        self._anchors: dict[str, np.ndarray] = {}
        self._anchor_keys: list[str] = []
        self._anchor_matrix: np.ndarray | None = None
        self._weights: dict[str, float] = dict(DEFAULT_WEIGHTS)

        # CLAP model on CPU
        self._clap_processor: ClapProcessor | None = None
        self._clap_model: ClapModel | None = None

        # Side index for dynamic seeds not present in the static catalogue
        self._side_index: faiss.IndexFlatIP | None = None
        self._extra_records: dict[str, dict] = {}
        self._loaded: bool = False

    def load(self) -> None:
        """Load FAISS index, metadata, anchors, weights, and CLAP model (CPU)."""
        logger.info("Loading recommendation index and artifacts...")

        def _resolve(p_str: str) -> Path:
            p = Path(p_str)
            if p.exists():
                return p
            for c in [p, Path("ml/inference") / p.name, Path("../ml/inference") / p.name]:
                if c.exists():
                    return c
            return p

        # 1. Weights
        weights_path = _resolve("ml/inference/rerank_weights.json")
        if weights_path.exists():
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    self._weights = json.load(f)
                logger.info("Loaded custom rerank weights from %s", weights_path)
            except Exception as exc:
                logger.warning("Error loading weights from %s: %s", weights_path, exc)

        # 2. Track metadata
        meta_path = _resolve(settings.track_metadata_path)
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info("Loaded %d track records from %s", len(self._metadata), meta_path)

        # 3. Track id map
        id_map_path = _resolve("ml/inference/track_id_map.json")
        if id_map_path.exists():
            with open(id_map_path, "r", encoding="utf-8") as f:
                self._track_id_map = json.load(f)
        else:
            self._track_id_map = {str(rec.get("id")): i for i, rec in enumerate(self._metadata)}

        # 4. FAISS index
        index_path = _resolve(settings.faiss_index_path)
        if index_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                logger.info("Loaded FAISS index from %s with %d entries", index_path, self._index.ntotal)
            except Exception as exc:
                logger.warning("Could not read FAISS index from %s: %s", index_path, exc)

        # 5. Anchors
        anchors_path = _resolve("ml/inference/anchors.json")
        if anchors_path.exists():
            try:
                with open(anchors_path, "r", encoding="utf-8") as f:
                    raw_anchors = json.load(f)
                self._anchor_keys = list(raw_anchors.keys())
                self._anchors = {k: np.array(v, dtype=np.float32) for k, v in raw_anchors.items()}
                self._anchor_matrix = np.stack([self._anchors[k] for k in self._anchor_keys], axis=0)
                logger.info("Loaded %d anchor embeddings from %s", len(self._anchor_keys), anchors_path)
            except Exception as exc:
                logger.warning("Error loading anchors from %s: %s", anchors_path, exc)

        # 6. Side index
        self._side_index = faiss.IndexFlatIP(512)

        # 7. CLAP model on CPU
        try:
            logger.info("Loading CLAP model on CPU...")
            try:
                self._clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music", local_files_only=True)
                self._clap_model = ClapModel.from_pretrained("laion/larger_clap_music", local_files_only=True)
            except Exception:
                logger.info("CLAP not in local cache; downloading laion/larger_clap_music...")
                self._clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_music")
                self._clap_model = ClapModel.from_pretrained("laion/larger_clap_music")
            self._clap_model.eval()
            logger.info("CLAP model ready.")
        except Exception as exc:
            logger.warning("CLAP model could not be loaded: %s", exc)

        self._loaded = True

    def describe(self, audio_bytes: bytes, meta: dict | None = None) -> dict:
        """
        Decode preview -> 11.5 descriptors + CLAP embedding + anchor vibes.
        Runs in 1.5 - 2.5s on CPU.
        """
        y_22k = decode_preview(audio_bytes)
        desc = describe_audio(y_22k, meta=meta, sr=22050)

        # CLAP embedding
        clap_emb = np.zeros(512, dtype=np.float32)
        vibes = []
        if self._clap_model is not None and self._clap_processor is not None:
            try:
                y_48k = librosa.resample(y_22k, orig_sr=22050, target_sr=48000, res_type="soxr_hq")
                inputs = self._clap_processor(audio=y_48k, sampling_rate=48000, return_tensors="pt")
                with torch.no_grad():
                    out = self._clap_model.get_audio_features(**inputs)
                    emb_t = _extract_tensor(out)
                clap_emb = emb_t.squeeze(0).cpu().numpy().astype(np.float32)
                norm = np.linalg.norm(clap_emb)
                if norm > 1e-8:
                    clap_emb = clap_emb / norm

                if self._anchor_matrix is not None and len(self._anchor_keys) > 0:
                    sims = self._anchor_matrix @ clap_emb
                    top2_idx = np.argsort(sims)[::-1][:2]
                    vibes = [self._anchor_keys[i] for i in top2_idx]
            except Exception as exc:
                logger.warning("Error generating CLAP embedding: %s", exc)

        desc["clap_vector"] = clap_emb.tolist()
        desc["vibes"] = vibes
        return desc

    def _compute_pair_score(self, seed: dict, cand: dict) -> tuple[float, list[str]]:
        """Compute 11.7 reranking features and weighted score."""
        w = self._weights

        # 1. CLAP cosine
        clap_a = np.array(seed.get("clap_vector") or [0.0] * 512, dtype=np.float32)
        clap_b = np.array(cand.get("clap_vector") or [0.0] * 512, dtype=np.float32)
        clap_cos = float(np.clip(np.dot(clap_a, clap_b), 0.0, 1.0)) if (len(clap_a) == 512 and len(clap_b) == 512) else 0.5

        # 2. s_tempo (11.5.1 posterior distance)
        dist_a = seed.get("felt_bpm_dist") or {}
        dist_b = cand.get("felt_bpm_dist") or {}
        if dist_a and dist_b:
            overlap = sum(min(dist_a.get(k, 0.0), dist_b.get(k, 0.0)) for k in dist_a)
            overlap_half = sum(min(dist_a.get(k, 0.0), dist_b.get(str(int(k) // 2), 0.0)) for k in dist_a if k.isdigit())
            overlap_double = sum(min(dist_a.get(k, 0.0), dist_b.get(str(int(k) * 2), 0.0)) for k in dist_a if k.isdigit())
            s_tempo = float(np.clip(overlap + 0.5 * overlap_half + 0.5 * overlap_double, 0.0, 1.0))
        else:
            bpm_a = float(seed.get("felt_bpm") or seed.get("bpm", 120))
            bpm_b = float(cand.get("felt_bpm") or cand.get("bpm", 120))
            ratio = bpm_a / max(1.0, bpm_b)
            diff_octave = min(abs(ratio - 1.0), abs(ratio - 0.5), abs(ratio - 2.0))
            s_tempo = float(np.clip(1.0 - diff_octave / 0.25, 0.0, 1.0))

        # 3. s_pulse
        s_pulse = float(np.clip(1.0 - abs(float(seed.get("pulse_rate", 0.5)) - float(cand.get("pulse_rate", 0.5))), 0.0, 1.0))

        # 4. s_groove
        kc_eq = 1.0 if seed.get("kick_class") == cand.get("kick_class") else 0.0
        bb_sim = 1.0 - abs(float(seed.get("backbeat", 0.5)) - float(cand.get("backbeat", 0.5)))
        hpb_sim = max(0.0, 1.0 - abs(float(seed.get("hits_per_bar", 4.0)) - float(cand.get("hits_per_bar", 4.0))) / 8.0)
        swing_sim = max(0.0, 1.0 - abs(float(seed.get("swing", 0.5)) - float(cand.get("swing", 0.5))) / 0.2)
        lag_sim = max(0.0, 1.0 - abs(float(seed.get("lag_ms", 0.0)) - float(cand.get("lag_ms", 0.0))) / 40.0)
        s_groove = float(np.clip(0.4 * kc_eq + 0.2 * bb_sim + 0.2 * hpb_sim + 0.1 * swing_sim + 0.1 * lag_sim, 0.0, 1.0))

        # 5. s_dark
        s_dark = float(np.clip(1.0 - abs(float(seed.get("darkness", 0.5)) - float(cand.get("darkness", 0.5))), 0.0, 1.0))

        # 6. s_grit
        s_grit = float(np.clip(1.0 - abs(float(seed.get("grit", 0.5)) - float(cand.get("grit", 0.5))), 0.0, 1.0))

        # 7. s_vocal
        syl_sim = max(0.0, 1.0 - abs(float(seed.get("syllable_rate", 2.0)) - float(cand.get("syllable_rate", 2.0))) / 8.0)
        sus_sim = 1.0 - abs(float(seed.get("sustain_ratio", 0.5)) - float(cand.get("sustain_ratio", 0.5)))
        pitch_sim = max(0.0, 1.0 - abs(float(seed.get("pitch_range", 12.0)) - float(cand.get("pitch_range", 12.0))) / 24.0)
        tune_sim = 1.0 - abs(float(seed.get("tuning_depth", 0.5)) - float(cand.get("tuning_depth", 0.5)))
        vib_sim = max(0.0, 1.0 - abs(float(seed.get("vibrato", 20.0)) - float(cand.get("vibrato", 20.0))) / 40.0)
        s_vocal = float(np.clip(np.mean([syl_sim, sus_sim, pitch_sim, tune_sim, vib_sim]), 0.0, 1.0))

        # 8. s_intimacy
        s_intimacy = float(np.clip(1.0 - abs(float(seed.get("intimacy", 0.5)) - float(cand.get("intimacy", 0.5))), 0.0, 1.0))

        # 9. s_restraint
        bucket_a = seed.get("restraint_bucket", "moderate")
        bucket_b = cand.get("restraint_bucket", "moderate")
        if bucket_a == bucket_b:
            s_restraint = 1.0
        elif (bucket_a, bucket_b) in [("restrained", "moderate"), ("moderate", "restrained"),
                                     ("moderate", "saturated"), ("saturated", "moderate")]:
            s_restraint = 0.5
        else:
            s_restraint = 0.0

        # 10. s_harm
        if seed.get("riff_flag") or cand.get("riff_flag"):
            s_harm = 0.5
        else:
            cpb_sim = max(0.0, 1.0 - abs(float(seed.get("changes_per_bar", 1.0)) - float(cand.get("changes_per_bar", 1.0))) / 4.0)
            lb_eq = 1.0 if seed.get("loop_bars") == cand.get("loop_bars") else 0.0
            s_harm = float(np.clip(0.5 * cpb_sim + 0.5 * lb_eq, 0.0, 1.0))

        # 11. s_mode
        if seed.get("mode_undefined") or cand.get("mode_undefined"):
            s_mode = 1.0
        else:
            pol_diff = abs(float(seed.get("mode_polarity", 0.0)) - float(cand.get("mode_polarity", 0.0)))
            s_mode = float(np.clip(1.0 - 0.5 * pol_diff, 0.0, 1.0))

        # 12. s_language
        s_language = language_affinity(seed.get("language"), cand.get("language"))

        features = {
            "clap_cosine": clap_cos,
            "s_tempo":     s_tempo,
            "s_pulse":     s_pulse,
            "s_groove":    s_groove,
            "s_dark":      s_dark,
            "s_grit":      s_grit,
            "s_vocal":     s_vocal,
            "s_intimacy":  s_intimacy,
            "s_restraint": s_restraint,
            "s_harm":      s_harm,
            "s_mode":      s_mode,
            "s_language":  s_language,
        }

        total_score = sum(w.get(k, 0.0) * val for k, val in features.items())
        total_score = float(np.clip(total_score, 0.05, 0.99))

        # matchReasons: 3 features with highest w_i * s_i
        weighted_terms = [(w.get(k, 0.0) * val, k) for k, val in features.items() if w.get(k, 0.0) > 0.0]
        weighted_terms.sort(key=lambda x: x[0], reverse=True)
        reasons = [FEATURE_DISPLAY_NAMES.get(k, k) for _, k in weighted_terms[:3]]

        return total_score, reasons

    def recommend(
        self,
        seed: dict,
        k: int = 20,
        cross_language_max: int = 6,
        diversity: float = 0.70,
    ) -> list[dict]:
        """
        Step 11.7 Recommend:
        1. Retrieve seed record (stored or cold described).
        2. ANN search (k=500) over catalogue and side index. Drop seed & same artist:title.
        3. Rerank with Section 11.7 feel scoring table.
        4. MMR with supply-scaled cross-language budget.
        5. Returns top k tracks.
        """
        if not self._loaded:
            self.load()

        seed_id = str(seed.get("id", ""))
        seed_record: dict | None = None

        # 1. Seed record lookup or cold describe
        if seed_id and seed_id in self._track_id_map:
            seed_record = self._metadata[self._track_id_map[seed_id]]
        elif seed_id and seed_id in self._extra_records:
            seed_record = self._extra_records[seed_id]
        else:
            # Cold seed
            seed_record = dict(seed)
            if seed.get("previewUrl") and not seed.get("clap_vector"):
                try:
                    with httpx.Client(timeout=10.0) as client:
                        resp = client.get(seed["previewUrl"])
                    if resp.status_code == 200:
                        desc = self.describe(resp.content, meta=seed)
                        seed_record.update(desc)
                except Exception as exc:
                    logger.warning("Failed to describe cold seed preview: %s", exc)

            if seed_id:
                self._extra_records[seed_id] = seed_record
                if self._side_index is not None and seed_record.get("clap_vector"):
                    cvec = np.array([seed_record["clap_vector"]], dtype=np.float32)
                    self._side_index.add(cvec)

        seed_clap = np.array([seed_record.get("clap_vector") or [0.0] * 512], dtype=np.float32)

        # 2. Retrieve candidates: ANN search k=500
        candidate_indices: set[int] = set()
        side_indices: set[int] = set()

        if self._index is not None and self._index.ntotal > 0:
            k_search = min(500, self._index.ntotal)
            _, I = self._index.search(seed_clap, k_search)
            for idx in I[0]:
                if idx >= 0:
                    candidate_indices.add(int(idx))

        if self._side_index is not None and self._side_index.ntotal > 0:
            k_side = min(100, self._side_index.ntotal)
            _, I_side = self._side_index.search(seed_clap, k_side)
            for s_idx in I_side[0]:
                if s_idx >= 0:
                    side_indices.add(int(s_idx))

        # Union candidate pool
        candidates_raw: list[dict] = []
        for idx in candidate_indices:
            if idx < len(self._metadata):
                candidates_raw.append(self._metadata[idx])

        extra_list = list(self._extra_records.values())
        for s_idx in side_indices:
            if s_idx < len(extra_list):
                candidates_raw.append(extra_list[s_idx])

        # Filter out seed id and matching artist:title
        seed_key = f"{_norm_artist(seed_record.get('artist', ''))}:{_norm_text(seed_record.get('title', ''))}"
        filtered_candidates: list[dict] = []
        for cand in candidates_raw:
            cand_id = str(cand.get("id", ""))
            if cand_id and cand_id == seed_id:
                continue
            cand_key = f"{_norm_artist(cand.get('artist', ''))}:{_norm_text(cand.get('title', ''))}"
            if cand_key == seed_key:
                continue
            filtered_candidates.append(cand)

        # 3. Rerank
        scored_candidates: list[tuple[dict, float, list[str]]] = []
        for cand in filtered_candidates:
            score, reasons = self._compute_pair_score(seed_record, cand)
            cand_res = dict(cand)
            cand_res["matchScore"] = score
            cand_res["matchReasons"] = reasons
            cand_res["languageAffinity"] = language_affinity(seed_record.get("language"), cand.get("language"))
            scored_candidates.append((cand_res, score, reasons))

        # Sort by rerank score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 4. MMR with supply-scaled language budget (Section 11.7.4)
        top_60 = scored_candidates[:60]
        same_count = sum(1 for c, _, _ in top_60 if (c.get("languageAffinity") or 0.0) >= 0.99)
        budget = cross_language_max if same_count >= 14 else k

        selected: list[dict] = []
        artist_counts: dict[str, int] = {}
        cross_lang_used = 0

        # MMR greedy selection
        remaining = list(scored_candidates)
        while len(selected) < k and remaining:
            best_idx = -1
            best_mmr_score = -1e9

            for idx, (cand, score, _) in enumerate(remaining):
                art = _norm_artist(cand.get("artist", ""))
                is_capped = bool(artist_counts.get(art, 0) >= 2)

                is_cross = bool((cand.get("languageAffinity") or 0.0) < 0.99)
                best_same = max(
                    [r_score for r_cand, r_score, _ in remaining if (r_cand.get("languageAffinity") or 0.0) >= 0.99],
                    default=0.0,
                )
                budget_exceeded = bool(is_cross and cross_lang_used >= budget and score < (best_same + 0.05))

                # Diversity penalty
                if selected:
                    cand_clap = np.array(cand.get("clap_vector") or [0.0] * 512, dtype=np.float32)
                    max_sim = max(
                        float(np.dot(cand_clap, np.array(s.get("clap_vector") or [0.0] * 512, dtype=np.float32)))
                        for s in selected
                    )
                else:
                    max_sim = 0.0

                mmr_val = diversity * score - (1.0 - diversity) * max_sim
                if is_capped:
                    mmr_val -= 100.0
                if budget_exceeded:
                    mmr_val -= 50.0

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = idx

            if best_idx < 0:
                # If strict constraints exhausted pool, pick highest remaining
                cand, _, _ = remaining.pop(0)
                selected.append(cand)
            else:
                cand, _, _ = remaining.pop(best_idx)
                art = _norm_artist(cand.get("artist", ""))
                artist_counts[art] = artist_counts.get(art, 0) + 1
                if (cand.get("languageAffinity") or 0.0) < 0.99:
                    cross_lang_used += 1
                selected.append(cand)

        return selected


recommender = MusicRecommender()


def load() -> None:
    recommender.load()


def describe(audio_bytes: bytes, meta: dict | None = None) -> dict:
    return recommender.describe(audio_bytes, meta=meta)


def recommend(seed: dict, k: int = 20, cross_language_max: int = 6, diversity: float = 0.70) -> list[dict]:
    return recommender.recommend(seed, k=k, cross_language_max=cross_language_max, diversity=diversity)
