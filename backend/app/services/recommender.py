"""
Echo Recommendation Engine (Section 12 rewrite).

Core functions:
- load() -> None: loads FAISS index, metadata, anchors, rerank weights, official CLAP model (CPU)
- describe(audio_bytes, meta) -> dict: decode_preview -> descriptors + CLAP + anchors
- recommend(seed, k, cross_language_max, diversity, key) -> list[dict]
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import faiss
import httpx
import librosa
import numpy as np
import soundfile as sf
import torch

try:
    import laion_clap
    HAS_LAION = True
except ImportError:
    HAS_LAION = False

try:
    from msclap import CLAP as MSCLAP
    from transformers import GPT2Tokenizer
    if not hasattr(GPT2Tokenizer, "encode_plus"):
        GPT2Tokenizer.encode_plus = lambda self, *args, **kwargs: self(*args, **kwargs)
    HAS_MSCLAP = True
except ImportError:
    HAS_MSCLAP = False

from app.audio.decode import decode_preview
from app.audio.features import describe_audio
from app.config import settings
from app.services.language_affinity import language_affinity

logger = logging.getLogger("recommender")

# Default weights from Section 12.6
DEFAULT_WEIGHTS: dict[str, float] = {
    "s_clap":      0.18,
    "s_anchor":    0.10,
    "s_tempo":     0.14,
    "s_pulse":     0.05,
    "s_groove":    0.10,
    "s_dark":      0.08,
    "s_grit":      0.08,
    "s_vocal":     0.07,
    "s_intimacy":  0.04,
    "s_restraint": 0.03,
    "s_meaning":   0.08,
    "s_language":  0.05,
    "s_harm":      0.00,
    "s_mode":      0.00,
}

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "s_tempo":     "Felt tempo",
    "s_pulse":     "Pulse rate",
    "s_groove":    "Groove & backbeat",
    "s_dark":      "Darkness & timbre",
    "s_grit":      "Grit & drive",
    "s_vocal":     "Vocal delivery",
    "s_intimacy":  "Intimacy",
    "s_restraint": "Restraint & headroom",
    "s_meaning":   "Lyrical theme & intent",
    "s_anchor":    "Anchor feel profile",
    "s_clap":      "Acoustic embedding",
    "s_language":  "Language affinity",
    "s_harm":      "Harmonic pacing",
    "s_mode":      "Tonal mode",
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


def make_track_key(artist: str, title: str) -> str:
    return f"{_norm_artist(artist)}:{_norm_text(title)}"


def _window_audio_frames(audio: np.ndarray, sr: int, window_sec: float) -> list[np.ndarray]:
    """Splits normalised audio into non-overlapping windows of window_sec duration. Drops final window < half duration."""
    win_len = int(sr * window_sec)
    half_win = win_len // 2
    windows = []
    idx = 0
    while idx + win_len <= len(audio):
        windows.append(audio[idx : idx + win_len])
        idx += win_len
    rem = len(audio) - idx
    if rem >= half_win:
        pad = np.zeros(win_len - rem, dtype=np.float32)
        windows.append(np.concatenate([audio[idx:], pad]))
    if not windows:
        windows.append(np.pad(audio, (0, max(0, win_len - len(audio)))))
    return windows


class MusicRecommender:
    """
    Online Serving Path for Echo (Section 12.1 - 12.6).
    """

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []
        self._metadata_header: dict[str, Any] = {}
        self._track_id_map: dict[str, int] = {}
        self._anchors: dict[str, np.ndarray] = {}
        self._anchor_keys: list[str] = []
        self._anchor_matrix: np.ndarray | None = None
        self._anchor_means: np.ndarray | None = None
        self._anchor_stds: np.ndarray | None = None
        self._weights: dict[str, float] = dict(DEFAULT_WEIGHTS)

        # Backbone on CPU
        self._clap_model: Any = None
        self._clap_embedder_type: str = "laion"
        self._dim: int = 512

        # Side index for dynamic cold seeds
        self._side_index: faiss.IndexFlatIP | None = None
        self._extra_records: dict[str, dict] = {}
        self._extra_records_path: Path = Path("ml/data/extra_records.jsonl")
        self._describe_locks: dict[str, asyncio.Lock] = {}
        self._loaded: bool = False

    def load(self) -> None:
        """Load FAISS index, metadata, anchors, weights, and official CLAP model (CPU)."""
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
                raw_meta = json.load(f)
            if isinstance(raw_meta, dict) and "tracks" in raw_meta:
                self._metadata = raw_meta["tracks"]
                self._metadata_header = {k: v for k, v in raw_meta.items() if k != "tracks"}
            elif isinstance(raw_meta, list):
                self._metadata = raw_meta
            logger.info("Loaded %d track records from %s", len(self._metadata), meta_path)

        # 3. Track id map (maps id, key, deezer_id, itunes_id -> row index)
        id_map_path = _resolve("ml/inference/track_id_map.json")
        if id_map_path.exists():
            with open(id_map_path, "r", encoding="utf-8") as f:
                self._track_id_map = json.load(f)
        else:
            self._track_id_map = {}

        # Ensure all key representations (key, deezer_id, itunes_id) are indexed
        for i, rec in enumerate(self._metadata):
            if rec.get("id"):
                self._track_id_map[str(rec["id"])] = i
            k = rec.get("key") or make_track_key(str(rec.get("artist", "")), str(rec.get("title", "")))
            self._track_id_map[k] = i
            if rec.get("deezer_id"):
                self._track_id_map[str(rec["deezer_id"])] = i
            if rec.get("itunes_id"):
                self._track_id_map[str(rec["itunes_id"])] = i
                self._track_id_map[f"it_{rec['itunes_id']}"] = i

        # 4. FAISS index
        index_path = _resolve(settings.faiss_index_path)
        if index_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
                self._dim = self._index.d
                logger.info("Loaded FAISS index from %s with %d entries, dim=%d", index_path, self._index.ntotal, self._dim)
            except Exception as exc:
                logger.warning("Could not read FAISS index from %s: %s", index_path, exc)

        # 5. Anchors
        anchors_path = _resolve("ml/inference/anchors.json")
        if anchors_path.exists():
            with open(anchors_path, "r", encoding="utf-8") as f:
                raw_anchors = json.load(f)
            if isinstance(raw_anchors, dict) and "prompts" in raw_anchors:
                prompt_dict = raw_anchors["prompts"]
                self._anchor_means = np.array(raw_anchors.get("means", []), dtype=np.float32)
                self._anchor_stds = np.array(raw_anchors.get("stds", []), dtype=np.float32)
            else:
                prompt_dict = raw_anchors

            self._anchor_keys = list(prompt_dict.keys())
            self._anchors = {k: np.array(prompt_dict[k], dtype=np.float32) for k in self._anchor_keys}
            self._anchor_matrix = np.stack([self._anchors[k] for k in self._anchor_keys], axis=0)
            logger.info("Loaded %d anchor embeddings from %s", len(self._anchor_keys), anchors_path)

        # 6. Side index
        self._side_index = faiss.IndexFlatIP(self._dim)

        # Load persistent extra records if present
        if self._extra_records_path.exists():
            try:
                with open(self._extra_records_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        rid = str(rec.get("id", ""))
                        if rid:
                            self._extra_records[rid] = rec
                            cvec = np.array([rec.get("clap_vector") or [0.0] * self._dim], dtype=np.float32)
                            self._side_index.add(cvec)
                logger.info("Loaded %d persistent extra records from %s", len(self._extra_records), self._extra_records_path)
            except Exception as exc:
                logger.warning("Error loading extra records: %s", exc)

        # 7. CLAP model on CPU
        embedder_type = os.environ.get("EMBEDDER", "laion").lower()
        if embedder_type == "laion":
            logger.info("Loading official LAION CLAP model on CPU...")
            ckpt_path = Path.home() / ".cache" / "clap" / "music_audioset_epoch_15_esc_90.14.pt"
            if not ckpt_path.exists():
                for alt in [
                    Path("ml/inference/music_audioset_epoch_15_esc_90.14.pt"),
                    Path("../ml/inference/music_audioset_epoch_15_esc_90.14.pt"),
                ]:
                    if alt.exists():
                        ckpt_path = alt
                        break
            if not ckpt_path.exists():
                raise FileNotFoundError(f"LAION CLAP checkpoint not found at {ckpt_path}")

            if not HAS_LAION:
                raise ImportError("laion_clap package is not installed. Install with `pip install laion_clap torchvision torchaudio`")

            self._clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
            self._clap_model.load_ckpt(str(ckpt_path))
            self._clap_embedder_type = "laion"
            self._dim = 512
            logger.info("Official LAION CLAP model ready.")
        elif embedder_type == "msclap":
            logger.info("Loading MSCLAP 2023 model on CPU...")
            if not HAS_MSCLAP:
                raise ImportError("msclap package is not installed. Install with `pip install msclap`")
            ms_ckpt = Path.home() / ".cache" / "clap" / "CLAP_weights_2023.pth"
            ms_fp = str(ms_ckpt) if ms_ckpt.exists() else None
            self._clap_model = MSCLAP(model_fp=ms_fp, version="2023", use_cuda=False)
            self._clap_embedder_type = "msclap"
            self._dim = 1024
            logger.info("MSCLAP 2023 model ready.")
        else:
            raise ValueError(f"Unknown EMBEDDER setting: {embedder_type}")

        # MANDATORY LOAD ASSERTIONS (Section 12.1):
        # 1. Anchor matrix: pairwise cosine max < 0.95 and min < 0.70, else raise
        if self._anchor_matrix is not None and len(self._anchor_keys) > 1:
            norms = np.linalg.norm(self._anchor_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed_anchors = self._anchor_matrix / norms
            sims = normed_anchors @ normed_anchors.T
            triu_idx = np.triu_indices(len(self._anchor_keys), k=1)
            pair_cos = sims[triu_idx]
            max_cos = float(np.max(pair_cos))
            min_cos = float(np.min(pair_cos))
            if max_cos >= 0.95 or min_cos >= 0.70:
                raise RuntimeError(
                    f"Anchor matrix failed mandatory diversity assertions: "
                    f"max_cosine={max_cos:.4f} (must be < 0.95), min_cosine={min_cos:.4f} (must be < 0.70)"
                )
            logger.info("Anchor matrix passed mandatory assertions: max=%.4f, min=%.4f", max_cos, min_cos)

        # 2. The first 50 catalogue vectors: pairwise cosine mean < 0.90 and every norm > 0, else raise
        if self._index is not None and self._index.ntotal >= 50:
            n_check = 50
            first_vecs = np.zeros((n_check, self._dim), dtype=np.float32)
            for i in range(n_check):
                first_vecs[i] = self._index.reconstruct(i)
            norms = np.linalg.norm(first_vecs, axis=1)
            if np.any(norms <= 0):
                raise RuntimeError("Catalogue vectors failed mandatory assertion: zero-norm vector found in first 50 vectors!")
            first_normed = first_vecs / np.maximum(norms[:, None], 1e-8)
            cat_sims = first_normed @ first_normed.T
            cat_triu = np.triu_indices(n_check, k=1)
            mean_cat_cos = float(np.mean(cat_sims[cat_triu]))
            if mean_cat_cos >= 0.90:
                raise RuntimeError(
                    f"Catalogue vectors failed mandatory diversity assertion: mean pairwise cosine={mean_cat_cos:.4f} (must be < 0.90)!"
                )
            logger.info("Catalogue vectors passed mandatory assertions: mean_cos=%.4f, all norms > 0", mean_cat_cos)

        self._loaded = True

    def get_track_record(self, track_id: str | None = None, key: str | None = None) -> dict | None:
        """Returns the stored record for an in-catalogue or cached track, or None."""
        if not self._loaded:
            self.load()
        if key and key in self._track_id_map:
            return self._metadata[self._track_id_map[key]]
        if track_id and str(track_id) in self._track_id_map:
            return self._metadata[self._track_id_map[str(track_id)]]
        if track_id and str(track_id) in self._extra_records:
            return self._extra_records[str(track_id)]
        if key and key in self._extra_records:
            return self._extra_records[key]
        return None

    def describe(self, audio_bytes: bytes, meta: dict | None = None) -> dict:
        """
        Decode preview -> 11.5 descriptors + official CLAP embedding + anchor vibes.
        Runs in 1.5 - 2.5s on CPU.
        """
        y_22k = decode_preview(audio_bytes)
        desc = describe_audio(y_22k, meta=meta, sr=22050)

        # Deterministic windowed embedding
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        clap_emb = np.zeros(self._dim, dtype=np.float32)
        vibes = []
        raw_anchor_cosines = []

        if self._clap_model is not None:
            if self._clap_embedder_type == "laion":
                y_48k = librosa.resample(y_22k, orig_sr=22050, target_sr=48000, res_type="soxr_hq")
                windows = _window_audio_frames(y_48k, 48000, 10.0)
                win_embs = []
                for w in windows:
                    emb = self._clap_model.get_audio_embedding_from_data(x=w[None, :], use_tensor=False)
                    emb = np.array(emb, dtype=np.float32).flatten()
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    win_embs.append(emb)
                clap_emb = np.mean(win_embs, axis=0)
                norm = np.linalg.norm(clap_emb)
                if norm > 0:
                    clap_emb = clap_emb / norm
            elif self._clap_embedder_type == "msclap":
                sr = getattr(self._clap_model.args, "sampling_rate", 44100)
                dur = getattr(self._clap_model.args, "duration", 7.0)
                y_target = librosa.resample(y_22k, orig_sr=22050, target_sr=sr, res_type="soxr_hq")
                windows = _window_audio_frames(y_target, sr, dur)
                win_embs = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, w in enumerate(windows):
                        tmp_wav = os.path.join(tmpdir, f"win_{i}.wav")
                        sf.write(tmp_wav, w, sr)
                        emb = self._clap_model.get_audio_embeddings([tmp_wav], resample=False)
                        if isinstance(emb, torch.Tensor):
                            emb = emb.detach().cpu().numpy()
                        emb = np.array(emb, dtype=np.float32).flatten()
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        win_embs.append(emb)
                clap_emb = np.mean(win_embs, axis=0)
                norm = np.linalg.norm(clap_emb)
                if norm > 0:
                    clap_emb = clap_emb / norm

            # Mandatory assertion on third-party model output
            assert np.linalg.norm(clap_emb) > 0.99, f"Audio embedding norm {np.linalg.norm(clap_emb)} is not unit norm"

            if self._anchor_matrix is not None and len(self._anchor_keys) > 0:
                raw_sims = self._anchor_matrix @ clap_emb
                raw_anchor_cosines = raw_sims.tolist()

                # Standardise anchor profile if catalogue means/stds exist
                if self._anchor_means is not None and self._anchor_stds is not None and len(self._anchor_means) == len(raw_sims):
                    std_sims = (raw_sims - self._anchor_means) / np.maximum(self._anchor_stds, 1e-6)
                    top2_idx = np.argsort(std_sims)[::-1][:2]
                    vibes = [self._anchor_keys[i] for i in top2_idx]
                    desc["anchor_profile"] = std_sims.tolist()
                else:
                    top2_idx = np.argsort(raw_sims)[::-1][:2]
                    vibes = [self._anchor_keys[i] for i in top2_idx]
                    desc["anchor_profile"] = raw_sims.tolist()

        desc["clap_vector"] = clap_emb.tolist()
        desc["vibes"] = vibes
        desc["raw_anchor_cosines"] = raw_anchor_cosines
        return desc

    def _compute_pair_score(self, seed: dict, cand: dict) -> tuple[float, list[str]]:
        """Compute Section 12.6 reranking features and weighted score."""
        w = self._weights

        # 1. s_clap (backbone cosine, clipped to [0, 1])
        clap_a = np.array(seed.get("clap_vector") or [0.0] * self._dim, dtype=np.float32)
        clap_b = np.array(cand.get("clap_vector") or [0.0] * self._dim, dtype=np.float32)
        dim = len(clap_a)
        if dim > 0 and len(clap_b) == dim and np.linalg.norm(clap_a) > 0 and np.linalg.norm(clap_b) > 0:
            s_clap = float(np.clip(np.dot(clap_a, clap_b), 0.0, 1.0))
        else:
            s_clap = 0.5

        # 2. s_anchor (standardised anchor-profile cosine mapped to [0, 1])
        prof_a = seed.get("anchor_profile")
        prof_b = cand.get("anchor_profile")
        if prof_a and prof_b and len(prof_a) == len(prof_b):
            pa = np.array(prof_a, dtype=np.float32)
            pb = np.array(prof_b, dtype=np.float32)
            na = np.linalg.norm(pa)
            nb = np.linalg.norm(pb)
            if na > 0 and nb > 0:
                raw_c = float(np.dot(pa, pb) / (na * nb))
                s_anchor = float(np.clip(0.5 * (raw_c + 1.0), 0.0, 1.0))
            else:
                s_anchor = 0.5
        else:
            s_anchor = 0.5

        # 3. s_tempo
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

        # 4. s_pulse
        s_pulse = float(np.clip(1.0 - abs(float(seed.get("pulse_rate", 0.5)) - float(cand.get("pulse_rate", 0.5))), 0.0, 1.0))

        # 5. s_groove
        kc_eq = 1.0 if seed.get("kick_class") == cand.get("kick_class") else 0.0
        bb_sim = 1.0 - abs(float(seed.get("backbeat", 0.5)) - float(cand.get("backbeat", 0.5)))
        hpb_sim = max(0.0, 1.0 - abs(float(seed.get("hits_per_bar", 4.0)) - float(cand.get("hits_per_bar", 4.0))) / 8.0)
        swing_sim = max(0.0, 1.0 - abs(float(seed.get("swing", 0.5)) - float(cand.get("swing", 0.5))) / 0.2)
        lag_sim = max(0.0, 1.0 - abs(float(seed.get("lag_ms", 0.0)) - float(cand.get("lag_ms", 0.0))) / 40.0)
        s_groove = float(np.clip(0.4 * kc_eq + 0.2 * bb_sim + 0.2 * hpb_sim + 0.1 * swing_sim + 0.1 * lag_sim, 0.0, 1.0))

        # 6. s_dark
        s_dark = float(np.clip(1.0 - abs(float(seed.get("darkness", 0.5)) - float(cand.get("darkness", 0.5))), 0.0, 1.0))

        # 7. s_grit
        s_grit = float(np.clip(1.0 - abs(float(seed.get("grit", 0.5)) - float(cand.get("grit", 0.5))), 0.0, 1.0))

        # 8. s_vocal (includes lyric_words_per_minute)
        syl_sim = max(0.0, 1.0 - abs(float(seed.get("syllable_rate", 2.0)) - float(cand.get("syllable_rate", 2.0))) / 8.0)
        sus_sim = 1.0 - abs(float(seed.get("sustain_ratio", 0.5)) - float(cand.get("sustain_ratio", 0.5)))
        pitch_sim = max(0.0, 1.0 - abs(float(seed.get("pitch_range", 12.0)) - float(cand.get("pitch_range", 12.0))) / 24.0)
        tune_sim = 1.0 - abs(float(seed.get("tuning_depth", 0.5)) - float(cand.get("tuning_depth", 0.5)))
        vib_sim = max(0.0, 1.0 - abs(float(seed.get("vibrato", 20.0)) - float(cand.get("vibrato", 20.0))) / 40.0)
        wpm_a = float(seed.get("lyric_words_per_minute") or 0.0)
        wpm_b = float(cand.get("lyric_words_per_minute") or 0.0)
        if wpm_a > 0 and wpm_b > 0:
            wpm_sim = max(0.0, 1.0 - abs(wpm_a - wpm_b) / 100.0)
            s_vocal = float(np.clip(np.mean([syl_sim, sus_sim, pitch_sim, tune_sim, vib_sim, wpm_sim]), 0.0, 1.0))
        else:
            s_vocal = float(np.clip(np.mean([syl_sim, sus_sim, pitch_sim, tune_sim, vib_sim]), 0.0, 1.0))

        # 9. s_intimacy
        s_intimacy = float(np.clip(1.0 - abs(float(seed.get("intimacy", 0.5)) - float(cand.get("intimacy", 0.5))), 0.0, 1.0))

        # 10. s_restraint
        bucket_a = seed.get("restraint_bucket", "moderate")
        bucket_b = cand.get("restraint_bucket", "moderate")
        if bucket_a == bucket_b:
            s_restraint = 1.0
        elif (bucket_a, bucket_b) in [("restrained", "moderate"), ("moderate", "restrained"),
                                     ("moderate", "saturated"), ("saturated", "moderate")]:
            s_restraint = 0.5
        else:
            s_restraint = 0.0

        # 11. s_meaning (theme vector cosine)
        has_l_a = bool(seed.get("has_lyrics"))
        has_l_b = bool(cand.get("has_lyrics"))
        theme_a = seed.get("theme_vec")
        theme_b = cand.get("theme_vec")
        has_both_lyrics = bool(has_l_a and has_l_b and theme_a and theme_b and len(theme_a) == len(theme_b))
        if has_both_lyrics:
            ta = np.array(theme_a, dtype=np.float32)
            tb = np.array(theme_b, dtype=np.float32)
            nta = np.linalg.norm(ta)
            ntb = np.linalg.norm(tb)
            if nta > 0 and ntb > 0:
                s_meaning = float(np.clip(0.5 * (float(np.dot(ta, tb) / (nta * ntb)) + 1.0), 0.0, 1.0))
            else:
                s_meaning = 0.5
        else:
            s_meaning = 0.5

        # 12. s_language
        s_language = language_affinity(seed.get("language"), cand.get("language"))

        # 13. s_harm
        if seed.get("riff_flag") or cand.get("riff_flag"):
            s_harm = 0.5
        else:
            cpb_sim = max(0.0, 1.0 - abs(float(seed.get("changes_per_bar", 1.0)) - float(cand.get("changes_per_bar", 1.0))) / 4.0)
            lb_eq = 1.0 if seed.get("loop_bars") == cand.get("loop_bars") else 0.0
            s_harm = float(np.clip(0.5 * cpb_sim + 0.5 * lb_eq, 0.0, 1.0))

        # 14. s_mode
        if seed.get("mode_undefined") or cand.get("mode_undefined"):
            s_mode = 1.0
        else:
            pol_diff = abs(float(seed.get("mode_polarity", 0.0)) - float(cand.get("mode_polarity", 0.0)))
            s_mode = float(np.clip(1.0 - 0.5 * pol_diff, 0.0, 1.0))

        features = {
            "s_clap":      s_clap,
            "s_anchor":    s_anchor,
            "s_tempo":     s_tempo,
            "s_pulse":     s_pulse,
            "s_groove":    s_groove,
            "s_dark":      s_dark,
            "s_grit":      s_grit,
            "s_vocal":     s_vocal,
            "s_intimacy":  s_intimacy,
            "s_restraint": s_restraint,
            "s_meaning":   s_meaning,
            "s_language":  s_language,
            "s_harm":      s_harm,
            "s_mode":      s_mode,
        }

        # Weight redistribution if either track has no lyrics (Section 12.3)
        w_eff = dict(w)
        if not has_both_lyrics and w_eff.get("s_meaning", 0.0) > 0.0:
            m_weight = w_eff.pop("s_meaning")
            audio_keys = ["s_clap", "s_anchor", "s_tempo", "s_pulse", "s_groove", "s_dark", "s_grit", "s_vocal", "s_intimacy", "s_restraint"]
            sum_audio = sum(w_eff.get(k, 0.0) for k in audio_keys)
            if sum_audio > 0:
                for k in audio_keys:
                    w_eff[k] += m_weight * (w_eff.get(k, 0.0) / sum_audio)

        total_score = sum(w_eff.get(k, 0.0) * val for k, val in features.items())
        total_score = float(np.clip(total_score, 0.05, 0.99))

        # matchReasons: 3 features with highest w_i * s_i
        weighted_terms = [(w_eff.get(k, 0.0) * val, k) for k, val in features.items() if w_eff.get(k, 0.0) > 0.0]
        weighted_terms.sort(key=lambda x: x[0], reverse=True)
        reasons = [FEATURE_DISPLAY_NAMES.get(k, k) for _, k in weighted_terms[:3]]

        return total_score, reasons

    def recommend(
        self,
        seed: dict,
        k: int = 20,
        cross_language_max: int = 6,
        diversity: float = 0.70,
        key: str | None = None,
    ) -> list[dict]:
        """
        Step 12 Recommend Path:
        1. Retrieve seed record by key or id (stored in catalogue or extra records).
        2. ANN search (k=500) over catalogue and side index. Drop seed & same artist:title.
        3. Rerank with Section 12.6 feel scoring table.
        4. MMR with supply-scaled cross-language budget.
        5. Returns top k tracks.
        """
        if not self._loaded:
            self.load()

        seed_id = str(seed.get("id", ""))
        seed_key = key or seed.get("key") or make_track_key(seed.get("artist", ""), seed.get("title", ""))
        seed_record: dict | None = None

        # 1. Lookup stored record by key or id
        if seed_key and seed_key in self._track_id_map:
            seed_record = self._metadata[self._track_id_map[seed_key]]
        elif seed_id and seed_id in self._track_id_map:
            seed_record = self._metadata[self._track_id_map[seed_id]]
        elif seed_id and seed_id in self._extra_records:
            seed_record = self._extra_records[seed_id]
        elif seed_key and seed_key in self._extra_records:
            seed_record = self._extra_records[seed_key]
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

            # Store in extra_records and side_index
            if seed_id:
                self._extra_records[seed_id] = seed_record
            if seed_key:
                self._extra_records[seed_key] = seed_record

            if self._side_index is not None and seed_record.get("clap_vector"):
                cvec = np.array([seed_record["clap_vector"]], dtype=np.float32)
                self._side_index.add(cvec)

            # Persist to ml/data/extra_records.jsonl (Section 12.5)
            try:
                self._extra_records_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._extra_records_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(seed_record) + "\n")
            except Exception as exc:
                logger.warning("Could not persist extra record: %s", exc)

        seed_clap = np.array([seed_record.get("clap_vector") or [0.0] * self._dim], dtype=np.float32)

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

        candidates_raw: list[dict] = []
        for idx in candidate_indices:
            if idx < len(self._metadata):
                candidates_raw.append(self._metadata[idx])

        extra_list = list(self._extra_records.values())
        for s_idx in side_indices:
            if s_idx < len(extra_list):
                candidates_raw.append(extra_list[s_idx])

        # Filter out seed id and matching artist:title
        filter_key = seed_key or make_track_key(seed_record.get("artist", ""), seed_record.get("title", ""))
        filtered_candidates: list[dict] = []
        for cand in candidates_raw:
            cand_id = str(cand.get("id", ""))
            if cand_id and cand_id == seed_id:
                continue
            cand_key = cand.get("key") or make_track_key(cand.get("artist", ""), cand.get("title", ""))
            if cand_key == filter_key:
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
                    cand_clap = np.array(cand.get("clap_vector") or [0.0] * self._dim, dtype=np.float32)
                    max_sim = max(
                        float(np.dot(cand_clap, np.array(s.get("clap_vector") or [0.0] * self._dim, dtype=np.float32)))
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


def recommend(seed: dict, k: int = 20, cross_language_max: int = 6, diversity: float = 0.70, key: str | None = None) -> list[dict]:
    return recommender.recommend(seed, k=k, cross_language_max=cross_language_max, diversity=diversity, key=key)
