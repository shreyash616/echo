"""
Echo Recommendation Engine (Section 12 rewrite).

Core functions:
- load() -> None: loads FAISS index, metadata, anchors, rerank weights, official CLAP model (CPU)
- describe(audio_bytes, meta) -> dict: decode_preview -> descriptors + CLAP + anchors
- recommend(seed, k, cross_language_max, diversity, key) -> list[dict]
"""
from __future__ import annotations

import asyncio
import functools
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

# T2 (FIX_PLAN_2): a raw-feature difference scores exp(-|a - b| / (TAU * std)), std from the
# metadata header's feature_std, so one catalogue standard deviation scores 0.37 on every axis
TAU = 1.0
# T3.4 as amended a0d9b43: weight of the polarity similarity when either mode is undefined
MODE_SHRINK = 0.5


def _zsim(a: float, b: float, std: float) -> float:
    return float(np.exp(-abs(a - b) / (TAU * max(std, 1e-6))))

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

# The axes in _compute_pair_score's features order; _score_pool's axis columns follow it
POOL_AXES: list[str] = [
    "s_clap", "s_anchor", "s_tempo", "s_pulse", "s_groove", "s_dark", "s_grit", "s_vocal",
    "s_intimacy", "s_restraint", "s_meaning", "s_language", "s_harm", "s_mode",
]

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


@functools.lru_cache(maxsize=None)
def _bench_norm(s: str) -> str:
    """tests/feel_benchmark.py normalize_title, copied once because the backend cannot import tests/:
    the seed filter drops a candidate that feel_benchmark.match_track would call the seed (P8)."""
    t = re.sub(r"\(.*?\)", "", s)
    t = re.sub(r"\[.*?\]", "", t)
    t = re.sub(r"[^\w\s]", "", t).strip().lower()
    return re.sub(r"\s+", " ", t)


def extract_raw_feel(r: dict) -> list[float]:
    """Extract raw 17-dimensional feel features in canonical S1 order."""
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
        self._cut_points: dict[str, Any] | None = None
        self._feature_std: dict[str, float] = {}
        self._track_id_map: dict[str, int] = {}
        self._anchors: dict[str, np.ndarray] = {}
        self._anchor_keys: list[str] = []
        self._anchor_matrix: np.ndarray | None = None
        self._anchor_means: np.ndarray | None = None
        self._anchor_stds: np.ndarray | None = None
        self._weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
        self._clap_matrix: np.ndarray | None = None

        # Backbone on CPU
        self._clap_model: Any = None
        self._clap_embedder_type: str = "laion"
        self._dim: int = 512

        # Side index for dynamic cold seeds
        self._side_index: faiss.IndexFlatIP | None = None
        self._extra_records: dict[str, dict] = {}
        self._extra_records_path: Path = Path("ml/data/extra_records.jsonl")
        self._describe_locks: dict[str, asyncio.Lock] = {}

        # Feel matrix & index (S1)
        self._feel_index: faiss.IndexFlatL2 | None = None
        self._feel_matrix: np.ndarray | None = None
        self._feel_axes: list[str] = [
            "log2_felt_bpm", "pulse_rate", "hits_per_bar", "darkness", "brightness",
            "low_share", "grit", "roughness", "intimacy", "reverb_tail",
            "loudness_range", "crest_factor", "lag_ms", "swing", "syllable_rate",
            "sustain_ratio", "mode_polarity"
        ]
        self._feel_mean: np.ndarray | None = None
        self._feel_std: np.ndarray | None = None
        self._feel_weights_vec: np.ndarray | None = None

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
                self._metadata_header = raw_meta.get("header", {k: v for k, v in raw_meta.items() if k != "tracks"})
                self._cut_points = self._metadata_header.get("cut_points")
                self._feature_std = dict(self._metadata_header.get("feature_std") or {})
                if not self._feature_std:
                    logger.error("track_metadata.json header has no feature_std; T2 similarities use std 1.0 on every axis")
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
                if self._index.ntotal > 0:
                    self._clap_matrix = self._index.reconstruct_n(0, self._index.ntotal)
                    logger.info("Extracted CLAP matrix from FAISS index: shape %s", self._clap_matrix.shape)
            except Exception as exc:
                logger.warning("Could not read FAISS index from %s: %s", index_path, exc)

        # 4b. Feel Matrix and Feel Index (S1)
        feel_mat_path = _resolve("ml/inference/feel_matrix.npy")
        if feel_mat_path.exists():
            try:
                self._feel_matrix = np.load(str(feel_mat_path))
                logger.info("Loaded feel matrix from %s: shape %s", feel_mat_path, self._feel_matrix.shape)
            except Exception as exc:
                logger.warning("Could not load feel matrix: %s", exc)

        feel_idx_path = _resolve("ml/inference/feel_index.faiss")
        if feel_idx_path.exists():
            try:
                self._feel_index = faiss.read_index(str(feel_idx_path))
                logger.info("Loaded FAISS feel index from %s with %d entries", feel_idx_path, self._feel_index.ntotal)
            except Exception as exc:
                logger.warning("Could not load feel index: %s", exc)

        # Header feel metadata
        if self._metadata_header:
            if "feel_axes" in self._metadata_header:
                self._feel_axes = self._metadata_header["feel_axes"]
            mean_dict = self._metadata_header.get("feel_mean", {})
            std_dict = self._metadata_header.get("feel_std", {})
            weights_dict = self._metadata_header.get("feel_weights", {})
            if mean_dict and std_dict:
                self._feel_mean = np.array([mean_dict.get(a, 0.0) for a in self._feel_axes], dtype=np.float32)
                self._feel_std = np.array([std_dict.get(a, 1.0) for a in self._feel_axes], dtype=np.float32)
                self._feel_std[self._feel_std == 0] = 1.0
            if weights_dict:
                self._feel_weights_vec = np.array([weights_dict.get(a, 1.0) for a in self._feel_axes], dtype=np.float32)

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
                try:
                    from huggingface_hub import hf_hub_download
                    logger.info("Downloading official LAION CLAP checkpoint to ~/.cache/clap/...")
                    cache_dir = Path.home() / ".cache" / "clap"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    dl_path = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt", local_dir=str(cache_dir))
                    ckpt_path = Path(dl_path)
                except Exception as dl_err:
                    raise FileNotFoundError(f"LAION CLAP checkpoint not found at {ckpt_path} and auto-download failed: {dl_err}")

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

    def get_track_index(self, record: dict) -> int | None:
        """Returns catalogue row index if record is in-catalogue, else None."""
        if not self._loaded:
            self.load()
        k = record.get("key") or make_track_key(str(record.get("artist", "")), str(record.get("title", "")))
        if k in self._track_id_map:
            return self._track_id_map[k]
        tid = record.get("id")
        if tid and str(tid) in self._track_id_map:
            return self._track_id_map[str(tid)]
        return None

    def get_feel_vector(self, record: dict) -> np.ndarray:
        """Returns 17-dimensional weighted standardised feel vector (S1)."""
        idx = self.get_track_index(record)
        if idx is not None and self._feel_matrix is not None and idx < len(self._feel_matrix):
            return self._feel_matrix[idx]
        raw = np.array(extract_raw_feel(record), dtype=np.float32)
        if self._feel_mean is not None and self._feel_std is not None and self._feel_weights_vec is not None:
            return ((raw - self._feel_mean) / self._feel_std) * self._feel_weights_vec
        return raw

    def update_feel_weights(self, new_weights: list[float] | np.ndarray) -> None:
        """Dynamically update feel weights and re-index feel space in memory."""
        if self._feel_weights_vec is None or self._feel_matrix is None:
            return
        if not hasattr(self, "_feel_unweighted_matrix") or self._feel_unweighted_matrix is None:
            safe_div = np.where(self._feel_weights_vec != 0, self._feel_weights_vec, 1.0)
            self._feel_unweighted_matrix = self._feel_matrix / safe_div
        new_vec = np.array(new_weights, dtype=np.float32)
        self._feel_weights_vec = new_vec
        self._feel_matrix = (self._feel_unweighted_matrix * new_vec).astype(np.float32)
        new_idx = faiss.IndexFlatL2(len(new_vec))
        new_idx.add(self._feel_matrix)
        self._feel_index = new_idx

    def get_clap_vector(self, track: dict | None) -> np.ndarray:
        """Retrieve unit-norm 512-d CLAP vector for catalogue or described track."""
        if not track:
            return np.zeros(self._dim, dtype=np.float32)
        if track.get("clap_vector") is not None:
            v = np.array(track["clap_vector"], dtype=np.float32)
            if v.ndim == 1 and len(v) == self._dim:
                return v
        k = track.get("key") or make_track_key(str(track.get("artist", "")), str(track.get("title", "")))
        idx = self._track_id_map.get(k)
        if idx is None and track.get("id"):
            idx = self._track_id_map.get(str(track["id"]))
        if idx is None and track.get("deezer_id"):
            idx = self._track_id_map.get(str(track["deezer_id"]))
        if idx is None and track.get("itunes_id"):
            idx = self._track_id_map.get(str(track["itunes_id"]))
            if idx is None:
                idx = self._track_id_map.get(f"it_{track['itunes_id']}")
        if idx is not None and self._clap_matrix is not None and 0 <= idx < len(self._clap_matrix):
            return self._clap_matrix[idx]
        return np.zeros(self._dim, dtype=np.float32)

    def describe(self, audio_bytes: bytes, meta: dict | None = None) -> dict:
        """
        Decode preview -> 11.5 descriptors + official CLAP embedding + anchor vibes.
        Runs in 1.5 - 2.5s on CPU.
        """
        y_22k = decode_preview(audio_bytes)
        desc = describe_audio(y_22k, meta=meta, sr=22050, cut_points=self._cut_points)

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

    def _compute_pair_score(self, seed: dict, cand: dict, return_debug: bool = False):
        """Compute Section 12.6 reranking features and weighted score."""
        w = self._weights

        # 1. s_clap (backbone cosine, clipped to [0, 1])
        clap_a = self.get_clap_vector(seed)
        clap_b = self.get_clap_vector(cand)
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

        # 3. s_tempo (Section 11.5.1 / 11.7)
        dist_a = seed.get("felt_bpm_dist") or {}
        dist_b = cand.get("felt_bpm_dist") or {}
        if dist_a and dist_b:
            log2_1_5 = 0.5849625
            d = 0.0
            for k_a, p_a in dist_a.items():
                try:
                    bpm_a = float(k_a)
                except ValueError:
                    continue
                for k_b, p_b in dist_b.items():
                    try:
                        bpm_b = float(k_b)
                    except ValueError:
                        continue
                    if bpm_a > 0 and bpm_b > 0:
                        x = abs(float(np.log2(bpm_a / bpm_b)))
                        if abs(x - 1.0) <= 0.25:
                            fold_x = min(x, 0.4)
                        elif abs(x - log2_1_5) <= 0.15:
                            fold_x = 0.7
                        else:
                            fold_x = min(1.0, x)
                        d += float(p_a) * float(p_b) * fold_x
            s_tempo = float(np.clip(1.0 - d, 0.0, 1.0))
        else:
            bpm_a = float(seed.get("felt_bpm") or seed.get("bpm", 120))
            bpm_b = float(cand.get("felt_bpm") or cand.get("bpm", 120))
            if bpm_a > 0 and bpm_b > 0:
                x = abs(float(np.log2(bpm_a / bpm_b)))
                if abs(x - 1.0) <= 0.25:
                    fold_x = min(x, 0.4)
                elif abs(x - 0.5849625) <= 0.15:
                    fold_x = 0.7
                else:
                    fold_x = min(1.0, x)
                s_tempo = float(np.clip(1.0 - fold_x, 0.0, 1.0))
            else:
                s_tempo = 0.5

        fs = self._feature_std

        def z(key: str, default: float) -> float:
            return _zsim(float(seed.get(key, default)), float(cand.get(key, default)), fs.get(key, 1.0))

        # 4. s_pulse
        s_pulse = z("pulse_rate", 0.5)

        # 5. s_groove (T3.5: swing is constant in the catalogue and left out; the other four
        # inner weights 0.4, 0.2, 0.2, 0.1 renormalised to sum to 1)
        kc_eq = 1.0 if seed.get("kick_class") == cand.get("kick_class") else 0.0
        bb_sim = z("backbeat", 0.5)
        hpb_sim = z("hits_per_bar", 4.0)
        lag_sim = z("lag_ms", 0.0)
        s_groove = float(np.clip((0.4 * kc_eq + 0.2 * bb_sim + 0.2 * hpb_sim + 0.1 * lag_sim) / 0.9, 0.0, 1.0))

        # 6. s_dark
        s_dark = z("darkness", 0.5)

        # 7. s_grit
        s_grit = z("grit", 0.5)

        # 8. s_vocal (includes lyric_words_per_minute); tuning_depth and words per minute have no
        # header std in T2's list and keep their linear terms
        syl_sim = z("syllable_rate", 2.0)
        sus_sim = z("sustain_ratio", 0.5)
        pitch_sim = z("pitch_range", 12.0)
        tune_sim = 1.0 - abs(float(seed.get("tuning_depth", 0.5)) - float(cand.get("tuning_depth", 0.5)))
        vib_sim = z("vibrato", 20.0)
        wpm_a = float(seed.get("lyric_words_per_minute") or 0.0)
        wpm_b = float(cand.get("lyric_words_per_minute") or 0.0)
        if wpm_a > 0 and wpm_b > 0:
            wpm_sim = max(0.0, 1.0 - abs(wpm_a - wpm_b) / 100.0)
            s_vocal = float(np.clip(np.mean([syl_sim, sus_sim, pitch_sim, tune_sim, vib_sim, wpm_sim]), 0.0, 1.0))
        else:
            s_vocal = float(np.clip(np.mean([syl_sim, sus_sim, pitch_sim, tune_sim, vib_sim]), 0.0, 1.0))

        # 9. s_intimacy
        s_intimacy = z("intimacy", 0.5)

        # 10. s_restraint (z-scored loudness range and crest factor; restraint_bucket is a display label)
        dz_lr = abs(float(seed.get("loudness_range", 0.0)) - float(cand.get("loudness_range", 0.0))) / max(fs.get("loudness_range", 1.0), 1e-6)
        dz_cf = abs(float(seed.get("crest_factor", 0.0)) - float(cand.get("crest_factor", 0.0))) / max(fs.get("crest_factor", 1.0), 1e-6)
        s_restraint = float(np.exp(-(dz_lr + dz_cf) / 2.0))

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

        # 13. s_harm (T3.4: no riff override; riff_flag equality is a 0.2 term inside it)
        cpb_sim = z("changes_per_bar", 1.0)
        lb_eq = 1.0 if seed.get("loop_bars") == cand.get("loop_bars") else 0.0
        riff_eq = 1.0 if bool(seed.get("riff_flag")) == bool(cand.get("riff_flag")) else 0.0
        s_harm = float(np.clip(0.4 * cpb_sim + 0.4 * lb_eq + 0.2 * riff_eq, 0.0, 1.0))

        # 14. s_mode (T3.4 as amended a0d9b43: z-scored polarity; an undefined mode shrinks it toward 0.7)
        sim_pol = z("mode_polarity", 0.0)
        if seed.get("mode_undefined") or cand.get("mode_undefined"):
            s_mode = 0.7 + MODE_SHRINK * (sim_pol - 0.7)
        else:
            s_mode = sim_pol

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

        if return_debug:
            return total_score, reasons, features, w_eff
        return total_score, reasons

    def _score_pool(self, seed: dict, cands: list[dict]) -> dict:
        """_compute_pair_score over a whole pool as arrays (P7): the same formulas term by term,
        so every axis and score agrees with the single-pair path to float rounding. Returns
        scores (n,), axes (n, 14) in POOL_AXES order, has_both (n,), w_eff for both cases, the
        reasons per candidate, language affinity (n,) and the candidates' CLAP rows (n, dim)."""
        w = self._weights
        n = len(cands)
        fs = self._feature_std

        def col(key: str, default: float) -> np.ndarray:
            return np.array([float(c.get(key, default)) for c in cands], dtype=np.float64)

        def z(key: str, default: float) -> np.ndarray:
            std = fs.get(key, 1.0)
            return np.exp(-np.abs(float(seed.get(key, default)) - col(key, default)) / (TAU * max(std, 1e-6)))

        def cosine_01(vec_a, key: str) -> tuple[np.ndarray, np.ndarray]:
            # 0.5 * (cosine + 1) clipped, 0.5 where either vector is missing, empty or zero
            out = np.full(n, 0.5)
            ok = np.zeros(n, dtype=bool)
            if not vec_a:
                return out, ok
            pa = np.array(vec_a, dtype=np.float32)
            rows = [i for i, c in enumerate(cands) if c.get(key) and len(c[key]) == len(vec_a)]
            ok[rows] = True
            if rows:
                pb = np.array([cands[i][key] for i in rows], dtype=np.float32)
                na = np.linalg.norm(pa)
                nb = np.linalg.norm(pb, axis=1)
                good = (nb > 0) & (na > 0)
                raw = (pb @ pa) / (na * np.where(good, nb, 1.0))
                vals = np.clip(0.5 * (raw.astype(np.float64) + 1.0), 0.0, 1.0)
                out[np.array(rows)[good]] = vals[good]
            return out, ok

        # 1. s_clap
        clap_a = self.get_clap_vector(seed)
        clap_b = np.stack([self.get_clap_vector(c) for c in cands]) if n else np.zeros((0, len(clap_a)), dtype=np.float32)
        if len(clap_a) > 0 and np.linalg.norm(clap_a) > 0:
            s_clap = np.where(np.linalg.norm(clap_b, axis=1) > 0,
                              np.clip((clap_b @ clap_a).astype(np.float64), 0.0, 1.0), 0.5)
        else:
            s_clap = np.full(n, 0.5)

        # 2. s_anchor
        s_anchor, _ = cosine_01(seed.get("anchor_profile"), "anchor_profile")

        # 3. s_tempo: the double loop over both felt-BPM distributions, flattened; bincount adds
        # each candidate's terms in the loop's order
        def fold(x: np.ndarray) -> np.ndarray:
            return np.where(np.abs(x - 1.0) <= 0.25, np.minimum(x, 0.4),
                            np.where(np.abs(x - 0.5849625) <= 0.15, 0.7, np.minimum(1.0, x)))

        def parse_dist(dist: dict) -> list[tuple[float, float]]:
            out = []
            for k_s, p in dist.items():
                try:
                    out.append((float(k_s), float(p)))
                except ValueError:
                    continue
            return out

        dist_a = seed.get("felt_bpm_dist") or {}
        has_dist = np.array([bool(dist_a and c.get("felt_bpm_dist")) for c in cands], dtype=bool)
        s_tempo = np.full(n, 0.5)
        if has_dist.any():
            seed_pairs = parse_dist(dist_a)
            flat_i, flat_bpm, flat_p = [], [], []
            for i in np.flatnonzero(has_dist):
                for bpm_b, p_b in parse_dist(cands[i]["felt_bpm_dist"]):
                    flat_i.append(i); flat_bpm.append(bpm_b); flat_p.append(p_b)
            flat_i = np.array(flat_i, dtype=np.int64)
            flat_bpm = np.array(flat_bpm, dtype=np.float64)
            flat_p = np.array(flat_p, dtype=np.float64)
            idx_all, terms_all = [], []
            for bpm_a, p_a in seed_pairs:
                valid = (flat_bpm > 0) & (bpm_a > 0)
                x = np.abs(np.log2(bpm_a / flat_bpm[valid]))
                idx_all.append(flat_i[valid])
                terms_all.append(p_a * flat_p[valid] * fold(x))
            d = np.bincount(np.concatenate(idx_all), weights=np.concatenate(terms_all), minlength=n) if idx_all else np.zeros(n)
            s_tempo[has_dist] = np.clip(1.0 - d[has_dist], 0.0, 1.0)
        if (~has_dist).any():
            bpm_a = float(seed.get("felt_bpm") or seed.get("bpm", 120))
            rest = np.flatnonzero(~has_dist)
            bpm_b = np.array([float(cands[i].get("felt_bpm") or cands[i].get("bpm", 120)) for i in rest], dtype=np.float64)
            ok = (bpm_b > 0) & (bpm_a > 0)
            x = np.abs(np.log2(bpm_a / np.where(ok, bpm_b, 1.0)))
            s_tempo[rest] = np.where(ok, np.clip(1.0 - fold(x), 0.0, 1.0), 0.5)

        # 4. s_pulse
        s_pulse = z("pulse_rate", 0.5)

        # 5. s_groove (T3.5, no swing)
        kc_eq = np.array([1.0 if seed.get("kick_class") == c.get("kick_class") else 0.0 for c in cands])
        s_groove = np.clip((0.4 * kc_eq + 0.2 * z("backbeat", 0.5) + 0.2 * z("hits_per_bar", 4.0)
                            + 0.1 * z("lag_ms", 0.0)) / 0.9, 0.0, 1.0)

        # 6. s_dark, 7. s_grit
        s_dark = z("darkness", 0.5)
        s_grit = z("grit", 0.5)

        # 8. s_vocal: mean of five terms, six where both sides have words per minute
        tune_sim = 1.0 - np.abs(float(seed.get("tuning_depth", 0.5)) - col("tuning_depth", 0.5))
        vocal_sum = z("syllable_rate", 2.0) + z("sustain_ratio", 0.5) + z("pitch_range", 12.0) + tune_sim + z("vibrato", 20.0)
        wpm_a = float(seed.get("lyric_words_per_minute") or 0.0)
        wpm_b = np.array([float(c.get("lyric_words_per_minute") or 0.0) for c in cands], dtype=np.float64)
        has_wpm = (wpm_b > 0) & (wpm_a > 0)
        wpm_sim = np.maximum(0.0, 1.0 - np.abs(wpm_a - wpm_b) / 100.0)
        s_vocal = np.clip(np.where(has_wpm, (vocal_sum + wpm_sim) / 6.0, vocal_sum / 5.0), 0.0, 1.0)

        # 9. s_intimacy
        s_intimacy = z("intimacy", 0.5)

        # 10. s_restraint
        dz_lr = np.abs(float(seed.get("loudness_range", 0.0)) - col("loudness_range", 0.0)) / max(fs.get("loudness_range", 1.0), 1e-6)
        dz_cf = np.abs(float(seed.get("crest_factor", 0.0)) - col("crest_factor", 0.0)) / max(fs.get("crest_factor", 1.0), 1e-6)
        s_restraint = np.exp(-(dz_lr + dz_cf) / 2.0)

        # 11. s_meaning
        s_meaning, theme_ok = cosine_01(seed.get("theme_vec") if seed.get("has_lyrics") else None, "theme_vec")
        has_both = theme_ok & np.array([bool(c.get("has_lyrics")) for c in cands], dtype=bool)
        s_meaning = np.where(has_both, s_meaning, 0.5)

        # 12. s_language
        seed_lang = seed.get("language")
        aff_cache: dict[Any, float] = {}
        for c in cands:
            lang = c.get("language")
            if lang not in aff_cache:
                aff_cache[lang] = language_affinity(seed_lang, lang)
        s_language = np.array([aff_cache[c.get("language")] for c in cands], dtype=np.float64)

        # 13. s_harm
        lb_eq = np.array([1.0 if seed.get("loop_bars") == c.get("loop_bars") else 0.0 for c in cands])
        riff_eq = np.array([1.0 if bool(seed.get("riff_flag")) == bool(c.get("riff_flag")) else 0.0 for c in cands])
        s_harm = np.clip(0.4 * z("changes_per_bar", 1.0) + 0.4 * lb_eq + 0.2 * riff_eq, 0.0, 1.0)

        # 14. s_mode
        sim_pol = z("mode_polarity", 0.0)
        undefined = np.array([bool(seed.get("mode_undefined") or c.get("mode_undefined")) for c in cands], dtype=bool)
        s_mode = np.where(undefined, 0.7 + MODE_SHRINK * (sim_pol - 0.7), sim_pol)

        axes = np.stack([s_clap, s_anchor, s_tempo, s_pulse, s_groove, s_dark, s_grit, s_vocal,
                         s_intimacy, s_restraint, s_meaning, s_language, s_harm, s_mode], axis=1) if n else np.zeros((0, len(POOL_AXES)))

        # Weight redistribution if either track has no lyrics (Section 12.3), as the single-pair path
        w_lyr = dict(w)
        w_nolyr = dict(w)
        if w_nolyr.get("s_meaning", 0.0) > 0.0:
            m_weight = w_nolyr.pop("s_meaning")
            audio_keys = ["s_clap", "s_anchor", "s_tempo", "s_pulse", "s_groove", "s_dark", "s_grit", "s_vocal", "s_intimacy", "s_restraint"]
            sum_audio = sum(w_nolyr.get(k, 0.0) for k in audio_keys)
            if sum_audio > 0:
                for k in audio_keys:
                    w_nolyr[k] += m_weight * (w_nolyr.get(k, 0.0) / sum_audio)
        wv_lyr = np.array([w_lyr.get(k, 0.0) for k in POOL_AXES], dtype=np.float64)
        wv_nolyr = np.array([w_nolyr.get(k, 0.0) for k in POOL_AXES], dtype=np.float64)
        W = np.where(has_both[:, None], wv_lyr[None, :], wv_nolyr[None, :])

        # The loop's sum(w * s) runs over the axes in order; the same order here
        terms = W * axes
        total = np.zeros(n)
        for j in range(len(POOL_AXES)):
            total = total + terms[:, j]
        scores = np.clip(total, 0.05, 0.99)

        # matchReasons: the three axes with the highest w * s among positive weights; a stable
        # sort keeps axis order on ties, as the loop's list.sort does
        ranked = np.argsort(-np.where(W > 0.0, terms, -np.inf), axis=1, kind="stable")[:, :3]
        names = [FEATURE_DISPLAY_NAMES.get(k, k) for k in POOL_AXES]
        reasons = [[names[j] for j in ranked[i] if W[i, j] > 0.0] for i in range(n)]

        return {
            "scores": scores, "axes": axes, "has_both": has_both, "w_lyr": w_lyr, "w_nolyr": w_nolyr,
            "reasons": reasons, "language": s_language, "clap": clap_b,
        }

    def _is_seed_copy(self, seed_record: dict, cand: dict, seed_id: str = "", seed_key: str | None = None) -> bool:
        """Whether cand is the seed's own song: the same id, the same artist:title key, or what
        feel_benchmark.match_track calls the same track (title and artist containment either way,
        _bench_norm), which catches a copy of the seed under another id and key (a side record)."""
        cand_id = str(cand.get("id", ""))
        if cand_id and cand_id == seed_id:
            return True
        filter_key = seed_key or make_track_key(seed_record.get("artist", ""), seed_record.get("title", ""))
        if (cand.get("key") or make_track_key(cand.get("artist", ""), cand.get("title", ""))) == filter_key:
            return True
        s_title, c_title = _bench_norm(str(seed_record.get("title", ""))), _bench_norm(str(cand.get("title", "")))
        s_art, c_art = _bench_norm(str(seed_record.get("artist", ""))), _bench_norm(str(cand.get("artist", "")))
        return bool((s_title in c_title or c_title in s_title) and (s_art in c_art or c_art in s_art))

    def recommend(
        self,
        seed: dict,
        k: int = 20,
        cross_language_max: int = 6,
        diversity: float = 0.70,
        key: str | None = None,
        debug: bool = False,
        return_pool: bool = False,
    ):
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

        seed_clap = self.get_clap_vector(seed_record)
        seed_clap_2d = seed_clap.reshape(1, -1)

        # 2. Retrieve candidates: Union of CLAP (500) and Feel (600) + Side Index (S1)
        clap_indices: set[int] = set()
        feel_indices: set[int] = set()
        side_indices: set[int] = set()

        if self._index is not None and self._index.ntotal > 0:
            k_search = min(settings.candidates_clap, self._index.ntotal)
            _, I = self._index.search(seed_clap_2d, k_search)
            for idx in I[0]:
                if idx >= 0:
                    clap_indices.add(int(idx))

        if self._feel_index is not None and self._feel_index.ntotal > 0:
            seed_feel = self.get_feel_vector(seed_record).reshape(1, -1).astype(np.float32)
            k_feel = min(settings.candidates_feel, self._feel_index.ntotal)
            _, I_feel = self._feel_index.search(seed_feel, k_feel)
            for idx in I_feel[0]:
                if idx >= 0:
                    feel_indices.add(int(idx))

        if self._side_index is not None and self._side_index.ntotal > 0:
            k_side = min(100, self._side_index.ntotal)
            _, I_side = self._side_index.search(seed_clap_2d, k_side)
            for s_idx in I_side[0]:
                if s_idx >= 0:
                    side_indices.add(int(s_idx))

        union_indices = clap_indices | feel_indices
        candidates_raw: list[dict] = []
        for idx in union_indices:
            if idx < len(self._metadata):
                cand = dict(self._metadata[idx])
                is_c = idx in clap_indices
                is_f = idx in feel_indices
                cand["pool_source"] = "both" if (is_c and is_f) else ("clap" if is_c else "feel")
                candidates_raw.append(cand)

        extra_list = list(self._extra_records.values())
        for s_idx in side_indices:
            if s_idx < len(extra_list):
                cand = dict(extra_list[s_idx])
                cand["pool_source"] = "clap"
                candidates_raw.append(cand)

        # Filter out the seed and copies of it (_is_seed_copy)
        filtered_candidates: list[dict] = []
        for cand in candidates_raw:
            if self._is_seed_copy(seed_record, cand, seed_id=seed_id, seed_key=seed_key):
                continue
            filtered_candidates.append(cand)

        # 3. Rerank (the whole pool as arrays; _compute_pair_score stays the single-pair reference)
        pool = self._score_pool(seed_record, filtered_candidates)
        scored_candidates: list[tuple[dict, float, list[str]]] = []
        for i, cand in enumerate(filtered_candidates):
            score = float(pool["scores"][i])
            reasons = pool["reasons"][i]
            cand_res = dict(cand)
            cand_res["matchScore"] = score
            cand_res["matchReasons"] = reasons
            cand_res["languageAffinity"] = float(pool["language"][i])
            if debug:
                cand_res["axes"] = {a: float(pool["axes"][i, j]) for j, a in enumerate(POOL_AXES)}
                cand_res["weights"] = dict(pool["w_lyr"] if pool["has_both"][i] else pool["w_nolyr"])
            scored_candidates.append((cand_res, score, reasons))

        # Sort by rerank score descending; the CLAP rows follow the same stable order
        order = sorted(range(len(scored_candidates)), key=lambda i: scored_candidates[i][1], reverse=True)
        scored_candidates = [scored_candidates[i] for i in order]
        pool_claps = pool["clap"][order] if order else pool["clap"]

        # 4. MMR with supply-scaled language budget (Section 11.7.4)
        top_60 = scored_candidates[:60]
        same_count = sum(1 for c, _, _ in top_60 if (c.get("languageAffinity") or 0.0) >= 0.99)
        budget = cross_language_max if same_count >= 14 else k

        selected: list[dict] = []
        cross_lang_used = 0

        # MMR greedy selection as arrays over the pool (P7). max_sim holds each candidate's highest
        # CLAP cosine to the picks so far and is updated by one matrix-vector product per pick;
        # np.argmax takes the first of equal values, as the loop's strict > did
        n_pool = len(scored_candidates)
        pool_scores = np.array([score for _, score, _ in scored_candidates], dtype=np.float64)
        is_cross_arr = np.array([bool((cand.get("languageAffinity") or 0.0) < 0.99) for cand, _, _ in scored_candidates], dtype=bool)
        art_names = [_norm_artist(cand.get("artist", "")) for cand, _, _ in scored_candidates]
        art_ids: dict[str, int] = {}
        art_codes = np.array([art_ids.setdefault(a, len(art_ids)) for a in art_names], dtype=np.int64)
        art_picks = np.zeros(len(art_ids), dtype=np.int64)
        remaining = np.ones(n_pool, dtype=bool)
        max_sim = np.zeros(n_pool, dtype=np.float64)

        while len(selected) < k and remaining.any():
            rem = np.flatnonzero(remaining)
            rem_scores = pool_scores[rem]
            rem_cross = is_cross_arr[rem]
            best_same = float(rem_scores[~rem_cross].max()) if (~rem_cross).any() else 0.0

            is_capped = art_picks[art_codes[rem]] >= 2
            budget_exceeded = rem_cross & (cross_lang_used >= budget) & (rem_scores < (best_same + 0.05))

            mmr_val = diversity * rem_scores - (1.0 - diversity) * (max_sim[rem] if selected else 0.0)
            mmr_val = np.where(is_capped, mmr_val - 100.0, mmr_val)
            mmr_val = np.where(budget_exceeded, mmr_val - 50.0, mmr_val)

            best = int(rem[int(np.argmax(mmr_val))])
            sims = (pool_claps @ pool_claps[best]).astype(np.float64)
            max_sim = sims if not selected else np.maximum(max_sim, sims)
            remaining[best] = False
            art_picks[art_codes[best]] += 1
            if is_cross_arr[best]:
                cross_lang_used += 1
            selected.append(scored_candidates[best][0])

        if return_pool:
            return selected, filtered_candidates
        return selected


recommender = MusicRecommender()


def load() -> None:
    recommender.load()


def describe(audio_bytes: bytes, meta: dict | None = None) -> dict:
    return recommender.describe(audio_bytes, meta=meta)


def recommend(
    seed: dict,
    k: int = 20,
    cross_language_max: int = 6,
    diversity: float = 0.70,
    key: str | None = None,
    debug: bool = False,
    return_pool: bool = False,
):
    return recommender.recommend(
        seed,
        k=k,
        cross_language_max=cross_language_max,
        diversity=diversity,
        key=key,
        debug=debug,
        return_pool=return_pool,
    )
