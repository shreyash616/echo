"""
ML-powered music recommendation service.

Loads the CnnMusicEncoder ONNX model + FAISS index at startup and serves
nearest-neighbour queries from raw audio bytes in <200 ms per request.

Pipeline:
  Audio bytes  →  librosa mel spectrogram  →  ONNX encoder  →  512-d embedding  →  FAISS search
"""
from __future__ import annotations

import io
import json
import logging
import random
from pathlib import Path

import faiss
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

from app.config import settings

logger = logging.getLogger(__name__)

# Must match ml/data/preprocess.py and ml/training/dataset.py
SR          = 22_050
DURATION    = 60.0   # 1-minute clip, skipping first 30s
SKIP_START  = 30.0   # skip intro/silence
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
CROP_FRAMES = 256


def _audio_bytes_to_mel(audio_bytes: bytes) -> np.ndarray:
    """
    Decode audio bytes (any format soundfile/librosa supports) to a
    (1, 128, CROP_FRAMES) float32 array ready for the ONNX encoder.
    """
    # Try soundfile first (fastest), fall back to librosa for mp3/m4a
    try:
        audio_buf = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_buf, dtype="float32", always_2d=False)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    except Exception:
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

    # Skip the first SKIP_START seconds, then take a random DURATION-second clip
    skip_samples = int(SR * SKIP_START)
    clip_samples = int(SR * DURATION)
    if len(y) > skip_samples + clip_samples:
        max_offset = len(y) - clip_samples
        start = random.randint(skip_samples, max_offset)
        y = y[start: start + clip_samples]
    elif len(y) > clip_samples:
        # Not long enough to skip 30s — just take the last DURATION seconds
        y = y[-clip_samples:]

    # Convert to mono float32
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)           # (128, T)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Centre-crop to CROP_FRAMES
    T = mel_db.shape[1]
    if T <= CROP_FRAMES:
        pad = CROP_FRAMES - T
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
    else:
        start  = (T - CROP_FRAMES) // 2
        mel_db = mel_db[:, start: start + CROP_FRAMES]

    # Shape: (1, 1, 128, CROP_FRAMES)
    return mel_db[np.newaxis, np.newaxis].astype(np.float32)


class MusicRecommender:
    """
    Wraps the ONNX encoder and FAISS index.
    Call load() once at startup; then use encode_audio() + recommend().
    """

    def __init__(self) -> None:
        self._sess:     ort.InferenceSession | None = None
        self._index:    faiss.Index          | None = None
        self._metadata: list[dict]                  = []
        self._input_name: str                       = "mel_spectrogram"
        self._loaded = False

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load ONNX model + FAISS index. Called once at server startup."""
        onnx_path  = Path(settings.onnx_model_path)
        index_path = Path(settings.faiss_index_path)
        meta_path  = Path(settings.track_metadata_path)

        if not onnx_path.exists():
            logger.warning(
                "ONNX model not found at %s — recommender disabled. "
                "Run the ML training pipeline first (see SETUP.md).",
                onnx_path,
            )
            return

        logger.info("Loading ONNX encoder from %s", onnx_path)
        self._sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name
        logger.info("ONNX encoder loaded.")

        if not index_path.exists():
            logger.warning("FAISS index not found at %s — recommender disabled", index_path)
            return

        logger.info("Loading FAISS index from %s", index_path)
        self._index = faiss.read_index(str(index_path))

        logger.info("Loading track metadata from %s", meta_path)
        with open(meta_path, encoding="utf-8") as f:
            self._metadata = json.load(f)

        self._loaded = True
        logger.info("Recommender ready — %d tracks indexed", self._index.ntotal)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Decode raw audio bytes and produce a (1, 512) L2-normalised embedding.
        """
        assert self._sess is not None, "Call load() first"
        mel = _audio_bytes_to_mel(audio_bytes)              # (1, 1, 128, T)
        out = self._sess.run(None, {self._input_name: mel})
        return out[0].astype(np.float32)                    # (1, 512)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(
        self,
        embedding: np.ndarray,
        exclude_id: str | None = None,
        k: int = 20,
    ) -> list[dict]:
        """Return up to k track dicts most similar to the given embedding."""
        if not self._loaded or self._index is None:
            logger.warning("Recommender not loaded; returning empty list")
            return []

        distances, indices = self._index.search(embedding, k + 5)

        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = dict(self._metadata[idx])
            if exclude_id and meta.get("id") == exclude_id:
                continue
            # cosine similarity ∈ [-1, 1] → map to [0, 1]
            meta["matchScore"] = float(np.clip((dist + 1) / 2, 0, 1))
            results.append(meta)
            if len(results) >= k:
                break

        return results


recommender = MusicRecommender()
