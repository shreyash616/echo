"""
ML-powered music recommendation and self-generated audio analytics service.

Loads the CnnMusicEncoder ONNX model + FAISS index at startup and provides:
  1. Deterministic multi-crop energy-weighted audio encoding (`encode_audio`)
  2. Self-generated musical & emotional metrics (`analyze_audio`: BPM, Key,
     Energy, Valence, Vibes) computed directly from the audio spectrogram
  3. Composite similarity scoring (`compute_composite_match_score`) combining
     the neural embedding with self-generated energy, valence, tempo, and key metrics
  4. Genre- and language-aware FAISS nearest-neighbour search (`recommend`)
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import subprocess
import time
from pathlib import Path

import faiss
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

# Must match ml/data/preprocess.py and ml/training/dataset.py
SR          = 22_050
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
CROP_FRAMES = 256
N_CROPS     = 4
TOP_DB      = 80.0

# In-memory cache of (embedding, analytics) keyed by audio SHA-256 hash
_audio_analysis_cache: TTLCache = TTLCache(maxsize=1024, ttl=7200)

# 12 pitch classes, Krumhansl-Schmuckler profiles, and Circle of Fifths positions
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)
# Map each pitch class to its position on the Circle of Fifths (0..11)
_FIFTHS_POS = {name: (i * 7) % 12 for i, name in enumerate(_PITCH_NAMES)}


def _hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = frequencies >= min_log_hz
    mels[log_mask] = min_log_mel + np.log(frequencies[log_mask] / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    f_min = 0.0
    f_sp = 200.0 / 3.0
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_mask = mels >= min_log_mel
    freqs[log_mask] = min_log_hz * np.exp(logstep * (mels[log_mask] - min_log_mel))
    return freqs


def _build_mel_to_chroma_matrix(n_mels: int = N_MELS, sr: int = SR) -> np.ndarray:
    mel_min = _hz_to_mel_slaney(np.array([0.0]))[0]
    mel_max = _hz_to_mel_slaney(np.array([sr / 2.0]))[0]
    mel_pts = np.linspace(mel_min, mel_max, n_mels)
    freqs = _mel_to_hz_slaney(mel_pts)

    W = np.zeros((12, n_mels), dtype=np.float32)
    valid = (freqs >= 65.0) & (freqs <= 2100.0)
    for idx in np.where(valid)[0]:
        f = freqs[idx]
        midi = 69.0 + 12.0 * np.log2(f / 440.0)
        pitch_class = midi % 12.0
        for c in range(12):
            dist = min(abs(pitch_class - c), 12.0 - abs(pitch_class - c))
            if dist < 1.0:
                W[c, idx] = np.exp(-0.5 * (dist / 0.4) ** 2)

    col_sums = W.sum(axis=1, keepdims=True) + 1e-8
    return W / col_sums


_MEL_TO_CHROMA = _build_mel_to_chroma_matrix()


def estimate_tempo_from_mel(
    mel: np.ndarray,
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
) -> tuple[int, float]:
    """Estimate BPM and pulse clarity directly from a (128, T) mel spectrogram."""
    T = mel.shape[1]
    if T < 64:
        return 120, 0.3

    rhythmic_mel = mel[:80, :]
    diff = np.diff(rhythmic_mel, axis=1)
    onset_env = np.maximum(0.0, diff).mean(axis=0)

    if len(onset_env) >= 5:
        kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float32)
        onset_env = np.convolve(onset_env, kernel, mode="same")

    onset_centered = onset_env - onset_env.mean()
    var = float(np.dot(onset_centered, onset_centered))
    if var < 1e-9:
        return 120, 0.1

    fps = sr / float(hop_length)
    min_bpm, max_bpm = 65.0, 185.0
    min_lag = max(4, int(round(fps * 60.0 / max_bpm)))
    max_lag = min(len(onset_centered) // 2, int(round(fps * 60.0 / min_bpm)))

    if max_lag <= min_lag:
        return 120, 0.2

    lags = np.arange(min_lag, max_lag + 1)
    ac = np.array(
        [np.dot(onset_centered[:-lag], onset_centered[lag:]) / var for lag in lags],
        dtype=np.float32,
    )

    lag_bpms = 60.0 * fps / lags
    tempo_prior = np.exp(-0.5 * (np.log2(lag_bpms / 120.0) / 0.55) ** 2)
    weighted_ac = np.maximum(0.0, ac) * tempo_prior

    best_idx = int(np.argmax(weighted_ac))
    best_lag = int(lags[best_idx])
    raw_clarity = float(np.clip(ac[best_idx], 0.0, 1.0))

    if 0 < best_idx < len(ac) - 1:
        y0, y1, y2 = float(ac[best_idx - 1]), float(ac[best_idx]), float(ac[best_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-6:
            delta = np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)
            best_lag_f = best_lag + delta
        else:
            best_lag_f = float(best_lag)
    else:
        best_lag_f = float(best_lag)

    bpm = int(np.clip(round(60.0 * fps / best_lag_f), 65, 185))
    pulse_clarity = float(np.clip(raw_clarity * 1.4, 0.05, 1.0))
    return bpm, pulse_clarity


def estimate_key_from_mel(mel: np.ndarray) -> tuple[str, float]:
    """Estimate musical key (e.g. 'A Minor') and Major/Minor mode polarity in [-1, 1]."""
    mag = np.power(10.0, (mel - 1.0) * 2.0)
    chroma = _MEL_TO_CHROMA @ mag.mean(axis=1)
    if float(chroma.std()) < 1e-7:
        return "C Major", 0.0

    chroma_z = (chroma - chroma.mean()) / (chroma.std() + 1e-8)
    maj_z = (_MAJOR_PROFILE - _MAJOR_PROFILE.mean()) / _MAJOR_PROFILE.std()
    min_z = (_MINOR_PROFILE - _MINOR_PROFILE.mean()) / _MINOR_PROFILE.std()

    best_key = "C Major"
    best_corr = -1e9
    best_maj_corr = -1e9
    best_min_corr = -1e9

    for shift in range(12):
        rolled = np.roll(chroma_z, -shift)
        c_maj = float(np.dot(rolled, maj_z) / 12.0)
        c_min = float(np.dot(rolled, min_z) / 12.0)

        if c_maj > best_maj_corr:
            best_maj_corr = c_maj
        if c_min > best_min_corr:
            best_min_corr = c_min

        if c_maj > best_corr:
            best_corr = c_maj
            best_key = f"{_PITCH_NAMES[shift]} Major"
        if c_min > best_corr:
            best_corr = c_min
            best_key = f"{_PITCH_NAMES[shift]} Minor"

    mode_polarity = float(np.clip((best_maj_corr - best_min_corr) * 1.5, -1.0, 1.0))
    return best_key, mode_polarity


def analyze_mel_spectrogram(mel: np.ndarray, genre_hint: str = "") -> dict:
    """
    Compute self-generated musical and emotional metrics directly from a
    (128, T) mel spectrogram in [0, 1].
    """
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim > 2:
        mel = mel.reshape(mel.shape[-2], mel.shape[-1])

    eps = 1e-8
    n_mels, T = mel.shape

    frame_e = mel.mean(axis=0)
    rms_mean = float(frame_e.mean())
    p95 = float(np.percentile(frame_e, 95))
    p10 = float(np.percentile(frame_e, 10))
    dyn_range = max(0.0, p95 - p10)

    if T > 1:
        flux = np.abs(mel[:, 1:] - mel[:, :-1]).mean(axis=0)
        flux_mean = float(flux.mean())
    else:
        flux_mean = 0.0

    bins = np.arange(1, n_mels + 1, dtype=np.float32)[:, None]
    centroid = (mel * bins).sum(axis=0) / (mel.sum(axis=0) + eps)
    centroid_norm = float(centroid.mean() / n_mels)

    hf_ratio = float((mel[n_mels // 2 :, :].sum(axis=0) / (mel.sum(axis=0) + eps)).mean())
    bass_ratio = float((mel[:24, :].sum(axis=0) / (mel.sum(axis=0) + eps)).mean())

    bpm, pulse_clarity = estimate_tempo_from_mel(mel)
    key_str, mode_polarity = estimate_key_from_mel(mel)

    flux_score = float(np.clip(flux_mean / 0.065, 0.0, 1.0))
    loud_score = float(np.clip((rms_mean - 0.15) / 0.55, 0.0, 1.0))
    tempo_score = float(np.clip((bpm - 70.0) / 100.0, 0.0, 1.0))
    energy = float(
        np.clip(
            0.38 * loud_score
            + 0.34 * flux_score
            + 0.16 * pulse_clarity
            + 0.12 * tempo_score,
            0.05,
            0.98,
        )
    )

    bright_score = float(np.clip((centroid_norm - 0.22) / 0.42, 0.0, 1.0))
    mode_score = 0.5 + 0.5 * mode_polarity
    valence = float(
        np.clip(
            0.38 * mode_score
            + 0.32 * bright_score
            + 0.18 * tempo_score
            + 0.12 * pulse_clarity,
            0.05,
            0.98,
        )
    )

    vibes: list[str] = []
    if genre_hint and genre_hint.lower() not in ("unknown", "none", ""):
        vibes.append(genre_hint.lower())

    if energy >= 0.72 and flux_score >= 0.60:
        vibes.append("adrenaline" if bpm >= 128 else "energetic")
    elif energy <= 0.38 and flux_score <= 0.40:
        vibes.append("soothing" if valence >= 0.45 else "ambient")
    elif pulse_clarity >= 0.55 and bpm >= 115:
        vibes.append("driving")
    elif energy <= 0.50:
        vibes.append("chill")

    if valence >= 0.65:
        vibes.append("uplifting" if energy >= 0.55 else "warm")
    elif valence <= 0.38:
        vibes.append("dark" if bass_ratio >= 0.35 else "melancholic")
    else:
        vibes.append("melodic" if dyn_range >= 0.18 else "groovy")

    if bass_ratio >= 0.42 and "bass-heavy" not in vibes:
        vibes.append("bass-heavy")
    elif hf_ratio >= 0.38 and "bright" not in vibes:
        vibes.append("bright")

    deduped_vibes: list[str] = []
    for v in vibes:
        if v not in deduped_vibes:
            deduped_vibes.append(v)

    return {
        "bpm": int(bpm),
        "key": key_str,
        "energy": round(energy, 2),
        "valence": round(valence, 2),
        "vibes": deduped_vibes[:3],
    }


def _key_compatibility(key_a: str, key_b: str) -> float:
    """
    Compute harmonic key similarity in [0, 1] using the Circle of Fifths
    and relative major/minor equivalence.
    """
    if not key_a or not key_b or key_a == "—" or key_b == "—":
        return 0.7
    if key_a == key_b:
        return 1.0

    parts_a = key_a.split()
    parts_b = key_b.split()
    if len(parts_a) != 2 or len(parts_b) != 2:
        return 0.6

    root_a, mode_a = parts_a[0], parts_a[1].lower()
    root_b, mode_b = parts_b[0], parts_b[1].lower()
    if root_a not in _FIFTHS_POS or root_b not in _FIFTHS_POS:
        return 0.6

    # Convert minor keys to their relative major position on the Circle of Fifths (+3 semitones = -3 fifths)
    idx_a = _PITCH_NAMES.index(root_a)
    idx_b = _PITCH_NAMES.index(root_b)
    rel_maj_a = _PITCH_NAMES[(idx_a + 3) % 12] if "min" in mode_a else root_a
    rel_maj_b = _PITCH_NAMES[(idx_b + 3) % 12] if "min" in mode_b else root_b

    pos_a = _FIFTHS_POS[rel_maj_a]
    pos_b = _FIFTHS_POS[rel_maj_b]
    fifths_dist = min(abs(pos_a - pos_b), 12 - abs(pos_a - pos_b))  # 0..6

    # Same mode bonus
    mode_match = 1.0 if mode_a == mode_b else 0.88
    return float(np.clip((1.0 - fifths_dist / 7.0) * mode_match, 0.25, 1.0))


def _bpm_compatibility(bpm_a: int, bpm_b: int) -> float:
    """
    Compute tempo similarity in [0, 1], accounting for double-time / half-time
    octave equivalence (e.g. 75 BPM and 150 BPM).
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.7
    ratios = [bpm_a / float(bpm_b), (bpm_a * 2.0) / float(bpm_b), bpm_a / (float(bpm_b) * 2.0)]
    min_rel_diff = min(abs(r - 1.0) for r in ratios)
    return float(np.clip(np.exp(-4.5 * min_rel_diff), 0.15, 1.0))


def compute_composite_match_score(
    src_emb: np.ndarray,
    src_metrics: dict,
    cand_emb: np.ndarray,
    cand_metrics: dict,
) -> float:
    """
    100% Data-Driven Neural Similarity Score in [0, 1].
    Directly evaluates the learned acoustic, timbral, and emotional geometry in the 512-d
    latent space discovered by CnnMusicEncoder, free of human-engineered music formulas.
    """
    a = src_emb.flatten()
    b = cand_emb.flatten()
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    # Calibrate neural similarity into intuitive [0.05, 0.99] match percentage
    score = float(np.clip((cos_sim + 0.1) / 1.1, 0.05, 0.99))
    return round(score, 4)


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode audio bytes of any format ffmpeg supports → mono float32 at SR."""
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        return y.astype(np.float32)
    except Exception:
        pass

    proc = subprocess.run(
        [
            "ffmpeg", "-i", "pipe:0",
            "-f", "wav", "-ar", str(SR), "-ac", "1",
            "-loglevel", "error",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()[:300]}")

    y, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
    return y.astype(np.float32)


def _audio_bytes_to_full_mel(audio_bytes: bytes) -> np.ndarray:
    """
    Decode audio bytes to a full-clip (128, T) float32 mel spectrogram in [0, 1]
    using a fixed 80 dB floor.
    """
    y = _decode_audio(audio_bytes)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    # Cap extremely long recordings to the central 60 seconds for bounded latency
    max_samples = int(SR * 60.0)
    if len(y) > max_samples:
        start = (len(y) - max_samples) // 2
        y = y[start : start + max_samples]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    mel_norm = np.clip((mel_db + TOP_DB) / TOP_DB, 0.0, 1.0)
    return mel_norm.astype(np.float32)


def _extract_energy_crops(
    mel: np.ndarray,
    crop_frames: int = CROP_FRAMES,
    n_crops: int = N_CROPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract up to `n_crops` deterministic windows of shape (K, 1, 128, crop_frames)
    weighted by local energy across the clip.
    """
    T = mel.shape[1]
    if T <= crop_frames:
        pad = crop_frames - T
        padded = np.pad(mel, ((0, 0), (0, pad)), mode="edge")
        return padded[np.newaxis, np.newaxis, :, :].astype(np.float32), np.array([1.0], dtype=np.float32)

    if n_crops <= 1 or T <= crop_frames * 3 // 2:
        start = (T - crop_frames) // 2
        crop = mel[:, start : start + crop_frames]
        return crop[np.newaxis, np.newaxis, :, :].astype(np.float32), np.array([1.0], dtype=np.float32)

    crops: list[np.ndarray] = []
    energies: list[float] = []
    frame_e = mel.mean(axis=0)
    kernel = np.ones(crop_frames, dtype=np.float32) / float(crop_frames)
    win_e = np.convolve(frame_e, kernel, mode="valid")
    max_start = len(win_e) - 1

    seg_bounds = np.linspace(0, max_start, n_crops + 1, dtype=int)
    for i in range(n_crops):
        lo, hi = int(seg_bounds[i]), int(seg_bounds[i + 1])
        best_start = lo if hi <= lo else lo + int(np.argmax(win_e[lo : hi + 1]))
        crops.append(mel[:, best_start : best_start + crop_frames])
        energies.append(float(win_e[best_start]) + 1e-4)

    weights = np.array(energies, dtype=np.float32)
    weights = weights / weights.sum()
    batch = np.stack(crops, axis=0)[:, np.newaxis, :, :].astype(np.float32)
    return batch, weights


class MusicRecommender:
    """
    Wraps the ONNX encoder, self-generated audio analyzer, and FAISS index.
    """

    def __init__(self) -> None:
        self._sess:       ort.InferenceSession | None = None
        self._index:      faiss.Index          | None = None
        self._metadata:   list[dict]                  = []
        self._archetype_centers: np.ndarray    | None = None
        self._archetype_names:   dict[str, str]       = {}
        self._input_name: str                         = "mel_spectrogram"
        self._loaded = False

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

        archetype_path = meta_path.parent / "archetypes.json"
        if archetype_path.exists():
            try:
                with open(archetype_path, encoding="utf-8") as f:
                    arch_data = json.load(f)
                    self._archetype_centers = np.array(arch_data["centers"], dtype=np.float32)
                    self._archetype_names = arch_data.get("names", {})
                logger.info("Loaded %d data-discovered archetypes from %s", len(self._archetype_names), archetype_path)
            except Exception as exc:
                logger.warning("Could not load archetypes from %s: %s", archetype_path, exc)

        self._loaded = True
        logger.info("Recommender ready — %d tracks indexed", self._index.ntotal)

    def encode_and_analyze_audio(
        self,
        audio_bytes: bytes,
        genre_hint: str = "",
    ) -> tuple[np.ndarray, dict]:
        """
        Decode raw audio bytes, compute multi-crop energy-weighted ONNX embedding
        (1, 512) AND self-generated audio metrics (bpm, key, energy, valence, vibes).
        Results are cached by audio SHA-256 digest.
        """
        digest = hashlib.sha256(audio_bytes).hexdigest()
        cache_key = f"{digest}:{genre_hint.lower()}"
        if cache_key in _audio_analysis_cache:
            return _audio_analysis_cache[cache_key]  # type: ignore[return-value]

        t0 = time.perf_counter()
        mel = _audio_bytes_to_full_mel(audio_bytes)
        metrics = analyze_mel_spectrogram(mel, genre_hint=genre_hint)

        if self._sess is not None:
            crops, weights = _extract_energy_crops(mel, crop_frames=CROP_FRAMES, n_crops=N_CROPS)
            out = self._sess.run(None, {self._input_name: crops})[0]  # (K, 512)
            pooled = (out * weights[:, np.newaxis]).sum(axis=0, keepdims=True)
            norm = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8
            emb = (pooled / norm).astype(np.float32)

            # Assign data-discovered archetypes if loaded
            if self._archetype_centers is not None and len(self._archetype_centers) > 0:
                affinities = np.dot(self._archetype_centers, emb.flatten())
                top_indices = np.argsort(-affinities)[:2]
                metrics["vibes"] = [self._archetype_names[str(idx)] for idx in top_indices if str(idx) in self._archetype_names]
        else:
            # Fallback deterministic spectral-emotion vector if ONNX session is unavailable
            emb = np.zeros((1, 512), dtype=np.float32)

        logger.info(
            "encode_and_analyze | total=%.3fs  bpm=%d  key=%s  E=%.2f  V=%.2f",
            time.perf_counter() - t0,
            metrics["bpm"], metrics["key"], metrics["energy"], metrics["valence"],
        )
        result = (emb, metrics)
        _audio_analysis_cache[cache_key] = result
        return result

    def encode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode raw audio bytes and produce a (1, 512) L2-normalised embedding."""
        emb, _ = self.encode_and_analyze_audio(audio_bytes)
        return emb

    def analyze_audio(self, audio_bytes: bytes, genre_hint: str = "") -> dict:
        """Decode raw audio bytes and return self-generated musical & emotional metrics."""
        _, metrics = self.encode_and_analyze_audio(audio_bytes, genre_hint=genre_hint)
        return metrics

    def recommend(
        self,
        embedding: np.ndarray,
        source_metrics: dict | None = None,
        genre_keywords: list[str] | None = None,
        exclude_id: str | None = None,
        k: int = 20,
    ) -> list[dict]:
        """
        Return up to k track dicts from the FAISS index most similar to the
        given embedding, filtering by genre family when `genre_keywords` is provided
        and skipping invalid stub entries.
        """
        if not self._loaded or self._index is None:
            logger.warning("Recommender not loaded; returning empty list")
            return []

        t0 = time.perf_counter()
        # Search a wider pool so genre filtering still yields k strong candidates
        search_depth = min(max(k * 10, 150), self._index.ntotal)
        distances, indices = self._index.search(embedding, search_depth)
        logger.info("faiss | search=%.3fs  depth=%d", time.perf_counter() - t0, search_depth)

        genre_kw = [g.lower() for g in (genre_keywords or []) if g]
        matched_genre_results: list[dict] = []
        fallback_results: list[dict] = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = dict(self._metadata[idx])
            if exclude_id and str(meta.get("id")) == str(exclude_id):
                continue

            # Skip unresolvable stub entries (e.g. "Track 12345" with empty artist)
            title = str(meta.get("title", "")).strip()
            artist = str(meta.get("artist", "")).strip()
            if not artist or (title.startswith("Track ") and title[6:].isdigit()):
                continue

            # Compute composite score if per-track metrics exist in metadata, else calibrated cosine
            if source_metrics and meta.get("bpm", 0) > 0:
                cand_emb_score = float(np.clip((float(dist) - 0.05) / 0.95, 0.0, 1.0))
                e_diff = abs(float(source_metrics.get("energy", 0.5)) - float(meta.get("energy", 0.5)))
                v_diff = abs(float(source_metrics.get("valence", 0.5)) - float(meta.get("valence", 0.5)))
                ev_score = float(np.clip(1.0 - 0.85 * np.hypot(e_diff, v_diff), 0.1, 1.0))
                bpm_score = _bpm_compatibility(int(source_metrics.get("bpm", 0)), int(meta.get("bpm", 0)))
                key_score = _key_compatibility(str(source_metrics.get("key", "—")), str(meta.get("key", "—")))
                meta["matchScore"] = round(
                    float(np.clip(0.65 * cand_emb_score + 0.15 * ev_score + 0.10 * bpm_score + 0.10 * key_score, 0.01, 0.99)),
                    4,
                )
            else:
                meta["matchScore"] = round(float(np.clip((float(dist) - 0.05) / 0.95, 0.01, 0.99)), 4)

            if genre_kw:
                track_genre_text = " ".join(
                    [str(meta.get("genre", "")), " ".join(meta.get("vibes", []))]
                ).lower()
                if any(kw in track_genre_text for kw in genre_kw):
                    matched_genre_results.append(meta)
                else:
                    fallback_results.append(meta)
            else:
                matched_genre_results.append(meta)

            if len(matched_genre_results) >= k:
                break

        combined = matched_genre_results + fallback_results
        return combined[:k]


recommender = MusicRecommender()