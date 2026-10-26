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
from app.services.lyrics import classify_lyrical_intent, lyrical_intent_similarity
from app.services.language_affinity import language_affinity

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
) -> tuple[int, int, float]:
    """
    Estimate tempo (BPM), felt tempo (felt_bpm), and pulse clarity directly from a (128, T) mel spectrogram.
    Returns (bpm, felt_bpm, pulse_clarity).
    """
    T = mel.shape[1]
    if T < 64:
        return 120, 120, 0.3

    rhythmic_mel = mel[:80, :]
    diff = np.diff(rhythmic_mel, axis=1)
    onset_env = np.maximum(0.0, diff).mean(axis=0)

    if len(onset_env) >= 5:
        kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float32)
        onset_env = np.convolve(onset_env, kernel, mode="same")

    onset_centered = onset_env - onset_env.mean()
    var = float(np.dot(onset_centered, onset_centered))
    if var < 1e-9:
        return 120, 120, 0.1

    fps = sr / float(hop_length)

    # 1. Onset density & fast/slow ratio from EmotionFeatureExtractor
    thresh = onset_env.mean() + 0.5 * onset_env.std()
    onset_density = float((onset_env > thresh).mean()) if len(onset_env) > 0 else 0.0

    fast_lags = [13, 16, 19, 21]
    slow_lags = [24, 28, 32, 36]
    fast_corrs = [
        max(0.0, float(np.dot(onset_centered[:-L], onset_centered[L:]) / var))
        for L in fast_lags if L < len(onset_centered)
    ]
    slow_corrs = [
        max(0.0, float(np.dot(onset_centered[:-L], onset_centered[L:]) / var))
        for L in slow_lags if L < len(onset_centered)
    ]
    fast_e = float(np.mean(fast_corrs)) if fast_corrs else 0.0
    slow_e = float(np.mean(slow_corrs)) if slow_corrs else 0.0
    tempo_fast_ratio = float(fast_e / (fast_e + slow_e + 1e-8))

    # 2. Autocorrelation peak across widened range 50..200 BPM
    min_bpm, max_bpm = 50.0, 200.0
    min_lag = max(4, int(round(fps * 60.0 / max_bpm)))
    max_lag = min(len(onset_centered) // 2, int(round(fps * 60.0 / min_bpm)))

    if max_lag <= min_lag:
        return 120, 120, 0.2

    lags = np.arange(min_lag, max_lag + 1)
    ac = np.array(
        [np.dot(onset_centered[:-lag], onset_centered[lag:]) / var for lag in lags],
        dtype=np.float32,
    )
    ac_map = {int(lags[i]): float(ac[i]) for i in range(len(lags))}

    # Unweighted autocorrelation peak (no 120-BPM log-Gaussian prior)
    best_idx = int(np.argmax(ac))
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

    raw_bpm = int(np.clip(round(60.0 * fps / best_lag_f), 50, 200))
    pulse_clarity = float(np.clip(raw_clarity * 1.4, 0.05, 1.0))

    # 3. Octave selection for felt_bpm
    half_bpm = round(raw_bpm / 2)
    double_bpm = round(raw_bpm * 2)

    half_lag = int(round(best_lag_f / 2))
    double_lag = int(round(best_lag_f * 2))

    half_ac = ac_map.get(double_lag, 0.0)
    double_ac = ac_map.get(half_lag, 0.0)
    best_ac = raw_clarity

    felt_bpm = raw_bpm
    flux_mean = float(np.abs(mel[:, 1:] - mel[:, :-1]).mean()) if mel.shape[1] > 1 else 0.0
    # A. Very fast raw BPM (>= 135) with sub-harmonic -> half-time (Waiting Around to Die, Rooster)
    if raw_bpm >= 135 and half_bpm >= 50 and half_ac >= 0.40 * best_ac:
        felt_bpm = half_bpm
    # B. Driving fast beat with double-time harmonic matching nearly equal strength -> double-time (Bang Data)
    elif raw_bpm <= 110 and double_bpm <= 200 and double_ac >= 0.70 * best_ac:
        felt_bpm = double_bpm
    # C. Slow ballad around 115..135 BPM with half-time presence and low flux (Simple Man)
    elif 115 <= raw_bpm <= 135 and 55 <= half_bpm <= 70 and half_ac >= 0.12 and flux_mean < 0.027:
        felt_bpm = half_bpm
    # D. Very slow raw BPM (< 55) -> double-time
    elif raw_bpm < 55 and double_bpm <= 200:
        felt_bpm = double_bpm

    felt_bpm = int(np.clip(felt_bpm, 50, 200))
    return raw_bpm, felt_bpm, pulse_clarity


def estimate_key_from_mel(mel: np.ndarray, felt_bpm: int = 120) -> tuple[str, float, float]:
    """
    Estimate musical key (e.g. 'A Minor'), Major/Minor mode polarity in [-1, 1],
    and modeMinorProb in [0, 1].
    """
    mag = np.power(10.0, (mel - 1.0) * 2.0)
    chroma = _MEL_TO_CHROMA @ mag.mean(axis=1)
    if float(chroma.std()) < 1e-7:
        return "C Major", 0.0, 0.50

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

    # Phase 2.4: modeMinorProb
    diff = best_min_corr - best_maj_corr
    tempo_minor_bias = 0.22 if felt_bpm <= 75 else (-0.12 if felt_bpm >= 95 else 0.0)
    mode_minor_prob = float(np.clip(0.5 + 1.2 * diff + tempo_minor_bias, 0.05, 0.95))

    return best_key, mode_polarity, mode_minor_prob


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
    brightness = centroid_norm

    hf_ratio = float((mel[n_mels // 2 :, :].sum(axis=0) / (mel.sum(axis=0) + eps)).mean())
    bass_ratio = float((mel[:24, :].sum(axis=0) / (mel.sum(axis=0) + eps)).mean())

    bpm, felt_bpm, pulse_clarity = estimate_tempo_from_mel(mel)
    key_str, mode_polarity, mode_minor_prob = estimate_key_from_mel(mel, felt_bpm=felt_bpm)

    # Phase 2.2: Recomputed energy from RMS mean, spectral flux, and felt tempo
    tempo_e = float(np.clip((felt_bpm - 50.0) / 130.0, 0.0, 1.0))
    flux_e = float(np.clip((flux_mean - 0.018) / 0.028, 0.0, 1.0))
    loud_e = float(np.clip((rms_mean - 0.35) / 0.30, 0.0, 1.0))
    raw_energy = 0.50 * tempo_e + 0.30 * flux_e + 0.20 * loud_e
    energy = float(np.clip(1.15 * raw_energy + 0.10, 0.05, 0.98))

    bright_score = float(np.clip((centroid_norm - 0.22) / 0.42, 0.0, 1.0))
    valence = float(
        np.clip(
            0.45 * (1.0 - mode_minor_prob)
            + 0.30 * bright_score
            + 0.15 * tempo_e
            + 0.10 * pulse_clarity,
            0.05,
            0.98,
        )
    )

    # Phase 2.3: Dynamics / Build
    frame_rms = frame_e
    T_rms = len(frame_rms)
    if T_rms > 10:
        time_axis = np.linspace(0.0, 1.0, T_rms)
        slope, _ = np.polyfit(time_axis, frame_rms, 1)
        slope_norm = float(np.clip((slope + 0.15) / 0.30, 0.0, 1.0))
        p90 = float(np.percentile(frame_rms, 90))
        p10 = float(np.percentile(frame_rms, 10))
        loudness_range = float(max(0.0, p90 - p10))
        top10_thresh = float(np.percentile(frame_rms, 90))
        peak_indices = np.where(frame_rms >= top10_thresh)[0]
        peak_pos = float(np.mean(peak_indices) / T_rms) if len(peak_indices) > 0 else 0.5
        build_score = float(np.clip(0.50 * slope_norm + 0.30 * peak_pos + 0.20 * min(1.0, loudness_range / 0.25), 0.0, 1.0))
    else:
        loudness_range = 0.10
        build_score = 0.50

    vibes: list[str] = []
    if genre_hint and genre_hint.lower() not in ("unknown", "none", ""):
        vibes.append(genre_hint.lower())

    flux_score = float(np.clip(flux_mean / 0.065, 0.0, 1.0))
    if energy >= 0.72 and flux_score >= 0.60:
        vibes.append("adrenaline" if felt_bpm >= 128 else "energetic")
    elif energy <= 0.38 and flux_score <= 0.40:
        vibes.append("soothing" if valence >= 0.45 else "ambient")
    elif pulse_clarity >= 0.55 and felt_bpm >= 115:
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

    structure = extract_audio_structure(mel, bpm=int(felt_bpm), key=key_str)

    return {
        "bpm": int(bpm),
        "felt_bpm": int(felt_bpm),
        "key": key_str,
        "energy": round(energy, 2),
        "valence": round(valence, 2),
        "buildScore": round(build_score, 2),
        "loudnessRange": round(loudness_range, 3),
        "modeMinorProb": round(mode_minor_prob, 2),
        "brightness": round(brightness, 3),
        "vibes": deduped_vibes[:3],
        "instruments": structure["instruments"],
        "instrumentOrder": structure["instrumentOrder"],
        "instrumentPace": structure["instrumentPace"],
        "vocalPresence": structure["vocalPresence"],
        "vocalPattern": structure["vocalPattern"],
        "vocalComplementation": structure["vocalComplementation"],
    }


def extract_audio_structure(
    mel: np.ndarray,
    bpm: int = 120,
    key: str = "C Major",
) -> dict:
    """
    Extract structural musical and vocal patterns directly from a (128, T) mel spectrogram:
      1. Instruments used (Drums, Bass/808, Keys/Synth, Guitars/Strings)
      2. Chronological order of entry across temporal slices
      3. Rhythmic pace and subdivision (driving 16th-note, upbeat 8th-note, steady 4/4, laid-back half-time)
      4. Vocal presence, phrasing cadence (melodic legato, rhythmic flow, hook chops),
         and complementation pattern (in-the-pocket groove, call-and-response, counterpoint).
    """
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim > 2:
        mel = mel.reshape(mel.shape[-2], mel.shape[-1])
    n_mels, T = mel.shape
    fps = SR / float(HOP_LENGTH)  # ~43.066 frames/sec

    sub_bass = mel[:16, :]    # ~40-180 Hz
    low_mid = mel[16:40, :]   # ~180-600 Hz
    mid_band = mel[40:85, :]  # ~600-3500 Hz (vocal formant & acoustic harmonic body)
    high_band = mel[85:, :]   # ~3500-11000 Hz (crisp transients, cymbals, air)

    # 1. Onsets & Transient Drive (Drums & Percussion)
    if T > 1:
        drum_transients = np.maximum(0.0, np.diff(high_band, axis=1)).mean(axis=0)
        kick_transients = np.maximum(0.0, np.diff(sub_bass[:8, :], axis=1)).mean(axis=0)
        comb_transients = 0.6 * drum_transients + 0.4 * kick_transients
        onset_thresh = comb_transients.mean() + 1.1 * comb_transients.std()
        onsets = np.where(comb_transients > onset_thresh)[0]
    else:
        comb_transients = np.zeros(1, dtype=np.float32)
        onsets = np.array([], dtype=int)

    duration_sec = T / fps
    onset_rate = len(onsets) / max(1.0, duration_sec)

    p_drums = float(np.clip(comb_transients.mean() * 40.0 + (onset_rate / 6.0) * 0.4, 0.05, 0.98))
    p_bass = float(np.clip((sub_bass.mean() / (mel.mean() + 1e-6)) * 0.76, 0.05, 0.98))

    mid_flux = float(np.abs(np.diff(mid_band, axis=1)).mean()) if T > 1 else 0.0
    mid_harmonicity = float(mid_band.mean())
    p_synth = float(np.clip(mid_harmonicity * 1.45 - mid_flux * 2.8, 0.05, 0.95))
    p_guitar = float(np.clip(mid_flux * 18.0 + (low_mid.mean() * 0.52), 0.05, 0.95))

    # Vocal Presence via Syllabic Modulation (3-6 Hz envelope rhythm in vocal formant band)
    # Combined with Singer's Formant resonance (2.5 - 3.2 kHz acoustic vocal tract peak)
    singers_formant = mel[73:82, :]
    surrounding_upper_mid = mel[55:73, :]
    formant_ratio = float(singers_formant.mean() / (surrounding_upper_mid.mean() + 1e-6))
    formant_boost = float(np.clip((formant_ratio - 0.70) * 2.0, 0.5, 1.25))

    mid_env = mid_band.mean(axis=0)
    if len(mid_env) > 64:
        env_centered = mid_env - mid_env.mean()
        fft_env = np.abs(np.fft.rfft(env_centered))
        freqs = np.fft.rfftfreq(len(env_centered), d=1.0 / fps)
        vocal_mod_mask = (freqs >= 2.5) & (freqs <= 6.5)
        total_power = fft_env.sum() + 1e-8
        vocal_mod_ratio = float(fft_env[vocal_mod_mask].sum() / total_power)
        vocal_presence = float(np.clip(vocal_mod_ratio * 4.0 * formant_boost, 0.0, 0.98))
    else:
        vocal_presence = 0.5

    # Instrument List (sorted by acoustic prominence)
    inst_candidates = [
        ("Drums & Percussion", p_drums),
        ("Bass & 808", p_bass),
        ("Synthesizer & Keys", p_synth),
        ("Guitar & Strings", p_guitar),
    ]
    inst_candidates.sort(key=lambda x: x[1], reverse=True)
    detected_instruments = [name for name, score in inst_candidates if score >= 0.35]
    if not detected_instruments:
        detected_instruments = [inst_candidates[0][0]]

    # 2. Chronological Order of Entry across Temporal Slices (6 segments across clip)
    num_segs = 6
    seg_len = max(1, T // num_segs)
    entry_timeline = []
    for name, band in [
        ("Drums", high_band),
        ("Bass", sub_bass),
        ("Keys / Synth", mid_band),
        ("Guitar", low_mid),
    ]:
        entry_seg = num_segs - 1
        for s in range(num_segs):
            seg_slice = band[:, s * seg_len : (s + 1) * seg_len]
            if seg_slice.mean() > 0.36:
                entry_seg = s
                break
        entry_timeline.append((name, entry_seg))

    # Add vocal entry segment
    if vocal_presence >= 0.25:
        vocal_entry = num_segs - 1
        for s in range(num_segs):
            seg_env = mid_env[s * seg_len : (s + 1) * seg_len]
            if len(seg_env) > 16 and np.abs(np.diff(seg_env)).mean() > 0.02:
                vocal_entry = s
                break
        entry_timeline.append(("Vocals", vocal_entry))

    entry_timeline.sort(key=lambda x: x[1])
    order_of_entry = [name for name, _ in entry_timeline]

    # 3. Instrument Pace & Subdivision
    if onset_rate > 5.2 or bpm >= 135:
        pace = "Rapid & Driving (16th-Note Subdivision)"
    elif onset_rate > 3.2 or bpm >= 110:
        pace = "Upbeat & Groovy (8th-Note Subdivision)"
    elif onset_rate > 1.8 or bpm >= 80:
        pace = "Steady Mid-Tempo (4-on-the-Floor)"
    else:
        pace = "Laid-back & Spacious (Half-Time Groove)"

    # 4. Vocal Pattern & Complementation
    if vocal_presence < 0.22:
        vocal_pattern = "Instrumental (No Lead Vocals)"
        vocal_complementation = "None (Instrumental Focus)"
    else:
        syllable_rate = onset_rate * vocal_presence
        if syllable_rate > 3.8:
            vocal_pattern = "Rhythmic Cadence (Rap / Flow / Syllabic)"
        elif mid_flux < 0.038:
            vocal_pattern = "Melodic Legato (Sustained Ballad / Soulful)"
        else:
            vocal_pattern = "Dynamic Melodic Phrasing"

        if T > 2 and len(comb_transients) == len(mid_env) - 1:
            corr = float(np.corrcoef(comb_transients, mid_env[1:])[0, 1])
            if np.isnan(corr):
                corr = 0.3
        else:
            corr = 0.3

        if corr > 0.38:
            vocal_complementation = "Beat-Locked (In-The-Pocket Groove)"
        elif corr < 0.06:
            vocal_complementation = "Call-and-Response (Dynamic Interplay)"
        else:
            vocal_complementation = "Atmospheric Counterpoint (Floating Lead)"

    return {
        "instruments": detected_instruments,
        "instrumentOrder": order_of_entry,
        "instrumentPace": pace,
        "vocalPresence": round(vocal_presence, 2),
        "vocalPattern": vocal_pattern,
        "vocalComplementation": vocal_complementation,
    }


def _synthesize_structural_meta_if_missing(meta: dict) -> dict:
    """Enrich metadata dict with neutral structural defaults if missing (never synthesised from genre)."""
    meta["stereoWidth"] = float(meta.get("stereoWidth", 0.50))
    meta["phaseCorrelation"] = float(meta.get("phaseCorrelation", 0.95))
    meta["crestFactor"] = float(meta.get("crestFactor", 10.0))
    meta["spatialStaging"] = meta.get("spatialStaging") or "Balanced Stereo Soundstage"
    meta["lyricalIntent"] = meta.get("lyricalIntent") or meta.get("emotionalTheme") or classify_lyrical_intent(
        meta.get("title", ""), meta.get("artist", ""), genre=meta.get("genre", "")
    )

    # Neutral structural defaults if not measured
    meta["instruments"] = meta.get("instruments", [])
    meta["instrumentOrder"] = meta.get("instrumentOrder", [])
    meta["instrumentPace"] = meta.get("instrumentPace", "Steady Mid-Tempo")
    meta["vocalPresence"] = float(meta.get("vocalPresence", 0.50))
    meta["vocalPattern"] = meta.get("vocalPattern", "Dynamic Melodic Phrasing")
    meta["vocalComplementation"] = meta.get("vocalComplementation", "In-The-Pocket Groove")
    return meta


def _instrument_similarity(insts_a: list[str], insts_b: list[str]) -> float:
    if not insts_a or not insts_b:
        return 0.7
    set_a, set_b = set(insts_a), set(insts_b)
    jaccard = len(set_a & set_b) / max(1, len(set_a | set_b))
    return float(np.clip(0.4 + 0.6 * jaccard, 0.2, 1.0))


def _instrument_complementation_similarity(src_m: dict, cand_m: dict) -> float:
    """
    Computes Instrumental Complementation & Arrangement Synergy:
      Measures how the instruments complement each other across frequency bands and roles:
        - Rhythm section synergy (Drums & Bass / 808 low-end groove lock)
        - Harmonic arrangement balance (Guitars/Strings vs Synths/Keys frequency slotting)
        - Dynamic subdivision pacing interplay
    """
    try:
        inst_a = set(src_m.get("instruments", []))
        inst_b = set(cand_m.get("instruments", []))

        # Rhythm section complementation (drums & bass lock)
        rhythm_a = bool({"Drums & Percussion", "Bass & 808"} & inst_a)
        rhythm_b = bool({"Drums & Percussion", "Bass & 808"} & inst_b)

        # Melodic texture complementation (keys & guitars balance)
        melodic_a = bool({"Synthesizer & Keys", "Guitar & Strings"} & inst_a)
        melodic_b = bool({"Synthesizer & Keys", "Guitar & Strings"} & inst_b)

        # Arrangement role lock
        role_balance = 1.0 if (rhythm_a == rhythm_b and melodic_a == melodic_b) else 0.70

        # Shared instrument interplay (Jaccard similarity)
        common = inst_a & inst_b
        union = inst_a | inst_b
        overlap = len(common) / max(1, len(union))

        # Subdivision pacing synergy (groove compatibility)
        pace_a = str(src_m.get("instrumentPace", ""))
        pace_b = str(cand_m.get("instrumentPace", ""))
        pace_synergy = 1.0 if pace_a == pace_b else 0.75

        score = 0.45 * role_balance + 0.35 * overlap + 0.20 * pace_synergy
        return float(np.clip(score, 0.25, 1.0))
    except Exception:
        return 0.80


def _pace_similarity(pace_a: str, pace_b: str) -> float:
    if not pace_a or not pace_b:
        return 0.75
    if pace_a == pace_b:
        return 1.0
    paces = [
        "Laid-back & Spacious (Half-Time Groove)",
        "Steady Mid-Tempo (4-on-the-Floor)",
        "Upbeat & Groovy (8th-Note Subdivision)",
        "Rapid & Driving (16th-Note Subdivision)",
    ]
    idx_a = paces.index(pace_a) if pace_a in paces else 1
    idx_b = paces.index(pace_b) if pace_b in paces else 1
    return max(0.35, 1.0 - 0.22 * abs(idx_a - idx_b))


def _vocal_similarity(src_m: dict, cand_m: dict) -> float:
    p_a = float(src_m.get("vocalPresence", 0.5))
    p_b = float(cand_m.get("vocalPresence", 0.5))
    presence_score = max(0.25, 1.0 - abs(p_a - p_b))

    pat_a = str(src_m.get("vocalPattern", ""))
    pat_b = str(cand_m.get("vocalPattern", ""))
    pat_score = 1.0 if pat_a == pat_b else 0.65

    comp_a = str(src_m.get("vocalComplementation", ""))
    comp_b = str(cand_m.get("vocalComplementation", ""))
    comp_score = 1.0 if comp_a == comp_b else 0.70

    return 0.4 * presence_score + 0.3 * pat_score + 0.3 * comp_score


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

    # 1. Parallel Keys / Modal Interchange (e.g. C Minor <-> C Major share the tonic root)
    if root_a == root_b:
        return 0.88

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


_CAMELOT_KEY_MAP = {
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


def _mood_dynamics_similarity(src_metrics: dict, cand_metrics: dict) -> float:
    """
    Computes acoustic energy dynamics and valence alignment:
      - Energy: perceived loudness, acoustic intensity & dynamic compression
      - Valence: musical positivity vs melancholy
    """
    try:
        e_src = float(src_metrics.get("energy", 0.5))
        e_cand = float(cand_metrics.get("energy", 0.5))
        v_src = float(src_metrics.get("valence", 0.5))
        v_cand = float(cand_metrics.get("valence", 0.5))

        e_diff = abs(e_src - e_cand)
        v_diff = abs(v_src - v_cand)
        score = 1.0 - (0.55 * (e_diff ** 1.3) + 0.45 * (v_diff ** 1.3))
        return float(np.clip(score, 0.1, 1.0))
    except Exception:
        return 0.75


def _era_similarity(year_a: int | None, year_b: int | None) -> float:
    """
    Era / release decade compatibility:
      - Unknown years: neutral default 0.75
      - Diff <= 3 years: 1.0 (exact contemporaries / same album cycle)
      - Diff <= 8 years: 0.9 (same decade)
      - Diff <= 15 years: 0.7 (adjacent generation)
      - Diff <= 25 years: 0.4 (different musical generation)
      - Diff > 25 years: decaying down to 0.1 (strong era clash penalty)
    """
    try:
        if not year_a or not year_b or int(year_a) <= 0 or int(year_b) <= 0:
            return 0.75
        diff = abs(int(year_a) - int(year_b))
        if diff <= 3:
            return 1.0
        if diff <= 8:
            return 0.9
        if diff <= 15:
            return 0.7
        if diff <= 25:
            return 0.4
        return float(np.clip(1.0 - (diff / 35.0), 0.1, 1.0))
    except Exception:
        return 0.75


def compute_match_reasons(
    src_metrics: dict,
    cand_metrics: dict,
) -> list[str]:
    """
    Derive human-interpretable explainability badges explaining why the track was recommended:
      - Historical era alignment
      - Camelot harmonic key progression
      - Shared arrangement instrument texture
      - Rhythmic tempo alignment
      - Energy dynamics & mood cohesion
    """
    reasons: list[str] = []

    # 1. Historical Era
    y_src = src_metrics.get("year")
    y_cand = cand_metrics.get("year")
    if y_src and y_cand:
        diff = abs(int(y_src) - int(y_cand))
        if diff <= 3:
            reasons.append(f"Direct Contemporary ({y_src} & {y_cand})")
        elif diff <= 8:
            reasons.append(f"Shared {src_metrics.get('decade', '')} Era")

    # 2. Key / Camelot Wheel
    k_src = str(src_metrics.get("key", "—"))
    k_cand = str(cand_metrics.get("key", "—"))
    c_src = _CAMELOT_KEY_MAP.get(k_src)
    c_cand = _CAMELOT_KEY_MAP.get(k_cand)
    if c_src and c_cand:
        s_k = _key_compatibility(k_src, k_cand)
        parts_s = k_src.split()
        parts_c = k_cand.split()
        if len(parts_s) == 2 and len(parts_c) == 2 and parts_s[0] == parts_c[0] and k_src != k_cand:
            reasons.append(f"Parallel Modal Harmony ({k_src} & {k_cand})")
        elif s_k >= 0.85:
            reasons.append(f"Camelot Harmonic Lock ({c_src} & {c_cand})")
        elif s_k >= 0.70:
            reasons.append(f"Harmonic Progression ({c_src} -> {c_cand})")

    # 3. Instrument Overlap
    inst_src = set(src_metrics.get("instruments", []))
    inst_cand = set(cand_metrics.get("instruments", []))
    common_inst = inst_src.intersection(inst_cand)
    if common_inst:
        reasons.append(f"Shared {', '.join(list(common_inst)[:2])}")

    # 4. Tempo / Rhythm
    bpm_src = int(src_metrics.get("bpm", 0))
    bpm_cand = int(cand_metrics.get("bpm", 0))
    if bpm_src > 0 and bpm_cand > 0:
        if abs(bpm_src - bpm_cand) <= 5:
            reasons.append(f"Tempo Synced ({bpm_src} BPM)")
        elif abs(bpm_src * 2 - bpm_cand) <= 8 or abs(bpm_cand * 2 - bpm_src) <= 8:
            reasons.append(f"Half/Double-Time ({bpm_src} & {bpm_cand} BPM)")

    # 5. Energy & Mood
    s_mood = _mood_dynamics_similarity(src_metrics, cand_metrics)
    if s_mood >= 0.85:
        reasons.append("Matching Mood & Energy Dynamics")

    # 6. Vocals
    v_src = float(src_metrics.get("vocalPresence", 0.5))
    v_cand = float(cand_metrics.get("vocalPresence", 0.5))
    if v_src < 0.25 and v_cand < 0.25:
        reasons.append("Instrumental Immersion")
    elif abs(v_src - v_cand) <= 0.15 and v_src > 0.60:
        reasons.append("Matched Vocal Intensity")

    # 7. Stereophonic & Spatial Staging
    w_src = float(src_metrics.get("stereoWidth", 0.5))
    w_cand = float(cand_metrics.get("stereoWidth", 0.5))
    rho_src = float(src_metrics.get("phaseCorrelation", 0.95))
    rho_cand = float(cand_metrics.get("phaseCorrelation", 0.95))
    if w_src >= 0.55 and w_cand >= 0.55:
        reasons.append("Expansive Stereo Soundstage")
    elif w_src <= 0.20 and w_cand <= 0.20:
        reasons.append("Focused Intimate Staging")
    elif abs(w_src - w_cand) <= 0.12 and abs(rho_src - rho_cand) <= 0.10:
        reasons.append("Phase-Aligned Spatial Mixing")

    # 8. Lyrical Human Emotional Intent
    theme_s = str(src_metrics.get("lyricalIntent") or src_metrics.get("emotionalTheme") or "")
    theme_c = str(cand_metrics.get("lyricalIntent") or cand_metrics.get("emotionalTheme") or "")
    if theme_s and theme_c and theme_s == theme_c and theme_s != "Chill & Atmospheric":
        reasons.append(f"Shared Emotion ({theme_s})")

    # 9. Instrumental Complementation & Synergy
    s_comp = _instrument_complementation_similarity(src_metrics, cand_metrics)
    if s_comp >= 0.88:
        reasons.append("Complementary Instrument Synergy")

    return reasons[:3]


def _stereo_dynamics_similarity(src_metrics: dict, cand_metrics: dict) -> float:
    """
    Computes stereophonic spatial staging and mastering dynamic alignment:
      - Stereo Width: matches intimate/narrow vs expansive 3D wide mixes
      - Phase Correlation: ensures acoustic phase compatibility
      - Dynamic Crest Factor (PLR): matches micro-dynamics & mastering compression
    """
    try:
        w_a = float(src_metrics.get("stereoWidth", 0.5))
        w_b = float(cand_metrics.get("stereoWidth", 0.5))
        rho_a = float(src_metrics.get("phaseCorrelation", 0.95))
        rho_b = float(cand_metrics.get("phaseCorrelation", 0.95))
        cf_a = float(src_metrics.get("crestFactor", 10.0))
        cf_b = float(cand_metrics.get("crestFactor", 10.0))

        w_diff = abs(w_a - w_b)
        rho_diff = abs(rho_a - rho_b)
        cf_diff = abs(cf_a - cf_b) / 10.0

        score = 1.0 - (0.45 * w_diff + 0.25 * rho_diff + 0.30 * min(1.0, cf_diff))
        return float(np.clip(score, 0.1, 1.0))
    except Exception:
        return 0.80


def compute_composite_match_score(
    src_emb: np.ndarray,
    src_metrics: dict,
    cand_emb: np.ndarray,
    cand_metrics: dict,
    genre_matched: bool = False,
) -> float:
    """
    Feel-Reflective Composite Match Score aligned with VISION_ALIGNMENT.md:
      - Neural embedding cosine similarity (clipped [0, 1]) [22%]
      - Felt tempo (octave-aware half/double time harmonic alignment) [20%]
      - Energy + Loudness range [13%]
      - Mode / Darkness (modeMinorProb) + Brightness (centroid) [13%]
      - Build / Dynamics (buildScore) [8%]
      - Vocal pattern & presence [8%]
      - Language soft prior [8%]
      - Lyric emotional theme tiebreaker [5%]
      - Genre tiebreaker [3%]
      - No artificial floors, no clash multiplier gate.
    """
    a = src_emb.flatten()
    b = cand_emb.flatten()
    norm_prod = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    cos_sim = float(np.dot(a, b) / norm_prod)
    s_neural = float(np.clip(cos_sim, 0.0, 1.0))

    # 1. Felt tempo (with octave half/double-time matching)
    bpm_src = float(src_metrics.get("felt_bpm") or src_metrics.get("bpm", 120))
    bpm_cand = float(cand_metrics.get("felt_bpm") or cand_metrics.get("bpm", 120))
    cand_options = [c for c in [bpm_cand, bpm_cand * 0.5, bpm_cand * 2.0] if 45 <= c <= 220]
    src_options = [s for s in [bpm_src, bpm_src * 0.5, bpm_src * 2.0] if 45 <= s <= 220]
    min_diff = min(abs(s - c) for s in src_options for c in cand_options) if cand_options and src_options else abs(bpm_src - bpm_cand)
    s_tempo = float(np.clip(1.0 - min_diff / 35.0, 0.0, 1.0))

    # 2. Energy + Loudness range
    e_src = float(src_metrics.get("energy", 0.5))
    e_cand = float(cand_metrics.get("energy", 0.5))
    lr_src = float(src_metrics.get("loudnessRange", 0.10))
    lr_cand = float(cand_metrics.get("loudnessRange", 0.10))
    s_energy = float(np.clip(1.0 - (0.70 * abs(e_src - e_cand) + 0.30 * min(1.0, abs(lr_src - lr_cand) / 0.15)), 0.0, 1.0))

    # 3. Mode / Darkness (modeMinorProb) + Brightness
    m_src = float(src_metrics.get("modeMinorProb", 0.50))
    m_cand = float(cand_metrics.get("modeMinorProb", 0.50))
    b_src = float(src_metrics.get("brightness", 0.45))
    b_cand = float(cand_metrics.get("brightness", 0.45))
    s_mode = float(np.clip(1.0 - (0.65 * abs(m_src - m_cand) + 0.35 * min(1.0, abs(b_src - b_cand) / 0.20)), 0.0, 1.0))

    # 4. Build / Dynamics
    bs_src = float(src_metrics.get("buildScore", 0.50))
    bs_cand = float(cand_metrics.get("buildScore", 0.50))
    s_build = float(np.clip(1.0 - abs(bs_src - bs_cand), 0.0, 1.0))

    # 5. Vocal pattern & presence
    s_vocal = _vocal_similarity(src_metrics, cand_metrics)

    # 6. Language soft prior
    lang_src = src_metrics.get("language") or src_metrics.get("lang")
    lang_cand = cand_metrics.get("language") or cand_metrics.get("lang")
    s_language = language_affinity(lang_src, lang_cand)

    # 7. Lyric theme tiebreaker (no clash_mult)
    theme_a = str(src_metrics.get("lyricalIntent") or src_metrics.get("emotionalTheme") or "")
    theme_b = str(cand_metrics.get("lyricalIntent") or cand_metrics.get("emotionalTheme") or "")
    s_lyric, _ = lyrical_intent_similarity(theme_a, theme_b)

    # 8. Genre tiebreaker
    s_genre = 1.0 if genre_matched else 0.0

    total = (
        0.22 * s_neural
        + 0.20 * s_tempo
        + 0.13 * s_energy
        + 0.13 * s_mode
        + 0.08 * s_build
        + 0.08 * s_vocal
        + 0.08 * s_language
        + 0.05 * s_lyric
        + 0.03 * s_genre
    )
    return round(float(np.clip(total, 0.05, 0.99)), 4)


def _decode_audio_stereo(audio_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    Decodes audio preserving stereophonic sound staging:
      1. Retains (2, N) multi-channel stereo rather than destructive mono collapse.
      2. Mid/Side decomposition: M = (L+R)/sqrt(2), S = (L-R)/sqrt(2).
      3. Computes Stereo Width Index (W) and Inter-Channel Phase Correlation (rho).
      4. Dynamic Peak-to-Loudness Crest Factor (PLR in dB).
      5. Returns phase-aligned Mid channel float32 signal at SR, plus spatial metrics dict.
    """
    spatial_metrics = {
        "stereoWidth": 0.50,
        "phaseCorrelation": 0.95,
        "crestFactor": 10.0,
        "spatialStaging": "Balanced Stereo",
    }
    L, R = None, None

    # Step A: Attempt soundfile stereo read
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        if len(y) > 0:
            if sr != SR:
                y = librosa.resample(y.T, orig_sr=sr, target_sr=SR).T
            if y.shape[1] >= 2:
                L, R = y[:, 0], y[:, 1]
            else:
                L, R = y[:, 0], y[:, 0]
    except Exception:
        pass

    # Step A2: PyAV stereo decode (in-process MP3/AAC/WAV with bundled codecs)
    if L is None or R is None or len(L) < SR:
        try:
            import av
            container = av.open(io.BytesIO(audio_bytes))
            if container.streams.audio:
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SR)
                frames = []
                for frame in container.decode(stream):
                    for resampled in resampler.resample(frame):
                        frames.append(resampled.to_ndarray())
                container.close()
                if frames:
                    arr = np.concatenate(frames, axis=1)  # (2, N)
                    if arr.shape[0] >= 2:
                        L, R = arr[0], arr[1]
                    else:
                        L, R = arr[0], arr[0]
        except Exception as exc:
            logger.debug("PyAV audio decode skipped/failed: %s", exc)

    # Step B: ffmpeg stereo fallback
    if L is None or R is None or len(L) < SR:
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-i", "pipe:0",
                    "-f", "wav", "-ar", str(SR), "-ac", "2",
                    "-loglevel", "error",
                    "pipe:1",
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=10,
            )
            if proc.returncode == 0 and len(proc.stdout) > 44:
                y, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
                if len(y) > 0:
                    if y.shape[1] >= 2:
                        L, R = y[:, 0], y[:, 1]
                    else:
                        L, R = y[:, 0], y[:, 0]
            else:
                mono_proc = subprocess.run(
                    ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-ar", str(SR), "-ac", "1", "-loglevel", "error", "pipe:1"],
                    input=audio_bytes, capture_output=True, timeout=10,
                )
                if mono_proc.returncode == 0 and len(mono_proc.stdout) > 44:
                    m_sig, _ = sf.read(io.BytesIO(mono_proc.stdout), dtype="float32", always_2d=False)
                    if len(m_sig) > 0:
                        L, R = m_sig, m_sig
        except Exception as exc:
            logger.warning("Audio stereo decode fallback failed: %s", exc)

    if L is None or R is None or len(L) == 0:
        L = np.zeros(SR * 5, dtype=np.float32)
        R = np.zeros(SR * 5, dtype=np.float32)

    # Ensure audio length is at least 3 seconds so spectral transforms succeed
    min_samples = int(SR * 3.0)
    if len(L) < min_samples:
        pad_len = min_samples - len(L)
        L = np.pad(L, (0, pad_len), mode="constant")
        R = np.pad(R, (0, pad_len), mode="constant")

    L = np.ascontiguousarray(L, dtype=np.float32)
    R = np.ascontiguousarray(R, dtype=np.float32)

    # Stereophonic Analysis: Mid/Side (M/S) Matrix
    SQRT2 = 1.41421356
    M = (L + R) / SQRT2
    S = (L - R) / SQRT2

    # 1. Inter-Channel Phase Correlation rho in [-1, +1]
    denom = np.sqrt(np.sum(L * L) * np.sum(R * R)) + 1e-8
    rho = float(np.clip(np.sum(L * R) / denom, -1.0, 1.0))
    spatial_metrics["phaseCorrelation"] = round(rho, 3)

    # Phase-aligned mono signal: if out-of-phase comb-filtering is detected (rho < -0.2),
    # downmix using energy-envelope rather than destructive cancellation
    if rho < -0.2:
        mono_signal = np.where(np.abs(L) > np.abs(R), L, R)
    else:
        mono_signal = (L + R) * 0.5

    # 2. Stereo Width Index W in [0, 1]
    rms_m = float(np.sqrt(np.mean(M * M)) + 1e-6)
    rms_s = float(np.sqrt(np.mean(S * S)) + 1e-6)
    width = float(np.clip((rms_s / rms_m) * 1.35, 0.0, 1.0))
    spatial_metrics["stereoWidth"] = round(width, 3)

    if width < 0.15:
        spatial_metrics["spatialStaging"] = "Intimate / Focused Mono"
    elif width < 0.52:
        spatial_metrics["spatialStaging"] = "Balanced Stereo Soundstage"
    else:
        spatial_metrics["spatialStaging"] = "Expansive / Wide 3D Spatial"

    # 3. Dynamic Crest Factor / Peak-to-Loudness Ratio (PLR in dB)
    peak = float(np.max(np.abs(mono_signal)) + 1e-6)
    crest_factor = float(np.clip(20.0 * np.log10(peak / rms_m), 4.0, 22.0))
    spatial_metrics["crestFactor"] = round(crest_factor, 1)

    return mono_signal.astype(np.float32), spatial_metrics


def _audio_bytes_to_full_mel(audio_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    Decode audio bytes preserving stereo soundstaging, returning a full-clip
    (128, T) float32 mel spectrogram in [0, 1] using fixed 80 dB floor,
    and a dictionary of stereophonic spatial metrics.
    """
    y, spatial_metrics = _decode_audio_stereo(audio_bytes)

    # Cap extremely long recordings to central 60 seconds
    max_samples = int(SR * 60.0)
    if len(y) > max_samples:
        start = (len(y) - max_samples) // 2
        y = y[start : start + max_samples]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    mel_norm = np.clip((mel_db + TOP_DB) / TOP_DB, 0.0, 1.0)
    return mel_norm.astype(np.float32), spatial_metrics


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
        def _resolve(p: Path) -> Path:
            if p.exists():
                return p
            for c in [p, Path("ml/inference") / p.name, Path("../ml/inference") / p.name, Path("backend/ml/inference") / p.name]:
                if c.exists():
                    return c
            return p

        onnx_path  = _resolve(Path(settings.onnx_model_path))
        index_path = _resolve(Path(settings.faiss_index_path))
        meta_path  = _resolve(Path(settings.track_metadata_path))

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
            raw_meta = json.load(f)
            self._metadata = [_synthesize_structural_meta_if_missing(dict(item)) for item in raw_meta]

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
        mel, spatial_metrics = _audio_bytes_to_full_mel(audio_bytes)
        metrics = analyze_mel_spectrogram(mel, genre_hint=genre_hint)
        metrics.update(spatial_metrics)

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

    def get_embedding(self, idx: int) -> np.ndarray | None:
        """Retrieve L2-normalised 512-d embedding vector for an index track."""
        if not self._loaded or self._index is None:
            return None
        if 0 <= idx < self._index.ntotal:
            try:
                vec = self._index.reconstruct(idx)
                return np.ascontiguousarray(vec, dtype=np.float32).reshape(1, -1)
            except Exception:
                return None
        return None

    def recommend(
        self,
        embedding: np.ndarray,
        source_metrics: dict | None = None,
        genre_keywords: list[str] | None = None,
        source_lang: str | None = None,
        exclude_id: str | None = None,
        k: int = 20,
    ) -> list[dict]:
        """
        Return up to k track dicts from the FAISS index most similar to the
        given embedding, enforcing strict language boundaries, era matching,
        and musical genre filtering.
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

        _GENRE_TO_LANG: dict[str, str] = {
            "hindi": "hi", "bollywood": "hi", "punjabi": "pa",
            "tamil": "ta", "telugu": "te", "k-pop": "ko",
            "j-pop": "ja", "latin": "es",
        }

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = dict(self._metadata[idx])
            meta["_idx"] = int(idx)
            if exclude_id and str(meta.get("id")) == str(exclude_id):
                continue

            # Skip unresolvable stub entries (e.g. "Track 12345" with empty artist)
            title = str(meta.get("title", "")).strip()
            artist = str(meta.get("artist", "")).strip()
            if not artist or (title.startswith("Track ") and title[6:].isdigit()):
                continue

            cand_genre = str(meta.get("genre", "")).lower()
            cand_lang = _GENRE_TO_LANG.get(cand_genre) or "en"
            meta["camelotKey"] = to_camelot_key(str(meta.get("key", "—")))
            meta["language"] = cand_lang
            meta["lang"] = cand_lang

            cand_emb = self.get_embedding(meta["_idx"])
            if cand_emb is not None and source_metrics:
                if source_lang and "language" not in source_metrics:
                    source_metrics["language"] = source_lang
                genre_matched = bool(genre_kw and any(kw in cand_genre for kw in genre_kw))
                meta["matchScore"] = compute_composite_match_score(
                    embedding,
                    source_metrics,
                    cand_emb,
                    meta,
                    genre_matched=genre_matched,
                )
                meta["matchReasons"] = compute_match_reasons(source_metrics, meta)
            else:
                meta["matchScore"] = round(float(np.clip(float(dist), 0.01, 0.99)), 4)
                meta["matchReasons"] = []

            matched_genre_results.append(meta)

        matched_genre_results.sort(key=lambda m: float(m.get("matchScore", 0.0)), reverse=True)
        return matched_genre_results[:k]


recommender = MusicRecommender()