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

    structure = extract_audio_structure(mel, bpm=int(bpm), key=key_str)

    return {
        "bpm": int(bpm),
        "key": key_str,
        "energy": round(energy, 2),
        "valence": round(valence, 2),
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
    mid_env = mid_band.mean(axis=0)
    if len(mid_env) > 64:
        env_centered = mid_env - mid_env.mean()
        fft_env = np.abs(np.fft.rfft(env_centered))
        freqs = np.fft.rfftfreq(len(env_centered), d=1.0 / fps)
        vocal_mod_mask = (freqs >= 2.5) & (freqs <= 6.5)
        total_power = fft_env.sum() + 1e-8
        vocal_mod_ratio = float(fft_env[vocal_mod_mask].sum() / total_power)
        vocal_presence = float(np.clip(vocal_mod_ratio * 4.6, 0.0, 0.98))
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
    """Enrich metadata dict with structural defaults if missing."""
    if meta.get("instruments") and meta.get("instrumentOrder") and meta.get("instrumentPace"):
        return meta

    genre = str(meta.get("genre", "")).lower()
    bpm = int(meta.get("bpm", 120))
    energy = float(meta.get("energy", 0.5))

    if any(k in genre for k in ("rock", "metal", "punk", "grunge", "indie")):
        insts = ["Guitar & Strings", "Drums & Percussion", "Bass & 808"]
        order = ["Guitar", "Drums", "Bass", "Vocals"]
    elif any(k in genre for k in ("hip-hop", "rap", "trap", "drill")):
        insts = ["Drums & Percussion", "Bass & 808", "Synthesizer & Keys"]
        order = ["Drums", "Bass", "Keys / Synth", "Vocals"]
    elif any(k in genre for k in ("electronic", "edm", "dance", "house", "techno", "dnb")):
        insts = ["Synthesizer & Keys", "Drums & Percussion", "Bass & 808"]
        order = ["Keys / Synth", "Drums", "Bass"]
    elif any(k in genre for k in ("acoustic", "folk", "classical")):
        insts = ["Guitar & Strings", "Acoustic Piano & Keys"]
        order = ["Keys / Synth", "Guitar", "Vocals"]
    else:
        insts = ["Drums & Percussion", "Synthesizer & Keys", "Bass & 808"]
        order = ["Keys / Synth", "Drums", "Bass", "Vocals"]

    if bpm >= 135 or energy >= 0.75:
        pace = "Rapid & Driving (16th-Note Subdivision)"
    elif bpm >= 110 or energy >= 0.55:
        pace = "Upbeat & Groovy (8th-Note Subdivision)"
    elif bpm >= 80:
        pace = "Steady Mid-Tempo (4-on-the-Floor)"
    else:
        pace = "Laid-back & Spacious (Half-Time Groove)"

    if any(k in genre for k in ("hip-hop", "rap", "trap")):
        vocal_pattern = "Rhythmic Cadence (Rap / Flow / Syllabic)"
        vocal_comp = "Beat-Locked (In-The-Pocket Groove)"
        vocal_pres = 0.88
    elif any(k in genre for k in ("classical", "ambient", "instrumental")):
        vocal_pattern = "Instrumental (No Lead Vocals)"
        vocal_comp = "None (Instrumental Focus)"
        vocal_pres = 0.05
    else:
        vocal_pattern = "Melodic Legato (Sustained Ballad / Soulful)" if energy < 0.5 else "Dynamic Melodic Phrasing"
        vocal_comp = "Beat-Locked (In-The-Pocket Groove)" if energy > 0.6 else "Call-and-Response (Dynamic Interplay)"
        vocal_pres = 0.78

    meta["instruments"] = meta.get("instruments") or insts
    meta["instrumentOrder"] = meta.get("instrumentOrder") or order
    meta["instrumentPace"] = meta.get("instrumentPace") or pace
    meta["vocalPresence"] = meta.get("vocalPresence", vocal_pres)
    meta["vocalPattern"] = meta.get("vocalPattern") or vocal_pattern
    meta["vocalComplementation"] = meta.get("vocalComplementation") or vocal_comp
    return meta


def _instrument_similarity(insts_a: list[str], insts_b: list[str]) -> float:
    if not insts_a or not insts_b:
        return 0.7
    set_a, set_b = set(insts_a), set(insts_b)
    jaccard = len(set_a & set_b) / max(1, len(set_a | set_b))
    return float(np.clip(0.4 + 0.6 * jaccard, 0.2, 1.0))


def _order_similarity(order_a: list[str], order_b: list[str]) -> float:
    if not order_a or not order_b:
        return 0.7
    first_match = 1.0 if order_a[0] == order_b[0] else 0.55
    common = set(order_a) & set(order_b)
    overlap = len(common) / max(1, len(set(order_a) | set(order_b)))
    return 0.5 * first_match + 0.5 * overlap


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
    Structural Composite Matching Score in [0, 1]:
      - 512-d neural embedding similarity (timbral & acoustic manifold) [45%]
      - Instruments used overlap [15%]
      - Arrangement progression & order of entry [10%]
      - Instrumental pace & subdivision [10%]
      - Vocal presence, phrasing pattern & complementation groove [10%]
      - Harmonic Key & BPM octave compatibility [10%]
    """
    a = src_emb.flatten()
    b = cand_emb.flatten()
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    s_neural = float(np.clip((cos_sim + 0.1) / 1.1, 0.05, 1.0))

    s_inst = _instrument_similarity(
        src_metrics.get("instruments", []),
        cand_metrics.get("instruments", []),
    )
    s_order = _order_similarity(
        src_metrics.get("instrumentOrder", []),
        cand_metrics.get("instrumentOrder", []),
    )
    s_pace = _pace_similarity(
        src_metrics.get("instrumentPace", ""),
        cand_metrics.get("instrumentPace", ""),
    )
    s_vocal = _vocal_similarity(src_metrics, cand_metrics)

    s_key = _key_compatibility(
        str(src_metrics.get("key", "—")),
        str(cand_metrics.get("key", "—")),
    )
    s_bpm = _bpm_compatibility(
        int(src_metrics.get("bpm", 0)),
        int(cand_metrics.get("bpm", 0)),
    )
    s_harmonic = 0.5 * s_key + 0.5 * s_bpm

    total = (
        0.45 * s_neural
        + 0.15 * s_inst
        + 0.10 * s_order
        + 0.10 * s_pace
        + 0.10 * s_vocal
        + 0.10 * s_harmonic
    )
    return round(float(np.clip(total, 0.05, 0.99)), 4)


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

            # Compute structural composite score incorporating neural, instruments, order, pace & vocals
            if source_metrics:
                cand_emb_score = float(np.clip((float(dist) + 0.1) / 1.1, 0.05, 1.0))
                s_inst = _instrument_similarity(source_metrics.get("instruments", []), meta.get("instruments", []))
                s_order = _order_similarity(source_metrics.get("instrumentOrder", []), meta.get("instrumentOrder", []))
                s_pace = _pace_similarity(source_metrics.get("instrumentPace", ""), meta.get("instrumentPace", ""))
                s_vocal = _vocal_similarity(source_metrics, meta)
                s_key = _key_compatibility(str(source_metrics.get("key", "—")), str(meta.get("key", "—")))
                s_bpm = _bpm_compatibility(int(source_metrics.get("bpm", 0)), int(meta.get("bpm", 0)))
                s_harmonic = 0.5 * s_key + 0.5 * s_bpm

                comp_score = (
                    0.45 * cand_emb_score
                    + 0.15 * s_inst
                    + 0.10 * s_order
                    + 0.10 * s_pace
                    + 0.10 * s_vocal
                    + 0.10 * s_harmonic
                )
                meta["matchScore"] = round(float(np.clip(comp_score, 0.05, 0.99)), 4)
            else:
                meta["matchScore"] = round(float(np.clip((float(dist) + 0.1) / 1.1, 0.01, 0.99)), 4)

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