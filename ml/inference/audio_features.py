"""
Self-generated audio analytics from mel spectrograms and raw waveforms.

Computes musical and emotional descriptors directly from the audio signal
without relying on any external pre-computed API features:
  - bpm:     Estimated tempo in beats per minute via onset-strength autocorrelation
  - key:     Musical key & mode (e.g. "A Minor", "C Major") via Krumhansl-Schmuckler
             chroma profile correlation mapped from mel frequency bands
  - energy:  Perceptual acoustic intensity in [0, 1] from RMS power, spectral flux,
             and high-frequency presence
  - valence: Musical mood/positivity in [0, 1] from harmonic mode (Major vs Minor),
             spectral brightness, and rhythmic bounce
  - vibes:   2–3 descriptive tags derived from the computed acoustic profile
"""
from __future__ import annotations

import numpy as np

SR         = 22_050
HOP_LENGTH = 512
N_MELS     = 128
TOP_DB     = 80.0

# 12 pitch classes and Krumhansl-Schmuckler key profiles
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)


def _hz_to_mel_slaney(frequencies: np.ndarray) -> np.ndarray:
    """Convert Hz to Slaney-style mel scale (matches librosa default)."""
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
    """Convert Slaney-style mel scale back to Hz."""
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
    """
    Build a (12, n_mels) projection matrix mapping mel bins to the 12 semitone
    pitch classes across musical octaves (65 Hz / C2 to 2093 Hz / C7).
    """
    mel_min = _hz_to_mel_slaney(np.array([0.0]))[0]
    mel_max = _hz_to_mel_slaney(np.array([sr / 2.0]))[0]
    mel_pts = np.linspace(mel_min, mel_max, n_mels)
    freqs = _mel_to_hz_slaney(mel_pts)

    W = np.zeros((12, n_mels), dtype=np.float32)
    valid = (freqs >= 65.0) & (freqs <= 2100.0)
    for idx in np.where(valid)[0]:
        f = freqs[idx]
        # MIDI note number (69 = A4 = 440 Hz, pitch class 9 = A, 0 = C)
        midi = 69.0 + 12.0 * np.log2(f / 440.0)
        pitch_class = midi % 12.0
        for c in range(12):
            # Circular distance in semitones
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
    """
    Estimate tempo (BPM) and pulse clarity from a (128, T) mel spectrogram
    using onset-strength envelope autocorrelation.

    Returns:
        (bpm, pulse_clarity) where bpm is in [65, 185] and pulse_clarity in [0, 1].
    """
    T = mel.shape[1]
    if T < 64:
        return 120, 0.3

    # Emphasise rhythmic bands (kick/snare/percussion: lower 80 mel bins)
    rhythmic_mel = mel[:80, :]
    diff = np.diff(rhythmic_mel, axis=1)
    onset_env = np.maximum(0.0, diff).mean(axis=0)  # (T - 1,)

    # Smooth slightly and zero-centre
    if len(onset_env) >= 5:
        kernel = np.array([0.15, 0.2, 0.3, 0.2, 0.15], dtype=np.float32)
        onset_env = np.convolve(onset_env, kernel, mode="same")

    onset_centered = onset_env - onset_env.mean()
    var = float(np.dot(onset_centered, onset_centered))
    if var < 1e-9:
        return 120, 0.1

    fps = sr / float(hop_length)  # ~43.066 frames/sec
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

    # Apply a perceptual tempo prior centered at 120 BPM (log-Gaussian)
    lag_bpms = 60.0 * fps / lags
    tempo_prior = np.exp(-0.5 * (np.log2(lag_bpms / 120.0) / 0.55) ** 2)
    weighted_ac = np.maximum(0.0, ac) * tempo_prior

    best_idx = int(np.argmax(weighted_ac))
    best_lag = int(lags[best_idx])
    raw_clarity = float(np.clip(ac[best_idx], 0.0, 1.0))

    # Sub-frame parabolic interpolation for finer BPM resolution
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

    bpm = int(round(60.0 * fps / best_lag_f))
    bpm = int(np.clip(bpm, 65, 185))
    pulse_clarity = float(np.clip(raw_clarity * 1.4, 0.05, 1.0))
    return bpm, pulse_clarity


def estimate_key_from_mel(mel: np.ndarray) -> tuple[str, float]:
    """
    Estimate musical key (e.g. 'C Major', 'A Minor') and harmonic mode score
    (positive for Major, negative for Minor) from a (128, T) mel spectrogram.
    """
    # Convert normalised log-mel [0, 1] back towards linear magnitude weighting
    mag = np.power(10.0, (mel - 1.0) * 2.0)  # emphasises dominant harmonic peaks
    chroma = _MEL_TO_CHROMA @ mag.mean(axis=1)  # (12,)
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

    # Mode polarity in [-1, 1]: > 0 means Major-leaning, < 0 means Minor-leaning
    mode_polarity = float(np.clip((best_maj_corr - best_min_corr) * 1.5, -1.0, 1.0))
    return best_key, mode_polarity


def analyze_mel_spectrogram(
    mel: np.ndarray,
    genre_hint: str = "",
    sr: int = SR,
    hop_length: int = HOP_LENGTH,
) -> dict:
    """
    Compute self-generated musical and emotional analytics directly from a
    (128, T) float32 mel spectrogram in [0, 1].

    Returns a dict with:
        bpm (int), key (str), energy (float 0-1), valence (float 0-1), vibes (list[str])
    """
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim == 3:
        mel = mel.squeeze(0)
    if mel.ndim == 4:
        mel = mel.squeeze(0).squeeze(0)

    eps = 1e-8
    n_mels, T = mel.shape

    # 1. Frame energy & robust dynamic range
    frame_e = mel.mean(axis=0)  # (T,)
    rms_mean = float(frame_e.mean())
    rms_std = float(frame_e.std())
    p95 = float(np.percentile(frame_e, 95))
    p10 = float(np.percentile(frame_e, 10))
    dyn_range = max(0.0, p95 - p10)

    # 2. Spectral flux (frame-to-frame change = adrenaline / aggression)
    if T > 1:
        flux = np.abs(mel[:, 1:] - mel[:, :-1]).mean(axis=0)
        flux_mean = float(flux.mean())
    else:
        flux_mean = 0.0

    # 3. Spectral brightness (normalised centroid in [0, 1]) & high-freq ratio
    bins = np.arange(1, n_mels + 1, dtype=np.float32)[:, None]
    centroid = (mel * bins).sum(axis=0) / (mel.sum(axis=0) + eps)
    centroid_norm = float(centroid.mean() / n_mels)  # [0, 1]

    hf_ratio = float((mel[n_mels // 2 :, :].sum(axis=0) / (mel.sum(axis=0) + eps)).mean())
    bass_ratio = float((mel[:24, :].sum(axis=0) / (mel.sum(axis=0) + eps)).mean())

    # 4. Tempo (BPM) & pulse clarity
    bpm, pulse_clarity = estimate_tempo_from_mel(mel, sr=sr, hop_length=hop_length)

    # 5. Musical key & Major/Minor mode polarity
    key_str, mode_polarity = estimate_key_from_mel(mel)

    # 6. Perceptual Energy in [0.05, 0.99]
    # Combines overall loudness (rms_mean), spectral flux (movement/aggression),
    # pulse clarity (beat drive), and upper-frequency presence.
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

    # 7. Perceptual Valence (mood: happy/bright vs dark/melancholic) in [0.05, 0.98]
    # Combines Major/Minor harmonic polarity, spectral brightness, tempo bounce, and pulse.
    bright_score = float(np.clip((centroid_norm - 0.22) / 0.42, 0.0, 1.0))
    mode_score = 0.5 + 0.5 * mode_polarity  # [0, 1]
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

    # 8. Self-generated Vibe tags (2–3 concise tags reflecting sound & emotion)
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

    # Deduplicate while preserving order, cap at 3 tags
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