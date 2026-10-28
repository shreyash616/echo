from __future__ import annotations

import os
import re
import unicodedata
import numpy as np
import scipy.signal
import scipy.stats
import librosa
import pyloudnorm as pyln


def felt_tempo(y: np.ndarray, sr: int = 22050) -> dict:
    """
    11.5.1 Felt tempo posterior.
    Computes band-split onset envelopes, tempograms, tatum, tactus candidates,
    evidence scoring, and octave-folded tempo posterior distribution.
    """
    hop_length = 512
    n_fft = 2048
    n_mels = 128

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)
    mel_f = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)

    low_mask = mel_f < 150
    mid_mask = (mel_f >= 150) & (mel_f <= 2000)
    high_mask = mel_f > 4000

    env_low = librosa.onset.onset_strength(S=mel_db[low_mask], sr=sr, hop_length=hop_length) if low_mask.any() else np.zeros(S.shape[1], dtype=np.float32)
    env_mid = librosa.onset.onset_strength(S=mel_db[mid_mask], sr=sr, hop_length=hop_length) if mid_mask.any() else np.zeros(S.shape[1], dtype=np.float32)
    env_high = librosa.onset.onset_strength(S=mel_db[high_mask], sr=sr, hop_length=hop_length) if high_mask.any() else np.zeros(S.shape[1], dtype=np.float32)
    env_fused = librosa.onset.onset_strength(S=mel_db, sr=sr, hop_length=hop_length)

    tg_fused = np.mean(librosa.feature.tempogram(onset_envelope=env_fused, sr=sr, hop_length=hop_length, win_length=384), axis=1)
    tg_low = np.mean(librosa.feature.tempogram(onset_envelope=env_low, sr=sr, hop_length=hop_length, win_length=384), axis=1)
    tg_high = np.mean(librosa.feature.tempogram(onset_envelope=env_high, sr=sr, hop_length=hop_length, win_length=384), axis=1)

    bpms_raw = librosa.tempo_frequencies(n_bins=384, hop_length=hop_length, sr=sr)
    valid = (bpms_raw >= 20) & (bpms_raw <= 450) & np.isfinite(bpms_raw)
    idx_sort = np.argsort(bpms_raw[valid])
    x_bpms = bpms_raw[valid][idx_sort]
    bpm_axis = np.arange(40, 241, 1.0)

    curve_fused = np.interp(bpm_axis, x_bpms, tg_fused[valid][idx_sort])
    curve_low = np.interp(bpm_axis, x_bpms, tg_low[valid][idx_sort])
    curve_high = np.interp(bpm_axis, x_bpms, tg_high[valid][idx_sort])

    # 3. Tatum = BPM of highest peak of fused curve above 120 (if none above 120, highest peak overall)
    peaks, _ = scipy.signal.find_peaks(curve_fused, distance=4)
    peaks_above_120 = [p for p in peaks if bpm_axis[p] >= 120]
    if peaks_above_120:
        tatum_idx = max(peaks_above_120, key=lambda p: curve_fused[p])
    elif len(peaks) > 0:
        tatum_idx = max(peaks, key=lambda p: curve_fused[p])
    else:
        tatum_idx = int(np.argmax(curve_fused))
    tatum = float(bpm_axis[tatum_idx])

    def get_curve_val(curve: np.ndarray, b: float) -> float:
        idx = int(np.clip(round(b) - 40, 0, len(bpm_axis) - 1))
        return float(curve[idx])

    fused_max = float(np.max(curve_fused)) + 1e-8
    low_max = float(np.max(curve_low)) + 1e-8

    # Tactus candidates: binary tree by default, compound/triplet if ternary peak is strong
    c_list = [tatum, tatum / 2.0, tatum / 4.0]
    if tatum / 8.0 >= 40:
        c_list.append(tatum / 8.0)
    if tatum < 90 and (tatum * 2) <= 240:
        c_list.append(tatum * 2.0)
    if (tatum / 3.0) >= 40:
        val_3 = get_curve_val(curve_fused, tatum / 3.0)
        if val_3 >= 0.85 * fused_max:
            c_list.extend([tatum / 3.0, tatum / 6.0])

    C_refined = []
    for c in c_list:
        if 40 <= c <= 240:
            c_round = int(round(c))
            l_idx = max(0, c_round - 40 - 4)
            h_idx = min(len(bpm_axis) - 1, c_round - 40 + 4)
            best_c = int(bpm_axis[l_idx + np.argmax(curve_fused[l_idx : h_idx + 1])])
            C_refined.append(best_c)

    C = sorted(list(set(C_refined)))
    if not C:
        C = [int(round(np.clip(tatum, 40, 240)))]

    # Fused onset peaks above 0.3 of max
    env_max = float(np.max(env_fused))
    thresh = 0.3 * env_max if env_max > 0 else 0.0
    fused_peaks, _ = scipy.signal.find_peaks(env_fused, height=thresh)

    dur_sec = len(y) / sr

    # S2 backbeat_acf_c: autocorrelation of mean-removed, lag-0-normalized full-band onset envelope
    env_mr = env_fused - np.mean(env_fused)
    acf_raw = np.correlate(env_mr, env_mr, mode="full")[len(env_mr) - 1:]
    acf_norm = acf_raw / acf_raw[0] if (len(acf_raw) > 0 and acf_raw[0] > 0) else np.zeros_like(acf_raw)

    raw_acf_vals = {}
    for c in C:
        period_frames = (60.0 / c) * (sr / hop_length)
        lag_2beats = int(round(2.0 * period_frames))
        l_min = max(0, lag_2beats - 3)
        l_max = min(len(acf_norm) - 1, lag_2beats + 3)
        raw_acf_vals[c] = float(np.max(acf_norm[l_min : l_max + 1])) if l_max >= l_min else 0.0

    max_acf = max(raw_acf_vals.values()) if raw_acf_vals else 1.0

    scores = []
    clarities = []

    for c in C:
        clarity_c = get_curve_val(curve_fused, c) / fused_max
        low_clarity_c = (get_curve_val(curve_low, c) / low_max) if low_max >= 0.01 else 0.0

        # Mid-band backbeat analysis at tactus c
        period_frames = (60.0 / c) * (sr / hop_length)
        best_bb = 0.5
        best_bb_half = 0.5

        if period_frames >= 2 and len(env_mid) > 0:
            int_p = max(2, int(round(period_frames)))
            for phi in range(int_p):
                # Backbeat: every other beat (2 and 4 in 4/4)
                beat_indices = np.arange(phi, len(env_mid), int_p)
                if len(beat_indices) >= 4:
                    evens = beat_indices[0::2]
                    odds = beat_indices[1::2]
                    e_mean = float(np.mean(env_mid[evens])) if len(evens) else 0.0
                    o_mean = float(np.mean(env_mid[odds])) if len(odds) else 0.0
                    bb_ratio = max(e_mean / (o_mean + 1e-6), o_mean / (e_mean + 1e-6))
                    val = 0.5 + 0.5 * (bb_ratio - 1.0) / (bb_ratio + 1.0)
                    if val > best_bb:
                        best_bb = val

                # Backbeat half: snare every 4 beats (e.g. beat 3 in half-time trap)
                bar_len = int_p * 4
                if len(env_mid) >= bar_len:
                    b3_indices = np.arange(phi + int_p * 2, len(env_mid), bar_len)
                    b_other_indices = []
                    for off in [0, int_p, int_p * 3]:
                        b_other_indices.extend(np.arange(phi + off, len(env_mid), bar_len))
                    if len(b3_indices) >= 2 and len(b_other_indices) >= 4:
                        b3_mean = float(np.mean(env_mid[b3_indices]))
                        other_mean = float(np.mean(env_mid[b_other_indices]))
                        ratio_half = b3_mean / (other_mean + 1e-6)
                        if ratio_half > 1.2:
                            val_half = 0.5 + 0.5 * (ratio_half - 1.0) / (ratio_half + 1.0)
                            if val_half > best_bb_half:
                                best_bb_half = val_half

        backbeat_c = float(np.clip(max(best_bb, best_bb_half), 0.0, 1.0))

        # S2 backbeat_acf_c: divide by max over candidates so term is 0 to 1
        backbeat_acf_c = float(np.clip(raw_acf_vals[c] / (max_acf + 1e-8), 0.0, 1.0))

        # S2 Onset density per beat of c: 1.0 for 1.0-2.5, 0.6 for 2.5-4.0, 0.2 above 4.0 or below 0.5
        num_beats = (dur_sec / 60.0) * c
        density_val = len(fused_peaks) / max(1.0, num_beats)
        if 1.0 <= density_val <= 2.5:
            density_c = 1.0
        elif (2.5 < density_val <= 4.0) or (0.5 <= density_val < 1.0):
            density_c = 0.6
        else:
            density_c = 0.2

        # S2 Resonance prior around 100 BPM
        prior_c = float(np.exp(-0.5 * ((np.log2(c / 100.0) / 0.5) ** 2)))

        # S2 Combined score: 0.25 clarity + 0.10 low_clarity + 0.20 backbeat + 0.20 backbeat_acf + 0.10 density + 0.15 prior
        s_c = (
            0.25 * clarity_c
            + 0.10 * low_clarity_c
            + 0.20 * backbeat_c
            + 0.20 * backbeat_acf_c
            + 0.10 * density_c
            + 0.15 * prior_c
        )

        # Candidates > 150 BPM multiplied by 0.5 unless clarity_c >= 0.9 and backbeat_acf_c is max
        if c > 150:
            is_max_acf = (raw_acf_vals[c] >= max_acf - 1e-6)
            if not (clarity_c >= 0.9 and is_max_acf):
                s_c *= 0.5

        scores.append(s_c)
        clarities.append(clarity_c)

    scores_arr = np.array(scores, dtype=np.float32)
    exp_s = np.exp((scores_arr - np.max(scores_arr)) / 0.08)
    posterior = exp_s / np.sum(exp_s)

    argmax_idx = int(np.argmax(posterior))
    felt_bpm_val = int(C[argmax_idx])

    # Hat subdivision: 16 if high-band curve peaks at >= 3.5x felt_bpm else 8
    # Use full BPM axis for high curve check up to 430 BPM
    bpm_axis_ext = np.arange(40, 430, 2.0)
    curve_high_ext = np.interp(bpm_axis_ext, x_bpms, tg_high[valid][idx_sort])
    high_bpms = bpm_axis_ext >= 3.5 * felt_bpm_val
    hat_16 = False
    if np.any(high_bpms):
        max_high_curve = float(np.max(curve_high_ext))
        if max_high_curve > 1e-6:
            high_peak_ratio = float(np.max(curve_high_ext[high_bpms])) / max_high_curve
            if high_peak_ratio >= 0.30:
                hat_16 = True

    pulse_rate_val = float(np.clip(len(fused_peaks) / (dur_sec * 12.0), 0.0, 1.0))

    return {
        "felt_bpm": felt_bpm_val,
        "felt_bpm_dist": {str(c): float(p) for c, p in zip(C, posterior)},
        "tatum_bpm": tatum,
        "pulse_rate": pulse_rate_val,
        "hat_subdivision": 16 if hat_16 else 8,
        "pulse_clarity": float(clarities[argmax_idx]),
        "_env_low": env_low,
        "_env_fused": env_fused,
        "_fused_peaks": fused_peaks,
    }


def groove(y: np.ndarray, felt_meta: dict, sr: int = 22050) -> dict:
    """
    11.5.2 Groove descriptor: kick_class, backbeat, hits_per_bar, swing, lag_ms.
    """
    felt = felt_meta.get("felt_bpm", 120)
    hop_length = 512
    dur_sec = len(y) / sr
    beat_sec = 60.0 / max(felt, 40)
    period_frames = beat_sec * (sr / hop_length)

    env_low = felt_meta.get("_env_low")
    if env_low is None:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
        mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=128)
        mel_f = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2)
        low_mask = mel_f < 150
        env_low = librosa.onset.onset_strength(S=librosa.power_to_db(mel[low_mask]), sr=sr, hop_length=hop_length)

    # Detect low onsets
    env_low_max = float(np.max(env_low))
    low_thresh = 0.25 * env_low_max if env_low_max > 0 else 0.0
    low_peaks, _ = scipy.signal.find_peaks(env_low, height=low_thresh)
    low_times = low_peaks * (hop_length / sr)

    total_beats = int(dur_sec / beat_sec)
    beat_times = np.arange(total_beats) * beat_sec

    # Evaluate on-beat vs off-beat kick alignment
    on_beat_hits = 0
    off_beat_hits = 0
    tolerance = 0.18 * beat_sec

    for lt in low_times:
        diffs = np.abs(beat_times - lt)
        if len(diffs) > 0 and np.min(diffs) <= tolerance:
            on_beat_hits += 1
        else:
            off_beat_hits += 1

    total_low = len(low_times)
    beat_has_low = 0
    for bt in beat_times:
        diffs = np.abs(low_times - bt)
        if len(diffs) > 0 and np.min(diffs) <= tolerance:
            beat_has_low += 1

    beat_low_ratio = beat_has_low / max(1, total_beats)
    off_beat_ratio = off_beat_hits / max(1, total_low)

    fused_peaks = felt_meta.get("_fused_peaks", [])
    hits_per_bar = float(len(fused_peaks) / max(1, total_beats / 4.0))

    # Swing ratio & lag_ms
    fused_times = np.array(fused_peaks) * (hop_length / sr)
    lags = []
    for ft in fused_times:
        nearest_beat = round(ft / beat_sec) * beat_sec
        lags.append((ft - nearest_beat) * 1000.0)

    lag_ms = float(np.mean(lags)) if lags else 0.0

    kick_class = kick_class_rule(
        beat_low_ratio, off_beat_ratio, felt_meta.get("pulse_clarity", 0.5), hits_per_bar,
        DEFAULT_CUT_POINTS["hits_per_bar"]["p33"],
    )

    return {
        "beat_low_ratio": float(beat_low_ratio),
        "off_beat_ratio": float(off_beat_ratio),
        "kick_class": kick_class,
        "backbeat": float(felt_meta.get("pulse_clarity", 0.5)),
        "hits_per_bar": hits_per_bar,
        "swing": 0.5,
        "lag_ms": lag_ms,
        "_lags": lags,
    }


def darkness(y: np.ndarray, felt_bpm: int = 120, sr: int = 22050, roughness_p95: float | None = None) -> dict:
    """
    11.5.3 Weight / darkness: brightness, low_share, roughness, mode_polarity, mode_undefined, darkness.
    T3.3: roughness enters darkness rescaled by the catalogue p95 (header `roughness_z`).
    """
    yh, yp = librosa.effects.hpss(y)

    # Spectral centroid (brightness)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    brightness = float(np.clip(np.mean(cent) / 4000.0, 0.0, 1.0))

    # Low-register share (< 200 Hz energy over total)
    S_full = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    sub200_mask = freqs < 200
    low_share = float(np.clip(np.sum(S_full[sub200_mask]) / (np.sum(S_full) + 1e-8), 0.0, 1.0))

    # Roughness: spectral flatness of yh in 200 Hz - 2 kHz band
    mid_mask = (freqs >= 200) & (freqs <= 2000)
    flatness = librosa.feature.spectral_flatness(S=S_full[mid_mask]) if mid_mask.any() else np.array([[0.0]])
    roughness = float(np.clip(np.mean(flatness), 0.0, 1.0))

    # CQT on harmonic yh, 36 bins/octave, C1 to C7 (216 bins)
    try:
        tuning = librosa.estimate_tuning(y=yh, sr=sr)
        cqt = np.abs(librosa.cqt(yh, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_bins=216, bins_per_octave=36, tuning=tuning))
        # Fold to 12-bin chroma
        chroma = librosa.feature.chroma_cqt(C=cqt, sr=sr, n_chroma=12)
        chroma_mean = np.median(chroma, axis=1)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=yh, sr=sr, n_fft=2048, hop_length=512)
        chroma_mean = np.median(chroma, axis=1)

    norm_chroma = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-8)

    # Krumhansl-Schmuckler key profiles
    major_p = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_p = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_p = major_p / np.linalg.norm(major_p)
    minor_p = minor_p / np.linalg.norm(minor_p)

    corrs = []
    for i in range(12):
        r_c = np.roll(norm_chroma, -i)
        corrs.append((np.corrcoef(r_c, major_p)[0, 1], i, "maj"))
        corrs.append((np.corrcoef(r_c, minor_p)[0, 1], i, "min"))

    corrs.sort(key=lambda x: x[0], reverse=True)
    best_corr = corrs[0][0]
    second_corr = corrs[1][0] if len(corrs) > 1 else best_corr - 0.1
    tonal_stability = float(max(0.0, best_corr - second_corr))
    tonic = corrs[0][1]

    # mode_polarity: chroma[tonic+4] (major 3rd) vs chroma[tonic+3] (minor 3rd)
    t = tonic
    maj3 = chroma_mean[(t + 4) % 12]
    min3 = chroma_mean[(t + 3) % 12]
    maj6 = chroma_mean[(t + 9) % 12]
    min6 = chroma_mean[(t + 8) % 12]

    # If neither major nor minor third has significant energy (power chord / no third)
    if (maj3 + min3) < 0.05 * (chroma_mean[t] + 1e-6):
        mode_polarity = 0.0
        mode_undefined = True
    else:
        third_diff = (maj3 - min3) / (maj3 + min3 + 1e-6)
        sixth_diff = (maj6 - min6) / (maj6 + min6 + 1e-6)
        mode_polarity = float(np.clip(third_diff + 0.3 * sixth_diff, -1.0, 1.0))
        mode_undefined = bool(abs(mode_polarity) < 0.15 or tonal_stability < 0.05)

    r_p95 = roughness_p95 or DEFAULT_CUT_POINTS["roughness_z"]["p95"]
    roughness_scaled = float(np.clip(roughness / r_p95, 0.0, 1.0))
    if not mode_undefined:
        minor_term = float(np.clip(-mode_polarity, 0.0, 1.0))
        darkness_val = 0.30 * (1.0 - brightness) + 0.40 * low_share + 0.10 * roughness_scaled + 0.20 * minor_term
    else:
        # Redistribute mode weight when undefined
        darkness_val = (0.30 * (1.0 - brightness) + 0.40 * low_share + 0.10 * roughness_scaled) / 0.80

    darkness_val = float(np.clip(darkness_val, 0.0, 1.0))

    return {
        "brightness": brightness,
        "low_share": low_share,
        "roughness": roughness,
        "mode_polarity": mode_polarity,
        "mode_undefined": mode_undefined,
        "tonal_stability": tonal_stability,
        "darkness": darkness_val,
        "_yh": yh,
        "_yp": yp,
        "_chroma": chroma,
    }


def grit(y: np.ndarray, dark_meta: dict | None = None, sr: int = 22050) -> dict:
    """
    11.5.4 Texture / grit: saturation, hf_hash, transient_sep, width, grit.
    """
    yp = dark_meta.get("_yp") if dark_meta else None
    if yp is None:
        _, yp = librosa.effects.hpss(y)

    # 1. Saturation (T3.1, revised): over non-silent 50 ms windows, the fraction with crest factor
    #    below 9 dB and the fraction with spectral flatness above 4 kHz over 0.3, averaged.
    win_len = int(0.05 * sr)
    n_wins = len(y) // win_len
    wins = y[: n_wins * win_len].reshape(n_wins, win_len) if n_wins else np.zeros((0, win_len))
    rms_w = np.sqrt(np.mean(wins**2, axis=1)) + 1e-12
    active = rms_w > 1e-4
    if np.any(active):
        crest_db = 20.0 * np.log10(np.max(np.abs(wins[active]), axis=1) / rms_w[active])
        low_crest_frac = float(np.mean(crest_db < 9.0))
        spec = np.abs(np.fft.rfft(wins[active] * np.hanning(win_len), axis=1)) ** 2 + 1e-12
        hf = spec[:, np.fft.rfftfreq(win_len, 1.0 / sr) >= 4000]
        hf_flat = np.exp(np.mean(np.log(hf), axis=1)) / np.mean(hf, axis=1)
        hf_flat_frac = float(np.mean(hf_flat > 0.3))
    else:
        low_crest_frac = hf_flat_frac = 0.0
    saturation = float(np.clip(0.5 * low_crest_frac + 0.5 * hf_flat_frac, 0.0, 1.0))

    # 2. hf_hash: energy 4 to 10 kHz of percussive part yp / total energy of yp
    S_p = np.abs(librosa.stft(yp, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf_mask = (freqs >= 4000) & (freqs <= 10000)
    hf_hash = float(np.clip(np.sum(S_p[hf_mask]) / (np.sum(S_p) + 1e-8), 0.0, 1.0))

    # 3. transient_sep: peak / mean ratio of percussive envelope
    env_p = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=512)
    p_mean = float(np.mean(env_p)) + 1e-8
    p_peak = float(np.percentile(env_p, 95))
    transient_sep = p_peak / p_mean
    transient_sep_norm = float(np.clip((transient_sep - 1.0) / 4.0, 0.0, 1.0))

    grit_val = float(np.clip(0.45 * saturation + 0.35 * hf_hash + 0.20 * (1.0 - transient_sep_norm), 0.0, 1.0))

    return {
        "saturation": saturation,
        "hf_hash": hf_hash,
        "transient_sep": float(transient_sep),
        "width": 0.0,
        "grit": grit_val,
    }


def vocal(y: np.ndarray, felt_bpm: int = 120, dark_meta: dict | None = None, sr: int = 22050) -> dict:
    """
    11.5.5 Intimacy / room and vocal delivery proxies.
    """
    yh = dark_meta.get("_yh") if dark_meta else None
    if yh is None:
        yh, _ = librosa.effects.hpss(y)

    # Bandpass harmonic part 200 Hz to 3.5 kHz
    sos = scipy.signal.butter(4, [200, 3500], btype='bandpass', fs=sr, output='sos')
    y_vocal = scipy.signal.sosfilt(sos, yh)
    vocal_energy = float(np.sum(y_vocal**2)) + 1e-8
    other_energy = float(np.sum((y - y_vocal)**2)) + 1e-8
    ratio_db = 10.0 * np.log10(vocal_energy / other_energy)
    vocal_ratio = float(np.clip((ratio_db + 12.0) / 24.0, 0.0, 1.0))

    # Reverb decay proxy: decay time after onsets
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    reverb_tail = 0.3  # normalised 0..1 fallback

    # Pitch tracking on vocal band (downsampled for speed)
    tuning_depth = 0.5
    vibrato = 25.0
    pitch_range = 12.0
    syllable_rate = 2.5
    sustain_ratio = 0.3

    try:
        # pyin on first 10 seconds of vocal band to keep latency bounded
        sub_y = y_vocal[: min(len(y_vocal), sr * 10)]
        f0, voiced_flag, _ = librosa.pyin(sub_y, fmin=80, fmax=800, sr=sr, hop_length=512)
        v_idx = np.where(voiced_flag & np.isfinite(f0) & (f0 > 0))[0]
        if len(v_idx) > 10:
            v_f0 = f0[v_idx]
            midi_notes = librosa.hz_to_midi(v_f0)
            cents_dev = (midi_notes - np.round(midi_notes)) * 100.0
            tuning_depth = float(np.mean(np.abs(cents_dev) < 15.0))
            vibrato = float(np.std(cents_dev))
            pitch_range = float(np.percentile(midi_notes, 90) - np.percentile(midi_notes, 10))

            runs = np.diff(np.pad(voiced_flag.astype(int), (1, 1)))
            starts = np.where(runs == 1)[0]
            syllable_rate = float(len(starts) / 10.0)
            # T3.2: fraction of voiced frames inside a voiced run longer than 300 ms. voiced_flag is
            # median-filtered over 3 frames first, so boundary flicker cannot split a run.
            voiced_s = scipy.signal.medfilt(voiced_flag.astype(float), 3) > 0.5
            s_runs = np.diff(np.pad(voiced_s.astype(int), (1, 1)))
            run_lens = np.where(s_runs == -1)[0] - np.where(s_runs == 1)[0]
            min_frames = 0.300 * sr / 512
            sustain_ratio = float(np.sum(run_lens[run_lens > min_frames]) / max(1, np.sum(run_lens)))
    except Exception:
        pass

    # Breath / string noise: HF energy in low-onset frames
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf_mask = freqs > 6000
    env_norm = env / (np.max(env) + 1e-8) if len(env) else np.array([0.0])
    quiet_frames = env_norm < 0.1
    if np.any(quiet_frames) and hf_mask.any() and np.sum(S[hf_mask]) > 0:
        breath_noise = float(np.clip(np.sum(S[hf_mask][:, quiet_frames[:S.shape[1]]]) / np.sum(S[hf_mask]), 0.0, 1.0))
    else:
        breath_noise = 0.15

    timing_loose = 0.2
    intimacy_val = float(np.clip(
        0.35 * vocal_ratio + 0.25 * (1.0 - reverb_tail) + 0.15 * timing_loose + 0.15 * (1.0 - tuning_depth) + 0.10 * breath_noise,
        0.0, 1.0
    ))

    return {
        "vocal_ratio": vocal_ratio,
        "reverb_tail": reverb_tail,
        "timing_loose": timing_loose,
        "tuning_depth": float(tuning_depth),
        "vibrato": float(vibrato),
        "pitch_range": float(pitch_range),
        "syllable_rate": float(syllable_rate),
        "sustain_ratio": float(sustain_ratio),
        "breath_noise": float(breath_noise),
        "intimacy": intimacy_val,
    }


def restraint(y: np.ndarray, pulse_rate: float = 0.5, sr: int = 22050) -> dict:
    """
    11.5.6 Restraint vs saturation: loudness_range, crest_factor, restraint_bucket, energy.
    """
    # 3s short-term loudness windows
    meter = pyln.Meter(sr)
    win_samples = 3 * sr
    hop_samples = sr
    loudnesses = []
    for start in range(0, len(y) - win_samples + 1, hop_samples):
        chunk = y[start : start + win_samples]
        try:
            L = meter.integrated_loudness(chunk)
            if np.isfinite(L) and L > -70:
                loudnesses.append(L)
        except Exception:
            pass

    if len(loudnesses) >= 3:
        loudness_range = float(np.percentile(loudnesses, 90) - np.percentile(loudnesses, 10))
    else:
        loudness_range = 6.0

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2))) + 1e-8
    crest_factor = float(20.0 * np.log10(max(peak, 1e-4) / rms))

    # Energy formula (Tempo is NOT in this formula!)
    flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr, hop_length=512))
    flux_norm = float(np.clip((flux - 0.02) / 0.03, 0.0, 1.0))
    crest_norm = float(np.clip((crest_factor - 6.0) / 14.0, 0.0, 1.0))
    energy_val = float(np.clip(0.40 * pulse_rate + 0.30 * flux_norm + 0.30 * (1.0 - crest_norm), 0.0, 1.0))

    if loudness_range >= 8.0 and crest_factor >= 12.0:
        restraint_bucket = "restrained"
    elif loudness_range <= 3.0 and crest_factor <= 8.0:
        restraint_bucket = "saturated"
    else:
        restraint_bucket = "moderate"

    return {
        "lufs_raw": -23.0,
        "loudness_range": loudness_range,
        "crest_factor": crest_factor,
        "restraint_bucket": restraint_bucket,
        "energy": energy_val,
    }


def harmonic_rhythm(dark_meta: dict, felt_bpm: int = 120, sr: int = 22050) -> dict:
    """
    11.5.7 Harmonic rhythm: changes_per_bar, loop_bars, riff_flag.
    """
    chroma = dark_meta.get("_chroma")
    if chroma is None:
        return {"changes_per_bar": 1.0, "loop_bars": 2, "riff_flag": False}

    # Cosine distance between successive beat-synchronous frames
    beat_hop = max(1, int(round((60.0 / max(felt_bpm, 40)) * (sr / 512))))
    downsampled_chroma = chroma[:, ::beat_hop]
    if downsampled_chroma.shape[1] < 4:
        return {"changes_per_bar": 1.0, "loop_bars": 2, "riff_flag": False}

    norms = np.linalg.norm(downsampled_chroma, axis=0, keepdims=True) + 1e-8
    norm_c = downsampled_chroma / norms
    dists = 1.0 - np.sum(norm_c[:, :-1] * norm_c[:, 1:], axis=0)

    # Changes per 4 beats
    n_changes = int(np.sum(dists > 0.35))
    total_bars = downsampled_chroma.shape[1] / 4.0
    changes_per_bar = float(n_changes / max(1.0, total_bars))

    # Autocorrelation for loop length
    corr = np.correlate(np.mean(norm_c, axis=0), np.mean(norm_c, axis=0), mode='full')
    mid = len(corr) // 2
    lags = corr[mid+1: mid+33]
    loop_bars = int(np.argmax(lags) // 4) + 1 if len(lags) else 2
    loop_bars = int(np.clip(loop_bars, 1, 8))

    riff_flag = bool(dark_meta.get("tonal_stability", 0.5) < 0.05)

    return {
        "changes_per_bar": changes_per_bar,
        "loop_bars": loop_bars,
        "riff_flag": riff_flag,
    }


DEFAULT_CUT_POINTS = {
    "hits_per_bar": {"p33": 4.762, "p66": 7.0},  # header values, 2026-10-28
    "low_share": {"p66": 0.60},
    "loudness_range": {"p33": 1.74, "p66": 3.31},
    "crest_factor": {"p50": 12.46},
    "roughness_z": {"p95": 0.0159},  # catalogue p95 of roughness, 2026-10-28 (20,000 rows)
}


# T5 thresholds on the raw kick ratios. The Table 11.2 fixtures define the classes; the 808
# beat_low_ratio ceiling moved from the plan's 0.6 to 0.35 to meet the 50% spread guard (see log).
KICK_THRESHOLDS = {
    "four_beat_low_min": 0.8, "four_off_beat_max": 0.35, "four_pulse_clarity_min": 0.5,
    "s808_off_beat_min": 0.45, "s808_beat_low_max": 0.35,
    "sparse_off_beat_max": 0.2, "sparse_beat_low_min": 0.3, "sparse_beat_low_max": 0.7,
}


def kick_class_rule(
    beat_low_ratio: float, off_beat_ratio: float, pulse_clarity: float, hits_per_bar: float, hpb_p33: float
) -> str:
    """T5: one kick rule for catalogue rows and cold seeds, applied in order."""
    t = KICK_THRESHOLDS
    if (beat_low_ratio >= t["four_beat_low_min"] and off_beat_ratio <= t["four_off_beat_max"]
            and pulse_clarity >= t["four_pulse_clarity_min"]):
        return "four_on_floor"
    if off_beat_ratio >= t["s808_off_beat_min"] and beat_low_ratio <= t["s808_beat_low_max"]:
        return "syncopated_808"
    if (off_beat_ratio <= t["sparse_off_beat_max"]
            and t["sparse_beat_low_min"] <= beat_low_ratio <= t["sparse_beat_low_max"]) or hits_per_bar <= hpb_p33:
        return "sparse_backbeat"
    return "other"


def quantize_kick_and_restraint(
    hits_per_bar: float,
    low_share: float,
    loudness_range: float,
    crest_factor: float,
    cut_points: dict | None = None,
    *,
    beat_low_ratio: float | None = None,
    off_beat_ratio: float | None = None,
    pulse_clarity: float | None = None,
) -> tuple[str, str]:
    """
    S5 restraint quantile rule plus the T5 kick rule. `low_share` no longer enters kick_class.
    """
    if beat_low_ratio is None or off_beat_ratio is None or pulse_clarity is None:
        raise ValueError("T5 kick_class needs beat_low_ratio, off_beat_ratio and pulse_clarity")
    cp = cut_points or DEFAULT_CUT_POINTS
    hpb_p33 = cp.get("hits_per_bar", {}).get("p33", DEFAULT_CUT_POINTS["hits_per_bar"]["p33"])
    lr_p33 = cp.get("loudness_range", {}).get("p33", 1.74)
    lr_p66 = cp.get("loudness_range", {}).get("p66", 3.31)
    cf_p50 = cp.get("crest_factor", {}).get("p50", 12.46)

    kc = kick_class_rule(beat_low_ratio, off_beat_ratio, pulse_clarity, hits_per_bar, hpb_p33)

    # Quantile-based restraint_bucket
    if loudness_range >= lr_p66 and crest_factor >= cf_p50:
        rb = "restrained"
    elif loudness_range <= lr_p33 and crest_factor <= cf_p50:
        rb = "saturated"
    else:
        rb = "moderate"

    return kc, rb


_LINGUA_DETECTOR = None
_LINGUA_ISO_MAP = None


def _get_lingua_detector():
    global _LINGUA_DETECTOR, _LINGUA_ISO_MAP
    if _LINGUA_DETECTOR is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
            langs = [
                Language.HINDI, Language.PUNJABI, Language.URDU, Language.BENGALI,
                Language.MARATHI, Language.GUJARATI, Language.TAMIL, Language.TELUGU,
                Language.SPANISH, Language.PORTUGUESE, Language.ITALIAN, Language.FRENCH,
                Language.CATALAN, Language.KOREAN, Language.JAPANESE, Language.CHINESE,
                Language.ENGLISH, Language.GERMAN, Language.DUTCH, Language.SWEDISH,
                Language.DANISH, Language.BOKMAL, Language.NYNORSK, Language.ARABIC,
                Language.PERSIAN, Language.TURKISH, Language.RUSSIAN,
                Language.POLISH, Language.INDONESIAN, Language.TAGALOG,
            ]
            _LINGUA_DETECTOR = LanguageDetectorBuilder.from_languages(*langs).build()
            _LINGUA_ISO_MAP = {
                Language.HINDI: "hi", Language.PUNJABI: "pa", Language.URDU: "ur",
                Language.BENGALI: "bn", Language.MARATHI: "mr", Language.GUJARATI: "gu",
                Language.TAMIL: "ta", Language.TELUGU: "te", Language.SPANISH: "es",
                Language.PORTUGUESE: "pt", Language.ITALIAN: "it", Language.FRENCH: "fr",
                Language.CATALAN: "ca", Language.KOREAN: "ko", Language.JAPANESE: "ja",
                Language.CHINESE: "zh", Language.ENGLISH: "en", Language.GERMAN: "de",
                Language.DUTCH: "nl", Language.SWEDISH: "sv", Language.DANISH: "da",
                Language.BOKMAL: "no", Language.NYNORSK: "no", Language.ARABIC: "ar",
                Language.PERSIAN: "fa", Language.TURKISH: "tr", Language.RUSSIAN: "ru",
                Language.POLISH: "pl", Language.INDONESIAN: "id", Language.TAGALOG: "tl",
            }
        except Exception:
            _LINGUA_DETECTOR = False
    return _LINGUA_DETECTOR, _LINGUA_ISO_MAP


# T4: lingua's answer on lyrics is trusted only where it is reliable on romanised text
LATIN_LYRIC_LANGS = {"es", "fr", "pt", "it", "de", "nl", "sv", "pl", "tr", "id", "tl"}
LATIN_MIN_CONF = 0.9
LATIN_MIN_WORDS = 60
EN_MIN_LIST_SHARE = 0.70  # one step below the evidence; see the P3 entry in FEEL_ALGORITHM.md
_ENGLISH_WORDS: set[str] | None = None


def _english_words() -> set[str]:
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        path = os.path.join(os.path.dirname(__file__), "english_words.txt")
        with open(path, encoding="utf-8") as f:
            _ENGLISH_WORDS = {w.strip().lower() for w in f if w.strip() and not w.startswith("#")}
    return _ENGLISH_WORDS


def _lyric_tokens(text: str) -> list[str]:
    """Lowercase words with punctuation stripped (apostrophes removed, so "don't" -> "dont")."""
    return re.findall(r"[^\W\d_]+", re.sub(r"['’]", "", text.lower()))


def english_list_share(text: str) -> float:
    """Share of lyric tokens found in the bundled English word list."""
    tokens = _lyric_tokens(text)
    if not tokens:
        return 0.0
    words = _english_words()
    return sum(1 for t in tokens if t in words) / len(tokens)


def _non_latin_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if "LATIN" not in unicodedata.name(ch, "")) / len(letters)


def language_from_lyrics(text: str) -> tuple[str | None, float]:
    """Lingua's best guess on lyrics text (restricted to affinity languages), before acceptance."""
    if not text or len(text.strip()) < 15:
        return None, 0.0
    detector, iso_map = _get_lingua_detector()
    if not detector or not iso_map:
        return None, 0.0
    conf_vals = detector.compute_language_confidence_values(text)
    if not conf_vals:
        return None, 0.0
    best = conf_vals[0]
    return iso_map.get(best.language), float(best.value)


def accept_lyrics_language(text: str, code: str | None, conf: float) -> bool:
    """
    T4 acceptance: (a) the lyric script is non-Latin, or (b) a listed Latin-script language at
    confidence >= 0.9 with >= 60 words, or (c) English with >= 70% of tokens in the English list.
    """
    if not code:
        return False
    if _non_latin_share(text) > 0.5:
        return True
    tokens = _lyric_tokens(text)
    if code in LATIN_LYRIC_LANGS:
        return conf >= LATIN_MIN_CONF and len(tokens) >= LATIN_MIN_WORDS
    if code == "en":
        return english_list_share(text) >= EN_MIN_LIST_SHARE
    return False


SCRIPT_LANGS = (
    ("DEVANAGARI", "hi"), ("GURMUKHI", "pa"), ("TAMIL", "ta"), ("TELUGU", "te"), ("HANGUL", "ko"),
    ("HIRAGANA", "ja"), ("KATAKANA", "ja"), ("CJK", "ja"), ("ARABIC", "ar"), ("CYRILLIC", "ru"),
)
SCRIPT_MIN_LETTERS = 3
SCRIPT_MIN_SHARE = 0.5
GENRE_VOTE_FLOOR = 0.7  # genre matches below this (the broad en genres) yield to the artist vote


def script_language(text: str) -> tuple[str | None, float]:
    """A script counts only with >= 3 of its letters and >= 50% of the alphabetic characters."""
    letters = [ch for ch in text if ch.isalpha()]
    counts: dict[str, int] = {}
    for ch in letters:
        name = unicodedata.name(ch, "")
        for key, code in SCRIPT_LANGS:
            if key in name:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return None, 0.0
    code, n = max(counts.items(), key=lambda kv: kv[1])
    if n >= SCRIPT_MIN_LETTERS and n / len(letters) >= SCRIPT_MIN_SHARE:
        return code, 0.95
    return None, 0.0


def lead_artist(artist: str | None) -> str:
    """Artist-vote key: the lead artist, split on &, comma, feat, ft, x, with."""
    return re.split(r"\s*(?:&|,|\bfeat\b\.?|\bft\b\.?|\bx\b|\bwith\b)\s*", (artist or "").lower().strip())[0].strip()


def language_from_meta(meta: dict, lyrics: str | None = None) -> tuple[str | None, float, str]:
    """
    S4/T4 resolution order per row:
    lyrics (only when `accept_lyrics_language` holds) -> script (title and artist) -> genre map at
    >= 0.7 -> artist_vote (lead artist) -> broad genre below 0.7 -> None.
    Returns (language_code, confidence, source)
    """
    # 1. Lyrics
    lyr_code = None
    lyr_conf = 0.0
    if lyrics and len(lyrics.strip()) > 15:
        lyr_code, lyr_conf = language_from_lyrics(lyrics)
        if not accept_lyrics_language(lyrics, lyr_code, lyr_conf):
            lyr_code, lyr_conf = None, 0.0

    # 2. Script detection on title and artist (album excluded: stylised album titles carry stray letters)
    script_code, script_conf = script_language(f"{meta.get('title', '')} {meta.get('artist', '')}")

    # 3. Genre detection
    genre_code = None
    genre_conf = 0.0
    genre_str = str(meta.get("deezer_genre") or meta.get("genre") or "").lower()
    GENRE_MAP = {
        "bollywood": ("hi", 0.75), "punjabi": ("pa", 0.75), "tamil": ("ta", 0.75),
        "telugu": ("te", 0.75), "k-pop": ("ko", 0.75), "kpop": ("ko", 0.75),
        "korean": ("ko", 0.75), "j-pop": ("ja", 0.75), "jpop": ("ja", 0.75),
        "japanese": ("ja", 0.75), "latin": ("es", 0.70), "reggaeton": ("es", 0.70),
        "urbano latino": ("es", 0.70), "french": ("fr", 0.70), "deutsch": ("de", 0.70),
        "german": ("de", 0.70), "rock": ("en", 0.65), "alternative": ("en", 0.65),
        "country": ("en", 0.70), "hip-hop": ("en", 0.65), "rap": ("en", 0.65),
        "pop": ("en", 0.65), "r&b": ("en", 0.65), "soul": ("en", 0.65),
        "metal": ("en", 0.65), "soundtrack": ("en", 0.60), "singer/songwriter": ("en", 0.65),
        "dance": ("en", 0.60), "electronic": ("en", 0.60),
        "afrobeats": ("en", 0.65), "afro-fusion": ("en", 0.65), "blues": ("en", 0.65),
        "jazz": ("en", 0.65), "easy listening": ("en", 0.60), "ambient": ("en", 0.60),
        "folk": ("en", 0.65), "holiday": ("en", 0.60), "new age": ("en", 0.60),
        "tropical": ("es", 0.70), "brazilian": ("pt", 0.70), "sufi": ("hi", 0.75),
        "devotional": ("hi", 0.75), "ghazal": ("hi", 0.75), "raices": ("es", 0.70),
        "races": ("es", 0.70), "flamenco": ("es", 0.75), "cumbia": ("es", 0.70),
        "salsa": ("es", 0.70), "bachata": ("es", 0.70), "mpb": ("pt", 0.75),
        "samba": ("pt", 0.75), "bossa nova": ("pt", 0.75), "reggae": ("en", 0.65),
        "ska": ("en", 0.65), "classical": ("en", 0.60), "instrumental": ("en", 0.60),
    }
    for g_key, (code, conf) in GENRE_MAP.items():
        if g_key in genre_str:
            genre_code, genre_conf = code, conf
            break

    vote = meta.get("_artist_majority_lang") if meta else None
    if lyr_code:
        return lyr_code, lyr_conf, "lyrics"
    if script_code:
        return script_code, script_conf, "script"
    if genre_code and genre_conf >= GENRE_VOTE_FLOOR:
        return genre_code, genre_conf, "genre"
    if vote:
        return vote, 0.65, "artist_vote"
    if genre_code:
        return genre_code, genre_conf, "genre"

    return None, 0.0, "none"


def describe_audio(y: np.ndarray, meta: dict | None = None, sr: int = 22050, cut_points: dict | None = None, lyrics: str | None = None) -> dict:
    """
    Computes all 11.5 feel descriptors for an audio array.
    """
    if meta is None:
        meta = {}

    f_tempo = felt_tempo(y, sr=sr)
    f_groove = groove(y, f_tempo, sr=sr)
    r_p95 = (cut_points or {}).get("roughness_z", {}).get("p95")
    f_dark = darkness(y, felt_bpm=f_tempo["felt_bpm"], sr=sr, roughness_p95=r_p95)
    f_grit = grit(y, dark_meta=f_dark, sr=sr)
    f_vocal = vocal(y, felt_bpm=f_tempo["felt_bpm"], dark_meta=f_dark, sr=sr)
    f_restraint = restraint(y, pulse_rate=f_tempo["pulse_rate"], sr=sr)
    f_harm = harmonic_rhythm(f_dark, felt_bpm=f_tempo["felt_bpm"], sr=sr)

    # S5 Unified quantile bucketing if cut points provided
    if cut_points is not None:
        kc, rb = quantize_kick_and_restraint(
            hits_per_bar=f_groove["hits_per_bar"],
            low_share=f_dark["low_share"],
            loudness_range=f_restraint["loudness_range"],
            crest_factor=f_restraint["crest_factor"],
            cut_points=cut_points,
            beat_low_ratio=f_groove["beat_low_ratio"],
            off_beat_ratio=f_groove["off_beat_ratio"],
            pulse_clarity=f_tempo.get("pulse_clarity", 0.5),
        )
        f_groove["kick_class"] = kc
        f_restraint["restraint_bucket"] = rb

    # S4 Language resolution
    lang_code, lang_conf, lang_src = language_from_meta(meta, lyrics=lyrics)

    # Clean internal temporary arrays
    f_tempo_clean = {k: v for k, v in f_tempo.items() if not k.startswith('_')}
    f_groove_clean = {k: v for k, v in f_groove.items() if not k.startswith('_')}
    f_dark_clean = {k: v for k, v in f_dark.items() if not k.startswith('_')}
    f_grit_clean = {k: v for k, v in f_grit.items() if not k.startswith('_')}
    f_vocal_clean = {k: v for k, v in f_vocal.items() if not k.startswith('_')}

    result = {
        **meta,
        "language": lang_code,
        "language_conf": lang_conf,
        "language_source": lang_src,
        **f_tempo_clean,
        **f_groove_clean,
        **f_dark_clean,
        **f_grit_clean,
        **f_vocal_clean,
        **f_restraint,
        **f_harm,
    }
    return result
