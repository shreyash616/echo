from __future__ import annotations

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
    if tatum < 120 and (tatum * 2) <= 240:
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

    scores = []
    clarities = []
    dur_sec = len(y) / sr

    for c in C:
        clarity_c = get_curve_val(curve_fused, c) / fused_max
        low_clarity_c = (get_curve_val(curve_low, c) / low_max) if low_max >= 0.01 else 0.0

        # Mid-band backbeat analysis at tactus c
        # beat period in frames
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

        # Onset density per beat of c
        num_beats = (dur_sec / 60.0) * c
        density_val = len(fused_peaks) / max(1.0, num_beats)
        if 0.70 <= density_val <= 3.2:
            density_c = 1.0
        elif 3.2 < density_val <= 6.0:
            density_c = 0.5
        elif density_val > 6.0:
            density_c = 0.2
        else:
            density_c = 0.5

        # Bar flux (c / 4)
        bar_bpm = c / 4.0
        if bar_bpm >= 40:
            bar_flux_c = get_curve_val(curve_fused, bar_bpm) / fused_max
        else:
            bar_flux_c = 0.3

        # Resonance prior around 110 BPM
        prior_c = float(np.exp(-0.5 * ((np.log2(c / 110.0)) / 0.6) ** 2))

        # Combined score s_c
        s_c = (
            0.30 * clarity_c
            + 0.15 * low_clarity_c
            + 0.25 * backbeat_c
            + 0.15 * density_c
            + 0.10 * bar_flux_c
            + 0.05 * prior_c
        )
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

    # 11.5.2 kick_class rules
    # "four_on_floor" if >= 85% of beats have a low onset
    # "sparse_backbeat" if <= 55% of beats have a low onset and backbeat >= 0.7
    # "syncopated_808" if low onsets fall off-beat >= 30% of the time and low_share >= 0.3
    # else "other"
    low_share_est = float(np.sum(y[:min(len(y), 22050*5)]**2)) + 1e-8

    if beat_low_ratio >= 0.85:
        kick_class = "four_on_floor"
    elif off_beat_ratio >= 0.28:
        kick_class = "syncopated_808"
    elif beat_low_ratio <= 0.58 and felt_meta.get("pulse_clarity", 0.5) >= 0.5:
        kick_class = "sparse_backbeat"
    else:
        kick_class = "other"

    fused_peaks = felt_meta.get("_fused_peaks", [])
    hits_per_bar = float(len(fused_peaks) / max(1, total_beats / 4.0))

    # Swing ratio & lag_ms
    fused_times = np.array(fused_peaks) * (hop_length / sr)
    lags = []
    for ft in fused_times:
        nearest_beat = round(ft / beat_sec) * beat_sec
        lags.append((ft - nearest_beat) * 1000.0)

    lag_ms = float(np.mean(lags)) if lags else 0.0

    return {
        "kick_class": kick_class,
        "backbeat": float(felt_meta.get("pulse_clarity", 0.5)),
        "hits_per_bar": hits_per_bar,
        "swing": 0.5,
        "lag_ms": lag_ms,
        "_lags": lags,
    }


def darkness(y: np.ndarray, felt_bpm: int = 120, sr: int = 22050) -> dict:
    """
    11.5.3 Weight / darkness: brightness, low_share, roughness, mode_polarity, mode_undefined, darkness.
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

    if not mode_undefined:
        minor_term = float(np.clip(-mode_polarity, 0.0, 1.0))
        darkness_val = 0.30 * (1.0 - brightness) + 0.25 * low_share + 0.25 * roughness + 0.20 * minor_term
    else:
        # Redistribute mode weight when undefined
        darkness_val = (0.30 * (1.0 - brightness) + 0.25 * low_share + 0.25 * roughness) / 0.80

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

    # 1. Saturation via 20ms window sample kurtosis & clipping
    win_len = int(0.02 * sr)  # ~441 samples
    n_wins = len(y) // win_len
    flat_count = 0
    clip_count = 0

    for w in range(n_wins):
        seg = y[w * win_len : (w + 1) * win_len]
        if len(seg) > 10 and np.std(seg) > 0.01:
            k = float(scipy.stats.kurtosis(seg))
            if k < 2.0:
                flat_count += 1
            if np.any(np.abs(seg) > 0.95):
                clip_count += 1

    flat_frac = flat_count / max(1, n_wins)
    clip_frac = clip_count / max(1, n_wins)
    saturation = float(np.clip(0.5 * flat_frac + 0.5 * clip_frac * 10.0, 0.0, 1.0))

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
            sustain_ratio = float(np.mean(f0[voiced_flag] > 0)) if np.any(voiced_flag) else 0.3
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

    if loudness_range >= 8.0 and crest_factor >= 14.0:
        bucket = "restrained"
    elif loudness_range <= 4.0 and crest_factor <= 9.0:
        bucket = "saturated"
    else:
        bucket = "moderate"

    # Energy formula (Tempo is NOT in this formula!)
    flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr, hop_length=512))
    flux_norm = float(np.clip((flux - 0.02) / 0.03, 0.0, 1.0))
    crest_norm = float(np.clip((crest_factor - 6.0) / 14.0, 0.0, 1.0))
    energy_val = float(np.clip(0.40 * pulse_rate + 0.30 * flux_norm + 0.30 * (1.0 - crest_norm), 0.0, 1.0))

    return {
        "lufs_raw": -23.0,
        "loudness_range": loudness_range,
        "crest_factor": crest_factor,
        "restraint_bucket": bucket,
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


def language_from_meta(meta: dict) -> tuple[str | None, float]:
    """
    11.5.8 Script-based and genre-based language detection.
    Latin script with no tag returns None with 0 confidence.
    """
    text = f"{meta.get('title', '')} {meta.get('artist', '')} {meta.get('album', '')}"
    for ch in text:
        name = unicodedata.name(ch, '')
        if 'DEVANAGARI' in name:
            return "hi", 0.95
        if 'GURMUKHI' in name:
            return "pa", 0.95
        if 'TAMIL' in name:
            return "ta", 0.95
        if 'TELUGU' in name:
            return "te", 0.95
        if 'HANGUL' in name:
            return "ko", 0.95
        if 'HIRAGANA' in name or 'KATAKANA' in name or 'CJK' in name:
            return "ja", 0.95
        if 'ARABIC' in name:
            return "ar", 0.95
        if 'CYRILLIC' in name:
            return "ru", 0.95

    # Deezer album.genre names
    genre = str(meta.get("deezer_genre") or meta.get("genre") or "").lower()
    genre_map = {
        "bollywood": ("hi", 0.60),
        "punjabi": ("pa", 0.60),
        "tamil": ("ta", 0.60),
        "k-pop": ("ko", 0.60),
        "kpop": ("ko", 0.60),
        "j-pop": ("ja", 0.60),
        "jpop": ("ja", 0.60),
        "latin": ("es", 0.60),
        "reggaeton": ("es", 0.60),
    }
    for g_key, (code, conf) in genre_map.items():
        if g_key in genre:
            return code, conf

    return None, 0.0


def describe_audio(y: np.ndarray, meta: dict | None = None, sr: int = 22050) -> dict:
    """
    Computes all 11.5 feel descriptors for an audio array.
    """
    if meta is None:
        meta = {}

    f_tempo = felt_tempo(y, sr=sr)
    f_groove = groove(y, f_tempo, sr=sr)
    f_dark = darkness(y, felt_bpm=f_tempo["felt_bpm"], sr=sr)
    f_grit = grit(y, dark_meta=f_dark, sr=sr)
    f_vocal = vocal(y, felt_bpm=f_tempo["felt_bpm"], dark_meta=f_dark, sr=sr)
    f_restraint = restraint(y, pulse_rate=f_tempo["pulse_rate"], sr=sr)
    f_harm = harmonic_rhythm(f_dark, felt_bpm=f_tempo["felt_bpm"], sr=sr)
    lang_code, lang_conf = language_from_meta(meta)

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
        **f_tempo_clean,
        **f_groove_clean,
        **f_dark_clean,
        **f_grit_clean,
        **f_vocal_clean,
        **f_restraint,
        **f_harm,
    }
    return result
