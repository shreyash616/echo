from __future__ import annotations

import io
import json
import os
import wave
import numpy as np
import scipy.signal

SR = 22050
DUR = 30.0  # 30.0 seconds
TOTAL_SAMPLES = int(SR * DUR)


def save_wav(path: str, samples: np.ndarray, sr: int = 22050):
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())


def make_click_72_backbeat() -> np.ndarray:
    y = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    beat_sec = 60.0 / 72.0
    num_beats = int(DUR / beat_sec)

    np.random.seed(42)
    for b in range(num_beats):
        t_b = b * beat_sec
        idx = int(t_b * SR)
        # Kick on beats 1 and 3 (0 and 2 mod 4) -> 50% of beats have low onset
        if b % 2 == 0:
            k_len = int(0.08 * SR)
            if idx + k_len < TOTAL_SAMPLES:
                t_k = np.linspace(0, 0.08, k_len)
                y[idx : idx + k_len] += 0.85 * np.sin(2 * np.pi * 60 * t_k) * np.exp(-t_k / 0.03)

        # Snare on beats 2 and 4 (1 and 3 mod 4)
        if b % 2 == 1:
            s_len = int(0.06 * SR)
            if idx + s_len < TOTAL_SAMPLES:
                noise = np.random.randn(s_len)
                sos = scipy.signal.butter(4, [1500, 2500], btype='bandpass', fs=SR, output='sos')
                s_filt = scipy.signal.sosfilt(sos, noise)
                t_s = np.linspace(0, 0.06, s_len)
                y[idx : idx + s_len] += 0.70 * s_filt * np.exp(-t_s / 0.02)

        # Hats on eighths (0.0 and beat_sec / 2.0)
        for off in [0.0, beat_sec / 2.0]:
            h_idx = int((t_b + off) * SR)
            h_len = int(0.02 * SR)
            if h_idx + h_len < TOTAL_SAMPLES:
                noise_h = np.random.randn(h_len)
                sos_h = scipy.signal.butter(4, 7000, btype='highpass', fs=SR, output='sos')
                h_filt = scipy.signal.sosfilt(sos_h, noise_h)
                t_h = np.linspace(0, 0.02, h_len)
                y[h_idx : h_idx + h_len] += 0.25 * h_filt * np.exp(-t_h / 0.008)

    return y


def make_click_144_fourfloor() -> np.ndarray:
    y = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    beat_sec = 60.0 / 144.0
    num_beats = int(DUR / beat_sec)

    np.random.seed(43)
    for b in range(num_beats):
        t_b = b * beat_sec
        idx = int(t_b * SR)
        # Kick on every beat
        k_len = int(0.06 * SR)
        if idx + k_len < TOTAL_SAMPLES:
            t_k = np.linspace(0, 0.06, k_len)
            y[idx : idx + k_len] += 0.85 * np.sin(2 * np.pi * 65 * t_k) * np.exp(-t_k / 0.025)

        # Hats on eighths, no snare
        for off in [0.0, beat_sec / 2.0]:
            h_idx = int((t_b + off) * SR)
            h_len = int(0.015 * SR)
            if h_idx + h_len < TOTAL_SAMPLES:
                noise_h = np.random.randn(h_len)
                sos_h = scipy.signal.butter(4, 7000, btype='highpass', fs=SR, output='sos')
                h_filt = scipy.signal.sosfilt(sos_h, noise_h)
                t_h = np.linspace(0, 0.015, h_len)
                y[h_idx : h_idx + h_len] += 0.25 * h_filt * np.exp(-t_h / 0.006)

    return y


def make_trap_96() -> np.ndarray:
    y = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    beat_sec = 60.0 / 96.0
    num_bars = int(DUR / (beat_sec * 4.0))

    np.random.seed(44)
    for bar in range(num_bars):
        bar_start = bar * beat_sec * 4.0

        # Kick on 1 and the "and" of 2 (offsets: 0.0, 1.5 * beat_sec)
        for k_off in [0.0, 1.5 * beat_sec]:
            idx = int((bar_start + k_off) * SR)
            decay_len = int(0.40 * SR)  # 50 Hz 808 with 400ms decay
            if idx + decay_len < TOTAL_SAMPLES:
                t_k = np.linspace(0, 0.40, decay_len)
                y[idx : idx + decay_len] += 1.0 * np.sin(2 * np.pi * 50 * t_k) * np.exp(-t_k / 0.15)

        # Snare on 2 and 4 (1.0 * beat_sec and 3.0 * beat_sec)
        for s_off in [1.0 * beat_sec, 3.0 * beat_sec]:
            s_idx = int((bar_start + s_off) * SR)
            s_len = int(0.07 * SR)
            if s_idx + s_len < TOTAL_SAMPLES:
                noise = np.random.randn(s_len)
                sos = scipy.signal.butter(4, [1500, 3000], btype='bandpass', fs=SR, output='sos')
                s_filt = scipy.signal.sosfilt(sos, noise)
                t_s = np.linspace(0, 0.07, s_len)
                y[s_idx : s_idx + s_len] += 0.70 * s_filt * np.exp(-t_s / 0.02)

        # Hats at 16ths (offsets: 0, 0.25, 0.5, 0.75, ... up to 4.0 beats)
        hat_16ths = np.arange(0, 4.0, 0.25) * beat_sec
        for h_off in hat_16ths:
            h_idx = int((bar_start + h_off) * SR)
            h_len = int(0.012 * SR)
            if h_idx + h_len < TOTAL_SAMPLES:
                noise_h = np.random.randn(h_len)
                sos_h = scipy.signal.butter(4, 7500, btype='highpass', fs=SR, output='sos')
                h_filt = scipy.signal.sosfilt(sos_h, noise_h)
                t_h = np.linspace(0, 0.012, h_len)
                y[h_idx : h_idx + h_len] += 0.25 * h_filt * np.exp(-t_h / 0.005)

    return y


def make_major_triad_pad() -> np.ndarray:
    # C major triad: C4 (261.63), E4 (329.63), G4 (392.00) sines sustained
    t = np.linspace(0, DUR, TOTAL_SAMPLES, endpoint=False)
    y = 0.33 * np.sin(2 * np.pi * 261.63 * t) + 0.33 * np.sin(2 * np.pi * 329.63 * t) + 0.33 * np.sin(2 * np.pi * 392.00 * t)
    return y.astype(np.float32)


def make_minor_triad_pad() -> np.ndarray:
    # C minor triad: C4 (261.63), Eb4 (311.13), G4 (392.00)
    t = np.linspace(0, DUR, TOTAL_SAMPLES, endpoint=False)
    y = 0.33 * np.sin(2 * np.pi * 261.63 * t) + 0.33 * np.sin(2 * np.pi * 311.13 * t) + 0.33 * np.sin(2 * np.pi * 392.00 * t)
    return y.astype(np.float32)


def make_power_chord_pad() -> np.ndarray:
    # Power chord: C3 (130.81), G3 (196.00), C4 (261.63) (no 3rd)
    t = np.linspace(0, DUR, TOTAL_SAMPLES, endpoint=False)
    y = 0.35 * np.sin(2 * np.pi * 130.81 * t) + 0.35 * np.sin(2 * np.pi * 196.00 * t) + 0.30 * np.sin(2 * np.pi * 261.63 * t)
    return y.astype(np.float32)


def make_pink_quiet() -> np.ndarray:
    # Pink noise with dynamics (high crest factor and loudness range >= 8 LU)
    np.random.seed(45)
    white = np.random.randn(TOTAL_SAMPLES)
    b, a = scipy.signal.butter(1, 0.05, btype='low')
    pink = scipy.signal.lfilter(b, a, white)
    pink = pink / (np.max(np.abs(pink)) + 1e-8)

    # Modulate envelope across 3s chunks: quiet periods vs louder bursts
    mod = np.ones(TOTAL_SAMPLES, dtype=np.float32)
    chunk = int(3.0 * SR)
    for i in range(0, TOTAL_SAMPLES, chunk * 2):
        mod[i : min(TOTAL_SAMPLES, i + chunk)] *= 0.15  # 16 dB quieter

    # Add sparse short spikes every 2s for high crest factor
    spikes = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    spike_interval = int(2.0 * SR)
    for s in range(0, TOTAL_SAMPLES, spike_interval):
        spikes[s : min(TOTAL_SAMPLES, s + int(0.005 * SR))] = 1.0

    y = pink * mod + 0.3 * spikes
    y = y / (np.max(np.abs(y)) + 1e-8) * 0.03
    return y.astype(np.float32)


def make_pink_clipped() -> np.ndarray:
    # Pink noise hard clipped at 0.3 then normalised
    np.random.seed(46)
    white = np.random.randn(TOTAL_SAMPLES)
    b, a = scipy.signal.butter(1, 0.05, btype='low')
    pink = scipy.signal.lfilter(b, a, white)
    pink = pink / (np.max(np.abs(pink)) + 1e-8)
    # hard-clipped at 0.3
    clipped = np.clip(pink, -0.3, 0.3)
    return clipped.astype(np.float32)


def make_bright_and_dark() -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(47)
    white = np.random.randn(TOTAL_SAMPLES)
    # Bright: band-limited noise 2 to 8 kHz
    sos_bright = scipy.signal.butter(4, [2000, 8000], btype='bandpass', fs=SR, output='sos')
    bright = scipy.signal.sosfilt(sos_bright, white)
    bright = (bright / (np.max(np.abs(bright)) + 1e-8) * 0.7).astype(np.float32)

    # Dark: band-limited noise 60 to 400 Hz
    sos_dark = scipy.signal.butter(4, [60, 400], btype='bandpass', fs=SR, output='sos')
    dark = scipy.signal.sosfilt(sos_dark, white)
    dark = (dark / (np.max(np.abs(dark)) + 1e-8) * 0.7).astype(np.float32)
    return bright, dark


def make_chord_cycle_2s() -> np.ndarray:
    # C, F, G, C triads changing every 2s at 120 BPM (1 change per bar)
    t = np.linspace(0, DUR, TOTAL_SAMPLES, endpoint=False)
    y = np.zeros(TOTAL_SAMPLES, dtype=np.float32)

    # Chords frequencies
    chords = [
        [261.63, 329.63, 392.00],  # C
        [293.66, 369.99, 440.00],  # D
        [329.63, 415.30, 493.88],  # E
        [369.99, 466.16, 554.37],  # F#
    ]

    chunk_len = int(2.0 * SR)
    n_chunks = TOTAL_SAMPLES // chunk_len
    for i in range(n_chunks):
        c_freqs = chords[i % 4]
        t_c = np.linspace(0, 2.0, chunk_len, endpoint=False)
        sig = 0.33 * np.sin(2 * np.pi * c_freqs[0] * t_c) + 0.33 * np.sin(2 * np.pi * c_freqs[1] * t_c) + 0.33 * np.sin(2 * np.pi * c_freqs[2] * t_c)
        y[i * chunk_len : (i + 1) * chunk_len] = sig

    return y


def _pluck(freq: float, dur: float, amp: float) -> np.ndarray:
    # Plucked string: decaying harmonics, brighter partials die faster, 3 ms attack
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for h in range(1, 9):
        if freq * h < SR / 2:
            y += (1.0 / h) * np.sin(2 * np.pi * freq * h * t) * np.exp(-t * (2.5 + 1.5 * h))
    y *= np.minimum(1.0, t / 0.003)
    return amp * y / (np.max(np.abs(y)) + 1e-8)


def make_fingerpicked_90() -> np.ndarray:
    # T5 fixture: Travis-style fingerpicking at 90 BPM, no drums. Thumb alternates E2/B2 on every
    # beat; fingers pluck G#3/B3/E4 on the off-beat eighths. Expected kick_class: not four_on_floor.
    y = np.zeros(TOTAL_SAMPLES, dtype=np.float64)
    beat_sec = 60.0 / 90.0
    bass = [82.41, 123.47]
    treble = [207.65, 246.94, 329.63, 246.94]
    for b in range(int(DUR / beat_sec)):
        for t_on, f, amp in [(b * beat_sec, bass[b % 2], 0.6), (b * beat_sec + beat_sec / 2, treble[b % 4], 0.4)]:
            idx = int(t_on * SR)
            p = _pluck(f, 1.2, amp)
            end = min(TOTAL_SAMPLES, idx + len(p))
            y[idx:end] += p[: end - idx]
    return (0.8 * y / (np.max(np.abs(y)) + 1e-8)).astype(np.float32)


def build_all_fixtures():
    out_dir = "tests/fixtures/audio"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating fixture audio files...")
    f1 = make_click_72_backbeat()
    save_wav(f"{out_dir}/click_72_backbeat.wav", f1)

    f2 = make_click_144_fourfloor()
    save_wav(f"{out_dir}/click_144_fourfloor.wav", f2)

    f3 = make_trap_96()
    save_wav(f"{out_dir}/trap_96.wav", f3)

    f4 = make_major_triad_pad()
    save_wav(f"{out_dir}/major_triad_pad.wav", f4)

    f5 = make_minor_triad_pad()
    save_wav(f"{out_dir}/minor_triad_pad.wav", f5)

    f6 = make_power_chord_pad()
    save_wav(f"{out_dir}/power_chord_pad.wav", f6)

    f7 = make_pink_quiet()
    save_wav(f"{out_dir}/pink_quiet.wav", f7)

    f8 = make_pink_clipped()
    save_wav(f"{out_dir}/pink_clipped.wav", f8)

    bright, dark = make_bright_and_dark()
    save_wav(f"{out_dir}/bright.wav", bright)
    save_wav(f"{out_dir}/dark.wav", dark)

    f10 = make_chord_cycle_2s()
    save_wav(f"{out_dir}/chord_cycle_2s.wav", f10)

    f11 = make_fingerpicked_90()
    save_wav(f"{out_dir}/fingerpicked_90.wav", f11)

    print("All fixtures written to tests/fixtures/audio/")


if __name__ == "__main__":
    build_all_fixtures()
