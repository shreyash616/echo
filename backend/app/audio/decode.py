from __future__ import annotations

import subprocess
import numpy as np
import pyloudnorm as pyln

FFMPEG_ARGS = [
    "ffmpeg",
    "-nostdin",
    "-loglevel",
    "error",
    "-i",
    "pipe:0",
    "-ac",
    "1",
    "-ar",
    "22050",
    "-f",
    "f32le",
    "-acodec",
    "pcm_f32le",
    "pipe:1",
]


def decode_preview(audio_bytes: bytes) -> np.ndarray:
    """
    Decodes audio bytes via ffmpeg, returns mono float32 at 22050 Hz,
    loudness-normalised to -23 LUFS with pyloudnorm, peak-limited to 0.99,
    trimmed to the first 30.0 s, zero-padded if shorter.
    """
    if not audio_bytes:
        return np.zeros(22050 * 30, dtype=np.float32)

    proc = subprocess.Popen(
        FFMPEG_ARGS,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate(input=audio_bytes)
    if proc.returncode != 0 or not stdout:
        return np.zeros(22050 * 30, dtype=np.float32)

    audio = np.frombuffer(stdout, dtype=np.float32).copy()
    target_len = 22050 * 30  # 30.0 seconds

    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    meter = pyln.Meter(22050)
    try:
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness) and loudness > -70.0:
            audio = pyln.normalize.loudness(audio, loudness, -23.0)
    except Exception:
        pass

    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    return audio.astype(np.float32)
