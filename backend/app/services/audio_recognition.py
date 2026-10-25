"""
AcoustID-based audio fingerprinting service (open source, free).
Uses Chromaprint (fpcalc) to generate a fingerprint, then queries the AcoustID API.
Docs: https://acoustid.org/webservice

Requires: fpcalc (Chromaprint CLI) installed and on PATH.
  Linux:   sudo apt install libchromaprint-tools
  macOS:   brew install chromaprint
  Windows: download from https://acoustid.org/chromaprint
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile

import httpx

from app.config import settings

logger = logging.getLogger("echo.acoustid")

ACOUSTID_API_URL = "https://api.acoustid.org/v2/lookup"


class AcoustIDRecognizer:
    """Fingerprints audio via Chromaprint (fpcalc) and looks up via AcoustID API."""

    def _fingerprint(self, audio_bytes: bytes) -> tuple[str, int]:
        """
        Transcode audio to WAV if needed, run fpcalc, return (fingerprint, duration_seconds).
        Raises RuntimeError/FileNotFoundError if fpcalc is not installed or audio cannot be decoded.
        """
        wav_bytes = audio_bytes
        try:
            ffmpeg_proc = subprocess.run(
                [
                    "ffmpeg", "-i", "pipe:0",
                    "-f", "wav", "-ar", "16000", "-ac", "1",
                    "-loglevel", "error",
                    "pipe:1",
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=10,
            )
            if ffmpeg_proc.returncode == 0 and len(ffmpeg_proc.stdout) > 44:
                wav_bytes = ffmpeg_proc.stdout
        except Exception as e:
            logger.debug("ffmpeg pre-transcode skipped: %s", e)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["fpcalc", "-plain", tmp_path],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                raise RuntimeError(f"fpcalc failed: {result.stderr.strip()}")

            lines = result.stdout.strip().splitlines()
            if len(lines) < 2:
                raise RuntimeError("fpcalc returned unexpected output")

            duration = int(float(lines[0]))
            fingerprint = lines[1]
            return fingerprint, duration
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def recognize(self, audio_bytes: bytes) -> dict:
        """Fingerprint audio bytes and query the AcoustID API. Returns raw JSON or error dict."""
        if not settings.acoustid_api_key:
            logger.info("AcoustID skipped: acoustid_api_key is empty")
            return {"status": "skipped"}

        try:
            fingerprint, duration = await asyncio.to_thread(self._fingerprint, audio_bytes)
        except Exception as exc:
            logger.warning("Chromaprint fingerprinting skipped/failed: %s", exc)
            return {"status": "error", "error": str(exc)}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    ACOUSTID_API_URL,
                    data={
                        "client": settings.acoustid_api_key,
                        "fingerprint": fingerprint,
                        "duration": duration,
                        "meta": "recordings",
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("AcoustID lookup failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def parse_result(self, raw: dict) -> dict | None:
        """
        Returns a simplified dict with title and artist.
        Returns None if no match found.
        """
        if raw.get("status") != "ok":
            return None

        results = raw.get("results", [])
        if not results:
            return None

        # Pick the highest-confidence result that has recording metadata
        for result in sorted(results, key=lambda r: r.get("score", 0), reverse=True):
            recordings = result.get("recordings", [])
            if recordings:
                rec = recordings[0]
                artists = ", ".join(a["name"] for a in rec.get("artists", []))
                return {
                    "title": rec.get("title", "Unknown"),
                    "artist": artists or "Unknown",
                    "acoustid": result.get("id", ""),
                    "score": result.get("score", 0),
                }

        return None


recognizer = AcoustIDRecognizer()
