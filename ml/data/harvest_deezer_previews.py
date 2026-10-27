"""
Raw Audio Preview Harvester for Echo (Zero External Pre-Computed Analytics).

Downloads raw 30-second chorus preview MP3s from the free public Deezer API
across diverse genres and languages, converts each raw audio preview into a
128-bin mel spectrogram (.npy) using Echo's fixed-floor preprocessing, and
writes a catalog metadata file (`deezer_catalog.json`) containing ONLY basic
catalog identity (id, title, artist, album, albumArtUrl, previewUrl, durationMs).

All embeddings, BPM, musical key, energy, valence, and vibe tags are then
computed 100% by Echo's own ML encoder and signal processing pipeline during
`train.py` and `build_index.py`.

Usage:
    python ml/data/harvest_deezer_previews.py \
        --output-specs ml/data/deezer_spectrograms \
        --output-json  ml/data/deezer_catalog.json \
        --tracks-per-query 50
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

DEEZER_BASE = "https://api.deezer.com"
SR          = 22_050
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
TOP_DB      = 80.0

def _get_ffmpeg_cmd() -> str:
    which = shutil.which("ffmpeg")
    if which:
        return which
    winget_paths = glob.glob(
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\**\ffmpeg.exe"),
        recursive=True,
    )
    if winget_paths:
        return winget_paths[0]
    return "ffmpeg"

# Curated multi-genre & multi-language seed queries to build a diverse mainstream catalog (Phase 4.2)
def _load_artist_seeds() -> list[tuple[str, str]]:
    p = Path(__file__).parent / "harvest_seed_artists.json"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                artists = json.load(f)
            return [(a, "seed") for a in artists]
        except Exception:
            pass
    return []

DEFAULT_SEED_QUERIES: list[tuple[str, str]] = _load_artist_seeds() + [
    # Rock / Grunge / Blues-Rock / Southern Rock
    ("classic rock", "rock"), ("southern rock", "rock"), ("blues rock", "rock"),
    ("grunge", "grunge"), ("garage rock", "rock"), ("psych rock", "psych"),
    ("alternative rock", "rock"), ("indie rock", "indie"), ("post punk", "rock"),
    ("britpop", "rock"), ("shoegaze", "indie"), ("dream pop", "indie"),
    ("metalcore", "metal"), ("hard rock", "rock"),
    # Folk / Acoustic / Alt-Country
    ("acoustic indie", "acoustic"), ("folk acoustic", "folk"), ("singer songwriter", "acoustic"),
    ("alt country", "country"), ("indie folk", "folk"), ("dark acoustic ballad", "folk"),
    # Hip-Hop / R&B / Soul / Trip-Hop
    ("hip hop", "hip-hop"), ("trap beats", "trap"), ("melodic rap", "hip-hop"),
    ("r&b soul", "r&b"), ("neo soul", "soul"), ("boom bap", "hip-hop"),
    ("trip hop", "trip-hop"), ("downtempo electronic", "downtempo"),
    # Electronic / Dance / Ambient
    ("house music", "house"), ("techno", "techno"), ("drum and bass", "dnb"),
    ("edm festival", "edm"), ("lo-fi beats", "lo-fi"), ("ambient chill", "ambient"),
    ("synthwave", "synthwave"), ("phonk", "phonk"),
    # Global / Latin / Cumbia / Reggae
    ("latin alt", "latin"), ("cumbia", "latin"), ("reggae", "reggae"),
    ("reggaeton hits", "latin"), ("afrobeats", "afrobeats"), ("french pop", "french"),
    # South Asian (Hindi / Punjabi / Tamil / Telugu / Ghazal / Sufi)
    ("bollywood hits", "hindi"), ("bollywood 70s", "hindi"), ("hindi indie", "hindi"),
    ("ghazal", "hindi"), ("sufi", "hindi"), ("hindi romantic", "hindi"),
    ("pritam bollywood", "hindi"), ("punjabi hits", "punjabi"), ("punjabi pop", "punjabi"),
    ("punjabi trap", "punjabi"), ("tamil hits", "tamil"), ("tamil 90s", "tamil"),
    ("anirudh ravichander", "tamil"), ("telugu hits", "telugu"),
    # East Asian
    ("k-pop hits", "k-pop"), ("k-indie", "k-pop"), ("j-pop anime", "j-pop"),
    ("j-rock", "j-rock"), ("city pop", "pop"),
    # Instrumental / Jazz / Classical
    ("jazz instrumental", "jazz"), ("piano solo", "classical"), ("cinematic soundtrack", "soundtrack"),
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.deezer.com/",
}


def _decode_mp3_bytes_to_mel(audio_bytes: bytes) -> np.ndarray | None:
    """Decode raw MP3/AAC preview bytes via PyAV or ffmpeg and extract fixed-floor mel spectrogram."""
    y = None
    # Method 1: PyAV (fast, in-process, no external ffmpeg required)
    try:
        import av
        container = av.open(io.BytesIO(audio_bytes))
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SR)
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        if chunks:
            y = np.concatenate(chunks, axis=1).squeeze()
    except Exception:
        y = None

    # Method 2: ffmpeg subprocess fallback
    if y is None or len(y) < SR * 5:
        try:
            ffmpeg_cmd = _get_ffmpeg_cmd()
            proc = subprocess.run(
                [
                    ffmpeg_cmd, "-i", "pipe:0",
                    "-f", "wav", "-ar", str(SR), "-ac", "1",
                    "-loglevel", "error",
                    "pipe:1",
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=15,
            )
            if proc.returncode == 0:
                y_sub, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
                if len(y_sub) >= SR * 5:
                    y = y_sub
        except Exception:
            pass

    if y is None or len(y) < SR * 5 or float(np.max(np.abs(y))) < 1e-4:
        return None

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    mel_norm = np.clip((mel_db + TOP_DB) / TOP_DB, 0.0, 1.0)
    return mel_norm.astype(np.float32)


def _fetch_tracks_for_query(client: httpx.Client, query: str, genre: str, limit: int) -> list[dict]:
    """Query Deezer search API with pagination up to `limit` tracks, with iTunes fallback."""
    collected: list[dict] = []
    index = 0
    page_size = min(50, limit)
    while len(collected) < limit:
        try:
            resp = client.get(
                f"{DEEZER_BASE}/search",
                params={"q": query, "limit": page_size, "index": index},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                break
            for item in data:
                preview = item.get("preview")
                if not preview:
                    continue
                artist = item.get("artist", {})
                album = item.get("album", {})
                art = album.get("cover_xl") or album.get("cover_medium") or album.get("cover", "")
                collected.append({
                    "id":          str(item["id"]),
                    "title":       item.get("title", ""),
                    "artist":      artist.get("name", ""),
                    "album":       album.get("title", ""),
                    "albumArtUrl": art,
                    "previewUrl":  preview,
                    "durationMs":  int(item.get("duration", 30)) * 1000,
                    "genre":       genre,
                })
                if len(collected) >= limit:
                    break
            index += len(data)
            if len(data) < page_size:
                break
            time.sleep(0.15)
        except Exception as exc:
            logger.debug("Deezer query %r failed at index=%d: %s", query, index, exc)
            break

    # If Deezer returned no previews (e.g. region-restricted), revert to iTunes Search API
    if len(collected) < limit:
        try:
            needed = limit - len(collected)
            resp = client.get(
                "https://itunes.apple.com/search",
                params={"term": query, "entity": "song", "limit": min(needed, 50)},
                timeout=10.0,
            )
            if resp.status_code == 200:
                for item in resp.json().get("results", []):
                    preview = item.get("previewUrl")
                    if not preview:
                        continue
                    art = (item.get("artworkUrl100") or "").replace("100x100bb", "600x600bb")
                    collected.append({
                        "id":          f"it_{item['trackId']}",
                        "title":       item.get("trackName", ""),
                        "artist":      item.get("artistName", ""),
                        "album":       item.get("collectionName", ""),
                        "albumArtUrl": art,
                        "previewUrl":  preview,
                        "durationMs":  int(item.get("trackTimeMillis", 30000)),
                        "genre":       genre,
                    })
                    if len(collected) >= limit:
                        break
        except Exception as exc:
            logger.debug("iTunes fallback query %r failed: %s", query, exc)

    return collected


def _download_and_process(item: dict, specs_dir: Path, fp16: bool) -> dict | None:
    """Download a single track's 30s preview MP3 and save its .npy mel spectrogram."""
    tid = item["id"]
    out_path = specs_dir / f"{tid}.npy"
    if out_path.exists():
        return item

    preview_url = item.get("previewUrl")
    if not preview_url:
        return None

    try:
        with httpx.Client(headers=_HEADERS, timeout=15.0) as client:
            resp = client.get(preview_url)
            resp.raise_for_status()
            audio_bytes = resp.content
    except Exception:
        return None

    mel = _decode_mp3_bytes_to_mel(audio_bytes)
    if mel is None:
        return None

    np.save(out_path, mel.astype(np.float16) if fp16 else mel)
    return item


def main() -> None:
    p = argparse.ArgumentParser(
        description="Harvest raw 30s chorus previews from Deezer and extract mel spectrograms"
    )
    p.add_argument("--output-specs",     default="ml/data/deezer_spectrograms",
                   help="Directory to save .npy mel spectrograms")
    p.add_argument("--output-json",      default="ml/data/deezer_catalog.json",
                   help="Path to save catalog track metadata JSON")
    p.add_argument("--tracks-per-query", type=int, default=50,
                   help="Max tracks to harvest per seed genre/artist query")
    p.add_argument("--workers",          type=int, default=8,
                   help="Concurrent download + spectrogram worker threads")
    p.add_argument("--fp16",             action="store_true",
                   help="Store spectrograms as float16")
    args = p.parse_args()

    specs_dir = Path(args.output_specs)
    specs_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing catalog if resuming
    catalog_by_id: dict[str, dict] = {}
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                for entry in json.load(f):
                    catalog_by_id[str(entry["id"])] = entry
            logger.info("Loaded %d existing tracks from %s", len(catalog_by_id), json_path)
        except Exception:
            pass

    logger.info("Searching Deezer across %d seed queries…", len(DEFAULT_SEED_QUERIES))
    candidates: dict[str, dict] = dict(catalog_by_id)
    with httpx.Client(headers=_HEADERS, timeout=10.0) as client:
        for query, genre in tqdm(DEFAULT_SEED_QUERIES, desc="Discovering tracks"):
            found = _fetch_tracks_for_query(client, query, genre, args.tracks_per_query)
            for t in found:
                candidates.setdefault(t["id"], t)

    logger.info("Discovered %d unique tracks with preview URLs. Extracting spectrograms…", len(candidates))

    valid_catalog: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_and_process, item, specs_dir, args.fp16): item["id"]
            for item in candidates.values()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Harvesting & Preprocessing"):
            res = fut.result()
            if res is not None:
                valid_catalog.append(res)

    valid_catalog.sort(key=lambda x: x["id"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(valid_catalog, f, ensure_ascii=False, indent=2)

    logger.info(
        "Harvest complete! Saved %d spectrograms to %s and metadata to %s",
        len(valid_catalog), specs_dir, json_path,
    )
    logger.info(
        "Next steps:\n"
        "  1. Train: python ml/training/train.py --spec-dir %s --output ml/checkpoints\n"
        "  2. Export: python ml/inference/export_onnx.py --checkpoint ml/checkpoints/best_model.pt\n"
        "  3. Index: python ml/inference/build_index.py --spec-dir %s --catalog-json %s",
        specs_dir, specs_dir, json_path,
    )


if __name__ == "__main__":
    main()