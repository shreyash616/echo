"""
Download audio features for ~100k tracks using the Spotify API.

Strategy:
  1. Load the FMA (Free Music Archive) metadata CSV (tracks.csv).
     Download from: https://github.com/mdeff/fma  →  fma_metadata.zip
  2. For each track, look up audio features on Spotify by ISRC or title+artist.
  3. Assign genre_cluster labels via k-means on the feature space.
  4. Save to ml/data/features_with_clusters.csv.

Usage:
    python ml/data/download_fma.py \
        --fma-tracks ml/data/fma_metadata/tracks.csv \
        --output ml/data/features_with_clusters.csv \
        --spotify-client-id YOUR_ID \
        --spotify-client-secret YOUR_SECRET \
        --n-clusters 50 \
        --limit 100000
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spotipy
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

BATCH_SIZE = 100  # Spotify allows up to 100 IDs per audio-features call


def get_spotify_client(client_id: str, client_secret: str) -> spotipy.Spotify:
    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        ),
        requests_timeout=15,
        retries=3,
    )


def search_spotify_id(sp: spotipy.Spotify, title: str, artist: str) -> str | None:
    """Look up a Spotify track ID by title + artist."""
    try:
        q = f"track:{title} artist:{artist}"
        res = sp.search(q=q, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        return items[0]["id"] if items else None
    except Exception:
        return None


def fetch_audio_features(sp: spotipy.Spotify, ids: list[str]) -> list[dict]:
    """Fetch audio features for up to 100 Spotify track IDs."""
    try:
        return sp.audio_features(ids) or []
    except Exception as e:
        logger.warning("audio_features error: %s", e)
        return []


def assign_clusters(features_df: pd.DataFrame, n_clusters: int) -> np.ndarray:
    """K-means cluster assignment on normalised feature vectors."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[FEATURE_COLS].values)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096)
    labels = kmeans.fit_predict(X)
    logger.info("K-means inertia: %.2f", kmeans.inertia_)
    return labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fma-tracks", required=True, help="Path to FMA tracks.csv")
    p.add_argument("--output", default="ml/data/features_with_clusters.csv")
    p.add_argument("--spotify-client-id", required=True)
    p.add_argument("--spotify-client-secret", required=True)
    p.add_argument("--n-clusters", type=int, default=50)
    p.add_argument("--limit", type=int, default=100_000)
    args = p.parse_args()

    sp = get_spotify_client(args.spotify_client_id, args.spotify_client_secret)

    # Load FMA metadata
    # FMA tracks.csv has a multi-level header — use header=[0,1]
    logger.info("Loading FMA metadata from %s", args.fma_tracks)
    raw = pd.read_csv(args.fma_tracks, index_col=0, header=[0, 1])

    # Flatten to usable columns
    titles = raw[("track", "title")].astype(str)
    artists = raw[("artist", "name")].astype(str)
    genres = raw[("track", "genre_top")].astype(str)

    meta = pd.DataFrame({
        "fma_id": raw.index.astype(str),
        "title": titles,
        "artist": artists,
        "genre": genres,
    }).head(args.limit)
    logger.info("Processing %d tracks", len(meta))

    # Step 1: Resolve Spotify IDs
    spotify_ids: list[str | None] = []
    for i, row in meta.iterrows():
        sid = search_spotify_id(sp, row["title"], row["artist"])
        spotify_ids.append(sid)
        if (len(spotify_ids)) % 500 == 0:
            logger.info("  Resolved %d/%d Spotify IDs", len(spotify_ids), len(meta))
        time.sleep(0.02)  # Gentle rate limiting

    meta["spotify_id"] = spotify_ids
    meta = meta.dropna(subset=["spotify_id"])
    logger.info("Resolved %d/%d tracks on Spotify", len(meta), args.limit)

    # Step 2: Fetch audio features in batches
    all_features: list[dict] = []
    ids_list = meta["spotify_id"].tolist()

    for i in range(0, len(ids_list), BATCH_SIZE):
        batch = ids_list[i:i + BATCH_SIZE]
        feats = fetch_audio_features(sp, batch)
        all_features.extend([f for f in feats if f is not None])
        if (i // BATCH_SIZE) % 20 == 0:
            logger.info("  Fetched features for %d/%d tracks", len(all_features), len(ids_list))
        time.sleep(0.05)

    features_df = pd.DataFrame(all_features)
    features_df = features_df.rename(columns={"id": "spotify_id"})
    features_df = features_df.dropna(subset=FEATURE_COLS)

    # Merge with metadata
    merged = meta.merge(features_df[["spotify_id"] + FEATURE_COLS], on="spotify_id", how="inner")
    merged = merged.rename(columns={"spotify_id": "id"})
    logger.info("Merged dataset: %d tracks with complete features", len(merged))

    # Step 3: Cluster
    logger.info("Running k-means with %d clusters…", args.n_clusters)
    merged["genre_cluster"] = assign_clusters(merged, args.n_clusters)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    logger.info("Saved to %s", output_path)

    # Summary
    cluster_dist = merged["genre_cluster"].value_counts()
    logger.info("Cluster distribution (top 10):\n%s", cluster_dist.head(10).to_string())


if __name__ == "__main__":
    main()
