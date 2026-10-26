#!/bin/bash
set -e

REPO="shreyash616/echo-ml-artifacts"
DEST="/app/ml/inference"

download_file() {
    local filename=$1
    echo "[startup] Syncing $filename from HF Hub ($REPO)..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$REPO',
    filename='$filename',
    repo_type='dataset',
    local_dir='$DEST',
)
"
    local filepath="$DEST/$filename"
    if [ ! -f "$filepath" ] || [ $(wc -c < "$filepath") -lt 1000 ]; then
        echo "[startup] ERROR: $filename missing or smaller than 1000 bytes (LFS pointer / failed download)!"
        exit 1
    fi
    echo "[startup] $filename synced."
}

download_file "music_encoder.onnx"
download_file "music_index.faiss"
download_file "track_metadata.json"
download_file "archetypes.json"

exec uvicorn main:app --host 0.0.0.0 --port 7860
