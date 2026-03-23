#!/bin/bash
set -e

REPO="shreyash616/echo-ml-artifacts"
DEST="/app/ml/inference"

download_if_missing() {
    local filename=$1
    local dest_path="$DEST/$filename"
    if [ ! -f "$dest_path" ]; then
        echo "[startup] Downloading $filename from HF Hub..."
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$REPO',
    filename='$filename',
    repo_type='dataset',
    local_dir='$DEST',
)
"
        echo "[startup] $filename ready."
    else
        echo "[startup] $filename already present, skipping."
    fi
}

download_if_missing "music_encoder.onnx"
download_if_missing "music_index.faiss"
download_if_missing "track_metadata.json"

exec uvicorn main:app --host 0.0.0.0 --port 7860
