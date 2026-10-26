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
    if [ ! -f "$filepath" ] || [ ! -s "$filepath" ]; then
        echo "[startup] ERROR: $filename missing or empty!"
        exit 1
    fi
    echo "[startup] $filename synced."
}

download_file "music_index.faiss"
download_file "track_metadata.json"
download_file "track_id_map.json"
download_file "anchors.json"
download_file "rerank_weights.json"
download_file "language_affinity.json"

# Optional artifacts
python -c "
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(repo_id='$REPO', filename='adapter.npz', repo_type='dataset', local_dir='$DEST')
    print('[startup] adapter.npz synced.')
except Exception as e:
    print('[startup] adapter.npz not present or skipped:', e)
"

exec uvicorn main:app --host 0.0.0.0 --port 7860
