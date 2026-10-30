#!/bin/bash
set -e

REPO="shreyash616/echo-ml-artifacts"
DEST="/app/ml/inference"

download_file() {
    local filename=$1
    echo "[startup] Syncing $filename from HF Hub ($REPO)..."
    python -c "
import os, time
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN') or None
for attempt in range(12):
    try:
        hf_hub_download(
            repo_id='$REPO',
            filename='$filename',
            repo_type='dataset',
            local_dir='$DEST',
            token=token,
        )
        print(f'[startup] Successfully downloaded $filename')
        break
    except Exception as e:
        print(f'[startup] Attempt {attempt+1} failed for $filename: {e}')
        if attempt == 11:
            raise e
        time.sleep(10 + 5 * attempt)
"
    local filepath="$DEST/$filename"
    if [ ! -f "$filepath" ] || [ ! -s "$filepath" ]; then
        echo "[startup] ERROR: $filename missing or empty!"
        exit 1
    fi
    echo "[startup] $filename synced."
}

download_file "music_index.faiss"
download_file "feel_matrix.npy"
download_file "feel_index.faiss"
download_file "track_metadata.json"
download_file "track_id_map.json"
download_file "anchors.json"
download_file "rerank_weights.json"
download_file "language_affinity.json"

# Optional artifacts
python -c "
import os
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN') or None
for opt in ['adapter.npz', 'theme_prompts.json']:
    try:
        hf_hub_download(repo_id='$REPO', filename=opt, repo_type='dataset', local_dir='$DEST', token=token)
        print(f'[startup] {opt} synced.')
    except Exception as e:
        print(f'[startup] {opt} not present or skipped:', e)
"

# Official LAION CLAP checkpoint
python -c "
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN') or None
cache_dir = Path.home() / '.cache' / 'clap'
cache_dir.mkdir(parents=True, exist_ok=True)
ckpt = cache_dir / 'music_audioset_epoch_15_esc_90.14.pt'
if not ckpt.exists():
    print('[startup] Downloading official LAION CLAP checkpoint...')
    hf_hub_download(repo_id='lukewys/laion_clap', filename='music_audioset_epoch_15_esc_90.14.pt', local_dir=str(cache_dir), token=token)
    print('[startup] LAION CLAP checkpoint ready.')
else:
    print('[startup] LAION CLAP checkpoint already cached.')
"

exec uvicorn main:app --host 0.0.0.0 --port 7860
