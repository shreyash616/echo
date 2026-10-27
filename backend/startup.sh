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
for opt in ['adapter.npz', 'theme_prompts.json']:
    try:
        hf_hub_download(repo_id='$REPO', filename=opt, repo_type='dataset', local_dir='$DEST')
        print(f'[startup] {opt} synced.')
    except Exception as e:
        print(f'[startup] {opt} not present or skipped:', e)
"

# Official LAION CLAP checkpoint
python -c "
from pathlib import Path
from huggingface_hub import hf_hub_download
cache_dir = Path.home() / '.cache' / 'clap'
cache_dir.mkdir(parents=True, exist_ok=True)
ckpt = cache_dir / 'music_audioset_epoch_15_esc_90.14.pt'
if not ckpt.exists():
    print('[startup] Downloading official LAION CLAP checkpoint...')
    hf_hub_download(repo_id='lukewys/laion_clap', filename='music_audioset_epoch_15_esc_90.14.pt', local_dir=str(cache_dir))
    print('[startup] LAION CLAP checkpoint ready.')
else:
    print('[startup] LAION CLAP checkpoint already cached.')
"

exec uvicorn main:app --host 0.0.0.0 --port 7860
