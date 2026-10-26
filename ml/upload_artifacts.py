"""
One-time script to upload ML inference artifacts to HF Hub.

Usage (from the echo/ root):
    HF_TOKEN=hf_xxx python ml/upload_artifacts.py   # Linux/macOS
    set HF_TOKEN=hf_xxx && python ml/upload_artifacts.py  # Windows CMD
"""
import os
from pathlib import Path
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN") or None

REPO_ID = "shreyash616/echo-ml-artifacts"
ARTIFACTS = [
    "ml/inference/music_index.faiss",
    "ml/inference/track_metadata.json",
    "ml/inference/track_id_map.json",
    "ml/inference/anchors.json",
    "ml/inference/rerank_weights.json",
    "ml/inference/language_affinity.json",
    "ml/inference/adapter.npz",
]

api = HfApi(token=token)

print(f"Creating dataset repo {REPO_ID} (if it doesn't exist)...")
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)

for path_str in ARTIFACTS:
    path = Path(path_str)
    print(f"Uploading {path.name}  ({path.stat().st_size / 1_048_576:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"  Done: {path.name}")

print("\nAll artifacts uploaded to:")
print(f"  https://huggingface.co/datasets/{REPO_ID}")
