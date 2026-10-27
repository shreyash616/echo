"""
Upload ML inference artifacts to HF Hub and prune dead files.

Usage (from the echo/ root):
    python ml/upload_artifacts.py
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, get_token

token = os.environ.get("HF_TOKEN") or get_token()
if not token:
    print("ERROR: HF_TOKEN not found in environment and no cached token found.", file=sys.stderr)
    sys.exit(1)

REPO_ID = "shreyash616/echo-ml-artifacts"

ARTIFACTS = [
    "ml/inference/music_index.faiss",
    "ml/inference/track_metadata.json",
    "ml/inference/track_id_map.json",
    "ml/inference/anchors.json",
    "ml/inference/theme_prompts.json",
    "ml/inference/rerank_weights.json",
    "ml/inference/language_affinity.json",
    "ml/inference/adapter.npz",
    "ml/inference/ADAPTER_MANIFEST.json",
    "ml/data/HARVEST_MANIFEST.json",
]

DEAD_FILES = [
    "music_encoder.onnx",
    "archetypes.json",
    "build_index.py",
]

api = HfApi(token=token)

print(f"Checking dataset repo {REPO_ID}...")
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)

# Delete dead files per Section 12.5
existing_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
for dead in DEAD_FILES:
    if dead in existing_files:
        print(f"Deleting dead file from HF dataset: {dead}...")
        try:
            api.delete_file(path_in_repo=dead, repo_id=REPO_ID, repo_type="dataset", commit_message=f"Remove deprecated artifact {dead}")
            print(f"  Deleted: {dead}")
        except Exception as e:
            print(f"  Failed to delete {dead}: {e}")

# Upload artifacts
for path_str in ARTIFACTS:
    path = Path(path_str)
    if not path.exists():
        print(f"WARNING: File {path_str} does not exist, skipping.")
        continue
    print(f"Uploading {path.name} ({path.stat().st_size / 1_048_576:.2f} MB)...")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Update {path.name} (20k catalogue, Section 12 feel algorithm)",
    )
    print(f"  Done: {path.name}")

print("\nAll artifacts uploaded to:")
print(f"  https://huggingface.co/datasets/{REPO_ID}")
