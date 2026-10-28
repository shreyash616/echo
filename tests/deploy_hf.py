from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "shreyash616/echo-backend"

files_to_upload = [
    ("backend/app/config.py", "app/config.py"),
    ("backend/app/models/schemas.py", "app/models/schemas.py"),
    ("backend/app/audio/features.py", "app/audio/features.py"),
    ("backend/app/services/deezer.py", "app/services/deezer.py"),
    ("backend/app/services/recommender.py", "app/services/recommender.py"),
    ("backend/app/services/audio_recognition.py", "app/services/audio_recognition.py"),
    ("backend/app/services/language_affinity.py", "app/services/language_affinity.py"),
    ("backend/app/services/lyrics.py", "app/services/lyrics.py"),
    ("backend/app/api/routes.py", "app/api/routes.py"),
    ("backend/requirements.txt", "requirements.txt"),
    ("backend/startup.sh", "startup.sh"),
    ("backend/static/index.html", "static/index.html"),
]

dataset_id = "shreyash616/echo-ml-artifacts"
artifacts_to_upload = [
    ("ml/inference/music_index.faiss", "music_index.faiss"),
    ("ml/inference/feel_matrix.npy", "feel_matrix.npy"),
    ("ml/inference/feel_index.faiss", "feel_index.faiss"),
    ("ml/inference/track_metadata.json", "track_metadata.json"),
    ("ml/inference/track_id_map.json", "track_id_map.json"),
    ("ml/inference/anchors.json", "anchors.json"),
    ("ml/inference/theme_prompts.json", "theme_prompts.json"),
    ("ml/inference/rerank_weights.json", "rerank_weights.json"),
    ("ml/data/HARVEST_MANIFEST.json", "HARVEST_MANIFEST.json"),
    ("ml/data/clap.npy", "ml/data/clap.npy"),
    ("ml/data/catalog_features.jsonl", "ml/data/catalog_features.jsonl"),
    ("ml/data/playlist_membership.jsonl", "ml/data/playlist_membership.jsonl"),
]

print("--- Uploading artifacts to dataset:", dataset_id, "---")
for local_path, repo_path in artifacts_to_upload:
    if os.path.exists(local_path):
        print(f"Uploading {local_path} -> {repo_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=f"Update artifact: {repo_path}"
        )
    else:
        print(f"Skipping missing artifact {local_path}")

print("--- Uploading backend files to space:", repo_id, "---")
for local_path, repo_path in files_to_upload:
    print(f"Uploading {local_path} -> {repo_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy update: {repo_path}"
    )

print("Restarting Space...")
try:
    api.restart_space(repo_id=repo_id)
    print("Space restart requested successfully.")
except Exception as e:
    print("Space restart notice:", e)

print("All files uploaded successfully!")
