from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "shreyash616/echo-backend"

files_to_upload = [
    ("backend/app/config.py", "app/config.py"),
    ("backend/app/services/deezer.py", "app/services/deezer.py"),
    ("backend/app/services/recommender.py", "app/services/recommender.py"),
    ("backend/app/services/audio_recognition.py", "app/services/audio_recognition.py"),
    ("backend/app/api/routes.py", "app/api/routes.py"),
    ("backend/app/services/lyrics.py", "app/services/lyrics.py"),
    ("backend/static/index.html", "static/index.html"),
]

for local_path, repo_path in files_to_upload:
    print(f"Uploading {local_path} -> {repo_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy update: {repo_path}"
    )

print("All files uploaded successfully!")
