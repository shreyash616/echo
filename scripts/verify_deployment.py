"""
Comprehensive deployment verification script.
Checks:
1. Hugging Face Dataset (shreyash616/echo-ml-artifacts)
2. Hugging Face Space (shreyash616/echo-backend)
3. Live Space Endpoints (health, root frontend, search, track, recommendations)
4. GitHub repository status
"""
import subprocess
import httpx
from huggingface_hub import HfApi, get_token

print("=== 1. VERIFYING HUGGING FACE DATASET: shreyash616/echo-ml-artifacts ===")
api = HfApi(token=get_token())
ds_files = api.list_repo_files(repo_id="shreyash616/echo-ml-artifacts", repo_type="dataset")
print(f"Total files in dataset: {len(ds_files)}")
print(f"Files: {ds_files}")

required_ds = [
    "music_index.faiss", "track_metadata.json", "track_id_map.json",
    "anchors.json", "theme_prompts.json", "rerank_weights.json",
    "language_affinity.json", "adapter.npz", "ADAPTER_MANIFEST.json", "HARVEST_MANIFEST.json"
]
missing_ds = [f for f in required_ds if f not in ds_files]
deprecated_present = [f for f in ["music_encoder.onnx", "archetypes.json", "build_index.py"] if f in ds_files]

print("Required artifacts status :", "ALL PRESENT" if not missing_ds else f"MISSING: {missing_ds}")
print("Deprecated artifacts status:", "CLEAN (NONE PRESENT)" if not deprecated_present else f"STILL PRESENT: {deprecated_present}")

print("\n=== 2. VERIFYING HUGGING FACE SPACE: shreyash616/echo-backend ===")
space_info = api.space_info(repo_id="shreyash616/echo-backend")
stage = space_info.runtime.stage if space_info.runtime else "None"
print("Space Runtime Stage:", stage)

print("\n=== 3. VERIFYING LIVE SPACE ENDPOINTS ===")
client = httpx.Client(base_url="https://shreyash616-echo-backend.hf.space", timeout=25.0)

# A. Health
r_health = client.get("/api/health")
print(f"GET /api/health -> HTTP {r_health.status_code}: {r_health.json()}")

# B. Web App Frontend
r_root = client.get("/")
print(f"GET / (Frontend index.html) -> HTTP {r_root.status_code} ({len(r_root.text)} bytes)")
has_vibe_map = "VIBE_COLOR_MAP" in r_root.text
has_theme_badge = "themePill" in r_root.text or "themeTag" in r_root.text
print(f"  Frontend Vibe Colour Map present: {has_vibe_map}")
print(f"  Frontend Theme Badge present     : {has_theme_badge}")

# C. Search
r_search = client.get("/api/search", params={"q": "Simple Man", "limit": 3})
search_data = r_search.json()
results = search_data.get("results", [])
print(f"GET /api/search?q=Simple+Man -> HTTP {r_search.status_code}: found {len(results)} results")
if results:
    first = results[0]
    print(f"  First result: {first.get('artist')} - {first.get('title')} (key: {first.get('key')})")

# D. Track Info
k = "lynyrd skynyrd:simple man"
r_track = client.get(f"/api/track/{k}", params={"key": k})
print(f"GET /api/track/{k} -> HTTP {r_track.status_code}")
if r_track.status_code == 200:
    tdata = r_track.json()
    print(f"  Track: {tdata.get('artist')} - {tdata.get('title')}")
    print(f"  Vibes: {tdata.get('vibes')}")
    print(f"  Theme: {tdata.get('theme')}")
    print(f"  BPM: {tdata.get('bpm')}, Felt BPM: {tdata.get('felt_bpm')}")

# E. Recommendations
r_rec = client.get(f"/api/recommendations/{k}", params={"key": k, "k": 5})
print(f"GET /api/recommendations/{k} -> HTTP {r_rec.status_code}")
if r_rec.status_code == 200:
    recs = r_rec.json()
    print(f"  Returned {len(recs)} recommendations:")
    for idx, item in enumerate(recs[:3], 1):
        print(f"    {idx}. {item.get('artist')} - {item.get('title')} | score: {item.get('matchScore'):.3f} | vibe: {item.get('vibes')} | reasons: {item.get('matchReasons')}")

print("\n=== 4. VERIFYING GITHUB REPOSITORY: shreyash616/echo ===")
git_head = subprocess.check_output(["git", "log", "-n", "1", "--oneline"], text=True).strip()
git_branch = subprocess.check_output(["git", "status", "-s", "-b"], text=True).strip()
print("Git Status:", git_branch)
print("Git Head  :", git_head)
