import urllib.request
import json

base_url = "https://shreyash616-echo-backend.hf.space"

# 1. Health check
with urllib.request.urlopen(f"{base_url}/api/health") as resp:
    data = json.loads(resp.read().decode())
    print("Live API Health:", data)

# 2. Search test
query = urllib.parse.quote("yeh vaada raha")
with urllib.request.urlopen(f"{base_url}/api/search?q={query}&limit=1") as resp:
    data = json.loads(resp.read().decode())
    results = data.get("results", [])
    if results:
        t = results[0]
        print(f"Found track: {t.get('title')} by {t.get('artist')} (ID: {t.get('id')})")
        
        # 3. Recommendations with MMR diversity and Camelot Keys
        rec_url = f"{base_url}/api/recommendations/{t.get('id')}?diversity=0.72"
        with urllib.request.urlopen(rec_url) as rec_resp:
            recs = json.loads(rec_resp.read().decode())
            print(f"Fetched {len(recs)} recommendations:")
            for i, r in enumerate(recs[:5], 1):
                title = r.get("title", "")
                artist = r.get("artist", "")
                key = r.get("camelotKey", "—")
                reasons = r.get("matchReasons", [])
                score = r.get("matchScore")
                print(f"  {i}. {title} by {artist} [{score}] | Key: {key} | Reasons: {reasons}")
