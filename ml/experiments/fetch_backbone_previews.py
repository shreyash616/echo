import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

TRACKS = [
    {"id": "simple_man", "title": "Simple Man", "artist": "Lynyrd Skynyrd"},
    {"id": "start_me_up", "title": "Start Me Up", "artist": "The Rolling Stones"},
    {"id": "rooster", "title": "Rooster", "artist": "Alice in Chains"},
    {"id": "levee", "title": "When the Levee Breaks", "artist": "Led Zeppelin"},
    {"id": "townes", "title": "Waiting Around to Die", "artist": "Townes Van Zandt"},
    {"id": "boyfriend", "title": "Boyfriend", "artist": "Karan Aujla"},
    {"id": "peaches", "title": "Peaches", "artist": "Justin Bieber"},
]

def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def search_itunes(title: str, artist: str, retries: int = 5):
    query = f"{title} {artist}"
    url = f"https://itunes.apple.com/search?term={urllib.parse.quote(query)}&country=US&entity=song&limit=10"
    req = urllib.request.Request(url, headers={"User-Agent": "EchoML/1.0"})
    
    data = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as e:
            if attempt == retries - 1:
                raise
            import time
            time.sleep(2 * (attempt + 1))
    
    norm_artist = normalise(artist)
    for result in data.get("results", []):
        tname = result.get("trackName", "")
        aname = result.get("artistName", "")
        preview = result.get("previewUrl")
        if not preview:
            continue
        # filter out live, remix, karaoke, cover, tribute, parentheses if possible
        low_tname = tname.lower()
        if any(bad in low_tname for bad in ["live", "remix", "karaoke", "cover", "tribute"]):
            continue
        if norm_artist in normalise(aname) or normalise(aname) in norm_artist:
            return {
                "trackName": tname,
                "artistName": aname,
                "previewUrl": preview,
                "trackId": result.get("trackId"),
            }
    # fallback to first result with preview
    for result in data.get("results", []):
        if result.get("previewUrl"):
            return {
                "trackName": result.get("trackName"),
                "artistName": result.get("artistName"),
                "previewUrl": result.get("previewUrl"),
                "trackId": result.get("trackId"),
            }
    return None

def main():
    audio_dir = Path("tests/previews")
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_items = []
    
    for t in TRACKS:
        print(f"Searching iTunes for {t['title']} - {t['artist']}...")
        res = search_itunes(t["title"], t["artist"])
        if not res:
            raise RuntimeError(f"Could not find iTunes preview for {t['title']} by {t['artist']}")
        
        # Download audio preview
        ext = ".m4a" if ".m4a" in res["previewUrl"] else ".mp3"
        local_path = audio_dir / f"{t['id']}{ext}"
        if not local_path.exists() or local_path.stat().st_size == 0:
            print(f"Downloading preview to {local_path}...")
            urllib.request.urlretrieve(res["previewUrl"], str(local_path))
        
        item = {
            "id": t["id"],
            "title": t["title"],
            "artist": t["artist"],
            "matched_title": res["trackName"],
            "matched_artist": res["artistName"],
            "preview_url": res["previewUrl"],
            "local_path": str(local_path).replace("\\", "/"),
        }
        out_items.append(item)
        try:
            print(f"  -> Found: {res['trackName']} by {res['artistName']} ({res['previewUrl']})")
        except UnicodeEncodeError:
            print(f"  -> Found: {res['trackName'].encode('ascii', 'replace').decode()} by {res['artistName'].encode('ascii', 'replace').decode()}")
    
    out_json = Path("tests/backbone_previews.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_items, f, indent=2)
    print(f"Saved {len(out_items)} previews to {out_json}")

if __name__ == "__main__":
    main()
