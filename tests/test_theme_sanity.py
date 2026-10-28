"""
tests/test_theme_sanity.py: Theme sanity assertion suite (S3).
Asserts:
1. No single theme is the argmax on more than 30% of rows in the catalogue.
2. At least 6 of the 10 themes each hold at least 3% share.
3. Reports the displayed theme for the 7 reference tracks and verifies at least 5 of 7 match:
   - Boyfriend: swagger
   - Mask Off: swagger
   - Get Lucky: celebration
   - Tum Hi Ho: devotion or longing
   - Hurt: grief or resignation
   - Rooster: defiance or grief
   - Simple Man: devotion or resignation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
import httpx
import pytest

SANITY_TARGETS = [
    {"title": "Boyfriend", "artist": "Karan Aujla", "expected": ["swagger"]},
    {"title": "Mask Off", "artist": "Future", "expected": ["swagger"]},
    {"title": "Get Lucky", "artist": "Daft Punk", "expected": ["celebration"]},
    {"title": "Tum Hi Ho", "artist": "Arijit Singh", "expected": ["devotion", "longing"]},
    {"title": "Hurt", "artist": "Johnny Cash", "expected": ["grief", "resignation"]},
    {"title": "Rooster", "artist": "Alice In Chains", "expected": ["defiance", "grief"]},
    {"title": "Simple Man", "artist": "Lynyrd Skynyrd", "expected": ["devotion", "resignation"]},
]


def test_theme_sanity(base_url: str | None = None):
    base_url = base_url or os.environ.get("ECHO_BASE_URL")
    # 1. Load catalogue track metadata if available
    meta_path = os.path.abspath("ml/inference/track_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracks = data.get("tracks", data) if isinstance(data, dict) else data
        N = len(tracks)
        assert N >= 1000, f"Catalogue too small: {N}"

        themes = [r.get("theme") for r in tracks if r.get("theme")]
        counts = Counter(themes)
        print("\n" + "=" * 80)
        print(f"THEME DISTRIBUTION ACROSS CATALOGUE ({len(themes)}/{N} tracks have theme badge):")
        for th, cnt in counts.most_common():
            pct = (cnt / N) * 100
            print(f"  {th:15s}: {cnt:5d} ({pct:5.2f}%)")

        # Assert 1: No single theme on > 30% of rows
        for th, cnt in counts.items():
            share = cnt / N
            assert share <= 0.30, f"Theme '{th}' exceeds 30% maximum share: {share*100:.2f}%"

        # Assert 2: At least 6 of 10 themes each hold at least 3% share
        themes_ge_3pct = [th for th, cnt in counts.items() if (cnt / N) >= 0.03]
        print(f"\nThemes with >= 3% share: {len(themes_ge_3pct)} / 10 ({themes_ge_3pct})")
        assert len(themes_ge_3pct) >= 6, (
            f"Expected at least 6 themes with >= 3% share, got {len(themes_ge_3pct)}: {themes_ge_3pct}"
        )
    else:
        tracks = []
        print("\n[NOTE] ml/inference/track_metadata.json absent; verifying reference targets against live API.")

    # Assert 3: Test displayed theme on the 7 reference tracks
    matches = 0
    print("\n" + "=" * 80)
    print(f"{'Target Track':25s} | {'Displayed Theme':18s} | {'Expected Themes':25s} | Match")
    print("-" * 80)

    for target in SANITY_TARGETS:
        t_name = target["title"].lower()
        a_name = target["artist"].lower()
        displayed_theme = None

        if base_url:
            # Query live API
            try:
                with httpx.Client(base_url=base_url, timeout=15.0) as client:
                    resp = client.get(f"/api/search", params={"q": f"{target['title']} {target['artist']}"})
                    if resp.status_code == 200:
                        data = resp.json()
                        track_list = data.get("results", data) if isinstance(data, dict) else data
                        if track_list and isinstance(track_list, list):
                            displayed_theme = track_list[0].get("theme")
            except Exception as e:
                print(f"API query failed for {target['title']}: {e}")

        if not displayed_theme:
            # Fallback to local catalogue lookup
            for r in tracks:
                if t_name in r.get("title", "").lower() and a_name in r.get("artist", "").lower():
                    displayed_theme = r.get("theme")
                    break

        is_match = displayed_theme in target["expected"]
        if is_match:
            matches += 1

        print(
            f"{target['title'] + ' - ' + target['artist']:25s} | "
            f"{str(displayed_theme):18s} | "
            f"{str(target['expected']):25s} | "
            f"{'YES' if is_match else 'NO'}"
        )

    print("=" * 80)
    print(f"Reference Track Matches: {matches}/7 (Required >= 5)")
    print("=" * 80)

    assert matches >= 5, f"Sanity targets failed: only {matches}/7 matched (required >= 5)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None, help="Base URL of live Space")
    args = parser.parse_args()
    test_theme_sanity(base_url=args.base_url)
