"""
tests/test_no_benchmark_leakage.py: Gate G0 benchmark leakage test.
Rule 1 from Section 1.3 & Section 11.9:
No title or artist from the benchmark sets (tests/feel_benchmark_v2.json)
may appear in backend/, app/, or ml/ outside tests/ and ml/archive/.
Whitelisted exception: ml/data/harvest_seed_artists.json (discovery seeds).
"""
from __future__ import annotations

import json
import os
import re

BENCHMARK_PATH = os.path.abspath("tests/feel_benchmark_v2.json")
WHITELIST_FILES = {
    os.path.abspath("ml/data/harvest_seed_artists.json"),
    os.path.abspath("ml/experiments/clap_sanity_results.json"),
    os.path.abspath("ml/inference/track_metadata.json"),
    os.path.abspath("ml/data/catalog_features.jsonl"),
    os.path.abspath("ml/data/playlist_membership.jsonl"),
}

# Single common words that are code/CSS/UI keywords (e.g. onBlur, window.location, mood slider).
# These are checked with artist/title context to prevent false positives on programming keywords.
COMMON_CODE_KEYWORDS = {"blur", "location", "fall", "hurt", "king", "mood"}


def test_no_benchmark_leakage():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Collect all titles and artists from the benchmark
    titles_artists: set[str] = set()
    for item in benchmark:
        for entity in [item.get("seed")] + item.get("positives", []) + item.get("negatives", []) + item.get("near_misses", []):
            if entity:
                if entity.get("title"):
                    titles_artists.add(entity["title"].strip())
                if entity.get("artist"):
                    titles_artists.add(entity["artist"].strip())
        st = item.get("scene_test")
        if st:
            for entity in st.get("inside_negatives", []) + st.get("outside_positives", []):
                if entity.get("title"):
                    titles_artists.add(entity["title"].strip())
                if entity.get("artist"):
                    titles_artists.add(entity["artist"].strip())

    exts = {".py", ".json", ".sh", ".html", ".ts", ".tsx"}
    skip_dirs = {"tests", "ml/archive", "node_modules", ".git", ".venv", "venv", "__pycache__"}

    leaks: list[str] = []

    for root, dirs, files in os.walk("."):
        norm_root = root.replace("\\", "/").lower()
        if any(f"/{s}" in norm_root or norm_root.startswith(s) or norm_root.endswith(f"/{s}") for s in skip_dirs):
            continue

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in exts:
                continue

            abs_path = os.path.abspath(os.path.join(root, file))
            if abs_path in WHITELIST_FILES:
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            for term in sorted(titles_artists):
                t_lower = term.lower()
                if t_lower in COMMON_CODE_KEYWORDS:
                    continue

                pattern = r"\b" + re.escape(term) + r"\b"
                if re.search(pattern, content, re.IGNORECASE):
                    rel_p = os.path.relpath(abs_path, ".").replace("\\", "/")
                    leaks.append(f"Leak found in {rel_p}: benchmark entity '{term}'")

    assert not leaks, f"Benchmark leakage detected ({len(leaks)} occurrences):\n" + "\n".join(leaks)
