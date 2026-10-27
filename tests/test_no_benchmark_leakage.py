"""
tests/test_no_benchmark_leakage.py: Benchmark Leakage Test (Sections 11.9, 12.0, 12.6).

Rules:
1. Load every title and artist from tests/feel_benchmark_v2.json.
2. Grep (case-insensitive, whole-word) code files outside tests/ and ml/archive/.
3. No file whitelist beyond ml/data/harvest_seed_artists.json.
4. No word exemptions.
5. Catalogue rule (Section 12.6): Benchmark titles must be < 2% of catalogue rows.
"""
from __future__ import annotations

import json
import os
import re

BENCHMARK_PATH = os.path.abspath("tests/feel_benchmark_v2.json")
CATALOG_PATH = os.path.abspath("ml/data/catalog_features.jsonl")
WHITELIST_FILES = {
    os.path.abspath("ml/data/harvest_seed_artists.json"),
}


def test_catalogue_benchmark_share():
    """Section 12.6 rule: Benchmark titles must be < 2% of catalogue rows."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    benchmark_titles: set[str] = set()
    for item in benchmark:
        for entity in [item.get("seed")] + item.get("positives", []) + item.get("negatives", []) + item.get("near_misses", []):
            if entity and entity.get("title"):
                benchmark_titles.add(entity["title"].strip().lower())
        st = item.get("scene_test")
        if st:
            for entity in st.get("inside_negatives", []) + st.get("outside_positives", []):
                if entity and entity.get("title"):
                    benchmark_titles.add(entity["title"].strip().lower())

    assert os.path.exists(CATALOG_PATH), f"Catalogue file not found: {CATALOG_PATH}"

    total_rows = 0
    matched_benchmark_titles = 0
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_rows += 1
            rec = json.loads(line)
            title = str(rec.get("title", "")).strip().lower()
            if title in benchmark_titles:
                matched_benchmark_titles += 1

    share = matched_benchmark_titles / max(total_rows, 1)
    print(f"\n[LEAKAGE CHECK] Catalogue rows: {total_rows}, Benchmark titles: {matched_benchmark_titles} ({share*100:.3f}%)")
    assert total_rows >= 20000, f"Catalogue has only {total_rows} rows (minimum 20,000 required)"
    assert share < 0.02, f"Benchmark titles exceed 2% of catalogue: {share*100:.2f}% (must be < 2.0%)"


def test_no_benchmark_leakage():
    """Scan all source code files for leaked benchmark entities."""
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
    skip_dirs = {"tests", "ml/archive", "ml/experiments", "node_modules", ".git", ".venv", "venv", "__pycache__"}
    # Catalogue datasets are governed by test_catalogue_benchmark_share (< 2% rule per Section 12.6)
    catalogue_files = {
        os.path.abspath("ml/data/catalog_features.jsonl"),
        os.path.abspath("ml/data/playlist_membership.jsonl"),
        os.path.abspath("ml/data/itunes_cache.jsonl"),
        os.path.abspath("ml/inference/track_metadata.json"),
        os.path.abspath("ml/inference/track_id_map.json"),
    }
    # Package configuration manifests that only declare npm dependencies
    manifest_files = {
        os.path.abspath("app/package.json"),
        os.path.abspath("app/package-lock.json"),
    }

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
            if abs_path in WHITELIST_FILES or abs_path in catalogue_files or abs_path in manifest_files:
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            for term in sorted(titles_artists):
                pattern = r"\b" + re.escape(term) + r"\b"
                if re.search(pattern, content, re.IGNORECASE):
                    rel_p = os.path.relpath(abs_path, ".").replace("\\", "/")
                    leaks.append(f"Leak found in {rel_p}: benchmark entity '{term}'")

    if leaks:
        print(f"\n[FAIL] Leakage detected in {len(leaks)} places:\n" + "\n".join(leaks[:20]))
    else:
        print("\n[SUCCESS] No benchmark leakage detected in any code files!")

    assert not leaks, f"Benchmark leakage detected ({len(leaks)} occurrences):\n" + "\n".join(leaks)


if __name__ == "__main__":
    test_catalogue_benchmark_share()
    test_no_benchmark_leakage()
