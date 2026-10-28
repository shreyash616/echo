"""
tests/test_axis_range.py: Asserts Rule 18 on TUNE seeds (T9).
Rule 18: Every per-axis similarity must move. report.py prints the axis standard deviation
over each seed's pool; an axis under 0.05 on three or more TUNE seeds is a bug to fix,
not a weight to lower.
Skips when offline artifacts are absent.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pytest

HF = os.environ.get("ECHO_ARTIFACT_DIR", "ml/inference")


def test_axis_range():
    meta_path = os.path.join(HF, "track_metadata.json")
    faiss_path = os.path.join(HF, "music_index.faiss")
    feel_path = os.path.join(HF, "feel_matrix.npy")
    if not (os.path.exists(meta_path) and os.path.exists(faiss_path) and os.path.exists(feel_path)):
        pytest.skip(f"Offline artifacts absent in {HF}")

    sys.path.insert(0, os.path.abspath("ml/experiments/offline_rerank"))
    from score_lib import Catalogue, AXES
    from report import load_benchmark

    try:
        cat = Catalogue()
    except Exception as e:
        pytest.skip(f"Failed to load Catalogue: {e}")

    benchmark_path = os.path.abspath("tests/feel_benchmark_v2.json")
    seeds = load_benchmark(benchmark_path, "tune")

    under_threshold_counts = {a: 0 for a in AXES}
    per_seed_stds = {}

    for seed_key, label, positives, negatives in seeds:
        if seed_key not in cat.id_map:
            continue
        si, idxs, srcs, F, flags = cat.feature_table(seed_key)
        stds = F.std(axis=0)
        per_seed_stds[label] = {a: float(stds[i]) for i, a in enumerate(AXES)}
        for i, a in enumerate(AXES):
            if stds[i] < 0.05:
                under_threshold_counts[a] += 1

    print("\n" + "=" * 80)
    print("RULE 18 AXIS RANGE CHECK (TUNE SEEDS)")
    print("=" * 80)
    for label, stds in per_seed_stds.items():
        print(f"\nSeed: {label}")
        for a in AXES:
            val = stds[a]
            mark = "FAIL (< 0.05)" if val < 0.05 else "OK"
            print(f"  {a:15s}: {val:.4f}  {mark}")

    print("\n" + "-" * 80)
    failing_axes = {a: count for a, count in under_threshold_counts.items() if count >= 3}
    if failing_axes:
        print(f"Axes failing Rule 18 (std < 0.05 on >= 3 seeds): {failing_axes}")
    else:
        print("All axes satisfy Rule 18 (no axis under 0.05 on 3 or more seeds).")
    print("=" * 80)

    assert not failing_axes, (
        f"Rule 18 violated: axes with std < 0.05 on 3 or more TUNE seeds: {failing_axes}"
    )


if __name__ == "__main__":
    test_axis_range()
