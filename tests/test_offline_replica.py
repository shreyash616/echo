"""
tests/test_offline_replica.py: Offline replica parity assertion (T9).
Runs report.py --validate-live against the live Space (or ECHO_BASE_URL)
and asserts max deviation between offline computed axes and live debug=1 axes is < 0.001.
Skips when artifacts are absent.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
import pytest

HF = os.environ.get("ECHO_ARTIFACT_DIR", "ml/inference")


def test_offline_replica():
    meta_path = os.path.join(HF, "track_metadata.json")
    faiss_path = os.path.join(HF, "music_index.faiss")
    feel_path = os.path.join(HF, "feel_matrix.npy")
    if not (os.path.exists(meta_path) and os.path.exists(faiss_path) and os.path.exists(feel_path)):
        pytest.skip(f"Offline artifacts absent in {HF}")

    sys.path.insert(0, os.path.abspath("ml/experiments/offline_rerank"))
    from score_lib import Catalogue, AXES
    from report import load_benchmark
    import httpx

    base_url = os.environ.get("ECHO_BASE_URL", "https://shreyash616-echo-backend.hf.space")

    try:
        cat = Catalogue()
    except Exception as e:
        pytest.skip(f"Failed to load Catalogue: {e}")

    benchmark_path = os.path.abspath("tests/feel_benchmark_v2.json")
    seeds = load_benchmark(benchmark_path, "tune")

    max_dev = 0.0
    client = httpx.Client(timeout=180.0)

    for seed_key, label, positives, negatives in seeds:
        if seed_key not in cat.id_map:
            continue
        si, idxs, srcs, F, flags = cat.feature_table(seed_key)
        try:
            resp = client.get(
                f"{base_url}/api/recommendations/x",
                params={"key": seed_key, "debug": 1, "diversity": 1.0},
            )
            assert resp.status_code == 200, f"Live query failed: {resp.text}"
            r = resp.json()
            res = r["results"] if isinstance(r, dict) else r
            for x in res:
                xi = cat.id_map.get(x.get("key") or "")
                if xi is None:
                    continue
                matches = np.where(idxs == xi)[0]
                if len(matches) == 0:
                    continue
                j = int(matches[0])
                for a_i, a in enumerate(AXES):
                    if a in x.get("axes", {}):
                        dev = abs(float(x["axes"][a]) - F[j, a_i])
                        if dev > max_dev:
                            max_dev = dev
        except httpx.ConnectError:
            pytest.skip(f"Cannot reach live base URL {base_url}")

    print(f"\nMax axis deviation vs {base_url} = {max_dev:.6f}")
    assert max_dev < 0.001, f"Max axis deviation {max_dev:.6f} exceeds limit 0.001"


if __name__ == "__main__":
    test_offline_replica()
