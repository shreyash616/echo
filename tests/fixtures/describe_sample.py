"""
tests/fixtures/describe_sample.py: P2 verification. Describes a deterministic sample of catalogue
rows from their decoded previews (ml/data/previews) with the header cut points, and prints the
catalogue std table for the T3 and T5 gates.

  py tests/fixtures/describe_sample.py --n 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.abspath("backend"))

PREVIEWS = os.path.abspath("ml/data/previews")
CATALOG = os.path.abspath("ml/data/catalog_features.jsonl")
META = os.path.abspath("ml/inference/track_metadata.json")
FIELDS = ["saturation", "sustain_ratio", "darkness", "roughness", "low_share", "brightness",
          "beat_low_ratio", "off_beat_ratio", "hits_per_bar", "grit", "intimacy"]


def _preview_path(row: dict) -> str | None:
    for name in (f"it_{row.get('itunes_id')}.m4a", f"{row.get('itunes_id')}.m4a", f"{row.get('id')}.m4a"):
        p = os.path.join(PREVIEWS, name)
        if os.path.exists(p):
            return p
    return None


def _describe(args: tuple[str, str, dict]) -> dict:
    import warnings
    warnings.filterwarnings("ignore")
    from app.audio.decode import decode_preview
    from app.audio.features import describe_audio
    key, path, cut_points = args
    with open(path, "rb") as f:
        y = decode_preview(f.read())
    d = describe_audio(y, sr=22050, cut_points=cut_points)
    return {"key": key, **{k: d[k] for k in FIELDS}, "kick_class": d["kick_class"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    with open(META, encoding="utf-8") as f:
        cut_points = json.load(f)["header"]["cut_points"]
    rows = [json.loads(l) for l in open(CATALOG, encoding="utf-8")]
    with_preview = [(r["key"], p) for r in rows if (p := _preview_path(r))]
    random.seed(a.seed)
    sample = random.sample(with_preview, a.n)
    print(f"catalogue rows {len(rows)}, with local preview {len(with_preview)}, sample {a.n} (seed {a.seed})")

    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(_describe, [(k, p, cut_points) for k, p in sample], chunksize=4))

    print(f"\n{'field':15s} {'std':>7s} {'mean':>7s} {'p5':>7s} {'p50':>7s} {'p95':>7s} {'share=1.0':>9s}")
    for k in FIELDS:
        v = np.array([o[k] for o in out], float)
        print(f"{k:15s} {v.std():7.3f} {v.mean():7.3f} {np.percentile(v, 5):7.3f} {np.percentile(v, 50):7.3f} "
              f"{np.percentile(v, 95):7.3f} {np.mean(v >= 0.999):9.2%}")
    kc = Counter(o["kick_class"] for o in out)
    print("\nkick_class: " + ", ".join(f"{c} {n / len(out):.1%}" for c, n in kc.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
