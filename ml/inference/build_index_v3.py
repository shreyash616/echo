"""
ml/inference/build_index_v3.py: columnar serving files (SCALE_PLAN section 6, WP2).

  python ml/inference/build_index_v3.py --out ml/inference

Rows come from build_index_v2's own build of ml/data/catalog_features.jsonl and ml/data/clap.npy
(run unchanged into a scratch directory, so every derived field, standardisation and cut point is
the code that produced the served rows), checked equal to the served track_metadata.json, then
written as columns:

  clap.f16.npy      n x 512 float16, the unit-norm CLAP rows
  feel_matrix.npy   n x 17 float32, as build_index_v2 writes it
  features.npz      one array per numeric field (float32; ints int64, flags bool), theme_vec,
                    anchor_profile, anchor_cosines as n x k float32, felt_bpm_dist as
                    dist_bpm / dist_p (n x 5, NaN padded) and dist_len
  strings.parquet   every text field (null kept), vibes as a list column
  keys.npy          the sorted 64-bit blake2b hashes of the lookup strings (id, key, itunes_id,
                    it_<itunes_id>), key_order.npy the row each one maps to, last row winning as
                    track_id_map.json does; fixed-width strings would be 86 MB here and 4 GB at a
                    million rows. The loaders confirm a hit against the row's own strings.
  header.json       build_index_v2's header plus the column list, row count and build commit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "backend")
import build_index_v2 as v2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_index_v3")

VEC_FIELDS = ("theme_vec", "anchor_profile", "anchor_cosines")


def key_hash(s: str) -> int:
    """The lookup hash the loaders share: blake2b, 8 bytes, as an unsigned 64-bit integer."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")
DIST_FIELD = "felt_bpm_dist"
DIST_WIDTH = 5


def v2_rows(served_header: dict) -> tuple[list[dict], dict, np.ndarray, np.ndarray]:
    """build_index_v2.main() with its outputs redirected to a scratch directory."""
    tmp = Path(tempfile.mkdtemp(prefix="echo_v3_"))
    for name in ("anchors.json", "theme_prompts.json"):
        shutil.copy(v2.INFERENCE_DIR / name, tmp / name)  # v2 reads its prompts from these and rewrites them
    (tmp / "track_metadata.json").write_text(json.dumps({"header": {"feel_weights": served_header["feel_weights"]}}),
                                             encoding="utf-8")  # v2 keeps the header's feel weights
    for attr, name in (("OUT_FAISS_PATH", "music_index.faiss"), ("OUT_METADATA_PATH", "track_metadata.json"),
                       ("OUT_ANCHORS_PATH", "anchors.json"), ("OUT_THEMES_PATH", "theme_prompts.json"),
                       ("OUT_ID_MAP_PATH", "track_id_map.json"), ("OUT_FEEL_MATRIX_PATH", "feel_matrix.npy"),
                       ("OUT_FEEL_FAISS_PATH", "feel_index.faiss")):
        setattr(v2, attr, tmp / name)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        v2.main()
    finally:
        sys.argv = argv
    built = json.loads((tmp / "track_metadata.json").read_text(encoding="utf-8"))
    import faiss
    index = faiss.read_index(str(tmp / "music_index.faiss"))
    clap = index.reconstruct_n(0, index.ntotal)
    feel = np.load(tmp / "feel_matrix.npy")
    shutil.rmtree(tmp, ignore_errors=True)
    return built["tracks"], built["header"], clap, feel


def kind_of(field: str, rows: list[dict]) -> str:
    if field in VEC_FIELDS:
        return "vec"
    if field == DIST_FIELD:
        return "dist"
    types = {type(r[field]) for r in rows if r[field] is not None}
    if not types:
        return "str"  # null on every row (deezer_id): kept as a text column of nulls
    if types <= {bool}:
        return "bool"
    if types <= {int}:
        return "int"
    if types <= {int, float}:
        return "float"
    if types <= {str}:
        return "str"
    if types <= {list}:
        return "strlist"
    raise TypeError(f"field {field}: unsupported types {types}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ml/inference")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    served = json.loads((v2.INFERENCE_DIR / "track_metadata.json").read_text(encoding="utf-8"))
    rows, header, clap, feel = v2_rows(served["header"])
    same = rows == served["tracks"] and header == served["header"]
    logger.info("v2 rows equal to the served track_metadata.json: %s", same)
    if not same:
        raise RuntimeError("build_index_v2's rows differ from the served track_metadata.json; not writing columns")
    n = len(rows)
    fields = list(rows[0].keys())
    if any(set(r.keys()) != set(fields) for r in rows):
        raise RuntimeError("rows do not share one field set")
    columns = [{"name": f, "kind": kind_of(f, rows)} for f in fields]
    # the JSON rows come in more than one field order (rows re-fetched in P4 were rebuilt); keep each row's
    orders = list(dict.fromkeys(tuple(r.keys()) for r in rows))
    order_of = {o: j for j, o in enumerate(orders)}

    # numeric and vector columns
    arrays: dict[str, np.ndarray] = {}
    for c in columns:
        f, k = c["name"], c["kind"]
        if k == "float":
            arrays[f] = np.array([r[f] for r in rows], dtype=np.float32)
        elif k == "int":
            arrays[f] = np.array([r[f] for r in rows], dtype=np.int64)
        elif k == "bool":
            arrays[f] = np.array([r[f] for r in rows], dtype=bool)
        elif k == "vec":
            width = {len(r[f]) for r in rows}
            if len(width) != 1:
                raise RuntimeError(f"{f}: lengths {width}")
            arrays[f] = np.array([r[f] for r in rows], dtype=np.float32)
        elif k == "dist":
            bpm = np.full((n, DIST_WIDTH), np.nan, dtype=np.float32)
            prob = np.full((n, DIST_WIDTH), np.nan, dtype=np.float32)
            length = np.zeros(n, dtype=np.int8)
            for i, r in enumerate(rows):
                items = list(r[f].items())
                if len(items) > DIST_WIDTH:
                    raise RuntimeError(f"row {i}: {len(items)} felt_bpm_dist entries")
                for j, (b, p) in enumerate(items):
                    if str(int(b)) != b:
                        raise RuntimeError(f"row {i}: felt_bpm_dist key {b!r} is not an integer string")
                    bpm[i, j], prob[i, j] = int(b), p
                length[i] = len(items)
            arrays["dist_bpm"], arrays["dist_p"], arrays["dist_len"] = bpm, prob, length
    arrays["field_order"] = np.array([order_of[tuple(r.keys())] for r in rows], dtype=np.int8)
    np.savez(out / "features.npz", **arrays)

    # text columns
    table = {}
    for c in columns:
        f, k = c["name"], c["kind"]
        if k == "str":
            table[f] = pa.array([r[f] for r in rows], type=pa.string())
        elif k == "strlist":
            table[f] = pa.array([r[f] for r in rows], type=pa.list_(pa.string()))
    pq.write_table(pa.table(table), out / "strings.parquet")

    # lookup: track_id_map.json's construction, last row winning
    id_map: dict[str, int] = {}
    for idx, rec in enumerate(rows):
        if rec.get("id"):
            id_map[str(rec["id"])] = idx
        id_map[rec.get("key") or v2.make_track_key(rec.get("artist", ""), rec.get("title", ""))] = idx
        if rec.get("deezer_id"):
            id_map[str(rec["deezer_id"])] = idx
        if rec.get("itunes_id"):
            id_map[str(rec["itunes_id"])] = idx
            id_map[f"it_{rec['itunes_id']}"] = idx
    hashed = sorted((key_hash(k), r) for k, r in id_map.items())
    hashes = np.array([h for h, _ in hashed], dtype=np.uint64)
    if len(np.unique(hashes)) != len(hashes):
        raise RuntimeError("blake2b-64 collision among the lookup strings")
    np.save(out / "keys.npy", hashes)
    np.save(out / "key_order.npy", np.array([r for _, r in hashed], dtype=np.int64))
    keys = hashed

    clap16 = clap.astype(np.float16)
    np.save(out / "clap.f16.npy", clap16)
    np.save(out / "feel_matrix.npy", feel)
    clap_err = float(np.abs(clap16.astype(np.float32) - clap).max())

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    h = dict(header)
    h.update({"version": "3.0", "rows": n, "columns": columns, "field_orders": [list(o) for o in orders], "dist_width": DIST_WIDTH,
              "build_commit": commit, "clap_f16_max_abs_error": clap_err})
    (out / "header.json").write_text(json.dumps(h, indent=1), encoding="utf-8")

    print("=" * 60)
    print(f"build_index_v3: {n} rows, {len(columns)} fields -> {out}")
    print(f"  numeric arrays {len(arrays)}, text columns {len(table)}, lookup keys {len(keys)}")
    print(f"  clap.f16 max abs error against the float32 rows: {clap_err:.3e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
