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
from collections import Counter
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


def v2_rows(served_header: dict, keep: Path | None = None) -> tuple[list[dict], dict, np.ndarray, np.ndarray]:
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
    if keep is not None:  # a shard build: the anchor and theme means and stds v2 rewrote over these rows are the served ones
        for name in ("anchors.json", "theme_prompts.json"):
            shutil.copy(tmp / name, keep / name)
    shutil.rmtree(tmp, ignore_errors=True)
    return built["tracks"], built["header"], clap, feel


REPO = "shreyash616/echo-ml-artifacts"
NON_LATIN_LANGS = {"hi", "pa", "ta", "te", "ko", "ja", "ar", "ru"}
LOCAL_SHARDS = Path("ml/data/harvest_v3_work")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _shard_file(shard: str, kind: str, name: str, want: str, local_root: Path) -> Path:
    """The shard's file from the runner's work folder when its SHA256 matches MANIFEST.json, else from the dataset."""
    local = local_root / shard / name
    if local.exists() and _sha256(local) == want:
        return local
    from huggingface_hub import hf_hub_download
    p = Path(hf_hub_download(REPO, f"catalog/v3/{name}", repo_type="dataset"))
    if _sha256(p) != want:
        raise RuntimeError(f"{shard} {kind}: SHA256 differs from MANIFEST.json")
    return p


def shards_to_v2_inputs(shards_dir: str, scratch: Path, local_root: Path) -> dict:
    """Every shard in MANIFEST.json, in shard order, written as build_index_v2's two inputs (catalog_features.jsonl and
    clap.npy) in scratch, streamed shard by shard into one preallocated CLAP array. The language fix runs here (WP7
    step 2, in place of WP6b's rule): a Latin-script lyric vote never overrides a meta vote for hi, pa, ta, te, ko, ja,
    ar or ru."""
    import gzip
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download
    from app.audio.features import LATIN_LYRIC_LANGS, language_from_meta

    manifest = json.loads(Path(hf_hub_download(REPO, f"{shards_dir}/MANIFEST.json", repo_type="dataset")).read_text(encoding="utf-8"))
    ids = sorted(manifest["shards"])
    total = sum(manifest["shards"][s]["songs"] for s in ids)
    logger.info("MANIFEST.json: %d shards, %d songs", len(ids), total)

    def fetch(s):
        sha = manifest["shards"][s]["sha256"]
        return (_shard_file(s, "features", f"shard_{s}.features.jsonl.gz", sha["features"], local_root),
                _shard_file(s, "clap", f"shard_{s}.clap.f16.npy", sha["clap"], local_root))

    clap = np.zeros((total, 512), dtype=np.float32)
    lang_before, lang_after, changed, n = Counter(), Counter(), Counter(), 0
    with open(scratch / "catalog_features.jsonl", "w", encoding="utf-8") as out, ThreadPoolExecutor(8) as pool:
        for s, (feat, vec) in zip(ids, pool.map(fetch, ids)):
            c = np.load(vec)
            rows = 0
            with gzip.open(feat, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    lang_before[r.get("language")] += 1
                    if r.get("language_source") == "lyrics" and r.get("language") in LATIN_LYRIC_LANGS:
                        lang, conf, src = language_from_meta({k: v for k, v in r.items() if k != "language"})
                        if lang in NON_LATIN_LANGS:
                            changed[f"{r['language']}->{lang}"] += 1
                            r["language"], r["language_conf"], r["language_source"] = lang, conf, src
                    lang_after[r.get("language")] += 1
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    rows += 1
            if rows != len(c) or rows != manifest["shards"][s]["songs"]:
                raise RuntimeError(f"{s}: {rows} rows, {len(c)} CLAP vectors, MANIFEST says {manifest['shards'][s]['songs']}")
            clap[n:n + rows] = c.astype(np.float32)
            n += rows
            logger.info("%s: %d rows (%d of %d)", s, rows, n, total)
    np.save(scratch / "clap.npy", clap)
    del clap
    return {"shards": len(ids), "songs": n, "language_fix": {"changed": sum(changed.values()), "by_change": dict(changed),
            "before": dict(lang_before.most_common()), "after": dict(lang_after.most_common())}}


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
    ap.add_argument("--shards", default=None, help="dataset folder of the v3 shards (catalog/v3): build from every shard in its MANIFEST.json")
    ap.add_argument("--local-shards", default=str(LOCAL_SHARDS), help="runner work folder checked first, by SHA256")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shard_info = None
    if args.shards:
        # the served header's feel weights, from the dataset root as deployed
        from huggingface_hub import hf_hub_download
        served_header = json.loads(Path(hf_hub_download(REPO, "header.json", repo_type="dataset")).read_text(encoding="utf-8"))
        scratch = Path(tempfile.mkdtemp(prefix="echo_v3_shards_"))
        shard_info = shards_to_v2_inputs(args.shards, scratch, Path(args.local_shards))
        v2.CATALOG_FEATURES_PATH, v2.CLAP_NPY_PATH = scratch / "catalog_features.jsonl", scratch / "clap.npy"
        rows, header, clap, feel = v2_rows(served_header, keep=out)
        shutil.rmtree(scratch, ignore_errors=True)
    else:
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
    if shard_info:
        h.update({"source": args.shards, "shards": shard_info["shards"], "language_fix": shard_info["language_fix"]})
    (out / "header.json").write_text(json.dumps(h, indent=1), encoding="utf-8")

    print("=" * 60)
    print(f"build_index_v3: {n} rows, {len(columns)} fields -> {out}")
    print(f"  numeric arrays {len(arrays)}, text columns {len(table)}, lookup keys {len(keys)}")
    print(f"  clap.f16 max abs error against the float32 rows: {clap_err:.3e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
