"""
tests/test_columnar_loader.py: the columnar catalogue (SCALE_PLAN section 6, WP2) against the JSON files it replaces.
Asserts (the first five when ml/inference holds both sets, skipped otherwise):
- every assembled row has the JSON row's fields in the JSON row's order and the same Python types;
- text, int and bool values are equal; every float equals the JSON value rounded to float32;
- every key of track_id_map.json looks up the same row, and an absent key looks up nothing;
- clap.f16.npy is the unit-norm CLAP rows to float16 precision (max abs error at most 5e-4);
- feel_matrix.npy is the matrix feel_index.faiss holds;
- the recommender loads the columnar set and returns the same row by key and by id;
- with a file missing or unreadable the recommender stays unloaded and logs the file (no JSON fallback).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("backend"))

D = Path(os.environ.get("ECHO_ARTIFACT_DIR", "ml/inference"))
COLUMNAR = ("header.json", "features.npz", "strings.parquet", "keys.npy", "key_order.npy", "clap.f16.npy", "feel_matrix.npy")
JSONSET = ("track_metadata.json", "track_id_map.json", "music_index.faiss", "feel_index.faiss")

needs_artifacts = pytest.mark.skipif(not all((D / f).exists() for f in COLUMNAR + JSONSET),
                                     reason="columnar and JSON artifacts not both present")


@pytest.fixture(scope="module")
def both():
    from app.services.recommender import ColumnarRows
    header = json.loads((D / "header.json").read_text(encoding="utf-8"))
    js = json.loads((D / "track_metadata.json").read_text(encoding="utf-8"))["tracks"]
    return header, ColumnarRows(D, header), js


def _same(a, b) -> bool:
    if isinstance(a, float):
        return isinstance(b, float) and b == float(np.float32(a))
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


@needs_artifacts
def test_rows_equal_json(both):
    header, rows, js = both
    assert len(rows) == len(js) == header["rows"]
    bad = []
    for i, a in enumerate(js):
        b = rows[i]
        if list(a) != list(b):
            bad.append((i, "field order"))
            continue
        bad += [(i, k) for k in a if not _same(a[k], b[k])]
    assert not bad, f"{len(bad)} mismatches, first {bad[:5]}"


@needs_artifacts
def test_lookup_equals_track_id_map(both):
    from app.services.recommender import KeyLookup
    _, rows, _ = both
    lk = KeyLookup(np.load(D / "keys.npy"), np.load(D / "key_order.npy"), rows)
    idm = json.loads((D / "track_id_map.json").read_text(encoding="utf-8"))
    bad = [k for k, v in idm.items() if lk.get(k) != v]
    assert not bad, f"{len(bad)} keys look up a different row, first {bad[:5]}"
    assert len(lk) == len(idm)
    assert lk.get("nobody at all:no such title") is None and "it_0" not in lk


@needs_artifacts
def test_clap_f16_and_feel_matrix():
    import faiss
    index = faiss.read_index(str(D / "music_index.faiss"))
    clap = index.reconstruct_n(0, index.ntotal)
    c16 = np.load(D / "clap.f16.npy")
    assert c16.dtype == np.float16 and c16.shape == clap.shape
    err = float(np.abs(c16.astype(np.float32) - clap).max())
    print(f"\nclap.f16 max abs error {err:.3e}")
    assert err <= 5e-4
    fi = faiss.read_index(str(D / "feel_index.faiss"))
    assert np.array_equal(np.load(D / "feel_matrix.npy"), fi.reconstruct_n(0, fi.ntotal))


@needs_artifacts
def test_recommender_loads_columnar(both):
    from app.services.recommender import ColumnarRows, recommender
    _, rows, js = both
    recommender.load()
    assert isinstance(recommender._metadata, ColumnarRows)
    for i in (0, 49, 4888, len(js) - 1):
        r = js[i]
        by_key = recommender.get_track_record(key=r["key"])
        by_id = recommender.get_track_record(track_id=r["id"])
        assert by_id == rows[recommender._track_id_map[r["id"]]]
        assert by_key == rows[recommender._track_id_map[r["key"]]]
    assert recommender._index.ntotal == len(js) and recommender._feel_index.ntotal == len(js)


# The no-fallback branch (reviewer 3b44f87): runs without artifacts, in a scratch directory
@pytest.mark.parametrize("case", ["missing", "unreadable"])
def test_incomplete_set_leaves_recommender_unloaded(case, tmp_path, monkeypatch, caplog):
    from app.services.recommender import COLUMNAR_FILES as FILES, MusicRecommender
    d = tmp_path / "ml" / "inference"
    d.mkdir(parents=True)
    for f in FILES:
        if case == "missing" and f == "strings.parquet":
            continue
        (d / f).write_bytes(b"not an artifact")
    monkeypatch.chdir(tmp_path)
    rec = MusicRecommender()
    with caplog.at_level("ERROR", logger="recommender"):
        rec.load()
    assert rec._loaded is False
    named = "strings.parquet" if case == "missing" else "header.json"
    assert any(named in r.getMessage() and "not loaded" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]
