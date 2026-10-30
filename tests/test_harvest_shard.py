"""
tests/test_harvest_shard.py: harvest_v3 end to end on one fixture artist (SCALE_PLAN section 5, WP4), into a
local store. Needs the network, ffmpeg and the CLAP checkpoint; skipped without them.
Asserts:
- the shard's features file and CLAP matrix have one row a song, in the same order, CLAP float16 and unit norm;
- the manifest's SHA256s are those of the files in the store, its describe commit is WP0's, it has seconds for A to F,
  and its dropped count is the dropped file's;
- every row has exactly the fields of a row of today's catalog_features.jsonl with the same has_lyrics;
- batched CLAP (64 windows a batch) equals P4's one-window-at-a-time path (redescribe_catalog.clap_windows) to 2e-3;
- the shard's previews are deleted.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "data"))
import harvest_v3 as hv  # noqa: E402

WP0_DESCRIBE_COMMIT = "e4c1b1bd2b4dd35a21560de82d280a587461ce99"
FIXTURE = ROOT / "tests" / "fixtures" / "harvest_fixture_shard.json"
CATALOG = ROOT / "ml" / "data" / "catalog_features.jsonl"

pytestmark = pytest.mark.skipif(not hv.rc.CLAP_CKPT.exists(), reason="LAION CLAP checkpoint not on this machine")


class ParityHarvester(hv.Harvester):
    """The real harvester; after stage D it embeds two songs the P4 way, before the previews are deleted."""

    def clap(self):
        out = super().clap()
        from app.audio.decode import decode_preview
        model = hv.rc.load_clap()
        self.parity = []
        for j, s in enumerate(self.songs[:2]):
            y = decode_preview((hv.PREVIEWS / f"{s['itunes_id']}.m4a").read_bytes())
            self.parity.append(float(np.abs(hv.rc.clap_windows(model, y) - out[j]).max()))
        return out


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("harvest")
    hv.WORK = tmp / "work"
    store = hv.LocalStore(tmp / "dataset")
    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    h = ParityHarvester("fixture_test", spec["artists"][:1], store, fixture=True, runner="test")
    manifest = h.run()
    return h, store, manifest


def test_files_and_manifest(shard):
    h, store, m = shard
    base = store.root
    feats = base / m["files"]["features"]
    clap = np.load(base / m["files"]["clap"])
    rows = [json.loads(l) for l in gzip.open(feats, "rt", encoding="utf-8")]
    assert m["songs"] == len(rows) == clap.shape[0] > 0 and clap.shape[1] == 512 and clap.dtype == np.float16
    assert np.allclose(np.linalg.norm(clap.astype(np.float32), axis=1), 1.0, atol=2e-3)
    assert [r["key"] for r in rows] == m["keys"]
    for kind, path in m["files"].items():
        assert hashlib.sha256((base / path).read_bytes()).hexdigest() == m["sha256"][kind], kind
    assert m["describe_commit"] == WP0_DESCRIBE_COMMIT
    assert set(m["seconds_per_stage"]) == set("ABCDEF")
    dropped = (base / m["files"]["dropped"]).read_text(encoding="utf-8").splitlines()
    assert m["dropped"] == len(dropped)
    stored = json.loads((base / f"{hv.PREFIX}/shard_fixture_test.manifest.json").read_text(encoding="utf-8"))
    assert stored["seconds_per_stage"] == m["seconds_per_stage"]


@pytest.mark.skipif(not CATALOG.exists(), reason="catalog_features.jsonl not on this machine")
def test_fields_match_catalogue(shard):
    _, store, m = shard
    # today's rows come in two field sets: with a lyric file, and without (no theme_vec, rule 17)
    ref: dict[bool, set] = {}
    with open(CATALOG, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ref.setdefault(bool(row.get("has_lyrics")), set(row))
            if len(ref) == 2:
                break
    rows = [json.loads(l) for l in gzip.open(store.root / m["files"]["features"], "rt", encoding="utf-8")]
    for r in rows:
        want = ref[bool(r.get("has_lyrics"))]
        missing, extra = want - set(r), set(r) - want
        assert not missing and not extra, (r["key"], sorted(missing), sorted(extra))


def test_batched_clap_equals_p4_path(shard):
    h, _, _ = shard
    print(f"\nbatched vs one-window CLAP, max abs difference: {h.parity}")
    assert h.parity and max(h.parity) <= 2e-3


def test_previews_deleted(shard):
    h, _, _ = shard
    left = [s for s in h.songs if (hv.PREVIEWS / f"{s['itunes_id']}.m4a").exists()]
    assert not left
