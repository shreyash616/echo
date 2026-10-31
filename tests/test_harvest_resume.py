"""
tests/test_harvest_resume.py: harvest_v3's claim and resume logic (SCALE_PLAN section 5, WP4), on a local store.
Asserts:
- the lowest shard that is neither finished nor claimed is taken;
- another runner's claim younger than 6 hours is skipped, an older one is taken over;
- a crash mid-shard leaves the runner's claim, and its next run takes that shard again, from stage A;
- a finished shard joins catalog/v3/MANIFEST.json, its keys join catalog/v3/keys.txt, and the claim is released;
- the next shard deduplicates against those keys;
- with the next shard's stage A overlapped (README c2f8090), a crash while holding two claims leaves both, and the
  restart takes the lower one first, then the other; the overlapped A deduplicates against the running shard's songs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml" / "data"))
import harvest_v3 as hv  # noqa: E402

SHARDS = ["a00000", "a00001", "a00002"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(hv, "WORK", tmp_path / "work")
    s = hv.LocalStore(tmp_path / "dataset")
    (tmp_path / "dataset" / "discovery").mkdir(parents=True)
    rows = [{"deezer_id": 100 + i, "name": f"artist {i}", "source": "related", "depth": 1} for i in range(6)]
    (tmp_path / "dataset" / "discovery" / "artists.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    spec = {"shard_size": 2, "shards": [{"shard": sid, "start": 2 * i, "end": 2 * i + 2} for i, sid in enumerate(SHARDS)]}
    (tmp_path / "dataset" / "discovery" / "shards.json").write_text(json.dumps(spec), encoding="utf-8")
    return s


class FakeHarvester:
    """Stands in for the network and GPU stages: records the stages it ran and what it was given."""
    calls: list[dict] = []
    crash_on: str | None = None

    def __init__(self, shard, artists, store, global_keys=None, global_itunes=None, fixture=False, runner=None):
        self.shard, self.artists, self.keys, self.runner = shard, artists, set(global_keys or ()), runner
        self.a_overlapped, self.songs = False, []

    def _stage(self, stage):
        self.entry["stages"].append(stage)
        if FakeHarvester.crash_on == f"{self.shard}:{stage}":
            raise RuntimeError(f"crash in {self.shard} stage {stage}")

    def stage_a(self):
        self.entry = {"shard": self.shard, "stages": [], "global_keys": set(self.keys),
                      "artists": [a["deezer_id"] for a in self.artists]}
        FakeHarvester.calls.append(self.entry)
        self._stage("A")
        self.songs = [{"key": f"artist {a['deezer_id'] - 100}:song", "itunes_id": str(a["deezer_id"])} for a in self.artists]

    def run_rest(self):
        for stage in "BCDEF":
            self._stage(stage)
        keys = [s["key"] for s in self.songs]
        return {"shard": self.shard, "songs": len(keys), "dropped": 0, "sha256": {}, "describe_commit": "x",
                "date": "d", "runner": self.runner, "keys": keys, "itunes_ids": [s["itunes_id"] for s in self.songs]}

    def run(self):
        self.stage_a()
        return self.run_rest()


def test_lowest_unclaimed_and_claims(store):
    assert hv.next_shard(store, SHARDS, "box") == "a00000"
    hv.claim(store, "a00000", "space", now=1_000_000.0)
    assert hv.next_shard(store, SHARDS, "box", now=1_000_000.0 + 3600) == "a00001"  # fresh foreign claim skipped
    assert hv.next_shard(store, SHARDS, "box", now=1_000_000.0 + 7 * 3600) == "a00000"  # stale claim taken over


def test_crash_resumes_same_shard_from_stage_a_then_finishes(store):
    FakeHarvester.calls, FakeHarvester.crash_on = [], "a00000:D"
    with pytest.raises(RuntimeError):
        hv.run_next(store, "box", harvester_cls=FakeHarvester)
    assert store.read_json(hv.claim_path("box", "a00000"))["shard"] == "a00000"   # the claim survives the crash
    assert hv.finished_shards(store) == {}
    FakeHarvester.crash_on = None
    m = hv.run_next(store, "box", harvester_cls=FakeHarvester)
    assert m["shard"] == "a00000"
    assert FakeHarvester.calls[1]["stages"] == list("ABCDEF")                  # rerun from stage A
    assert FakeHarvester.calls[1]["artists"] == [100, 101]
    assert list(hv.finished_shards(store)) == ["a00000"]
    assert store.read_json(hv.claim_path("box", "a00000")) is None               # claim released
    keys, ids = hv.global_seen(store)
    assert keys == {"artist 0:song", "artist 1:song"} and ids == {"100", "101"}
    m2 = hv.run_next(store, "box", harvester_cls=FakeHarvester)
    assert m2["shard"] == "a00001"
    assert FakeHarvester.calls[2]["global_keys"] == {"artist 0:song", "artist 1:song"}  # dedupe set passed on
    assert store.read_json("catalog/v3/MANIFEST.json")["songs"] == 4


def test_nothing_left(store):
    for sid in SHARDS:
        hv.finish(store, FakeHarvester(sid, [], store, runner="box").run(), "box")
    assert hv.next_shard(store, SHARDS, "box") is None
    assert hv.run_next(store, "box", harvester_cls=FakeHarvester) is None


def test_two_claims_crash_then_restart_lower_first(store):
    FakeHarvester.calls, FakeHarvester.crash_on = [], "a00000:D"
    with pytest.raises(RuntimeError):
        hv.run_shards(store, "box", 3, harvester_cls=FakeHarvester)
    held = sorted(c["shard"] for c in hv.claims(store) if c["runner"] == "box")
    assert held == ["a00000", "a00001"]                                         # both claims survive the crash
    pre = next(c for c in FakeHarvester.calls if c["shard"] == "a00001")
    assert pre["global_keys"] == {"artist 0:song", "artist 1:song"}            # overlapped A saw the running shard's songs
    FakeHarvester.crash_on = None
    FakeHarvester.calls = []
    done = hv.run_shards(store, "box", 2, harvester_cls=FakeHarvester)
    assert [m["shard"] for m in done] == ["a00000", "a00001"]                   # lower claim first, then the other
    assert [c["stages"] for c in FakeHarvester.calls] == [list("ABCDEF")] * 2    # each from stage A
    assert hv.claims(store) == []                                                # both released
    assert list(hv.finished_shards(store)) == ["a00000", "a00001"]
