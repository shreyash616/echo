"""
tests/test_discover_bollywood.py: ml/data/discover_bollywood.py (SCALE_PLAN WP6-IN) on saved Wikipedia wikitext and saved
iTunes India JSON (tests/fixtures/bollywood), no network.
Asserts:
- a year's film list is the italic links of its tables, disambiguators stripped, no cast, studio or footnote source;
- an album is accepted on title prefix, release year within one, and Bollywood genre or a seed credit, and only then;
- a row is stage A's song plus the discovery fields, and rows without a preview or with an unclean name are skipped;
- end to end: rows are unique by key and iTunes id and miss catalog/v3/keys.txt, artist rows are Bollywood only,
  the MANIFEST counts agree with the rows, shards.json covers the rows in order, and the cap stops the requests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "data"))
import discover_bollywood as db  # noqa: E402
import harvest_v3 as hv  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "bollywood"
STAGE_A = {"id", "itunes_id", "deezer_id", "key", "title", "artist", "album", "albumArtUrl", "previewUrl", "durationMs",
           "genre", "year", "decade"}
EXTRA = {"film", "film_year", "collection_id", "discovery_source"}
SEEDS = {db._norm_str(s) for s in db.SEED_ARTISTS}


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def films(year: int) -> list[str]:
    return db.parse_films(load(f"wiki_{year}.json")["parse"]["wikitext"])


def test_film_lists():
    f75, f13, f25 = films(1975), films(2013), films(2025)
    assert {"Sholay", "Pratigya", "Khel Khel Mein", "Chupke Chupke", "Deewaar", "Aa Jaa Sanam"} <= set(f75)
    assert {"Dhoom 3", "Aashiqui 2", "Table No. 21", "Inkaar"} <= set(f13)
    for fl in (f75, films(1994), f13, f25):
        assert len(fl) > 60 and len(fl) == len({t.casefold() for t in fl})
        assert not any("film)" in t or "''" in t or "[[" in t for t in fl)
    # cast and studios are plain links; the newspaper sits in a footnote template
    assert not {"Dharmendra", "Amitabh Bachchan", "Yash Raj Films", "Aditya Datt"} & set(f75 + f13)
    assert "Hindustan Times" not in f25


def test_album_acceptance():
    sholay = [a["collectionId"] for a in load("itunes_album_sholay.json")["results"] if db.accept_album(a, "Sholay", 1975, SEEDS)]
    # the 1975 soundtrack (Bollywood) and its remaster (Soundtrack genre, R.D. Burman credited); not the 2015 mashup,
    # the RRR song or the hip-hop singles
    assert sholay == [1443485527, 6811025777]
    a2 = [a["collectionId"] for a in load("itunes_album_aashiqui2.json")["results"] if db.accept_album(a, "Aashiqui 2", 2013, SEEDS)]
    # not the 1989 Aashiqui, the 2022 reissue, or the Tamil, Bengali, Gujarati and EP versions
    assert a2 == [1073359412]
    # the 1990 film Aashiqui takes its own 1989 soundtrack, not Aashiqui 2's albums
    a1 = [a["collectionId"] for a in load("itunes_album_aashiqui2.json")["results"] if db.accept_album(a, "Aashiqui", 1990, SEEDS)]
    assert a1 == [1073377141]
    # the remaster is Soundtrack genre: accepted through R.D. Burman's credit only
    assert [a["collectionId"] for a in load("itunes_album_sholay.json")["results"] if db.accept_album(a, "Sholay", 1975, set())] == [1443485527]


def clean_tracks(items: list[dict]) -> list[dict]:
    from harvest_v2 import _is_clean_track_name
    return [it for it in items if it.get("wrapperType") == "track" and it.get("previewUrl")
            and _is_clean_track_name(it.get("trackName", ""))]


def test_track_rows():
    items = load("itunes_lookup_aashiqui2.json")["results"]
    rows = [r for r in (db.track_row(it, "film", "Aashiqui 2", 2013) for it in items) if r]
    assert len(rows) == len(clean_tracks(items)) == 10  # 11 songs with previews, "Tum Hi Ho (Remix)" skipped
    for r in rows:
        assert set(r) == STAGE_A | EXTRA
        assert r["film"] == "Aashiqui 2" and r["film_year"] == 2013 and r["collection_id"] == 1073359412
        assert r["id"] == f"it_{r['itunes_id']}" and r["decade"] == "2010s" and r["genre"] == "Bollywood"
    it = dict(next(x for x in items if x.get("wrapperType") == "track"))
    assert db.track_row({**it, "previewUrl": None}, "film") is None
    assert db.track_row({**it, "trackName": "Tum Hi Ho (Remix)"}, "film") is None


class FakeFetcher(db.Fetcher):
    """Answers from the fixtures: two years' lists, two films' albums, one album's songs, one singer's search."""

    def __init__(self):
        self.counts, self.calls = __import__("collections").Counter(), []

    def get(self, host, url, params, spacing):
        self.calls.append((host, url.rsplit("/", 1)[-1], dict(params)))
        self.counts[f"{host} 200"] += 1
        if host == "wikipedia":
            year = params["page"].rsplit(" ", 1)[-1]
            return load(f"wiki_{year}.json") if (FIX / f"wiki_{year}.json").exists() else {"error": {"code": "missingtitle"}}
        if url.endswith("lookup"):
            return load("itunes_lookup_aashiqui2.json") if params["id"] == "1073359412" else {"results": []}
        if params.get("entity") == "album":
            return {"Sholay": load("itunes_album_sholay.json"), "Aashiqui 2": load("itunes_album_aashiqui2.json")}.get(
                params["term"], {"results": []})
        return load("itunes_artist_kishore.json") if params["term"] == "Kishore Kumar" else {"results": []}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "WORK", tmp_path / "work")
    store = hv.LocalStore(tmp_path / "dataset")
    tum_hi_ho = next(it for it in load("itunes_lookup_aashiqui2.json")["results"] if it.get("trackName") == "Tum Hi Ho")
    seen_key = db.track_row(tum_hi_ho, "film")["key"]
    fetch = FakeFetcher()
    rows, manifest = db.discover(fetch, {seen_key}, set(), range(1975, 1977), seeds=["Kishore Kumar"])
    db.write(store, rows, manifest)
    return store, rows, manifest, fetch, seen_key


def test_end_to_end(run):
    store, rows, m, fetch, seen_key = run
    assert rows and len({r["key"] for r in rows}) == len(rows) == len({r["itunes_id"] for r in rows})
    assert all(set(r) == STAGE_A | EXTRA for r in rows)
    film = [r for r in rows if r["discovery_source"] == "film"]
    artist = [r for r in rows if r["discovery_source"] == "artist"]
    assert not film  # Sholay's albums answer no songs in the fake, so 1975's film rows are empty
    assert artist and all(r["genre"] == "Bollywood" and r["film"] is None for r in artist)
    assert m["rows"] == len(rows) and sum(m["by_decade"].values()) == len(rows)
    assert m["by_source"] == {"artist": len(artist)}
    assert m["films"]["by_year"]["1975"] == {"listed": len(films(1975)), "matched": 1, "page": True}
    assert m["films"]["by_year"]["1976"]["page"] is False and m["films"]["matched"] == 1
    assert m["albums_accepted"] == 1  # with only Kishore Kumar as seed, the remaster's R.D. Burman credit does not count
    # one album search a film, one lookup an accepted album, one song search a name (the seed, then the credits)
    kinds = [(c[0], c[1], c[2].get("entity")) for c in fetch.calls]
    assert kinds.count(("wikipedia", "api.php", None)) == 2
    assert kinds.count(("itunes", "search", "album")) == len(films(1975))
    assert kinds.count(("itunes", "lookup", "song")) == 1
    assert kinds.count(("itunes", "search", "song")) == m["artist_names"] == m["artists_searched"] == 2  # + R.D. Burman
    assert all(c[2].get("country") == "IN" for c in fetch.calls if c[0] == "itunes")  # the India storefront throughout
    stored = [json.loads(l) for l in (store.root / "discovery/bollywood/tracks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stored == rows
    spec = json.loads((store.root / "discovery/bollywood/shards.json").read_text(encoding="utf-8"))
    assert spec["tracks"] == "discovery/bollywood/tracks.jsonl"
    assert spec["shards"][0]["start"] == 0 and spec["shards"][-1]["end"] == len(rows)
    assert all(s["source"] == "tracks" and s["end"] - s["start"] <= db.SHARD_ROWS for s in spec["shards"])
    assert [s["shard"] for s in spec["shards"]] == [f"b{i:05d}" for i in range(len(spec["shards"]))]
    assert json.loads((store.root / "discovery/bollywood/MANIFEST.json").read_text(encoding="utf-8"))["rows"] == len(rows)


def test_film_rows_and_keys_txt(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "WORK", tmp_path / "work")
    items = load("itunes_lookup_aashiqui2.json")["results"]
    seen_key = db.track_row(next(it for it in items if it.get("trackName") == "Tum Hi Ho"), "film")["key"]
    rows, m = db.discover(FakeFetcher(), {seen_key}, set(), range(2013, 2014), seeds=[])
    film = [r for r in rows if r["discovery_source"] == "film"]
    assert film and seen_key not in {r["key"] for r in rows}
    assert all(r["film"] == "Aashiqui 2" and r["film_year"] == 2013 for r in film)
    assert len(film) == len(clean_tracks(items)) - 1
    assert m["by_decade"] == {"2010s": len(rows)}


def test_cap_stops_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "WORK", tmp_path / "work")
    fetch = FakeFetcher()
    rows, m = db.discover(fetch, set(), set(), range(1975, 1976), seeds=["Kishore Kumar", "Lata Mangeshkar"], max_tracks=5)
    assert len(rows) == 5 and m["artists_searched"] == 1
    assert sum(1 for c in fetch.calls if c[2].get("entity") == "song" and c[1] == "search") == 1


def test_shard_spec_sizes():
    rows = [{"key": str(i)} for i in range(2 * db.SHARD_ROWS + 7)]
    spec = db.shard_spec(rows)
    assert [(s["start"], s["end"]) for s in spec["shards"]] == [(0, db.SHARD_ROWS), (db.SHARD_ROWS, 2 * db.SHARD_ROWS),
                                                                  (2 * db.SHARD_ROWS, 2 * db.SHARD_ROWS + 7)]
