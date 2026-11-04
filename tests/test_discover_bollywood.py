"""
tests/test_discover_bollywood.py: ml/data/discover_bollywood.py (SCALE_PLAN WP6-IN) on saved Wikipedia wikitext and saved
iTunes India JSON (tests/fixtures/bollywood), no network.
Asserts:
- a year's film list is the italic links of its tables, disambiguators stripped, no cast, studio or footnote source;
- an album is accepted on title prefix, release year within one, and Bollywood genre or a seed credit, and only then;
- a row is stage A's song plus the discovery fields, and rows without a preview or with an unclean name are skipped;
- end to end: rows are unique by key and iTunes id and miss catalog/v3/keys.txt, artist rows are Bollywood only,
  the MANIFEST counts agree with the rows, shards.json covers the rows in order, and the cap stops the requests.
README 871d37d:
- a seed matches its credit with or without spaced initials ("A.R. Rahman", "A. R. Rahman"), and is searched once;
- an unclosed note template costs only the rest of its line, logged once a page; a self-closing ref with a "/" in
  an attribute does not swallow the films up to the next </ref>;
- an iTunes request that fails eight times stops the run, with the status counts, instead of reading as a miss;
  Wikipedia's reads as a missing page;
- the MANIFEST counts film-path rows by genre.
README 8756c5f, round 2:
- its names are the round 1 rows' credits in first-appearance order, without round 1's names;
- round 1's names are rebuilt from the cached film pass, and an offline Fetcher sends nothing;
- its rows miss round 1's, and shards2.json starts at b00010 over tracks2.jsonl.
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
SEEDS = {db._name_key(s) for s in db.SEED_ARTISTS}


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
    assert m["film_rows_by_genre"] == {"Bollywood": len(film)}


def test_cap_stops_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "WORK", tmp_path / "work")
    fetch = FakeFetcher()
    rows, m = db.discover(fetch, set(), set(), range(1975, 1976), seeds=["Kishore Kumar", "Lata Mangeshkar"], max_tracks=5)
    assert len(rows) == 5 and m["artists_searched"] == 1
    assert sum(1 for c in fetch.calls if c[2].get("entity") == "song" and c[1] == "search") == 1


def test_seed_credit_with_spaced_initials(tmp_path, monkeypatch):
    album = {"collectionName": "Roja (Original Motion Picture Soundtrack)", "releaseDate": "1992-08-15T07:00:00Z",
             "primaryGenreName": "Soundtrack", "artistName": "A. R. Rahman"}
    assert db.accept_album(album, "Roja", 1992, SEEDS)
    assert not db.accept_album({**album, "artistName": "Ilaiyaraaja"}, "Roja", 1992, SEEDS)
    monkeypatch.setattr(db, "WORK", tmp_path / "work")
    fetch = FakeFetcher()
    _, m = db.discover(fetch, set(), set(), range(1975, 1975), seeds=["A.R. Rahman", "A. R. Rahman", "Kishore Kumar"])
    assert m["artist_names"] == m["artists_searched"] == 2
    assert [c[2]["term"] for c in fetch.calls] == ["A.R. Rahman", "Kishore Kumar"]


def test_unclosed_template_keeps_later_films(caplog):
    text = ("{|\n|-\n| ''[[Alpha (1980 film)|Alpha]]'' {{efn|source ''[[Hindustan Times]]''\n|-\n| ''[[Beta]]''\n"
            "|-\n| ''[[Gamma]]''{{cite web|title=''[[Somewhere]]''\n|-\n| ''[[Delta]]''{{efn|''[[Closed]]''}}\n|}\n")
    with caplog.at_level("WARNING", logger="discover_bollywood"):
        assert db.parse_films(text, 1980) == ["Alpha", "Beta", "Gamma", "Delta"]
    warnings = [r.getMessage() for r in caplog.records]
    assert warnings == ["1980: 2 unclosed note template(s), each cut to the end of its line"]


def test_ref_with_slash_in_attribute():
    text = "{|\n|-\n| ''[[Alpha]]''<ref name=\"a/b\"/>\n|-\n| ''[[Beta]]''<ref>x</ref>\n|-\n| ''[[Gamma]]''\n|}\n"
    assert db.parse_films(text) == ["Alpha", "Beta", "Gamma"]


def test_exhausted_itunes_stops_the_run(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(db, "PAUSE", tmp_path / "PAUSE")
    monkeypatch.setattr(db.time, "sleep", lambda s: None)
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(403)))
    fetch = db.Fetcher(tmp_path / "cache", client)
    with pytest.raises(SystemExit, match="8 attempts failed") as e:
        fetch.itunes("search", {"term": "Sholay", "entity": "album"})
    assert "'itunes 403': 8" in str(e.value) and "'itunes exhausted': 1" in str(e.value)
    assert fetch.wiki(1975) is None and fetch.counts["wikipedia 403"] == 8
    assert not list((tmp_path / "cache").iterdir())
    ok = db.Fetcher(tmp_path / "cache", httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404))))
    assert ok.itunes("lookup", {"id": "1"}) == [] and ok.counts["itunes 404"] == 1  # a plain non-200 is a miss


def test_shard_spec_sizes():
    rows = [{"key": str(i)} for i in range(2 * db.SHARD_ROWS + 7)]
    spec = db.shard_spec(rows)
    assert [(s["start"], s["end"]) for s in spec["shards"]] == [(0, db.SHARD_ROWS), (db.SHARD_ROWS, 2 * db.SHARD_ROWS),
                                                                  (2 * db.SHARD_ROWS, 2 * db.SHARD_ROWS + 7)]


def test_round2_names_skip_round1():
    rows1 = [{"artist": "Arijit Singh, Shreya Ghoshal"}, {"artist": "New Singer & Arijit Singh"},
             {"artist": "A. R. Rahman"}, {"artist": "Another Voice, New Singer"}]
    names, skipped = db.round2_names(rows1, ["Arijit Singh", "Shreya Ghoshal", "A.R. Rahman"])
    assert names == ["New Singer", "Another Voice"]  # first appearance order, one per name
    assert skipped == 3  # spaced initials still match round 1's name


def test_round1_names_rebuilt_from_the_cache(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(db, "WORK", tmp_path / "work")
    fetch = FakeFetcher()
    _, m = db.discover(fetch, set(), set(), range(1975, 1977), seeds=["Kishore Kumar"])
    terms = [c[2]["term"] for c in fetch.calls if c[1] == "search" and c[2].get("entity") == "song"]
    replay = FakeFetcher()
    assert db.round1_names(replay, range(1975, 1977), ["Kishore Kumar"]) == terms and len(terms) == m["artist_names"]
    assert not any(c[2].get("entity") == "song" and c[1] == "search" for c in replay.calls)  # no artist search
    def refuse(req):
        raise AssertionError("an offline Fetcher sent a request")
    off = db.Fetcher(tmp_path / "cache", httpx.Client(transport=httpx.MockTransport(refuse)), offline=True)
    assert off.itunes("search", {"term": "Sholay", "entity": "album"}) == [] and off.counts == {"itunes not cached": 1}


def test_round2_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "WORK", tmp_path / "work")
    store = hv.LocalStore(tmp_path / "dataset")
    kishore = [it for it in load("itunes_artist_kishore.json")["results"] if it.get("primaryGenreName") == "Bollywood"]
    already = next(r for r in (db.track_row(it, "artist") for it in kishore) if r)  # a round 1 row
    rows1 = [{**already, "artist": "Kishore Kumar & Lata Mangeshkar"}]
    fetch = FakeFetcher()
    rows, m = db.discover_round2(fetch, set(), set(), rows1, ["Lata Mangeshkar"])
    assert [c[2]["term"] for c in fetch.calls] == ["Kishore Kumar"]
    assert rows and already["key"] not in {r["key"] for r in rows} and already["itunes_id"] not in {r["itunes_id"] for r in rows}
    assert all(r["genre"] == "Bollywood" and r["discovery_source"] == "artist" for r in rows)
    assert m["names_searched"] == m["artist_names"] == 1 and m["names_skipped_as_round_1"] == 1
    assert m["rows"] == len(rows) and m["max_tracks"] == db.ROUND2_MAX_TRACKS
    db.write(store, rows, m, suffix="2", first_shard=10)
    spec = json.loads((store.root / "discovery/bollywood/shards2.json").read_text(encoding="utf-8"))
    assert spec["tracks"] == "discovery/bollywood/tracks2.jsonl" and spec["shards"][0]["shard"] == "b00010"
    stored = [json.loads(l) for l in (store.root / "discovery/bollywood/tracks2.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stored == rows
    assert json.loads((store.root / "discovery/bollywood/MANIFEST2.json").read_text(encoding="utf-8"))["names_searched"] == 1
    assert not (store.root / "discovery/bollywood/tracks.jsonl").exists()  # round 1's files are left alone
