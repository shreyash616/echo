# Echo: scale plan, one million songs (round 3, "make it work" scope)

Status: written 2026-10-28 against commit `5693980` on `main`. This plan starts only after
`FIX_PLAN_2.md` section 6 passes on the 20,000-row catalogue; it reuses that describe code unchanged
and runs it a million times. Scope is "make it work": the catalogue exists, loads on the free Space,
and the benchmark can be run on it. Latency, index compression and catalogue stratification are
deferred and listed in section 9 so nobody builds them early. Where this file and `FIX_PLAN_2.md`
disagree, `FIX_PLAN_2.md` wins until its section 6 passes, then this file wins.

Sections: 1 why the catalogue is the search space, 2 machines and the division of labour, 3 the
50 GB disk budget, 4 discovery, 5 the harvester, 6 storage and index at a million, 7 serving
changes, 8 tests and acceptance, 9 deferred, 10 rules, 11 order and gates, 12 amendment to
`FIX_PLAN_2.md` T1.

## 1. Why the catalogue is the search space

Every suggestion is a row of the catalogue found through the two FAISS indexes. A song that has not
been fetched and described has no vector and cannot be returned. Today the catalogue is 20,000 songs
chosen by a hand list of 117 artists and the artists that recur in Deezer mood playlists, about
2,700 artists at seven or eight songs each, 85% labelled English. The best scoring over that set can
only reorder it. The owner's requirement before any outside user tries the app is at least
1,000,000 described songs. Nothing is labelled by a human at any point; a song enters when the
machine has listened to its 30-second preview.

## 2. Machines and the division of labour

| Job | Where | Why there |
|---|---|---|
| Discovery: related-artist walk, charts, genre pages on Deezer | the serving Space (`shreyash616/echo-backend`), as a background task or a one-off script run through its shell | Deezer answers these endpoints only from outside India |
| iTunes lookups: artist id, song list with preview URLs | the Windows box and the Space, split by artist shard | each IP has its own rate budget (about 20 requests a minute); two runners halve the wall time |
| Preview fetch, decode, DSP describe, lyrics, language, theme | the Windows box (Ryzen 7600, 12 threads), under WSL2 | about 1 s per song per thread: a million in roughly 28 CPU-hours |
| CLAP vectors | the Windows box, RTX 3080 | batched HTSAT at 48 kHz, three 10 s windows a song: a few hours for a million |
| Shard upload, manifest, index build | the Windows box writes to the HF dataset `shreyash616/echo-ml-artifacts`; the index is built from the dataset | one source of truth; the Mac and the Space never hold a private copy |

The Windows box runs everything under WSL2 (Ubuntu): ffmpeg, librosa, pyloudnorm, laion_clap with
CUDA, sentence-transformers, lingua, httpx, huggingface_hub. Native Windows paths and audio libraries
are not supported by this plan. Expected wall time for a million: four to five days, bounded by the
iTunes rate limit and the preview downloads (about 1 TB in total), not by compute.

## 3. The 50 GB disk budget on the Windows box

Previews are never kept beyond the shard being processed. The budget, with headroom:

| Item | Size | Note |
|---|---|---|
| WSL2 Ubuntu, Python venv, torch with CUDA, laion_clap checkpoint, MiniLM, lingua models | about 9 GB | one-off |
| Previews for the current shard (5,000 songs at about 1 MB) | at most 5 GB | deleted when the shard's upload is verified |
| Finished shard outputs waiting for upload (features gz, CLAP float16, manifest) | about 10 MB a shard | deleted after upload verification |
| Lyrics text cache (`ml/data/lyrics/<id>.txt`) | about 2 KB a song, 2 GB for a million | kept, it is re-used by re-describes |
| Decoded audio | none on disk | ffmpeg pipes to memory, as `decode.py` does today |
| Logs and manifests | under 1 GB | |
| Total steady state | under 20 GB | leaves 30 GB headroom |

Hard rule: the harvester checks free disk before starting a shard and refuses to start one with
less than 8 GB free. It never downloads a second shard's previews while the first is on disk.
`ml/data/previews` is a scratch directory for the current shard only; it is gitignored and never
uploaded.

## 4. Discovery: building the artist universe (on the Space)

Goal: an artist list of 80,000 to 120,000 artists with Deezer ids, written to the dataset as
`discovery/artists.jsonl` (fields: `deezer_id`, `name`, `nb_fan`, `source`, `depth`).

1. Seeds: every artist in `tests/feel_benchmark_v2.json` (seeds, positives, negatives), the 117
   names in `ml/data/harvest_seed_artists.json` resolved to Deezer ids, and the top 100 artists from
   Deezer charts for each of SG, IN, US, GB, ES, MX, BR, AR, KR, JP, FR, DE, IT, PK, BD, ID, PH, TR,
   NG, ZA (endpoint `/chart/{country}/artists` or `/editorial/{id}/charts`, whichever Deezer serves).
2. Breadth-first walk over `/artist/{id}/related` (20 per call), depth-first from no seed; stop at
   120,000 unique artists or when a depth has no new artists. Keep `nb_fan` for the per-artist cap.
3. Genre pages: `/genre` then `/genre/{id}/artists` for every genre, appended with `source =
   "genre"`, so small scenes (garage, psych, qawwali, dangdut, enka) are present even if the walk
   never reaches them.
4. Rate limits: Deezer allows about 50 requests per 5 seconds; sleep on 4xx and retry with
   backoff; the walk is resumable from the last written line.
5. Done when: `discovery/artists.jsonl` has at least 80,000 rows, at least 30 countries among the
   chart seeds, and the file's SHA256 is in `discovery/MANIFEST.json` with the counts per source.

The artist list is then split into shards of 250 artists (`discovery/shards.json`: shard id to
artist id range). At a cap of 20 songs an artist, 250 artists is about 5,000 songs, one harvest
shard.

## 5. The harvester (on the Windows box; the Space may take shards too)

`ml/data/harvest_v3.py`, one process per shard, staged so disk and GPU use are predictable:

**Stage A, resolve.** For each artist in the shard: iTunes `search?term=<name>&entity=musicArtist`
to get the artist id (first result whose name matches after normalisation), then
`lookup?id=<artistId>&entity=song&limit=200&country=US`. Keep songs with a `previewUrl`, in iTunes'
order, at most 20 an artist, deduplicated by canonical `artist:title` key against the global key set
in the manifest. Respect 20 requests a minute; on 403 or 429 back off exponentially and keep going.
Output: the shard's song list in memory.

**Stage B, fetch.** Download every preview to `ml/data/previews/<itunes_id>.m4a` with 8 parallel
connections. Songs that fail after one retry are written to the shard's `dropped.jsonl` with the
reason and are not described.

**Stage C, describe.** `describe_audio` from `backend/app/audio/features.py` over the shard's
previews using a `multiprocessing.Pool` of 10 workers, with the header cut points of the current
catalogue passed in (for `kick_class` and `restraint_bucket`). Output: one descriptor dict a song.

**Stage D, CLAP.** One process, CUDA: for all songs in the shard, decode at 48 kHz, cut three
consecutive 10 s windows from 0 s as `recommender.py` does, batch 64 windows through the official
LAION loader (`CLAP_Module(enable_fusion=False, amodel="HTSAT-base")` with the music checkpoint),
mean the three windows, L2-normalise, store float16. The audio tower weights are loaded once per
shard.

**Stage E, text.** LRCLIB fetch for songs without a cached lyric file (paced at 5 a second), then
language (round 2 T4 resolution order), theme chunks (round 2 T8), `has_lyrics` true only when a
lyric file exists.

**Stage F, write and upload.** Three files to the dataset under `catalog/v3/`:
`shard_00042.features.jsonl.gz` (one row a song, the same fields as `catalog_features.jsonl`
today), `shard_00042.clap.f16.npy` (n x 512 float16, row order equal to the features file),
`shard_00042.manifest.json` (artist range, song count, dropped count, SHA256 of both files,
describe code commit hash, date). After `hf_hub_download` of the uploaded manifest confirms the
SHA256s, the shard id is appended to `catalog/v3/MANIFEST.json` and the shard's previews are deleted.

**Resume.** On start the harvester reads `catalog/v3/MANIFEST.json` and `discovery/shards.json`,
takes the lowest unfinished shard not claimed by another runner (`claims/<runner>.json` with a
timestamp; a claim older than 6 hours is stale), and runs stages A to F. A crash mid-shard restarts
that shard from stage A; at 5,000 songs that costs under an hour. Two runners (Windows and the
Space) never collide because claims are per shard.

**Throughput check, built in.** Each shard manifest records seconds per stage. After the first
ten shards the log entry states songs per hour and the projected finish date; the owner decides
whether to add the Space as a second runner.

## 6. Storage and index at a million

The serving format changes from one JSON file to columns, because `json.load` of the metadata at a
million rows needs about 20 GB of RAM and the free Space has 16. `ml/inference/build_index_v3.py`
reads every finished shard from the dataset and writes to the dataset root:

| File | Shape | Loaded by |
|---|---|---|
| `clap.f16.npy` | n x 512 float16 (1 GB at a million) | converted to float32 at load into a flat `IndexFlatIP` (2 GB RAM); the file is also the replica's input |
| `feel_matrix.npy` | n x 17 float32 | flat `IndexFlatL2`, 68 MB |
| `features.npz` | one float32 array a numeric descriptor (`felt_bpm`, `pulse_rate`, `darkness`, `grit`, ... plus `felt_bpm_dist` as two arrays of candidates and probabilities, `theme_vec` as n x 10, `anchor_profile` as n x 24) | numpy, memory-mapped |
| `strings.parquet` | `key`, `id`, `itunes_id`, `title`, `artist`, `album`, `albumArtUrl`, `previewUrl`, `language`, `language_source`, `theme`, `vibes`, `kick_class`, `restraint_bucket` | pyarrow, loaded once, about 300 MB |
| `keys.npy` plus `key_order.npy` | sorted keys and the row permutation | `np.searchsorted` for `artist:title` lookup, replacing `track_id_map.json` |
| `header.json` | means, stds, cut points, feel axes and weights, counts, build commit | as today |

`recommender.py` gets one loader change: `get_track_record(idx)` assembles a dict from the columns
on demand instead of indexing a list of dicts. `score_lib.py` reads the same files, so
`report.py` keeps working with `ECHO_ARTIFACT_DIR` pointing at the dataset download.

RAM on the Space at a million: about 2 GB for the CLAP index, 70 MB feel index, under 1 GB for
features and strings, plus the CLAP model for cold seeds. Under 5 GB total.

## 7. Serving changes (minimal)

- Loader as in section 6. No other change to the recommendation path; pool sizes and weights stay
  what round 2 left.
- `/api/search` unchanged. Cold seeds unchanged: a searched song outside the catalogue is described
  on request and kept in `extra_records.jsonl` as today.
- The result list keeps at most 2 songs by the same artist (today's MMR artist handling), which at
  a million rows is what stops one artist's 20 songs filling the top 20.
- Startup must finish inside the Space's health timeout: load the float16 file and convert in
  chunks; assert `index.ntotal == len(keys)` and the same four assertions round 1 added.

## 8. Tests and acceptance

Tests, all new, under `tests/`:

- `test_harvest_shard.py`: runs stages A to F on a fixture shard of 5 artists (not benchmark
  artists) against the live iTunes API, asserts every output file, the manifest SHA256s, the
  per-artist cap, and that `ml/data/previews` is empty afterwards.
- `test_harvest_resume.py`: kills the harvester after stage C on the fixture shard, restarts it,
  asserts the shard completes once and the manifest lists it once.
- `test_columnar_loader.py`: builds the section 6 files from three shards, loads them through
  `recommender.py`, asserts `get_track_record` round-trips every field of a sample of 200 rows
  against the shard JSON.
- The existing suite (golden features, no gates, leakage, octave truth, candidate pool, theme
  sanity, describe parity, constant fields, live overlap, replica validation) runs unchanged on the
  million-row build.

Acceptance, measured and pasted with commands, after the full build is deployed:

| Check | Required |
|---|---|
| Rows in the catalogue (`header.json`) | at least 1,000,000, every row with a CLAP vector and a describe from audio (`rows_described_from_audio == rows`) |
| Distinct artists | at least 60,000; no artist above 20 rows |
| Benchmark share | all 149 benchmark entities present; under 0.1% of rows |
| Language | `null` at most 10%; `language_source = lyrics` on at least 60% of rows with a lyric file; no single language above 70% |
| Felt tempo | above 150 BPM at most 10%, below 60 at most 5% |
| Theme | largest share of non-null at most 30% |
| Replica vs live (`report.py --validate-live`) | deviation under 0.001 |
| Candidate pool (`report.py` pooled) | at least as many TUNE positives pooled as the round 2 result on 20,000 |
| Per-seed checks from `FIX_PLAN_2.md` section 6 | still hold (Simple Man, Waiting Around to Die, Boyfriend) |
| Negatives in top 10 | 0 |
| `test_live_overlap.py` | passes |
| Space health | `recommender_loaded: true`; any in-catalogue request answers within 10 s (speed is deferred, failing to answer is not) |
| Blind listen | the owner rates 40 suggestions (10 per world, titles hidden, 15 s each) as "same mood: yes/no"; at least 24 yes, no world under 4 |

Recall at 20 on the benchmark is reported but has no threshold at a million: more right answers
and more distractors both enter, and the per-seed checks plus the blind listen are the quality
gate at this scale.

## 9. Deferred, deliberately

Not in this round, do not build: IVF or product-quantised indexes, latency targets below 10 s,
feel-bucket stratification of the catalogue, language quotas beyond the 70% ceiling, on-demand
growth from user searches, duplicate-version collapsing (remasters, live versions) beyond the
`artist:title` key, any learned model. Each is a follow-up with its own measurement once the
million-row catalogue works.

## 10. Rules, in addition to `FIX_PLAN_2.md` section 3

21. No stage of this plan runs over more than one fixture shard until `FIX_PLAN_2.md` section 6
    has passed and the describe code commit is recorded in every shard manifest.
22. A shard is complete only when its two data files are on the dataset with matching SHA256s and
    the shard id is in `catalog/v3/MANIFEST.json`. Anything else is not counted.
23. Previews live on disk only for the shard in progress. Free disk is checked before every shard;
    under 8 GB free, the harvester stops and logs it.
24. Every number in a log entry about the harvest comes from a manifest file on the dataset, named
    in the entry.
25. The serving Space is never used as a harvester while users can reach it; if the Space takes
    shards, it is a separate Space.

## 11. Order and gates

1. Round 2 passes (its section 6). Gate: the pasted table.
2. Section 6 storage and loader, built and deployed on the existing 20,000 rows. Gate:
   `test_columnar_loader.py`, the full existing suite, `report.py --validate-live` under 0.001.
   Doing this first means the loader is proven before the data gets big.
3. Discovery on the Space. Gate: section 4 "done when".
4. Harvester on one fixture shard, then ten real shards. Gate: the two harvest tests, the throughput
   check, previews directory empty, free disk unchanged after the shards.
5. The full run to 1,000,000, both runners if the owner chooses. Gate: `catalog/v3/MANIFEST.json`
   row sum at least 1,000,000.
6. Index build and deploy. Gate: section 8 acceptance.

## 12. Amendment to `FIX_PLAN_2.md` T1

T1 said "fetch the preview into `ml/data/previews/<id>.m4a`". With a 50 GB disk the 20,000-row
data pass must already work the way section 5 does: fetch, describe, delete, in batches of at most
5,000, with `dropped.jsonl` for failures. Keeping 20,000 previews (about 20 GB) on the Windows box
is not allowed. The rest of T1 stands.
