# Echo: scale plan, one million songs (round 3, "make it work" scope)

Status: written 2026-10-28 against commit `5693980` on `main`. This plan starts only after
`FIX_PLAN_2.md` section 6 passes on the 20,000-row catalogue; it reuses that describe code unchanged
and runs it a million times. Scope is "make it work": the catalogue exists, loads on the free Space,
and the benchmark can be run on it. Latency, index compression and catalogue stratification are
deferred and listed in section 9 so nobody builds them early. Where this file and `FIX_PLAN_2.md`
disagree, `FIX_PLAN_2.md` wins until its section 6 passes, then this file wins.

Sections: 0 how to work this plan (read first), 1 why the catalogue is the search space, 2 machines
and the division of labour, 3 the 50 GB disk budget, 4 discovery, 5 the harvester, 6 storage and
index at a million, 7 serving changes, 8 tests and acceptance, 9 deferred, 10 rules, 11 work
packets and gates, 12 amendment to `FIX_PLAN_2.md` T1.

## 0. How to work this plan (binding for the implementing agent)

The work is cut into packets (section 11). Each packet is small enough for one sitting, names the
only files it may touch, and ends with a verification block whose output is pasted, verbatim, into
`SCALE_PROGRESS.md` under the packet's heading. The protocol:

1. Start a packet by writing its heading and start time in `SCALE_PROGRESS.md`, then the list of
   files you intend to touch. Touching a file outside the packet's list is a failed packet.
2. Do only what the packet says. A good idea that belongs to a later packet goes into a
   "Noticed" line in the progress entry, not into the code.
3. Run the packet's verification block exactly as written and paste the full output. If any check
   fails, fix it inside the packet and re-run; never carry a failing check into the next packet and
   never edit a check to make it pass.
4. End the packet with the line `PACKET <id> READY FOR REVIEW` and stop. Do not start the next
   packet until the owner replies `GO <next id>`. The owner, or an auditor the owner runs, re-runs
   the verification block before replying; a packet whose checks do not reproduce is reopened.
5. One commit per packet, message `Packet <id>: <packet title>`, pushed before the review request.
   The commit touches only the packet's files plus `SCALE_PROGRESS.md`.
6. If a packet cannot be completed as written (an API changed, a number is unreachable), stop,
   write what was found with the command output, and end with `PACKET <id> BLOCKED`. Do not
   improvise a different design.

The point of the protocol is that a mistake costs one packet, not the whole run. A million-song
harvest executed on a wrong describe or a wrong storage format is a week lost; the gates in
section 11 exist to make that impossible.

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
| Preview fetch, decode, DSP describe, lyrics, language, theme | the Windows box (Ryzen 7600, 12 threads), native Windows Python 3.13, the stack that described P4 | about 1 s per song per thread: a million in roughly 28 CPU-hours |
| CLAP vectors | the Windows box, RTX 3080 | batched HTSAT at 48 kHz, three 10 s windows a song: a few hours for a million |
| Shard upload, manifest, index build | the Windows box writes to the HF dataset `shreyash616/echo-ml-artifacts`; the index is built from the dataset | one source of truth; the Mac and the Space never hold a private copy |

The Windows box runs everything on the native Windows Python 3.13 venv that described P4 (amended
2026-10-30; WSL2 was the original text, but FIX_PLAN_2 rule 20b makes the describing stack the reference,
and that stack is native Windows, `pip freeze` recorded in FEEL_ALGORITHM.md): ffmpeg, librosa,
pyloudnorm, laion_clap with CUDA, sentence-transformers, lingua, httpx, huggingface_hub. Scripts use
`pathlib` and no shell-specific syntax so they run unchanged on Windows and Linux. Expected wall time for a million: four to five days, bounded by the
iTunes rate limit and the preview downloads (about 1 TB in total), not by compute.

## 3. The 50 GB disk budget on the Windows box

Previews are never kept beyond the shard being processed. The budget, with headroom:

| Item | Size | Note |
|---|---|---|
| Python venv, torch with CUDA, laion_clap checkpoint, MiniLM, lingua models (already present from P4) | about 9 GB | one-off |
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
   NG, ZA, TW, HK, SA, EG, AE, VN, TH, MY (endpoint `/chart/{country}/artists` or `/editorial/{id}/charts`, whichever Deezer serves).
2. Breadth-first walk over `/artist/{id}/related` (20 per call), depth-first from no seed; stop at
   120,000 unique artists or when a depth has no new artists. Keep `nb_fan` for the per-artist cap.
3. Genre pages: `/genre` then `/genre/{id}/artists` for every genre, appended with `source =
   "genre"`, so small scenes (garage, psych, qawwali, dangdut, enka) are present even if the walk
   never reaches them.
4. Rate limits: Deezer allows about 50 requests per 5 seconds; sleep on 4xx and retry with
   backoff; the walk is resumable from the last written line.
5. Done when: `discovery/artists.jsonl` has at least 80,000 rows, at least 28 countries among the
   chart seeds (every one listed in step 1), and the file's SHA256 is in `discovery/MANIFEST.json` with the counts per source.

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

**Stage E, text.** LRCLIB `/api/get` with `duration` and `album_name`, then without the album,
then `/api/search` by track and artist; `Retry-After` honoured. The dump is not an option: it is
one 47.8 GB gzipped SQLite file (`lrclib-db-dump-20261012T090639Z.sqlite3.gz`, listed by
`lrclib-db-dumps.bu3nnyut4y9jfkdg.workers.dev` on 2026-10-29), over the box's whole budget before
decompression. The flow is `fetch_lyrics_remote.py`'s: cleaned title, `/api/get` without the album, `/api/search`
by fields, then `q=`; five 503 retries at 1 s. Rate, measured (amended 2026-10-30): LRCLIB answers 503 to
about 60% of requests from the registered application as from anyone, independent of our pace, so a worker
does about 6 rows a minute whatever its spacing (P4: 13,866 rows in 2 h 15 min on 8 workers; WP1: 300 rows
in 18 min on one). Stage E therefore runs as a pool of 24 fetchers over disjoint row ranges, about 1,000
requests a minute app-wide against the 6,000 allowance, roughly 150 rows a minute, a million rows in about
5 days overlapped with the fetch and describe stages. Do not go wider than 24; LRCLIB is already shedding.
The application headers come from `backend/.env`; WP1 stops BLOCKED if they are absent. Then
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
| `keys.npy` plus `key_order.npy` | sorted blake2b-64 hashes of the keys and the row permutation (amended 2026-10-30 at WP2: fixed-width strings were 86 MB at 20,000 rows against 480 KB) | `np.searchsorted` on the hash, then the exact key confirmed from `strings.parquet`, replacing `track_id_map.json` |
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
| Language | `null` at most 10%; `language_source = lyrics` on at least 60% of rows with a lyric file; no single language above 70%; at least 10 languages with at least 10,000 rows each |
| Lyric coverage per language | for every language with at least 2,000 rows, `header.json` records the share with a lyric file, the `language_source` distribution and the theme null share; the share with a lyric file is at least 0.5 for en, es, pt, fr, de, it; every other language is reported, lowest ten listed in the WP6 report |
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

## 11. Work packets and gates

Each packet: scope, files it may touch, verification block (run and paste), stop. Packets run in
this order; none starts before the previous one's `GO`.

**WP0. Preconditions.** Scope: confirm `FIX_PLAN_2.md` section 6 passed, and record the describe
code commit that every shard will carry. Files: `SCALE_PROGRESS.md` only.
Verify and paste:
```
git log -1 --format=%H -- backend/app/audio/features.py
ECHO_ARTIFACT_DIR=ml/inference python ml/experiments/offline_rerank/report.py --split tune --validate-live https://shreyash616-echo-backend.hf.space | tail -3
```
Gate (amended 2026-10-30 after FIX_PLAN_2 P8): the `FIX_PLAN_2.md` section 6 table pasted with every row
measured; replica deviation under 0.001. The engineering rows must pass: axes std, rows described, lyric
language share, 14 seeds on their language, deviation, p95, CI, overlap, negatives in top 10. The coverage
rows (TUNE recall@10 and recall@20, pooled positives, the four per-seed rows) are recorded pass or fail
and do not block: the 20,000-row catalogue is the ceiling they measure, and this plan is the remedy. If
an engineering row has not passed, this packet ends `BLOCKED` and nothing below starts.

**WP1. Windows box environment.** Scope (amended 2026-10-30): the native Windows Python 3.13 venv that described P4, with ffmpeg, librosa,
pyloudnorm, laion_clap with CUDA, sentence-transformers, lingua, pyarrow, huggingface_hub; the
LAION music checkpoint downloaded; HF write token for the dataset in the environment; the
owner's LRCLIB server-side application in `backend/.env` as `LRCLIB_APPLICATION_ID` and
`LRCLIB_SECRET_KEY` (registered 2026-10-29; `ml/data/fetch_lyrics_remote.py` reads that file and also accepts
`LRCLIB_APP_ID` / `LRCLIB_API_KEY` from the environment; the values never enter the repository or a log). Files:
`SETUP.md` (a Windows box section), `ml/data/requirements-harvest.txt` (exact pins from that venv's `pip freeze`).
Verify and paste:
```
uname -a && nvidia-smi --query-gpu=name,memory.total --format=csv && df -h ~ | tail -1
python -c "import torch, laion_clap, librosa, lingua, pyarrow; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q tests/test_features_golden.py
```
Gate: CUDA true on the RTX 3080; golden tests pass on this machine; at least 30 GB free; 300
sequential `/api/get` calls with the application headers at 10 a second return no 401 and no 429 (paste
the status counts; the values are read from `backend/.env`, never printed). The 503 share is reported, not
gated: measured 2026-10-30 at 58.8% of 48,711 requests with the registered application, the same as
unregistered, absorbed by the fetcher's five 1 s retries to 1 error in 13,866 rows.

**WP2. Columnar storage and loader, on the existing 20,000 rows.** Scope: `build_index_v3.py`
writing the section 6 files from the current `catalog_features.jsonl` and `clap.npy`; the loader
change in `recommender.py`; `score_lib.py` reading the same files. Files: `ml/inference/build_index_v3.py`
(new), `backend/app/services/recommender.py` (loader only), `ml/experiments/offline_rerank/score_lib.py`,
`tests/test_columnar_loader.py` (new), `tests/deploy_hf.py` (file list), `backend/startup.sh` (the six columnar downloads), `backend/requirements.txt` (`pyarrow` pin) (both added 2026-10-30). No fallback to the JSON files: a missing columnar file leaves `recommender_loaded` false.
Verify and paste:
```
python ml/inference/build_index_v3.py --out ml/inference && ls -la ml/inference | grep -E "clap.f16|feel_matrix|features.npz|strings.parquet|keys.npy|header.json"
python -m pytest -q tests/test_columnar_loader.py tests/test_features_golden.py tests/test_no_gates.py tests/test_no_benchmark_leakage.py
python tests/deploy_hf.py && curl -s https://shreyash616-echo-backend.hf.space/api/health
ECHO_ARTIFACT_DIR=ml/inference python ml/experiments/offline_rerank/report.py --split tune --validate-live https://shreyash616-echo-backend.hf.space | tail -3
python tests/test_live_overlap.py --base-url https://shreyash616-echo-backend.hf.space | tail -2
```
Gate: all tests pass; health `recommender_loaded: true`; replica deviation under 0.001 against the
redeployed Space; overlap passes; `report.py` pooled and hits equal to the WP0 numbers (the storage
change must not move a single rank).

WP2 passed review 2026-10-30 at 2c67304 (Deploy 41086058209, Tests 41085686410 green): deviation 0.0000,
20 of 20 on all nine seeds, pooled 36 and 3 / 5 / 0.0170 unchanged, float16 max abs 6.1e-05 with no served
rank moved. Departures recorded: ints int64, flags bool, every JSON field carried (adds `deezer_id`, `genre`,
`decade`, `language_conf`, `anchor_cosines`), keys as hashes (above). p95 rose from 0.105 s to 0.225 s
because each call assembles about 3,200 candidate dicts from columns; WP7's serving change has `_score_pool`
read columns by index instead.

**WP3. Discovery on the Space.** Scope: `ml/data/discover_artists.py` implementing section 4,
run on the Space as a background task (amended 2026-10-30: nobody has a shell on the Space), writing
`discovery/artists.jsonl`, `discovery/MANIFEST.json`, `discovery/shards.json` to the dataset, with a
checkpoint to the dataset every 5,000 artists and resume from it. Trigger: `POST /api/admin/discovery/start`
and `GET /api/admin/discovery/status`, both requiring an `X-Admin-Token` header equal (constant-time) to the
Space secret `ECHO_ADMIN_TOKEN`; when that secret is unset the routes return 404, so no surface exists.
Status reports progress counts and whether `HF_TOKEN` is present and its role (`HfApi.whoami`), never a
value. The owner sets `ECHO_ADMIN_TOKEN` and a write-scoped `HF_TOKEN` in the Space settings. No deploy
while discovery runs, since a deploy restarts the Space. Files: `ml/data/discover_artists.py` (new),
`backend/app/api/routes.py` (the two routes), `backend/requirements.txt` only if a library is missing,
`ml/data/harvest_seed_artists.json` (read only).
Verify and paste:
```
python -c "from huggingface_hub import hf_hub_download as d; import json; m=json.load(open(d('shreyash616/echo-ml-artifacts','discovery/MANIFEST.json',repo_type='dataset'))); print(m)"
```
Gate: at least 80,000 artists; counts per source (chart, related, genre) all above zero; at least
28 chart countries (every one in section 4 step 1); `shards.json` with artist ranges covering every artist exactly once.

**WP4. Harvester on one fixture shard.** Scope: `ml/data/harvest_v3.py` stages A to F and the
claim and resume logic of section 5, run on `tests/fixtures/harvest_fixture_shard.json` (5 artists
that are not in the benchmark). Files: `ml/data/harvest_v3.py` (new), `tests/test_harvest_shard.py`
(new), `tests/test_harvest_resume.py` (new), the fixture file, `.gitignore` (previews scratch).
Verify and paste:
```
python -m pytest -q tests/test_harvest_shard.py tests/test_harvest_resume.py
ls ml/data/previews | wc -l
python -c "from huggingface_hub import hf_hub_download as d; import json; print(json.load(open(d('shreyash616/echo-ml-artifacts','catalog/v3/shard_fixture.manifest.json',repo_type='dataset'))))"
```
Gate: both tests pass; previews directory empty; the fixture shard manifest on the dataset shows
`describe_commit` equal to WP0's hash, seconds per stage, dropped count, and SHA256s that match the
uploaded files.

**WP5. Ten real shards.** Scope: run `harvest_v3.py` on shards 0 to 9 from `discovery/shards.json`
on the Windows box. No code changes except bug fixes inside `harvest_v3.py`, each named in the
progress entry. Files: `ml/data/harvest_v3.py`, `SCALE_PROGRESS.md`.
Verify and paste:
```
python -c "from huggingface_hub import hf_hub_download as d; import json; m=json.load(open(d('shreyash616/echo-ml-artifacts','catalog/v3/MANIFEST.json',repo_type='dataset'))); print(len(m['shards']), sum(x['rows'] for x in m['shards']), sum(x['dropped'] for x in m['shards']))"
df -h ~ | tail -1 && ls ml/data/previews | wc -l
```
Gate: 10 shards complete, at least 40,000 rows, dropped under 10%; free disk within 3 GB of WP1's
value; previews directory empty; a projected finish date for a million at the measured songs per
hour, stated in the entry. If the projection exceeds 14 days, stop and report before WP6.

**WP6. The full run.** Scope: run to at least 1,000,000 rows; optionally a second runner on a
separate Space following rule 25. No code changes; a bug fix reopens WP4 and re-verifies the
fixture shard before the run resumes. Files: `SCALE_PROGRESS.md` (one line a day: shards done,
rows, dropped, free disk).
Verify and paste: the WP5 manifest command, daily, and at the end.
Gate: manifest row sum at least 1,000,000; every shard manifest carries the WP0 describe commit.

**Note for WP7 (from FIX_PLAN_2 P9, 2026-10-30).** One call in 100 on the Space takes about 0.42 s; the cause is a CPython generation-2 garbage collection (0.29 s) over the metadata object graph, since each call builds fresh candidate dicts. WP7's serving change should remove it: `gc.freeze()` after the artifacts are loaded, or raised generation thresholds, or no per-call dict construction with the columnar store; measure p95 and max over 100 calls before and after.

**WP7. Index build and deploy at a million.** Scope: `build_index_v3.py` over all shards from the
dataset, upload, redeploy. Files: `ml/inference/build_index_v3.py` (bug fixes only),
`tests/deploy_hf.py`.
Verify and paste: the whole of section 8's acceptance table, with every command, plus the blind
listen sheet (40 rows: world, hidden title revealed after rating, yes or no).
Gate: section 8, every row. Recall at 20 is reported without a threshold.

A packet's gate is checked by the owner before `GO`. The owner may also run
`ml/experiments/offline_rerank/report.py` and the live tests independently at any time; a
disagreement with a pasted output reopens the packet.

## 12. Amendment to `FIX_PLAN_2.md` T1

T1 said "fetch the preview into `ml/data/previews/<id>.m4a`". With a 50 GB disk the 20,000-row
data pass must already work the way section 5 does: fetch, describe, delete, in batches of at most
5,000, with `dropped.jsonl` for failures. Keeping 20,000 previews (about 20 GB) on the Windows box
is not allowed. The rest of T1 stands.
