# SCALE_PLAN progress

Packet log for `SCALE_PLAN.md` (section 0 protocol). Each entry: heading, start time, files,
the verification block's output verbatim, the gate, `READY FOR REVIEW` or `BLOCKED`.

## WP0. Preconditions

- Started: 2026-10-30 20:30 IST after GO WP0 (README e889acb; gate as amended at 06948ee).
- Files: `SCALE_PROGRESS.md` only.
- Served state: Hugging Face Space runtime sha `6b768fbf`; pools `candidates_clap` 300, `candidates_feel` 3000;
  rerank weights = `ml/training/weight_search_log.jsonl` evaluation 1748 (FIX_PLAN_2 P8); feel-pool weights
  as P7 (header `feel_weights`); catalogue 19,999 rows.
- Describe commit every shard will carry (`git log -1 --format=%H -- backend/app/audio/features.py`):
  **`e4c1b1bd2b4dd35a21560de82d280a587461ce99`** (2026-10-28, "Packet P3: T4 amendments A+B+L and script fix").

### Verification block, verbatim

```
$ git log -1 --format=%H -- backend/app/audio/features.py
e4c1b1bd2b4dd35a21560de82d280a587461ce99

$ ECHO_ARTIFACT_DIR=ml/inference python ml/experiments/offline_rerank/report.py --split tune --validate-live https://shreyash616-echo-backend.hf.space | tail -3

SUMMARY split=tune: positives 81, pooled 36, not pooled 45, hits@10 3, hits@20 5, MRR 0.0170, negatives in top10 0
LIVE VALIDATION: max axis deviation vs https://shreyash616-echo-backend.hf.space = 0.0000 (must be < 0.001)
```

(Run on the Windows box as `py -3 ...`, the same interpreter as every FIX_PLAN_2 packet.)

### FIX_PLAN_2 section 6, from the P9 entry in FEEL_ALGORITHM.md (measured live at `6b768fbf`)

Kind per the amended WP0 gate. Engineering rows must pass; coverage rows are recorded and do not block.

| # | Check | Kind | Measured | Required | Result |
|---|---|---|---|---|---|
| 1 | TUNE positives in the candidate pool (`report.py` pooled) | coverage | 36 of 81 | at least 35 of 81 | pass |
| 2 | Owner positives in the pool | coverage | 8 of 11 | at least 8 of 11 | pass |
| 3 | TUNE recall@20 (`feel_benchmark.py --split tune --base-url`) | coverage | 5 of 81 | at least 10 of 81 | fail |
| 4 | TUNE recall@10 | coverage | 3 of 81 | at least 7 of 81 | fail |
| 5 | Negatives in top 10 | engineering | 0 | 0 | pass |
| 6 | `test_live_overlap.py` | engineering | passes; every pair 0 of 10 and 0 of 20 | still passes | pass |
| 7 | Simple Man top 20 | coverage | When the Levee Breaks, rank 9 | Levee or Stairway | pass |
| 8 | Waiting Around to Die top 20 | coverage | none of No Surprises, Me Gustas Tu, Black | at least one | fail |
| 9 | Boyfriend top 10 | coverage | 0 of Antidote, For A Reason, Baller, Roll With Me | at least 2 | fail |
| 10 | Bang Data top 10 | coverage | felt_bpm 81 to 107, all within 43 to 172; mean grit 0.416 against at least 0.346 | every felt_bpm 0.5x to 2x, mean grit at least seed minus 0.1 | pass |
| 11 | Feel axes with pool std under 0.05 on 3 or more TUNE seeds | engineering | none (only `s_language` on Tomboy, 0.0457, one seed) | none | pass |
| 12 | Rows described from decoded audio | engineering | 19,999 of 19,999 | all catalogue rows | pass |
| 13 | `language_source = lyrics` share of rows with a lyric file | engineering | 96.74% (16,840 of 17,408) | at least 60% | pass |
| 14 | 14 benchmark seeds on their declared language | engineering | 14 of 14 | 14 of 14 | pass |
| 15 | Replica vs live axis deviation | engineering | 0.0000 (P9 and the block above) | under 0.001 | pass |
| 16 | Server-side p95, in-catalogue seed | engineering | 0.105 s over 20 calls (0.128 s over 100) | under 0.300 s | pass |
| 17 | CI on main with the T9 list | engineering | green; latest https://github.com/shreyash616/echo/actions/runs/41075478508 on `e889acb`; `tests.yml` runs all six T9 tests | green with the T9 list | pass |
| 18 | HOLDOUT recall@20 | reported | 1 of 14, read once in P9 | reported once | reported |

### Gate

- Engineering rows (5, 6, 11 to 17): all nine pass.
- Replica deviation: 0.0000, under 0.001.
- Coverage rows recorded: 1, 2, 7 and 10 pass; 3, 4, 8 and 9 fail. Per the amended gate they do not block:
  the 20,000-row catalogue is the ceiling they measure (P9's `unpooled_ranks.py`: 45 TUNE positives
  unpooled, 34 of them deeper than feel rank 5,000).

### Carried into SCALE_PLAN (already written there, per README e889acb)

- The owner's terminal windows (a `tail -f` of the describe log and of the CLAP batch log, with per-batch
  CLAP INFO lines added first) from the first WP that describes audio.
- `LRCLIB_APPLICATION_ID` / `LRCLIB_SECRET_KEY` in `backend/.env` (WP1).
- Dedupe by iTunes id at harvest (WP2); tests must not write serving state (WP7); exact pins in
  `backend/requirements.txt` (WP7, rule 20b); the generation-2 collection outlier (WP7).

### Noticed

- `ml/data/HARVEST_MANIFEST.json`'s `describe_commit` is `05def45` (the repository head when P4's re-describe
  ran); `backend/app/audio/features.py` itself last changed at `e4c1b1b`, the hash above. The two describe the
  same feature code; the shard manifests will carry `e4c1b1b` as WP0 specifies.

PACKET WP0 READY FOR REVIEW

## WP1. Windows box environment

- Started: 2026-10-30 20:45 IST after GO WP1 (README 1f08e71).
- Files: `SETUP.md` (a WSL2 section), `ml/data/requirements-harvest.txt`, `SCALE_PROGRESS.md`.

### Found before any install

```
PS> wsl --status; wsl -l -v
The Windows Subsystem for Linux is not installed. You can install by running 'wsl.exe --install'.

PS> nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
NVIDIA GeForce RTX 3080, 610.88, 10240 MiB

PS> Get-PSDrive C   (free)
135.3 GB

Windows 11 Pro build 26200
```

WSL2 is not installed on the box. Installing it enables Windows optional features as administrator and
needs a reboot, which is the owner's to do, not the implementing agent's; so is putting the Hugging Face
write token into the WSL environment. The driver (610.88) carries CUDA for WSL2, and C: has 135 GB free
against the 30 GB gate. Owner steps, then the packet resumes:

1. Administrator PowerShell: `wsl --install -d Ubuntu-24.04`, reboot, open Ubuntu once and create the Linux user.
2. In Ubuntu: `sudo apt update && sudo apt install -y ffmpeg python3-venv python3-pip git` (needs the Linux
   password, so the owner runs it).
3. After the agent has made the venv: the HF write token into the WSL environment (`hf auth login`, or
   `HF_TOKEN` in `~/.bashrc`), typed by the owner.

PACKET WP1 BLOCKED

### Resumed under the amended scope (README c2453cb): native Windows, no WSL2

- Amended 2026-10-30 20:50 IST: the native Windows Python 3.13.12 that described P4 is the harvest stack (FIX_PLAN_2
  rule 20b); SCALE_PLAN sections 2, 3 and WP1 drop WSL2. The interpreter is the python.org install itself,
  `C:\Users\Shreyash Padhi\AppData\Local\Programs\Python\Python313\python.exe`, not a venv.
- Files: `SETUP.md` (section 4, Windows box), `ml/data/requirements-harvest.txt` (new), `SCALE_PROGRESS.md`.
- Installed: `pyarrow==26.0.0` only (`pip install --dry-run pyarrow` listed it alone; nothing else moved).
  Already present: ffmpeg 9.0.2 (gyan.dev full build), huggingface_hub 2.3.0 with the cached login, the LAION
  checkpoint `%USERPROFILE%\.cache\clap\music_audioset_epoch_15_esc_90.14.pt` (2,352,471,003 bytes).

### Verification block, verbatim

```
PS> nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 3080, 610.88, 10240 MiB

PS> py -3 -c "import sys; print(sys.version)"
3.13.12 (tags/v3.13.12:1cbe481, Feb  3 2026, 18:22:25) [MSC v.1944 64 bit (AMD64)]

PS> (Get-PSDrive C).Free
145127411712   (135.2 GB)

PS> py -3 -c "import torch, laion_clap, librosa, lingua, pyarrow; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
True NVIDIA GeForce RTX 3080

PS> py -3 tests/test_features_golden.py
(no output, exit 0: the file defines pytest tests and has no __main__ block, so run directly it executes none)

PS> py -3 -m pytest -q tests/test_features_golden.py
12 passed, 3 warnings in 25.92s

PS> py -3 ml/data/fetch_lyrics_remote.py --tracks ml/inference/track_metadata.json --out <scratch>/wp1_lrclib --limit 300 --spacing 0.1
20:42:07 300 rows to fetch (19699 skipped), spacing 0.10 s
20:42:07 registered app: yes
20:58:28 status after 300 rows: {'get 200': 278, 'get 404': 15, 'get 503': 435, 'search 200': 1, 'search 503': 2}
20:58:28 retrying 21 errors once
21:00:10 done {'fetched': 294, 'error': 22, 'none': 5, 'error_after_retry': 1} in 18 min; status {'get 200': 297, 'get 404': 21, 'get 503': 474, 'search 200': 3, 'search 503': 8}
```

(The `error on <id>` lines, one per row whose five 503 retries ran out, are left out above; there were 22.)

LRCLIB: 803 requests, 792 of them `/api/get`: 200 x 297, 404 x 21, 503 x 474; `/api/search`: 200 x 3, 503 x 8.
**No 401, no 429.** 503 share 60.0% of all requests (59.8% of `/api/get`), against 58.8% measured in P4; the
fetcher's retries left 1 row in error out of 300. The credentials were read from `backend/.env` by the fetcher;
no value was printed.

### Stack identity against the P2 reference (rule 20b)

| package | P2 reference (FEEL_ALGORITHM P2 log) | `pip freeze` now |
|---|---|---|
| librosa | 1.0.0 | 1.0.0 |
| numpy | 2.4.3 | 2.4.3 |
| scipy | 1.18.1 | 1.18.1 |
| laion_clap | 1.1.7 | 1.1.7 |
| torch | 2.6.0+cu124 | 2.6.0+cu124 |
| pyloudnorm | 0.2.0 | 0.2.0 |

All 24 packages of the P2 reference line are identical (faiss-cpu, lingua-language-detector, llvmlite, numba,
sentence-transformers, soundfile, torchaudio, torchvision, transformers and laion_clap's runtime deps
included). Same interpreter, same stack as P4's describe, so the golden parity question does not arise.

### Gate

- CUDA true on the RTX 3080: yes.
- Golden tests pass on this machine: 12 of 12 under pytest.
- At least 30 GB free: 135.2 GB.
- No 401 and no 429 in the LRCLIB calls: none; 503 share reported (60.0%).

### Noticed

- `python tests/test_features_golden.py`, as the block writes it, runs no test (no `__main__`); the pytest
  line above is the run. Changing the test file is outside WP1.
- 300 rows took 18 minutes at 0.1 s spacing: the 503s each cost a 1 s retry sleep, so the spacing is not
  the rate limit at a 60% 503 share. That bears on WP5's lyric throughput estimate.
- The packet's code went out in `f01813f` without this entry (a script error on my side); this entry is the
  follow-up commit.

PACKET WP1 READY FOR REVIEW

## WP2. Columnar storage and loader, on the existing 20,000 rows

- Started: 2026-10-30 21:10 IST after GO WP2 (README 8ea9733).
- Files (seven, the list as amended at 3b44f87): `ml/inference/build_index_v3.py` (new), `backend/app/services/recommender.py`
  (loader only), `ml/experiments/offline_rerank/score_lib.py`, `tests/test_columnar_loader.py` (new), `tests/deploy_hf.py`
  (file list), `backend/startup.sh` (six downloads), `backend/requirements.txt` (`pyarrow==26.0.0`); `SCALE_PROGRESS.md`.
- Commits: `15db12e` (the five files), `2c67304` (no JSON fallback, `startup.sh`, the pin).

### What was built

- `build_index_v3.py` runs `build_index_v2.main()` unchanged with its outputs redirected to a scratch directory (so every
  derived field, standardisation and cut point is the code that produced the served rows), checks those rows and that
  header equal the served `track_metadata.json` (`True`), and writes: `clap.f16.npy` (19,999 x 512 float16),
  `feel_matrix.npy` (byte-identical to the served one), `features.npz` (52 arrays), `strings.parquet` (17 text columns),
  `keys.npy` / `key_order.npy`, `header.json` (v2's header plus the column list, the two field orders, rows, build commit).
- `recommender.py`: `ColumnarRows` (a row is a dict assembled on demand from the columns: the JSON row's fields, field
  order and Python types), `KeyLookup` (`get` / `in` / `[]` by id, key, itunes id, as `track_id_map.json`), and
  `_load_columnar`: CLAP float16 to float32 into a flat `IndexFlatIP`, the feel matrix into `IndexFlatL2`. **No JSON
  fallback**: a missing or unreadable file is logged by name and the recommender stays unloaded (`/api/health` false).
- `score_lib.Catalogue` reads the same files through the Space's own `ColumnarRows` and `KeyLookup` (rows materialised
  once, as the replica iterates them) and raises on a missing file; no JSON path left.

### Verification block, verbatim

```
$ python ml/inference/build_index_v3.py --out ml/inference && ls -la ml/inference | grep -E "clap.f16|feel_matrix|features.npz|strings.parquet|keys.npy|header.json"

============================================================
STEP R4 / S1 INDEX BUILD COMPLETED SUCCESSFULLY
Total Catalogue Rows:       19999
CLAP FAISS Index Total:     19999 (dim 512)
Feel FAISS Index Total:     19999 (dim 17)
ID Map Entries:             59953
Anchor Max Cosine:          0.8032 (< 0.95)
Anchor Min Cosine:          -0.0431 (< 0.70)
First 50 Mean Cosine:       0.6542 (< 0.90)
kick_class shares:
  other               : 41.20%
  syncopated_808      : 42.30%
  sparse_backbeat     : 15.67%
  four_on_floor       : 0.83%
restraint_bucket shares:
  moderate            : 51.62%
  restrained          : 24.30%
  saturated           : 24.08%
============================================================
============================================================
build_index_v3: 19999 rows, 66 fields -> ml\inference
  numeric arrays 52, text columns 17, lookup keys 59953
  clap.f16 max abs error against the float32 rows: 6.104e-05
============================================================
-rw-r--r-- 1 Shreyash Padhi 197121 20479104 Oct 30 21:29 clap.f16.npy
-rw-r--r-- 1 Shreyash Padhi 197121  9312707 Oct 30 21:29 features.npz
-rw-r--r-- 1 Shreyash Padhi 197121  1360060 Oct 30 21:29 feel_matrix.npy
-rw-r--r-- 1 Shreyash Padhi 197121    13007 Oct 30 21:29 header.json
-rw-r--r-- 1 Shreyash Padhi 197121   479752 Oct 30 21:29 keys.npy
-rw-r--r-- 1 Shreyash Padhi 197121  3191038 Oct 30 21:29 strings.parquet

$ python -m pytest -q tests/test_columnar_loader.py tests/test_features_golden.py tests/test_no_gates.py tests/test_no_benchmark_leakage.py
21 passed, 5 warnings in 55.59s

$ python tests/deploy_hf.py && curl -s https://shreyash616-echo-backend.hf.space/api/health
Uploading ml/inference/clap.f16.npy -> clap.f16.npy...
Uploading ml/inference/features.npz -> features.npz...
Uploading ml/inference/strings.parquet -> strings.parquet...
Uploading ml/inference/keys.npy -> keys.npy...
Uploading ml/inference/key_order.npy -> key_order.npy...
Uploading ml/inference/header.json -> header.json...
Restarting Space...
All files uploaded successfully!
{"status":"ok","recommender_loaded":true}   (the old container, still up during the restart; the rebuilt one below)

deploy.yml https://github.com/shreyash616/echo/actions/runs/41086058209 success; Space RUNNING 21:34:17 on sha 1095b439, then:
$ curl -s https://shreyash616-echo-backend.hf.space/api/health
{"status":"ok","recommender_loaded":true}

$ ECHO_ARTIFACT_DIR=ml/inference python ml/experiments/offline_rerank/report.py --split tune --validate-live https://shreyash616-echo-backend.hf.space | tail -3

SUMMARY split=tune: positives 81, pooled 36, not pooled 45, hits@10 3, hits@20 5, MRR 0.0170, negatives in top10 0
LIVE VALIDATION: max axis deviation vs https://shreyash616-echo-backend.hf.space = 0.0000 (must be < 0.001)

$ python tests/test_live_overlap.py --base-url https://shreyash616-echo-backend.hf.space | tail -2
[SUCCESS] ALL PAIRS PASSED OVERLAP THRESHOLDS (Rule 9 Satisfied)!

```

### Beyond the block

```
align_p8.py (side record recreated with /api/track/67238735), served weights, 300 / 3000:
pools 300 / 3000; served weights from ml/inference/rerank_weights.json
| seed | shared of 20 | live pool | side records in live pool | replica only | live only |
|---|---|---|---|---|---|
| Simple Man | 20 (same order) | 3215 | 1 | - | - |
| Waiting Around to Die | 20 (same order) | 3172 | 1 | - | - |
| Bang Data | 20 (same order) | 3139 | 1 | - | - |
| Boyfriend | 20 (same order) | 3131 | 1 | - | - |
| Hurt | 20 (same order) | 3191 | 1 | - | - |
| Tum Hi Ho | 20 (same order) | 3201 | 1 | - | - |
| Mask Off | 20 (same order) | 3219 | 1 | - | - |
| Get Lucky | 20 (same order) | 3221 | 0 | - | - |
| Tomboy | 20 (same order) | 3177 | 1 | - | - |

minimum shared over the TUNE seeds: 20 of 20 (gate: at least 18 on every seed)

$ python tests/test_candidate_pool.py --base-url https://shreyash616-echo-backend.hf.space
  Owner Positives in Union Pool: 8 / 11 (Required >= 8 of 11)
  All Positives in Union Pool:   35 / 67
  X-Process-Time per call (s):   0.196, 0.201, 0.224, 0.201, 0.198, 0.177, 0.179, 0.184, 0.183, 0.192, 0.182, 0.172, 0.209, 0.190, 0.177, 0.193, 0.241, 0.207, 0.208, 0.175
  Server-side p95 Process Time:  0.225 s (Required < 0.300 s)

$ python -m pytest -q tests/
29 passed, 1 skipped, 5 warnings in 150.60s (0:02:30)

Tests https://github.com/shreyash616/echo/actions/runs/41085686410 green (attempt 2, rerun after the deploy; attempt 1 ran before it)
```

### Float16 CLAP and every rank

- Max absolute difference between the float16-rounded-back CLAP rows and the float32 unit-norm rows from `clap.npy`:
  **6.104e-05**. Deviation against the columnar Space 0.0000; the same replica against the JSON-served Space before the
  deploy read 0.0001, which is how the live check shows the Space loads the columnar set.
- Served ranks: `report.py` pooled 36, hits@10 3, hits@20 5, MRR 0.0170, negatives 0, exactly WP0's; `align_p8.py` 20 of 20
  in the same order on all nine, Get Lucky included. No seed's top 20 changed order.
- Below the served lists, for the record: replica JSON against replica columnar on the nine TUNE seeds, the pools are
  identical, axis differences at most 7.3e-05, score differences at most 1.7e-05, and positions 1 to 60 of the pre-MMR
  order (the window the language budget reads) identical on every seed. Deeper, near-tied candidates swap: the first
  differing pre-MMR rank is 97 to 290 by seed, 119 to 241 positions differ among about 3,200. With float32 CLAP and the
  float32 columns instead, 2 to 8 positions differ on five seeds (first at rank 268 or deeper) and none on four: the
  deep swaps are mostly float16, as expected.

### Gate

- All tests pass: the block's four files 21 passed; the full suite 29 passed, 1 skipped; Tests green.
- Health `recommender_loaded: true` on the rebuilt Space (`1095b439`).
- Deviation 0.0000, under 0.001. Overlap passes.
- `report.py` pooled 36 and 3 / 5 / 0.0170 as WP0; `align_p8.py` 20 of 20 on all nine.
- p95 reported: **0.225 s** over 20 calls (P9: 0.105 s); under the 0.300 s serving gate.

### Noticed

- p95 doubled because each call assembles about 3,200 candidate dicts from the columns (about 10.6 microseconds a row,
  35 ms a call on the box) where it used to copy stored dicts. `_score_pool` reading the columns by row index instead of
  through dicts would remove it; that is a serving change for WP7, not a loader change.
- Departures from section 6's table, each to keep the rows identical or the files small: ints are stored int64 and
  flags bool (not float32), so `felt_bpm`, `year`, `loop_bars` stay ints; `strings.parquet` and `features.npz` carry every
  field of the JSON row (adding `deezer_id`, `genre`, `decade`, `language_conf`, `anchor_cosines` to the table's lists);
  `keys.npy` holds sorted blake2b-64 hashes of the lookup strings, not the strings, since fixed-width numpy strings were
  86 MB at 20,000 rows (about 4 GB at a million) against 480 KB; a hash hit is confirmed against the row's own strings,
  and `test_columnar_loader` checks all 59,953 keys against `track_id_map.json`.
- `build_index_v3.py` reaches the rows through `build_index_v2`; at a million rows it reads shards instead (WP6).

PACKET WP2 READY FOR REVIEW

## WP3. Discovery on the Space

- Started: 2026-10-30 21:50 IST after GO WP3 (README 221a5e2); scope amended at 618dedd.
- Files: `backend/app/services/discovery.py` (new; first written as `ml/data/discover_artists.py`, moved), `backend/app/data/discovery_seeds.json`
  (new, generated), `ml/data/make_discovery_seeds.py` (new), `backend/app/api/routes.py` (the two routes), `SCALE_PROGRESS.md`.
  `backend/requirements.txt` unchanged (httpx and huggingface_hub are already there).
- Commits: `b2596f7` (first version), `7dbde4c` (moved into the backend, ids-only seeds, 28 charts with `chart_kind`).

### What was built

- **Charts.** `/chart/{country}/artists` ignores the country (SG and 0 return the same list); Deezer serves country charts as the
  "Deezer Charts" account's playlists (user 637006841, 102 playlists), which cover 22 of the 28 countries. `discovery.CHARTS` pins
  country to playlist id: 22 `official`, and for the six without one a `user` playlist, picked 2026-10-30 as the most-followed
  current-hits list, preferring lists created 2024 or later when any has followers:
  IN 5714603022 "Bollywood Top 50" (Filtr Arabia, a label account; 2,114 fans, created 2019, the best 2024+ list had 19),
  PK 15038204963 "TRENDING PAKISTANI SONGS 2025" (6 fans, 2026-03), BD 13175326563 "Top Bangla Songs 2024" (2 fans, 2024-10),
  TW 10561367522 "Top 50 - Taiwan" (5 fans, 2022; no 2024+ list had a fan), HK 13617318101 "Hong Kong 2025" (2 fans, 2025-03),
  VN 14760476603 "Vietnamese (V-Pop) 2026" (8 fans, 2025-12).
- **Seeds.** `make_discovery_seeds.py` resolves the 117 names of `harvest_seed_artists.json` and the 115 lead-artist names of the
  benchmark on the box: 116 exact, 3 top hit, 0 unresolved, **119 unique Deezer ids**, written with counts only. The file holds
  ids, not names: `test_no_benchmark_leakage` greps `.json` files outside `tests/` for benchmark artist names as well as
  titles and whitelists only `harvest_seed_artists.json`, so a names file in `backend/app/data/` would fail it; the walk needs
  only the ids.
- **Walk.** `Discovery.run`: seeds (`/artist/{id}`), charts (top 100 artists of each playlist), genre pages, then the related
  walk in file order to 120,000; `nb_fan` filled for rows that lack it; client at 50 requests per 5 s with backoff on
  quota errors, 4xx/5xx and network errors. Checkpoint every 5,000 artists: `discovery/artists.jsonl` and `discovery/progress.json`
  (rows expanded) to the dataset; a restart resumes with the queue = rows from `expanded` on. At the end `MANIFEST.json` (counts
  per source, chart countries with `chart_kind`, official count, depths, SHA256) and `shards.json` (250 a shard, row ranges).
- **Routes.** `POST /api/admin/discovery/start`, `GET /api/admin/discovery/status`: 404 unless `ECHO_ADMIN_TOKEN` is set;
  `X-Admin-Token` checked with `hmac.compare_digest` (403 otherwise); start refuses with 412 when `HF_TOKEN` is unset and 409
  while a run is alive; one background thread; status gives progress, checkpoint, error, `hf_token_present` and the role from
  `whoami`, never a value. Local TestClient: no secret 404 / 404; wrong token 403; right token 200; start without `HF_TOKEN` 412.

### Local dry run (box, `--no-upload`, cap 4,000)

```
{'rows': 4000, 'counts_per_source': {'seed': 119, 'chart': 1314, 'genre': 43, 'related': 2524}, 'chart_country_count': 28,
 'official_chart_countries': 22, 'unresolved_seed_ids': 0, 'depths': {'0': 1476, '1': 2524}, 'requests': 1794,
 'failed_requests': 0, 'elapsed_s': 463}
kinds: Counter({'official': 22, 'user': 6}) | user charts (artists, new): IN (52, 51), PK (49, 39), BD (73, 73), TW (39, 31), HK (71, 70), VN (100, 100)
sha ok: True | shards exact: True | unique: True
```

About 3.9 requests a second (latency-bound) and about 11 new artists per related call, so 120,000 is roughly an hour of walk.

### Deploy

- `deploy.yml` https://github.com/shreyash616/echo/actions/runs/41094984706 success; Space RUNNING 22:31:55 on sha `12e51b50`;
  the Space repository holds `app/services/discovery.py` and `app/data/discovery_seeds.json`, no `.env`.
- `/api/health` `{"status":"ok","recommender_loaded":true}`; `--validate-live` 0.0000, `report.py` 36, 3 / 5 / 0.0170 unchanged.
- With `ECHO_ADMIN_TOKEN` unset: `GET /api/admin/discovery/status` `{"detail":"Not Found"}` HTTP 404; `POST .../start` HTTP 404.

WP3 DEPLOYED. Waiting for the owner's two Space secrets; discovery is not started.

### Noticed

- `tests/deploy_hf.py` uploads an explicit file list that has neither new file; a deploy through it alone would leave the Space
  without them (it does not delete them once there). `deploy.yml` uploads the backend folder and carries both; WP7 item.

### Discovery run on the Space

- Owner set `ECHO_ADMIN_TOKEN` (2026-10-30 22:45 IST; `HF_TOKEN` was already a Space secret). Status, with the token read from the
  gitignored `backend/.env` and never printed: `HTTP 200`, `running false`, `hf_token_present true`, `hf_token_role "write"`.
- `POST /api/admin/discovery/start`: `{"started": true, "started_at": "2026-10-30T17:20:10+0000"}` (22:50:10 IST).
- First checkpoint on the dataset (downloaded back): `progress.json` artists 6,447, expanded 539, requests 720, failures 0,
  t 118 s; `artists.jsonl` 6,447 rows: seed 119, chart 1,314 (28 countries), genre 10, related 5,004. Space rate about 6 requests
  a second, about 12 new artists per related call: 120,000 in roughly half an hour.
- Genre new artists 10 here against 43 in the box dry run: Deezer's `/genre/{id}/artists` appears to answer by the requester's
  location, and from the Space it overlaps the charts more. Above zero, as the gate asks.
- No deploy until the run finishes (it would restart the Space).

### Run finished

- Status (admin route, 2026-10-31): `running false`, `phase "done"`, `error null`, started 2026-10-30T17:20:10Z, finished
  22:04:22Z (16,864 s, 4.7 h, not the half hour estimated from the first checkpoint: the related calls slowed as the walk deepened
  and new artists per call fell). Artists 120,000, expanded 95,236, requests 95,417 by the status / 96,741 by the manifest
  (the manifest counts the closing `nb_fan` fills), failures 0. `stopped_by "cap"`: the walk reached the 120,000 cap.

### Verification block, verbatim (warnings from `huggingface_hub` about symlinks removed)

```
python -c "from huggingface_hub import hf_hub_download as d; import json; m=json.load(open(d('shreyash616/echo-ml-artifacts','discovery/MANIFEST.json',repo_type='dataset'))); print(m)"
{'rows': 120000, 'sha256_artists_jsonl': '6ca1a1534f1de2cc77b9ef3e1a531df25e05d51442e9e88b0df7f7a3287b408e', 'counts_per_source': {'seed': 119, 'chart': 1314, 'genre': 10, 'related': 118557}, 'chart_countries': ['AE', 'AR', 'BD', 'BR', 'DE', 'EG', 'ES', 'FR', 'GB', 'HK', 'ID', 'IN', 'IT', 'JP', 'KR', 'MX', 'MY', 'NG', 'PH', 'PK', 'SA', 'SG', 'TH', 'TR', 'TW', 'US', 'VN', 'ZA'], 'chart_country_count': 28, 'chart_kind': {'AE': 'official', 'AR': 'official', 'BD': 'user', 'BR': 'official', 'DE': 'official', 'EG': 'official', 'ES': 'official', 'FR': 'official', 'GB': 'official', 'HK': 'user', 'ID': 'official', 'IN': 'user', 'IT': 'official', 'JP': 'official', 'KR': 'official', 'MX': 'official', 'MY': 'official', 'NG': 'official', 'PH': 'official', 'PK': 'user', 'SA': 'official', 'SG': 'official', 'TH': 'official', 'TR': 'official', 'TW': 'user', 'US': 'official', 'VN': 'user', 'ZA': 'official'}, 'official_chart_countries': 22, 'charts': {'SG': {'title': 'Top Singapore', 'playlist_id': 1313620765, 'chart_kind': 'official', 'artists': 86, 'new': 84}, 'US': {'title': 'Top USA', 'playlist_id': 1313621735, 'chart_kind': 'official', 'artists': 65, 'new': 57}, 'GB': {'title': 'Top UK', 'playlist_id': 1111142221, 'chart_kind': 'official', 'artists': 77, 'new': 34}, 'ES': {'title': 'Top Spain', 'playlist_id': 1116190041, 'chart_kind': 'official', 'artists': 53, 'new': 35}, 'MX': {'title': 'Top Mexico', 'playlist_id': 1111142361, 'chart_kind': 'official', 'artists': 54, 'new': 43}, 'BR': {'title': 'Top Brazil', 'playlist_id': 1111141961, 'chart_kind': 'official', 'artists': 67, 'new': 59}, 'AR': {'title': 'Top Argentina', 'playlist_id': 1279119721, 'chart_kind': 'official', 'artists': 60, 'new': 31}, 'KR': {'title': 'Top South Korea', 'playlist_id': 1362510315, 'chart_kind': 'official', 'artists': 87, 'new': 77}, 'JP': {'title': 'Top Japan', 'playlist_id': 1362508955, 'chart_kind': 'official', 'artists': 74, 'new': 60}, 'FR': {'title': 'Top France', 'playlist_id': 1109890291, 'chart_kind': 'official', 'artists': 73, 'new': 51}, 'DE': {'title': 'Top Germany', 'playlist_id': 1111143121, 'chart_kind': 'official', 'artists': 78, 'new': 9}, 'IT': {'title': 'Top Italy', 'playlist_id': 1116187241, 'chart_kind': 'official', 'artists': 77, 'new': 13}, 'ID': {'title': 'Top Indonesia', 'playlist_id': 1116188761, 'chart_kind': 'official', 'artists': 79, 'new': 53}, 'PH': {'title': 'Top Philippines', 'playlist_id': 1362518895, 'chart_kind': 'official', 'artists': 80, 'new': 40}, 'TR': {'title': 'Top Turkey', 'playlist_id': 1116189071, 'chart_kind': 'official', 'artists': 72, 'new': 6}, 'NG': {'title': 'Top Nigeria', 'playlist_id': 1362516565, 'chart_kind': 'official', 'artists': 69, 'new': 59}, 'ZA': {'title': 'Top South Africa', 'playlist_id': 1362528775, 'chart_kind': 'official', 'artists': 83, 'new': 20}, 'SA': {'title': 'Top Saudi Arabia', 'playlist_id': 1362521285, 'chart_kind': 'official', 'artists': 79, 'new': 59}, 'EG': {'title': 'Top Egypt', 'playlist_id': 1362501615, 'chart_kind': 'official', 'artists': 64, 'new': 47}, 'AE': {'title': 'Top United Arab Emirates', 'playlist_id': 1362491345, 'chart_kind': 'official', 'artists': 69, 'new': 24}, 'TH': {'title': 'Top Thailand', 'playlist_id': 1362524475, 'chart_kind': 'official', 'artists': 75, 'new': 45}, 'MY': {'title': 'Top Malaysia', 'playlist_id': 1362515675, 'chart_kind': 'official', 'artists': 62, 'new': 44}, 'IN': {'title': 'Bollywood Top 50 (Filtr Arabia)', 'playlist_id': 5714603022, 'chart_kind': 'user', 'artists': 52, 'new': 51}, 'PK': {'title': 'TRENDING PAKISTANI SONGS 2025', 'playlist_id': 15038204963, 'chart_kind': 'user', 'artists': 49, 'new': 39}, 'BD': {'title': 'Top Bangla Songs 2024', 'playlist_id': 13175326563, 'chart_kind': 'user', 'artists': 73, 'new': 73}, 'TW': {'title': 'Top 50 - Taiwan', 'playlist_id': 10561367522, 'chart_kind': 'user', 'artists': 39, 'new': 31}, 'HK': {'title': 'Hong Kong 2025', 'playlist_id': 13617318101, 'chart_kind': 'user', 'artists': 71, 'new': 70}, 'VN': {'title': 'Vietnamese (V-Pop) 2026', 'playlist_id': 14760476603, 'chart_kind': 'user', 'artists': 100, 'new': 100}}, 'depths': {'0': 1443, '1': 9831, '2': 24330, '3': 34032, '4': 31519, '5': 18845}, 'expanded': 95236, 'stopped_by': 'cap', 'unresolved_seed_ids': 0, 'requests': 96741, 'failed_requests': 0, 'shards': 480, 'shard_size': 250, 'elapsed_s': 17050, 'finished_at': '2026-10-30T22:04:20+0000'}
```

Beyond the block: `artists.jsonl` and `shards.json` downloaded from the dataset and checked against the manifest
(`scratchpad/verify_wp3.py`):

```
rows 120000 | manifest rows 120000
sha ok: True
unique ids: True
sources: {'seed': 119, 'chart': 1314, 'genre': 10, 'related': 118557}
chart kinds: {'official': 22, 'user': 6} | countries 28
shards 480 size 250 | every artist exactly once: True
first {'shard': 'a00000', 'start': 0, 'end': 250, 'first_deezer_id': 816, 'last_deezer_id': 219242275} last {'shard': 'a00479', 'start': 119750, 'end': 120000, 'first_deezer_id': 107943672, 'last_deezer_id': 195921677}
```

### Gate

- At least 80,000 artists: 120,000. Met.
- Counts per source above zero: chart 1,314, related 118,557, genre 10. Met (genre is thin; see Noticed).
- At least 28 chart countries with `chart_kind` each, at least 22 `official`: 28 countries, 22 `official`, 6 `user`. Met.
- `shards.json` covering every artist exactly once: 480 shards of 250, rows 0 to 120,000, each row covered once. Met.
- The SHA256 of `artists.jsonl` on the dataset equals the manifest's `6ca1a153...`. Met.

PACKET WP3 READY FOR REVIEW.

### Noticed (run)

- Genre added 10 new artists from the Space against 43 in the box dry run; every other genre-page artist was already there from
  the seeds and charts. Above zero as the gate asks; a reviewer call whether genre needs another source.
- Depths 0 to 5 (1,443 / 9,831 / 24,330 / 34,032 / 31,519 / 18,845): 85% of the artists are 3 or more related hops from a seed
  or chart. Shards run in file order, so the early shards are the near ones; that is the order section 5 asks for.

## WP4. Harvester on one fixture shard

- Started: 2026-10-30 23:05 IST, as the 21:48 GO allows once discovery was checkpointing (first checkpoint 22:52).
- Files: `ml/data/harvest_v3.py` (new), `tests/test_harvest_shard.py` (new), `tests/test_harvest_resume.py` (new),
  `tests/fixtures/harvest_fixture_shard.json` (new: Khruangbin, Altin Gun, Fatoumata Diawara, Snarky Puppy, The Marias;
  none in the benchmark or the harvest seeds, checked), `.gitignore` (`/ml/data/harvest_v3_work`; `/ml/data/previews`
  was already ignored), `SCALE_PROGRESS.md`.

### What was built

`harvest_v3.py`, stages A to F of section 5, reusing the P4 code instead of copying it:
- A: iTunes artist search (first result whose `harvest_v2._norm_str` name equals the artist's), lookup of 200 songs, songs
  with a preview in iTunes order, `harvest_v2._is_clean_track_name`, at most 20 an artist, deduplicated by key and
  iTunes id against the global set and the shard; harvest_v2's record fields; 3 s spacing (20 a minute), exponential
  backoff on 403 / 429.
- B: 8 connections, one retry, failures to the dropped list. C: `redescribe_catalog._describe_one` over 10 workers with
  `header.json`'s cut points. D: P4's decode and 48 kHz resample, `_window_audio_frames`, 64 windows a batch, mean,
  L2-normalise, float16; anchors and vibes as P4; one INFO line a batch to `<shard>.clap.log`.
- E: `fetch_lyrics_remote.fetch_lyrics` with the application headers, 24 fetchers over disjoint ranges; then P4's theme
  (`apply_lyric_theme`), words per minute and language (artist majority, then `language_from_meta`).
- F: features, CLAP, dropped and manifest files to `catalog/v3/`, every file downloaded back and hashed against the
  manifest; outside a fixture the shard joins `catalog/v3/MANIFEST.json` and its keys `catalog/v3/keys.txt`, and the
  claim is released; the shard's previews are deleted.
- Claims and resume: `catalog/v3/claims/<runner>.json`; the lowest shard neither finished nor claimed by another runner
  within 6 hours; a runner's own surviving claim (a crash) is taken again first, from stage A.
- harvest_v2 and the recommender import torch at module level, so the harvester imports them inside the stages; the
  10 spawned describe workers then load no torch (checked with `-X importtime`).

### The owner's windows

Both opened in the Terminal panel before the run and left up: `Get-Content -Path ml\data\harvest_v3_work\fixture.describe.log -Wait`
(stages A to C, E, F) and `Get-Content -Path ml\data\harvest_v3_work\fixture.clap.log -Wait` (stage D, one line a batch:
batch index of total, windows, device, seconds, songs a second, `torch.cuda.memory_allocated` in MB).

```
CLAP batch 1/5 | windows 64 | device cuda | 184.11 s | 0.12 songs/s | cuda 778 MB
CLAP batch 2/5 | windows 64 | device cuda | 0.88 s | 0.23 songs/s | cuda 778 MB
CLAP batch 3/5 | windows 64 | device cuda | 0.65 s | 0.34 songs/s | cuda 778 MB
CLAP batch 4/5 | windows 64 | device cuda | 0.67 s | 0.46 songs/s | cuda 778 MB
CLAP batch 5/5 | windows 44 | device cuda | 0.45 s | 0.54 songs/s | cuda 778 MB
CLAP done: 100 songs, 300 windows, 186.8 s
```

### Verification block, verbatim

```
$ python ml/data/harvest_v3.py --shard-file tests/fixtures/harvest_fixture_shard.json
shard fixture done in 420 s: 100 songs, 0 dropped, 100 previews deleted

$ python -m pytest -q tests/test_harvest_shard.py tests/test_harvest_resume.py
7 passed, 1 warning in 294.28s (0:04:54)

$ ls ml/data/previews | wc -l
0

$ python -c "from huggingface_hub import hf_hub_download as d; import json; print(json.load(open(d('shreyash616/echo-ml-artifacts','catalog/v3/shard_fixture.manifest.json',repo_type='dataset'))))"
{"shard": "fixture", "fixture": true, "runner": "Shreyash-PC", "artists": 5, "first_deezer_id": 5328540, "last_deezer_id": 12607183, "artists_unresolved": 0, "songs": 100, "dropped": 0, "dropped_by_stage": {}, "lyrics": {"fetched": 40, "none": 14, "error": 33}, "files": {"features": "catalog/v3/shard_fixture.features.jsonl.gz", "clap": "catalog/v3/shard_fixture.clap.f16.npy", "dropped": "catalog/v3/shard_fixture.dropped.jsonl"}, "sha256": {"features": "98339e75714412df684da364de136982a284609581127dd8f0cd3127bb978f69", "clap": "4130572ef966a3e7b7663b80dec3b37c20817cb559e22df8b42152b3b52e71c4", "dropped": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}, "clap_shape": [100, 512], "describe_commit": "e4c1b1bd2b4dd35a21560de82d280a587461ce99", "date": "2026-10-30T18:24:25+00:00", "seconds_per_stage": {"A": 47.2, "B": 24.8, "C": 33.0, "D": 209.1, "E": 94.8, "F": 9.1}}
(plus "keys" and "itunes_ids", 100 each)
```

Uploaded files hashed independently after a fresh download:
```
catalog/v3/shard_fixture.features.jsonl.gz: 98339e75714412df684da364de136982a284609581127dd8f0cd3127bb978f69 matches
catalog/v3/shard_fixture.clap.f16.npy: 4130572ef966a3e7b7663b80dec3b37c20817cb559e22df8b42152b3b52e71c4 matches
catalog/v3/shard_fixture.dropped.jsonl: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 matches
```
Only `catalog/v3/shard_fixture.*` is on the dataset under `catalog/`; nothing under `discovery/` was touched.

### Gate

- Both tests pass (7 of 7). `test_harvest_shard` runs one fixture artist end to end into a local store: rows and CLAP aligned,
  float16 unit rows, SHA256s, describe commit WP0's, seconds for A to F, dropped count, every row's field set equal to
  today's `catalog_features.jsonl` row with the same `has_lyrics` (two sets: 65 fields, and 64 without `theme_vec`),
  batched CLAP equal to P4's one-window path (max abs difference 3.4e-08 and 4.1e-08), previews deleted.
  `test_harvest_resume`: lowest unclaimed shard, fresh foreign claim skipped, stale one taken over, crash in stage D leaves
  the claim and the rerun starts at A, finish writes the manifest and keys and releases the claim, the next shard gets the keys.
- Previews directory empty. Fixture manifest: `describe_commit` `e4c1b1bd2b4dd35a21560de82d280a587461ce99` (WP0's), seconds
  per stage A 47.2, B 24.8, C 33.0, D 209.1, E 94.8, F 9.1, dropped 0, SHA256s matching the uploaded files.

### Noticed

- **CLAP batch 64 costs a one-time stall of 3 to 5 minutes per process** (184 s here; 185, 235 and 328 s in three separate
  measurements), on the first 64-window call only; peak memory is 3.3 GB of the 3080's 10 GB, so it is not memory. Measured
  in fresh processes: batch 16 first call 0.93 s and steady 8 ms a window; batch 32 first call 0.89 s, 11 ms a window;
  batch 64 first call 185 s, 15 ms a window. At one process per shard and about 200 shards that is about 10 hours of stall,
  and 64 is also the slowest per window. Recommendation: 32. Section 5 says 64, so the fixture ran at 64.
- Stage E: 33 of 87 lyric fetches ended in error after the five 503 retries (38%). `fetch_lyrics_remote`'s own main runs a
  second pass over its errors; the harvester's stage E does not yet. A second pass is the cheap fix, for WP5.
- `test_harvest_shard` takes about 5 minutes (network, the CLAP stall); it is not in `tests.yml`.

PACKET WP4 READY FOR REVIEW

## WP5. Shards 0 to 9 on the box

- Started: 2026-10-31 00:38 IST after GO WP5 (README ce741ca). One runner only, the Windows box (`Shreyash-PC`), for WP5 and WP6.
- Files: `ml/data/harvest_v3.py` (the four fixes and the log pair), `.github/workflows/tests.yml` (`test_harvest_resume`),
  `SCALE_PROGRESS.md`. Commit `b24f216`.

### The fixes before shard 0, as ce741ca named them

1. Stage D streams: songs are decoded and windowed only as the next batch needs them; a batch's windows are dropped once
   embedded, so memory holds one batch of audio. Stage C keeps each preview's decoded length (`_n22`, not written) so the
   batch total in the log line is exact without a second decode (`_window_count`).
2. `CLAP_BATCH = 32`.
3. Rule 23: `shutil.disk_usage` on the previews drive before every shard; under 8 GB free (`redescribe_catalog.MIN_FREE_BYTES`)
   it logs and stops. The free space at shard start is recorded in each manifest (`free_gb_at_start`).
4. Stage E: each row's fetch catches any exception, counted as `error` with its type; a second pass runs over the error rows;
   stage E refuses to start when `registered_app()` gives no `Lrclib-Api-Key`.

Also: an iTunes 200 without JSON is retried rather than killing the shard; `test_harvest_resume` in `tests.yml`.

### The owner's two windows, once for the whole run

Both loggers also write `ml/data/harvest_v3_work/current.describe.log` and `current.clap.log`, truncated at each shard start.
`Get-Content -Wait` follows a truncated file (checked: after a truncate it printed the next shard's lines from the top). The two
windows now run `Get-Content -Path ml\data\harvest_v3_work\current.describe.log -Encoding UTF8 -Wait` and the same for
`current.clap.log`; `-Encoding UTF8` because the default code page showed Altin Gun's dotless i as mojibake.

### Fixture shard re-verified after the code change (the WP6 rule)

```
$ python ml/data/harvest_v3.py --shard-file tests/fixtures/harvest_fixture_shard.json
stage A done in 49.8 s | B 25.7 s | C 34.6 s | D 28.8 s | E 61.9 s | F: uploaded and read back, SHA256s match
shard fixture done in 213 s: 100 songs, 0 dropped, 100 previews deleted

current.clap.log:
CLAP batch 1/10 | windows 32 | device cuda | 1.31 s | 2.88 songs/s | cuda 778 MB
CLAP batch 2/10 | windows 32 | device cuda | 0.21 s | 4.14 songs/s | cuda 778 MB
...
CLAP batch 10/10 | windows 12 | device cuda | 0.08 s | 6.10 songs/s | cuda 778 MB
CLAP done: 100 songs, 300 windows, 16.4 s

$ python -m pytest -q tests/test_harvest_shard.py tests/test_harvest_resume.py
7 passed, 1 warning in 98.64s (0:01:38)

$ ls ml/data/previews | wc -l
0

catalog/v3/shard_fixture.manifest.json (downloaded): songs 100, dropped 0, describe_commit e4c1b1bd2b4dd35a21560de82d280a587461ce99,
seconds_per_stage A 49.8 B 25.7 C 34.6 D 28.8 E 61.9 F 10.8, free_gb_at_start 133.9,
sha256 features e39a826a..., clap 8c738a57..., dropped e3b0c442...
```

- Stage D 209 s to 29 s: no first-call stall at batch 32 (1.31 s), 0.21 s a batch after.
- The CLAP file's SHA256 changed from WP4's (4130572e to 8c738a57): a different batch size reorders the GPU arithmetic, so float16
  bits move within rounding; the batched-against-one-window parity in `test_harvest_shard` still passes.
- Stage E on this rerun fetched only the 45 rows that had no lyric file after WP4, that is the rows that had already failed
  (33 errors, 14 none, less any the test run fetched): first pass 5 fetched, 9 none, 31 errors; the second pass over the 31
  fetched 5 more and left 23. Biased sample; shard 0 gives the real rate.
- Free disk at the start 133.9 GB (WP1: 135.2 GB).

Shards 0 to 9 wait for `discovery/shards.json`, written when discovery finishes.

### Shards 0 and 1, sequential stages (code `b24f216`)

Run as `py -3 ml/data/harvest_v3.py --next --runner Shreyash-PC`, one process a shard, from 2026-10-31 03:36 IST. Numbers
from the manifests on the dataset (`catalog/v3/shard_a0000{0,1}.manifest.json`, `catalog/v3/MANIFEST.json`):

```
a00000: 250 artists, 0 unresolved, 4936 songs, 10 dropped (C 10), free_gb_at_start 133.9, describe_commit e4c1b1bd...
  seconds_per_stage A 1551.8 B 1101.3 C 1240.1 D 723.8 E 1236.4 F 17.1 (sum 5870.5)
  lyrics first_pass none 94 fetched 1357 error 980; second_pass fetched 411 none 88 error 481; error_after_retry 481
a00001: 250 artists, 1 unresolved, 4886 songs, 0 dropped, free_gb_at_start 132.8, describe_commit e4c1b1bd...
  seconds_per_stage A 1556.0 B 1064.1 C 1208.3 D 739.7 E 2612.3 F 44.5 (sum 7224.9)
  lyrics first_pass fetched 2342 none 197 error 2182; second_pass fetched 807 none 212 error 1163; error_after_retry 1163
MANIFEST.json: songs 9822, shards a00000 (4936, features 9e3f3cf5...), a00001 (4886, features 85d34618...)
```

These `lyrics` counters are read as `none + fetched + error`. The manifests also carry each error under
`error HTTPStatusError`, which counts the same rows again (README 3949be4). Fixed in `c938999`: the exception types are now a
separate `error_types` breakdown.

**Against the fixture.** WP5's fixture rerun had 100 songs: A 49.8, B 25.7, C 34.6, D 28.8, E 61.9, F 10.8 s. Shard 0 has 49.4
times the fixture's songs. Its stages against 49.4 times the fixture's:
- **A:** 1,552 s against 2,460. A is the iTunes pacing, 500 calls at 3 s. The fixture's A spreads the first call's latency over
  only 5 artists, so per artist it is 6.2 s on the shard against 10.0 s on the fixture.
- **B, C and D:** all under linear. B 1,101 against 1,270; C 1,240 against 1,709; D 724 against 1,422, which is 0.15 s a song.
- **E:** not comparable, because the fixture rerun fetched only rows that had failed before. Between shards: shard 0's E took
  1,236 s and shard 1's 2,612 s, for 2,431 and 4,721 rows without a lyric file.

**Stage D memory.** The main process's working set was sampled every 15 s (`ml/data/harvest_v3_work/wp5_ram.csv`).
- The peak is the first sample in D, while the CLAP checkpoint loads: 2,133 MB on shard 0 and 2,055 MB on shard 1.
- After that it stays between 700 and 830 MB for the whole stage, flat, with no growth as songs are embedded.
- The highest of any stage is C: 2,856 MB across the main process and its 10 describe workers. The box has 31,850 MB.

**Real stage E rates.** These count only rows without a lyric file: 2,505 of shard 0's songs already had one from P4's
catalogue, and 165 of shard 1's.

| | Shard 0 | Shard 1 |
|---|---|---|
| Rows without a lyric file | 2,431 | 4,721 |
| First-pass errors | 40.3% | 46.2% |
| Fetched, after the second pass | 72.7% | 66.7% |
| None, after the second pass | 7.5% | 8.7% |
| Still in error, after the second pass | 19.8% (481) | 24.6% (1,163) |
| Still in error, as a share of the shard's songs | 9.7% | 23.8% |
| Stage E time | 1,236 s | 2,612 s |

Every error is an `HTTPStatusError`. Shard 1's E took twice as long because the error share rose, and each error costs
`fetch_lyrics_remote`'s 503 retries and sleeps.

**Rate before the change.**
- **Both shards:** 9,822 songs in 13,095 s of stage time, 2,700 songs an hour. A million at that rate is 370 hours,
  15.4 days.
- **Shard 0 alone:** 3,027 an hour, 13.8 days.
- **Shard 1 alone:** 2,435 an hour, 17.1 days.

That is over the 14-day line; README c2f8090's changes below answer it.

**Disk.**
- Free at shard start: 133.9 GB (a00000), 132.8 GB (a00001), 132.8 GB (a00002). WP1 was 135.2 GB.
- The harvester's own files: 51 MB in `harvest_v3_work` (about 20 MB a shard of outputs and read-back copies) and 93 MB of
  lyric files. Previews: 0.
- The 1.1 GB that went between shard 0's start and shard 1's is outside those directories. Watched shard by shard.

### The three changes of README c2f8090, at the a00001 / a00002 boundary (code `c938999`)

The runner was held at the boundary. a00001 finished at 07:16:29 IST, with F read back. The loop's next process was ended 3 s
after it started, before it claimed anything; the dataset had no claims afterwards.

1. **Stage E overlapped with B to D.**
   - `run_rest` starts the 24 fetchers in a thread at the start of B, over stage A's songs, and joins it after D. Theme, words
     per minute and language stay after the join.
   - `seconds_per_stage["E"]` is the fetchers' wall time.
   - The stages no longer add up to the shard's wall time, so `seconds_other` carries:
     - `E_wait_after_D`, the join's wait;
     - `E_theme_language`;
     - `wall`, the shard's wall time: B to F, plus A when A did not run under the previous shard.
2. **The next shard's A overlapped with this shard's B to F.**
   - `--next --shards N` runs N shards in one process. When a shard's A finishes, the next shard
     (`next_shard(..., exclude={this})`) is claimed and its `stage_a` runs in a thread.
   - That A deduplicates against `keys.txt` and this shard's A songs, which reach `keys.txt` only when this shard finishes.
   - One module-wide iTunes pacer (`_ITUNES_LAST` under a lock), so the limit of 20 a minute holds across the two.
   - Claims are one file a shard, `claims/<runner>.<shard>.json`, at most two a runner.
     - `finish` deletes this runner's claims on the shard, whatever their file name.
     - `next_shard` takes the runner's own unfinished claims first, lowest first.
   - `manifest["a_overlapped"]` says which shards' A was hidden.
   - The owner's `current.*.log` is opened once a process and each line names its shard, so both windows show both shards
     while they overlap.
3. **Lyric passes until the error share is small.**
   - After each pass, another runs over the rows in error while they exceed 2% of stage A's songs.
   - At most four passes in all, the first included. That is my reading of "at most four passes".
   - The manifest's `lyrics` is now `songs, had_file, without_file, passes: [{rows, none, fetched, error, error_types}],
     error_after_retry`.

`test_harvest_resume` gets the case asked for:
- A crash in a00000's D, after a00001's overlapped A has run, leaves both claims.
- The restart takes a00000 then a00001, each from A, and releases both.
- The overlapped A saw a00000's songs as taken.

The two existing claim assertions now read `hv.claim_path("box", "a00000")`, the new file name, with the same meaning.

### Fixture shard and the two tests after `c938999` (the WP6 rule)

```
$ python ml/data/harvest_v3.py --shard-file tests/fixtures/harvest_fixture_shard.json      (real 3m5.6s, exit 0)
fixture | stage A done in 59.4 s
fixture | stage E start (lyric fetch, beside B to D)
fixture | stage B start
fixture | E: pass 1 over 35 rows: 8 fetched, 17 none, 10 error {'HTTPStatusError': 10}
fixture | stage B done in 26.9 s
fixture | stage C start
fixture | E: pass 2 over 10 rows: 3 fetched, 1 none, 6 error {'HTTPStatusError': 6}
fixture | E: pass 3 over 6 rows: 2 fetched, 2 none, 2 error {'HTTPStatusError': 2}
fixture | E: lyrics for 35 of 100 songs without a file, 3 passes, 2 still in error (2.0% of the songs)
fixture | stage E lyric fetch done in 48.6 s
fixture | stage C done in 45.2 s
fixture | stage D start
fixture | stage D done in 34.1 s
fixture | stage E theme, words per minute and language done in 5.5 s
fixture | F: uploaded and read back catalog/v3/shard_fixture.features.jsonl.gz, catalog/v3/shard_fixture.clap.f16.npy, catalog/v3/shard_fixture.dropped.jsonl, SHA256s match
fixture | shard fixture done in 122 s: 100 songs, 0 dropped, 100 previews deleted

$ python -m pytest -q tests/test_harvest_shard.py tests/test_harvest_resume.py
........                                                                 [100%]
8 passed, 1 warning in 82.52s (0:01:22)        (the warning: timm's FutureWarning on timm.models.layers)

$ ls ml/data/previews | wc -l
0

catalog/v3/shard_fixture.manifest.json (downloaded): songs 100, dropped 0, describe_commit e4c1b1bd2b4dd35a21560de82d280a587461ce99,
seconds_per_stage A 59.4 B 26.9 E 48.6 C 45.2 D 34.1 F 8.2, seconds_other E_wait_after_D 0.0 E_theme_language 5.5 wall 179.3,
a_overlapped False, free_gb_at_start 132.8,
lyrics songs 100 had_file 65 without_file 35 passes [35: none 17 fetched 8 error 10 | 10: none 1 fetched 3 error 6 | 6: none 2 fetched 2 error 2] error_after_retry 2,
sha256 features d9b7072d..., clap 8c738a57..., dropped e3b0c442...
```

- E finished inside C, and the join waited 0.0 s. The fixture's wall time was 179 s, against 213 s before.
- The CLAP SHA256 8c738a57 equals the previous fixture run's, so stage D is unchanged.
- The features SHA256 moved because stage E fetched 5 more lyrics in passes 2 and 3, which changes those rows' lyric fields.

**The overlap on real code.** A local store with two shards of two fixture artists each
(`--next --shards 2 --store <scratch> --runner smoke`, exit 0), from `current.describe.log`:

```
07:25:17 s00000 | stage A done in 29.7 s
07:25:17 s00001 | stage A start
07:25:17 s00000 | stage E start (lyric fetch, beside B to D)
07:25:17 s00000 | stage B start
07:25:25 s00000 | stage C start
07:25:29 s00001 | stage A done in 12.0 s
07:25:49 s00000 | stage D start
07:26:21 s00000 | shard s00000 done in 64 s: 40 songs, 0 dropped, 40 previews deleted
07:26:21 s00001 | stage E start (lyric fetch, beside B to D)
07:26:21 s00001 | stage B start
07:27:11 s00001 | shard s00001 done in 50 s: 40 songs, 0 dropped, 40 previews deleted
```
Both shards are in the local `MANIFEST.json`, `claims/` is empty afterwards, and previews are 0.

### Shards 2 to 9, overlapped (code `c938999`)

Started 2026-10-31 07:27 IST in one process: `py -3 ml/data/harvest_v3.py --next --shards 8 --runner Shreyash-PC`. a00002 was
claimed first. The owner's two windows stay up on `current.describe.log` and `current.clap.log`.

**Expected before measuring**, from shard 1:
- Two more lyric passes, over about 1,163 and 600 rows, add roughly 700 s to E.
- So E, at about 3,300 s, may now be longer than B + C + D (about 3,000 s) and set the shard's pace.
- Four passes may leave about 6% of the songs in error rather than 2%.

The after rate below comes from the manifests, not from this estimate.

**CI.** Tests on `6a04b7b` (41135855564) failed twice, both times on `test_candidate_pool`'s live p95, and both after discovery
finished at 22:04Z:
- 0.568 s at 22:09Z;
- 0.313 s on the rerun at 22:16Z.

No green run yet; the run on this entry's commit is the next one.

### CI and the Space restart (README 70690e9)

**Tests runs after discovery ended (22:04Z):**

| Run | Commit | Result |
|---|---|---|
| 41135855564 | `6a04b7b` | red, p95 0.568 s; rerun red, 0.313 s |
| 41136344566 | `3949be4` | red |
| 41156708810 | `c2f8090` | green (01:43Z) |
| 41157769469 | `c938999` | green |
| 41158075211 | `7c14c83` | red, p95 0.354 s |
| 41158112863 | `ce1980e` | green |

The `7c14c83` and `ce1980e` runs started 28 s apart, so they overlapped on the Space.

**Restart.** `HfApi().restart_space("shreyash616/echo-backend")` at 02:22:04Z: no code and no upload, sha `12e51b50` before
and after. The Space was `RUNNING` at 02:23:24Z with `/api/health` `{"status":"ok","recommender_loaded":true}`. Admin status
afterwards read `running false`, `started_at null`, `finished_at null`: a fresh process, without discovery's state.

**Live p95 over 20 calls**, `py -3 -m pytest -q -s tests/test_candidate_pool.py` from the box, three runs each:
- before the restart: 0.293 s, 0.272 s, 0.303 s;
- after the restart: 0.337 s (the first run, against a cold process), then 0.238 s and 0.212 s.

Once warm, p95 is back near P9's 0.225 s.

**Tests re-run on `ce1980e` after the restart:** 41158112863 attempt 2, success, finished 02:30:35Z. This is the green run
WP5 names.

### Noticed

- **Logging error in the runner tab.** At 07:31 IST the runner tab printed `--- Logging error --- ... UnicodeEncodeError:
  'charmap'`. Python's stdout through the PowerShell pipe is cp1252, and an artist name outside it could not be printed. It
  came from logging's `handleError`, not a crash: the shard went on, and the UTF-8 `<shard>.describe.log` and
  `current.describe.log` have the line. The next runner start should set `PYTHONIOENCODING=utf-8`; not changed mid-run.
- **`MAX_CLAIMS`** is declared and not read; the bound of two is structural (README 70690e9). Left as is.

### a00002 and a00003 on `c938999`

From the manifests on the dataset:

| | a00002 | a00003 |
|---|---|---|
| Songs | 4,650 | 4,613 |
| Dropped | 11 (C) | 9 (C) |
| A (s) | 1,525.9 | 1,500.2 |
| B (s) | 1,174.1 | 1,336.4 |
| C (s) | 1,220.7 | 1,275.7 |
| D (s) | 756.5 | 767.5 |
| E, lyric fetch (s) | 3,058.5 | 3,748.9 |
| F (s) | 78.2 | 73.5 |
| `E_wait_after_D` (s) | 0.0 | 369.3 |
| `E_theme_language` (s) | 91.2 | 88.6 |
| Wall (s) | 4,846.6 | 3,911.0 |
| `a_overlapped` | False (first shard of the process) | True |

**Lyric passes (rows: fetched / none / error):**
- a00002: 4,379: 2,164 / 219 / 1,996; 1,996: 771 / 247 / 978; 978: 342 / 154 / 482; 482: 119 / 122 / 241. 241 left,
  5.2% of 4,661.
- a00003: 4,416: 1,903 / 274 / 2,239; 2,239: 770 / 264 / 1,205; 1,205: 301 / 255 / 649; 649: 138 / 200 / 311. 311 left,
  6.7% of 4,622.

**a00003's measured rate**, the first shard with both overlaps: 4,613 songs in 3,911 s of wall time is 4,246 songs an hour.
A million at that rate is 235 hours, 9.8 days. On a00003, E set the pace: B + C + D = 3,380 s against E's 3,749 s.

`MANIFEST.json`: 19,085 songs over a00000 to a00003.

### The LRCLIB probe (README 0b03ff7)

Command: `fetch_lyrics_remote.py --tracks <a00002_nofile_100.jsonl> --out <scratch> --spacing 0.75`. One thread, the
registered app. The input is the first 100, in shard order, of the 977 a00002 songs with no lyric file. These mix rows E left
in error with rows LRCLIB has no lyrics for.

```
probe 1, 08:58 to 09:23 IST, while a00003's E ran its 24 threads on the same key:
done {'fetched': 5, 'none': 76, 'error': 68, 'error_after_retry': 19} in 25 min; status {'get 200': 5, 'get 404': 288, 'get 503': 22, 'search 200': 178, 'search 503': 632}
probe 2, 09:57 to 10:23 IST, the runner held, no other LRCLIB traffic from the box:
done {'fetched': 5, 'none': 81, 'error': 62, 'error_after_retry': 14} in 26 min; status {'get 200': 5, 'get 404': 288, 'search 200': 184, 'search 503': 708}
```

- In probe 2, every error is `503 Service Unavailable` on `/api/search` (62 of 62); `/api/get` never answers 503 when the
  box is quiet.
- `/api/search` answers 503 on 708 of 892 calls (79%), at one request every 0.75 s, after `_get`'s own five 1 s retries.
- The first pass failed 48 of 100 rows with no load, against 49 under load.

The failures are not load-dependent. They are LRCLIB's search endpoint refusing about four calls in five from this key or this
IP, whatever the rate. Only the 22 `get 503` in probe 1 track our own load.

### The lyric pacer and per-pass status (code `c441bea`)

- **`fetch_lyrics_remote.py`.** `_LAST` is read, slept on and stamped under `_LAST_LOCK`, the shape of `_ITUNES_LOCK`.
  `LRCLIB_STATUS` increments under `_STATUS_LOCK`; `+=` on a Counter is not atomic across threads, and the per-pass deltas
  need exact counts.
- **`harvest_v3.py`.** Each pass takes `LRCLIB_STATUS` before and after. The difference is logged with the pass and stored as
  `lyrics.passes[].status`, keyed `"get 404"` etc. as the script prints them, and it includes `_get`'s own 429 / 503 retries.
  `LYRIC_MAX_PASSES` 4, `LYRIC_ERROR_SHARE` 0.02, `LYRIC_WORKERS` 24 and `LYRIC_SPACING_S` 0.06 are unchanged.
- **Pacer check.** A stand-in client sleeps 50 ms a request and records start times; 24 threads make 5 rows each:

  ```
  c441bea:          360 requests over 21.65 s, min gap 60.0 ms, mean 60.3 ms
  HEAD before it:   360 requests over 0.98 s, min gap 0.0 ms, mean 2.7 ms, 367 requests a second
  ```

  On the real network the unlocked burst was bounded by latency, not by the 0.06 s spacing.

### Fixture shard and the two tests after `c441bea`

Runner held, so nothing else on the key:

```
$ python ml/data/harvest_v3.py --shard-file tests/fixtures/harvest_fixture_shard.json      (exit 0)
fixture | stage A done in 93.6 s
fixture | stage E start (lyric fetch, beside B to D)
fixture | stage B start
fixture | E: pass 1 over 22 rows: 1 fetched, 21 none, 0 error {}; status {'get 200': 20, 'get 404': 5, 'get 503': 4, 'search 200': 4, 'search 503': 1}
fixture | E: lyrics for 22 of 100 songs without a file, 1 passes, 0 still in error (0.0% of the songs)
fixture | stage E lyric fetch done in 11.9 s
fixture | stage C done in 59.2 s
fixture | stage D done in 47.3 s
fixture | shard fixture done in 227 s: 100 songs, 0 dropped, 100 previews deleted

$ python -m pytest -q tests/test_harvest_shard.py tests/test_harvest_resume.py
8 passed, 1 warning in 109.43s (0:01:49)

catalog/v3/shard_fixture.manifest.json (downloaded): songs 100, dropped 0, describe_commit e4c1b1bd...,
lyrics.passes[0].status {'get 200': 20, 'get 404': 5, 'get 503': 4, 'search 200': 4, 'search 503': 1},
sha256 clap 8c738a57 (unchanged), features 967383de, dropped e3b0c442; free_gb_at_start 133.8
```

**Previews.** `ls ml/data/previews | wc -l` gave 11 after the fixture, all written 09:57:10 to 09:57:12. They were a00004's
stage B, cut by the hold; the fixture deleted its own 100. The 11 were removed (a00004 fetches them again), leaving 0.

**Hold.** The runner was ended at 09:57:12, 5 s into a00004's B, after `finish(a00003)` returned. Claims afterwards:
`Shreyash-PC.a00004.json`, `Shreyash-PC.a00005.json`. a00004's overlapped A (1,481 s) was lost, as accepted.

### Shards 4 to 9 on `c441bea`

Started 10:33 IST in the runner tab with `PYTHONIOENCODING=utf-8`, `--next --shards 6`. a00004 resumed from its own claim,
from stage A. The owner's two windows are up.

### Noticed: free disk fell 15 GB at 10:31, the Windows pagefile

**What changed.**
- a00004 started at 10:33 with 118.7 GB free; the fixture at 10:24 had 133.8 GB.
- The harvester's own files account for none of it: `harvest_v3_work` 88 MB, lyrics 121 MB, previews 0, HF cache 4.8 GB
  (unchanged), `%TEMP%` 1.1 GB.
- `C:\pagefile.sys` is 43.7 GB on disk, last written 10:31:34; `Win32_PageFileUsage` reports a base of 29,860 MB. Windows grew
  its system-managed pagefile by about 14.5 GB under commit pressure. It does not shrink until a reboot.

**Commit pressure.**
- Commit charge at 10:40: 51.9 GB of a 60.3 GB limit; free RAM 0.1 GB.
- Process private bytes sum to 16.5 GB; nonpaged and paged pools are 3.6 and 2.8 GB. About 29 GB of commit is in no process
  and no pool that `Get-Process` and the memory counters show.
- The box had about 31 GB of its 31.8 GB RAM in use at the first WP5 sample, at 03:42, with the harvester at 0.8 GB. So this
  predates WP5. The harvester's peak, 2.9 GB in stage C, is what tips commit over the pagefile's base.

**What I did not do.** I did not change pagefile settings or touch other processes, since both are for the owner.

**For the gate.** The WP5 line "free disk within 3 GB of WP1's 135.2 GB" now reads 118.7 GB. All but about 1 GB of the drop is
the pagefile, not harvest data. A reboot would return the pagefile to its base, but it ends the runner.

### a00004 lost in stage D: CUDA out of memory (11:59 IST)

**Run.** `c441bea`, `--shards 6`, started 10:33. a00004 resumed from its own claim; a00005's A ran under it, 10:58 to about 11:23.

**Stages before the crash** (from `a00004.describe.log`):
- A 1,495.8 s, B 1,514.2 s, C 1,300.9 s.
- D started 11:45:50.
- E pass 1 over 4,312 rows: 1,203 fetched, 327 none, 2,782 error (`HTTPStatusError` 2,731, `ReadTimeout` 51); status
  `{'get 200': 1188, 'get 404': 2399, 'get 502': 153, 'get 503': 19856, 'search 200': 808, 'search 502': 4, 'search 503': 4243}`.
  `/api/get` is now shedding too: 19,856 503s against 3,587 answers. At 10:23 it gave none.

**The crash.**
- Batch 384 of 414 was logged at 11:58:56 (`cuda 778 MB`, flat through D, as on every shard).
- The next `get_audio_embedding_from_data` raised `RuntimeError: CUDA error: out of memory` in
  `laion_clap/clap_module/htsat.py:807` (`interpolate(torch.sigmoid(x)...)`).
- The runner logged `=== exit 1 after 5162 s` at 11:59:06.

**Evidence, as found; nothing changed.**
- **GPU.** After the crash, `nvidia-smi` shows 1,882 of 10,240 MiB used, by desktop apps. No process started between 11:30
  and the crash.
- **Box RAM.** The harvester's working set was trimmed to 190 to 480 MB through D (`wp5_ram.csv`; 700 to 830 MB on shards 0
  to 2). Box RAM in use was 31.4 to 31.8 GB of 31.85 GB, and free RAM was 0.1 to 0.3 GB.
- **Commit.** 51.7 of 60.3 GB charged after the harvester exited; available 8.6 GB.
- **Windows events.** None from the resource-exhaustion detector in the last 6 hours.

The likeliest reading is host-side memory pressure under WDDM, with the CUDA allocation failing although VRAM was free. I
cannot prove it from here, and per README dcb05e7 I did not look further into the box's memory.

**State.**
- `catalog/v3/MANIFEST.json`: 19,085 songs, a00000 to a00003; nothing of a00004 on the dataset.
- Claims: `Shreyash-PC.a00004.json` and `Shreyash-PC.a00005.json` are held. A restart re-runs a00004 from A, then a00005.
- a00004's 4,412 previews (about 4.5 GB, written from 10:58) were removed; previews 0.
- Free disk 118 GB. The pagefile is still 43.7 GB, last written 10:31:34.

The runner is not restarted: BLOCKED in the channel.

### LRCLIB rate probe, spacing 0.25 s, in-pass breaker, lyric error list (code `08a1083`)

**Rate probe (README 3927a8a point 1).**
- One thread, `fetch_lyrics_remote.py`, 200 fresh a00004 songs a spacing. a00004's rows were re-resolved with the
  harvester's `resolve()` into a scratch `WORK`: artists 0 to 13, 14 to 27 and 28 to 41, one set a run, so LRCLIB's cache
  could not clean a later run.
- The share of failed get calls does not fall with spacing, and 502s appear at 1.0 s. No spacing is clean.
- README 045f348: LRCLIB is down, not throttling (12 × 502 on 12 gets from the reviewer's machine at 13:40).

```
0.25 s, 12:10-12:46: done {'fetched': 139, 'error': 100, 'none': 35, 'error_after_retry': 26} in 36 min; status {'get 200': 124, 'get 404': 180, 'get 503': 798, 'search 200': 95, 'search 503': 353}
0.50 s, 12:49-13:14: done {'fetched': 170, 'error': 71, 'none': 15, 'error_after_retry': 15} in 25 min; status {'get 200': 169, 'get 404': 71, 'get 503': 660, 'search 200': 37, 'search 503': 126}
1.00 s, 13:17-13:37: done {'fetched': 97, 'error': 176, 'none': 15, 'error_after_retry': 88} in 20 min; status {'get 200': 94, 'get 404': 87, 'get 502': 130, 'get 503': 375, 'search 200': 39, 'search 503': 127}
```

**`08a1083` (README 045f348).**
- **Spacing.** `LYRIC_SPACING_S` 0.25: 4 requests a second app-wide under `_LAST_LOCK`.
- **Breaker, inside the pass.** Each fetcher checks the pass's status delta after every row. From 300 calls on
  (`LYRIC_BREAKER_MIN_CALLS`), `get 503 + search 503` at or above half the pass's calls (`LYRIC_503_BREAKER`) sets an event,
  and every fetcher returns before its next row. Rows not tried join the rows in error; the pass records `untried`;
  `lyrics.stopped_by` is `"lrclib_503"`, otherwise `"error_share"`, `"max_passes"` or `"no_rows"`.
- **Error list.** Stage F writes `shard_<id>.lyrics_error.txt`, the kept songs still in lyric error, one key a line. It is
  uploaded and read back with the shard's files and listed under `files` and `sha256`.
- **Tests.** `test_harvest_resume` gets two cases, with a stub fetcher:
  - 3,000 rows at 6 search-503 calls a row: the breaker fires within one pass, with over 2,000 rows never tried and all 3,000
    in the error list;
  - a pass of 200 calls never trips, and a pass of 1,000 calls with no 503s never trips.

**Fixture and the two tests after `08a1083`.**

```
$ python ml/data/harvest_v3.py --shard-file tests/fixtures/harvest_fixture_shard.json      (exit 0)
fixture | E: pass 1 over 21 rows: 0 fetched, 0 none, 21 error {'HTTPStatusError': 21}, 0 not tried; status {'get 502': 21}; 503 0 of 21 calls
fixture | E: pass 2 over 21 rows: ... status {'get 502': 21}; 503 0 of 21 calls      (passes 3 and 4 the same)
fixture | E: lyrics for 21 of 100 songs without a file, 4 passes, 21 still in error (21.0% of the songs)
fixture | stage E lyric fetch done in 31.2 s
fixture | stage D done in 76.9 s
fixture | shard fixture done in 190 s: 100 songs, 0 dropped, 100 previews deleted

$ python -m pytest -q tests/test_harvest_shard.py tests/test_harvest_resume.py
10 passed, 1 warning in 128.28s (0:02:08)

$ ls ml/data/previews | wc -l
0

catalog/v3/shard_fixture.manifest.json (downloaded): songs 100, dropped 0, describe_commit e4c1b1bd...,
files {features, clap, dropped, lyrics_error: catalog/v3/shard_fixture.lyrics_error.txt},
sha256 features 2c5f6b96, clap 8c738a57 (unchanged), dropped e3b0c442, lyrics_error 6260f720;
lyrics stopped_by "max_passes", error_after_retry 21; shard_fixture.lyrics_error.txt downloaded: 21 lines.
```

- With LRCLIB down, all 21 fixture rows without a lyric file end in error. That is the expected result (README 045f348), not a
  failure.
- **Noticed, for the reviewer.** LRCLIB now answers 502, not 503, and the breaker counts 503 only, so a 502 outage does not
  trip it. A 502 comes back at once (`_get` retries only 429 and 503), so each row is one call. Four passes over about 4,400
  rows at 0.25 s would be about 4,400 s, against B + C + D at about 3,380 s. Counting 502 with 503 is a one-line change;
  not made without a call.
