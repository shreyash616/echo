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
