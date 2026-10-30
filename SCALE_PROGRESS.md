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
