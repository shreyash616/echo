# Echo: fix plan, round 2 (operative)

Status: written 2026-10-28 against commit `8fac3b2` on `main`, after the round 1 plan in
`FIX_PLAN.md` was executed and measured. This file is now the operative plan; `FIX_PLAN.md` is the
record of round 1 and its section 1 audit stays valid; `FEEL_ALGORITHM.md` stays the specification.
Where this file and either of them disagree, this file wins. Numbered references like 11.8 or 12.6
point into `FEEL_ALGORITHM.md`; "section N" means a section of this file.

Sections: 1 what round 1 achieved and what it did not, 2 the measurement tool that every change in
this file is evaluated with before a deploy, 3 rules, 4 fixes T1 to T10, 5 order and gates,
6 acceptance.

## 1. What round 1 achieved, and what it did not

**Measured on the live Space (`tests/feel_benchmark.py --split tune --base-url`), 2026-10-28:**

| Metric | Before round 1 | After round 1 | Round 1 target |
|---|---|---|---|
| TUNE recall@20 | 5 of 81 | 3 of 81 | at least 10 |
| TUNE recall@10 | 4 of 81 | 1 of 81 | at least 7 |
| Owner positives in the candidate pool | 2 of 11 | 6 of 11 | at least 6 |
| All benchmark positives in the pool | not measured | 20 of 81 | - |
| Negatives in top 10 | 0 | 0 | 0 |
| HOLDOUT recall@20 (reported once) | - | 2 of 14 | reported |

**Verified as done** (code and artifacts, not the log): union candidate pool with a real 17-axis
feel index (`feel_matrix.npy` 20000 x 17, `feel_index.faiss`), tempo scorer rewritten as specified
(`features.py:71-194`) with the octave fixture green in CI, lingua resolver with a `language_source`
field, one quantile bucket function used by both the index build and cold describe, `debug=1`
payload with axes, weights and pool source, placeholder fields and the KEY stat removed, Last.fm and
related-artist code removed, adapter manifest corrected, large binaries moved to the HF dataset and
out of git, CI green, stale tests and plots deleted, round 1 log corrections written.

**Claimed in the log but contradicted by the code or artifacts:**

| Claim | Reality | Evidence |
|---|---|---|
| "Re-described 20,000 rows from decoded previews" | `redescribe_catalog.py` recomputes tempo only when a preview file exists under `ml/data/previews`; that directory is absent and gitignored; every other row keeps its old values inside a bare `except: pass` | `ml/data/redescribe_catalog.py:143-169` |
| Theme reads the whole lyric in 40-word chunks | Chunking runs only for rows with a cached lyric file. All other rows get `theme_vec = anchor_cosines @ M_cal`, a projection of the CLAP audio anchors, and are stamped `has_lyrics = True`. 12 of 20,000 rows have `language_source = lyrics` | `redescribe_catalog.py:258-268` |
| Language from lyrics; 14 of 14 seeds correct | 19,192 of 20,000 labels come from the genre map. Boyfriend is `en` at 0.999 confidence, Tum Hi Ho is `tr`, Tomboy is `en`. The Indic override only fires below 0.8 confidence. Null share 2.2%, not 0.5% | `features.py:786-796`, live `/api/track` |
| Server-side p95 0.38 ms | The middleware writes seconds; the test strips the suffix "ms". Real server p95 over 20 calls is about 550 ms | `backend/main.py:46`, `tests/test_candidate_pool.py:80` |
| Catalogue above 150 BPM 7.9% | Artifacts say 3.46% | `HARVEST_MANIFEST.json` |
| `harvest_v2.py` fixed | Untouched: loads the theme bundle as `{name: prompt}` and sets `theme` at harvest | `ml/data/harvest_v2.py:684-687,852` |
| Absolute kick thresholds deleted | Still present, overridden only when cut points are passed | `features.py:289-296` |
| Workflow lists the nine tests | Candidate pool, theme sanity, describe parity and no-constant-fields are missing from `tests.yml` | `.github/workflows/tests.yml` |
| Weight search: "no TUNE regression" | Measured against a 1 of 81 baseline taken after the rebuild. The rebuild itself took recall from 5 to 1 | `ml/inference/WEIGHTS_REPORT.md` |
| Served feel weights are the plan's | Header and `build_index_v2.py:70-88` use intimacy 0.8 and crest 0.4; the plan said 1.2 and 0.6; `search_weights.py:62` tuned on 1.2 and 0.6 and never wrote back | `track_metadata.json` header |

**Why recall fell, from the offline replica (section 2), which matches the live scoring exactly:**

1. **The feel axes have no dynamic range.** Across the whole catalogue: `grit` std 0.038 (p5 0.260,
   p95 0.385), `intimacy` std 0.051, `darkness` std 0.082, `roughness` mean 0.005, `saturation`
   0.489 plus or minus 0.023, `sustain_ratio` 1.0 on 95% of rows. The similarities at
   `recommender.py:653,664,667,684` are `1 - |a - b|` on these raw values, so `s_grit` is 0.95 to
   1.00 for every candidate of every seed (std over the pool 0.01 to 0.02), `s_dark` and
   `s_intimacy` 0.85 to 1.00, while `s_clap`, `s_anchor`, `s_restraint` and `s_meaning` swing from
   0 or 0.5 to 1.0. A weighted sum is decided by the axes that move; no weight on grit can matter.
   This, not the weights, is why the search put 0.21 on language and 0.18 on anchors and why the
   feel pool's candidates lose to CLAP's.
2. **Three features are constants by construction.** `saturation` (`features.py:427`) is
   `0.5 * flat_frac + 0.5 * clip_frac * 10`; after LUFS normalisation to a 0.99 peak `clip_frac`
   is 0 and `flat_frac` is about 0.98, so it is 0.49 everywhere, and `grit` (`:442`) takes 0.45 of
   its value from it. `sustain_ratio` (`:496`) is `mean(f0[voiced] > 0)`, which is 1.0 whenever any
   frame is voiced, because f0 on voiced frames is positive by definition. `roughness` (`:329`) is
   mean spectral flatness, which is 0.00 to 0.02 for tonal music and so contributes nothing to
   `darkness` at weight 0.25.
3. **Restraint bucket equality zeroes the stillness world.** Waiting Around to Die bucketed
   "restrained", all three of its owner positives "saturated", so `s_restraint = 0.0` for each
   (`recommender.py` bucket rule; buckets from `features.py:655-661` quantiles on loudness range and
   crest factor, which measure mastering loudness, not restraint).
4. **`riff_flag` is a kill switch on harmony.** `recommender.py:718` sets `s_harm = 0.5` when either
   track has `riff_flag`; for Boyfriend the entire pool is 0.5, so the 0.11 weight is spent on a
   constant. `s_mode` is 1.0 for every candidate when the seed's mode is undefined (three of the
   four owner seeds).
5. **Language labels penalise the right answers.** Boyfriend `en`, its Punjabi positives `pa`
   (affinity 0.70) under the heaviest weight in the file (0.21).
6. **Retrieval is still the ceiling.** 61 of 81 TUNE positives never enter the pool. For the rock
   and folk seeds CLAP top-500 contributes one of them; the feel index carries the rest, on feel
   axes that themselves have the range problem above.

Reweighting cannot fix this: over 2,000 random weightings and coordinate ascent on the replica, zero
satisfy the round 1 per-seed checks, and the one knife-edge vector that does (language 0.43, mode
0.29) scores 0 of 14 on the other five TUNE seeds.

## 2. The measurement tool: `ml/experiments/offline_rerank/`

`score_lib.py` reimplements `_compute_pair_score` and the union pool; `report.py` ranks every
benchmark positive under any weights file, prints each positive's axes next to the top 3, the
per-axis standard deviation over the pool (an axis under 0.02 is inert for that seed), and, with
`--validate-live <space>`, the maximum deviation between its axes and the Space's `debug=1` axes.
On 2026-10-28 it reproduced the live pools exactly (sizes 1061 / 1030 / 1031 / 996) and every axis
with deviation 0.0000.

```
ECHO_ARTIFACT_DIR=<dir> python ml/experiments/offline_rerank/report.py --split tune \
    [--weights w.json] [--k-clap 500 --k-feel 600] [--validate-live https://shreyash616-echo-backend.hf.space]
```

`<dir>` holds `track_metadata.json`, `music_index.faiss`, `feel_matrix.npy`, `feel_index.faiss`,
`rerank_weights.json`, downloaded from the HF dataset root or written by `build_index_v2.py`.
Baseline output on 2026-10-28, served weights: positives 81, pooled 20, hits@10 0, hits@20 0,
MRR 0.0011; HOLDOUT: positives 14, pooled 5, hits@20 1.

**Simulated on 2026-10-28 with the current artifacts** (replica, pure score, no diversity pass):
T2 + T3.4 alone take TUNE from 0 to 1 hit of 81; adding T6 pools 26 of 81 positives (from 20);
adding T7's constrained search reaches 3 hits at both @10 and @20 (HOLDOUT 0 of 14, once). All three
section 6 per-seed checks still fail (Levee 1,104, No Surprises 83, Antidote 111). Rule 18 passes after
T2 on paper, but the rescaled axes are rescaled constants: `swing` is exactly 0.5 on every row,
`saturation` 0.49, `sustain_ratio` 1.0 on 95%. So the scoring steps are worth about 3 hits on the
current data; everything beyond that depends on T1 and T3.1 to T3.3 producing features that spread,
and on the feel index being rebuilt on them. Any step whose replica result is below these numbers has
regressed and is not deployed.

Every scoring change in `recommender.py` is mirrored in `score_lib.py` in the same commit, and
`report.py --validate-live` must print a deviation under 0.001 after the deploy. The replica is how
T2 to T7 are evaluated before anything is deployed.

## 3. Rules, in addition to `FIX_PLAN.md` section 2 and 12.0

17. A data pass counts only what it actually processed. The manifest records `previews_decoded`,
    `rows_described_from_audio`, `rows_with_lyric_file`; a row without a decoded preview is dropped
    from the catalogue, never carried over. `has_lyrics` is true only when a lyric file was read.
    No bare `except: pass` in any script under `ml/`.
18. Every per-axis similarity must move. `report.py` prints the axis standard deviation over each
    seed's pool; an axis under 0.05 on three or more TUNE seeds is a bug to fix, not a weight to
    lower.
19. Units are checked before a number is logged: a timing test asserts on a value whose unit it
    has read from the code that writes it.
20. The replica first, the Space second. A step's gate is measured with `report.py`; the deploy
    confirms it with `--validate-live` and the live benchmark; both outputs go in the log entry.

## 4. Fixes

**T1. The data pass, for real.**
- Problem: section 1, "re-described" claim.
- Change: `ml/data/redescribe_catalog.py` becomes a full pass: for every row, fetch the preview
  from its `previewUrl` (iTunes US; re-resolve through `resolve_itunes_preview` when the URL is
  dead) into `ml/data/previews/<id>.m4a`, decode, run `describe_audio` with the header cut points,
  compute the CLAP windows, read the lyric file (fetch from LRCLIB when absent), run T8's theme and
  T4's language. Rows whose preview cannot be fetched after one retry are dropped and listed in
  `ml/data/DROPPED.jsonl` with the reason. Write `HARVEST_MANIFEST.json` with the three counts of
  rule 17, the SHA256 of `catalog_features.jsonl`, and the distributions of `felt_bpm`,
  `kick_class`, `restraint_bucket`, `language`, `language_source`, `theme`. Then
  `build_index_v2.py`, then `deploy_hf.py`.
- Disk: fetch, describe and delete in batches of at most 5,000 previews; never keep the full set on
  disk (`SCALE_PLAN.md` section 12).
- Done when: `previews_decoded = rows_described_from_audio = catalogue rows >= 19,000`;
  `rows_with_lyric_file >= 0.9 x rows`; the SHA in the manifest matches the uploaded file; the
  seven octave fixtures still pass; `report.py` runs on the new artifacts.
- Artifacts: the manifest, `DROPPED.jsonl`, the rebuilt index files on the HF dataset.

**T2. Give every feel axis a scale: z-scored distances, not raw differences.**
- Problem: section 1, item 1.
- Change: `build_index_v2.py` writes `feature_std` into the metadata header for `darkness`, `grit`,
  `intimacy`, `pulse_rate`, `brightness`, `low_share`, `roughness`, `loudness_range`,
  `crest_factor`, `syllable_rate`, `sustain_ratio`, `pitch_range`, `vibrato`, `swing`, `lag_ms`,
  `hits_per_bar`, `backbeat`, `changes_per_bar`. In `_compute_pair_score` every `1 - |a - b|` and
  `1 - |a - b| / constant` term becomes `exp(-|a - b| / (tau * std_k))` with `tau = 1.0`, so a
  one-standard-deviation difference scores 0.37 and an identical value 1.0 on every axis alike.
  `s_restraint` becomes `exp(-(|dz_loudness_range| + |dz_crest_factor|) / 2)`; the bucket stays a
  display label only. `s_vocal` and `s_groove` keep their inner weights but use the z-scored
  terms. `score_lib.py` mirrored in the same commit.
- Done when: `report.py` shows, for every TUNE seed, no feel axis with pool std under 0.05 except
  those T3 lists as still pending; `--validate-live` deviation under 0.001 after deploy.
- Artifacts: the header, the report output in the log entry.

**T3. Fix the three constant features and the harmony kill switch.**
- Problem: section 1, items 2 and 4.
- Change, all in `backend/app/audio/features.py` unless stated:
  1. `saturation` (`:427`): replace with the fraction of 50 ms windows whose crest factor is below
     6 dB plus the fraction whose spectral flatness above 4 kHz exceeds 0.5, each 0 to 1, averaged.
     Done when catalogue std is at least 0.08 and the golden "saturated" fixture scores above the
     golden "restrained" fixture by at least 0.3.
  2. `sustain_ratio` (`:496`): the fraction of voiced frames that belong to a voiced run longer than
     300 ms. Done when catalogue std is at least 0.10 and it is not 1.0 on more than 20% of rows.
  3. `roughness` (`:329`): keep spectral flatness but rescale by the catalogue p95 at index time
     (`roughness_z` in the header), and drop its weight inside `darkness` to 0.10, moving 0.15 to
     `low_share`. Done when `darkness` std rises above 0.10.
  4. `riff_flag` (`recommender.py:718`): delete the override. `s_harm` is always the
     `changes_per_bar` and `loop_bars` similarity, with a `riff_flag` equality term at 0.2 inside
     it. `s_mode` when either mode is undefined becomes 0.7, not 1.0. Done when `report.py` shows
     `s_harm` and `s_mode` pool std above 0.05 on at least three TUNE seeds.
- Artifacts: `tests/test_features_golden.py` updated expectations with the fixture values printed,
  the catalogue std table in the manifest.

**T4. Language that is right for romanised lyrics.**
- Problem: section 1, item 5 and the audit row.
- Change in `features.py:701-806`: lingua's answer is accepted only when (a) the lyric script is
  non-Latin, or (b) the detected language is one of `es, fr, pt, it, de, nl, sv, pl, tr, id, tl`
  with confidence at least 0.9 and at least 60 words of lyrics, or (c) the detected language is
  `en` and at least 55% of the lyric tokens are in a bundled 10,000-word English list
  (`backend/app/audio/english_words.txt`, from the public-domain word frequency lists). Otherwise
  the row falls through to script, genre map, artist vote, then None. `language_source` is added to
  the `/api/track` and `/api/recommendations` responses.
- Done when: all 14 benchmark seeds resolve to their declared language on the live Space; the
  manifest shows `language_source = lyrics` on at least 60% of rows that have a lyric file;
  `language = null` at most 10%.
- Artifacts: the histogram in the manifest, the 14-seed table in the log.

**T5. Kick class from the kick, not from the hit count.**
- Problem: `features.py:650` labels `four_on_floor` when `hits_per_bar >= p66`, so fingerpicked
  folk and No Surprises are four-on-the-floor.
- Change in `quantize_kick_and_restraint`: `four_on_floor` when `beat_low_ratio >= p75` and
  `off_beat_ratio <= p50`; `syncopated_808` when `off_beat_ratio >= p66`; `sparse_backbeat` when
  `hits_per_bar <= p33`; else `other`. Delete the absolute thresholds at `features.py:289-296`.
- Done when: no class holds more than 50% of rows; the golden four-on-the-floor fixture classifies
  as `four_on_floor` and the fingerpicked fixture does not; `tests/test_describe_parity.py` 18 of
  20 or better.

**T6. Pool sizes and the latency test.**
- Change: `config.py` `candidates_clap = 300`, `candidates_feel = 1500`. `tests/test_candidate_pool.py:80`
  parses `X-Process-Time` as seconds. The feel weights in `build_index_v2.py:70-88` are set to the
  values the search in T7 writes back (one source, the header).
- Done when: at least 8 of 11 owner positives and at least 35 of 81 TUNE positives are in the pool
  (`report.py` "pooled"); server-side p95 under 0.300 s over 20 in-catalogue calls.

**T7. Weight search on the replica, with range guards.**
- Change: `search_weights.py` evaluates on `score_lib` (no HTTP), objective
  `0.5 x hits@20 + 0.3 x MRR + 0.2 x hits@10` over all TUNE positives with unpooled ones counted
  as misses, minus 0.5 per negative in the top 10; constraints: no axis above 0.25,
  `s_clap + s_anchor <= 0.25`, `s_language <= 0.08`, `s_mode <= 0.05`; 20 random restarts in the
  simplex plus coordinate ascent; every evaluation logged; the feel-pool weights searched in the
  same run and written back to the header through `build_index_v2.py`. HOLDOUT evaluated once at
  the end.
- Done when: the log has at least 500 evaluations; `WEIGHTS_REPORT.md` shows the replica's
  before and after for TUNE and the single HOLDOUT number; the chosen weights beat the T2 to T6
  state on the replica and the live benchmark agrees within one hit.

**T8. Theme from lyrics only.**
- Change: remove the anchor projection fallback at `redescribe_catalog.py:258-268`; `theme_vec`
  exists only for rows with a lyric file; `has_lyrics` follows rule 17; `theme = null` otherwise.
  Fix `harvest_v2.py:684-687` to read the bundle through the same reader as `build_index_v2.py`
  and stop setting `theme` at harvest. `theme_prompts.json` stats recomputed on the new catalogue.
- Done when: largest theme share at most 30% of non-null rows; at least 6 themes at 3% or more;
  `tests/test_theme_sanity.py` 5 of 7 or better on the live Space; no row has `has_lyrics = true`
  without a lyric file.

**T9. Tests and workflow.**
- Change: add candidate pool, theme sanity, describe parity and no-constant-fields to `tests.yml`;
  `test_no_constant_fields.py` checks `/api/track` as well as `/api/recommendations`;
  `tests/test_offline_replica.py` runs `report.py --validate-live` and asserts deviation under
  0.001 (skips when artifacts are absent); `tests/test_axis_range.py` asserts rule 18 on the
  TUNE seeds.
- Done when: a green run on main with all of them listed, URL in the log.

**T10. Log corrections for round 1.**
- Change: dated correction entries in `FEEL_ALGORITHM.md` for the nine contradicted claims in
  section 1 (do not edit the originals). Rule 12 applies to every new entry.

## 5. Order and gates

1. T9 first (so the replica validation and range tests exist), then T10.
2. T3 (features) + T5 (kick) + T4 (language) + T8 (theme) in one change set, then T1 (the full
   data pass, which is what makes them real), one index build, one deploy. Gate: T1, T3, T4, T5,
   T8 "done when" on the artifacts; `report.py --validate-live` deviation under 0.001.
3. T2 (z-scored similarities). Gate: rule 18 on the replica; validate-live after deploy.
4. T6 (pools, latency). Gate: pooled counts on the replica, p95 on the Space.
5. T7 (weights). Gate: replica before and after, live benchmark within one hit.
6. Section 6 measured on the live Space, HOLDOUT once.

A step that fails its gate is not deployed. Weights are not touched before step 5.

## 6. Acceptance (live Space, measured, pasted into the log with the commands)

| Check | Today | Required |
|---|---|---|
| TUNE positives in the candidate pool (`report.py` pooled) | 20 of 81 | at least 35 of 81 |
| Owner positives in the pool | 6 of 11 | at least 8 of 11 |
| TUNE recall@20 (`feel_benchmark.py --split tune --base-url`) | 3 of 81 | at least 10 of 81 |
| TUNE recall@10 | 1 of 81 | at least 7 of 81 |
| Negatives in top 10 | 0 | 0 |
| `test_live_overlap.py` | 0 of 20 | still passes |
| Simple Man top 20 | none | contains When the Levee Breaks or Stairway to Heaven |
| Waiting Around to Die top 20 | none | contains No Surprises, Me Gustas Tu or Black |
| Boyfriend top 10 | none | at least 2 of Antidote, For a Reason, Baller, Roll With Me |
| Bang Data top 10 | passes | every `felt_bpm` within 0.5x to 2x of the seed, mean grit at least the seed's minus 0.1 |
| Feel axes with pool std under 0.05 on 3 or more TUNE seeds | grit, dark, intimacy, pulse, vocal, mode | none |
| Rows described from decoded audio | unknown (previews absent) | all catalogue rows |
| `language_source = lyrics` share of rows with a lyric file | 0.06% | at least 60% |
| 14 benchmark seeds on their declared language | 11 of 14 | 14 of 14 |
| Replica vs live axis deviation | 0.0000 | under 0.001 after every deploy |
| Server-side p95, in-catalogue seed | about 0.55 s | under 0.300 s |
| CI on main | green, four tests missing | green with the section T9 list |
| HOLDOUT recall@20 | 2 of 14 | reported once at the end |

Rooster, Tidal Wave and Black (Danger Mouse) stay outside the required set for the reason given in
`FIX_PLAN.md` section 1; if T2 and T3 bring them into the top 20 on the replica, say so in the log,
and do not tune toward them.
