# Echo: fix plan after the second audit (round 1, executed; superseded by `FIX_PLAN_2.md`)

Round 1 was executed on 2026-10-28 and audited; the audit and the round 2 plan are in
`FIX_PLAN_2.md`, which is now the operative file. This file stays as the record of round 1.

Status: written 2026-10-28 against commit `ed7af17` on `main`, from measurements on the live Space
(`https://shreyash616-echo-backend.hf.space`) and on the published artifacts. This file is the
operative plan. `FEEL_ALGORITHM.md` stays the specification (sections 11 and 12 say what each
descriptor, score and artifact is); this file says what is wrong with the implementation of it, what
to fix, what to delete, in what order, and what counts as done. Where this file and
`FEEL_ALGORITHM.md` disagree, this file wins. `VISION_ALIGNMENT.md` is historical.

Sections: 1 what the audit found, 2 rules, 3 fixes S1 to S10, 4 order and gates, 5 acceptance,
6 cleanup C1 to C11.
Numbered references of the form 11.8, 12.0, 12.4 or 12.7 point into `FEEL_ALGORITHM.md`; "section N"
without a dot means a section of this file.

Everything in section 1 was measured on commit `ed7af17`, on the live Space, and on the LFS artifacts
(`catalog_features.jsonl`, `clap.npy`, `adapter.npz`). Nothing here is taken from the progress log.
The implementing agent executes section 3 in the order of section 4, does the cleanup in section 6 at the
steps named there, and must satisfy section 5. Rules 1 to 11 in `FEEL_ALGORITHM.md` section 12.0 still
bind; section 2 adds five more.

## 1. What the audit found

**Progress-log claims against the code and artifacts.**

| Claim in the progress log | What is actually there | Evidence |
|---|---|---|
| R7: adapter trained, gate passed (3.70% to 4.94%) | Serving never loads `adapter.npz`; `grep -rn adapter backend/app` is empty. No code computes a before/after recall; `train_adapter.py:223-230` writes only pairs, playlists, epochs, loss, device, shape. "Before 3.70%" contradicts R6's own 4.94% baseline. | `ml/training/train_adapter.py:223-230`, `backend/app/services/recommender.py` |
| R7: PMI affinity over 62 pairs | 60 of 62 values equal the hand table in `FEEL_ALGORITHM.md` section 5; only es:en and ru:en moved. 18,352 of 20,000 rows (91.8%) have `language = null`, so `s_language` returns the UNKNOWN constant 0.75 on almost every pair and the cross-language budget never fires. | `ml/inference/language_affinity.json`, `language_affinity.py:37,72-73` |
| R7: weights optimised by coordinate ascent | One weight moved one step (s_anchor 0.10 to 0.08) then renormalised. The loop is first-improvement greedy with step 0.02 and logs only improvements to stderr; no evaluation log exists. | `ml/training/search_weights.py:122-151` |
| R3: harvest per 12.4 | `deezer_id` is null on all 20,000 rows (`harvest_v2.py:467` hardcodes None; `resolve_itunes_preview` is never called with one). 14,876 rows come from iTunes artist expansion, 5,124 from playlist membership. | `ml/data/harvest_v2.py:343,467,838` |
| R2: in-catalogue p95 24.62 ms | Live p95 855 ms; `tests/test_live_api.py` fails against the Space. The test measures HTTP round trip from India (health RTT 650 ms), so the number was never comparable. | `tests/test_live_api.py:64-87` |
| R8: CI green | Every run of `tests.yml` on main is red: no ffmpeg step, the 10 golden tests fail, later steps never run. | `.github/workflows/tests.yml` |
| R9: client parity | `key` is now the canonical `artist:title` string (`routes.py:57-61`), and both clients render it as the musical KEY stat. | `SongCard.tsx:63-64`, `static/index.html:1582-1584` |
| 12.3: meaning axis | Every one of 40 live results across the four seeds shows `theme = resignation`. Catalogue raw argmax: resignation 14,391, grief 5,524, escape 84, devotion 1. | `harvest_v2.py:827-830` |

What is genuinely done and must not be redone: official LAION CLAP with 10 s consecutive windows and
load-time assertions (`recommender.py:120-135,301,320-354`); 20,000 rows with descriptors computed
from decoded audio (`harvest_v2.py:800`); `clap.npy` (20000, 512) unit norm, zero zero-rows, sample
mean pairwise cosine 0.50; quantile cut points in the metadata header; key routing and in-flight
dedup; `test_live_overlap.py` passing 0/20 live.

**Live quality on the four TUNE seeds (fetched 2026-10-28).**

| Seed | Live top 10 | Owner positives in top 10 |
|---|---|---|
| Simple Man | Whitesnake, Scorpions x2, Deep Purple x2, CCR, Bowie, Kings of Leon, a second Skynyrd. 7 of 10 tagged "Raw and driving". | 0 of 3 |
| Waiting Around to Die | Tyler Childers x2, Johnny Cash x2, Nick Drake, Iron and Wine, Doc Watson. All "Folk story". | 0 of 3 |
| Bang Data | Maluma x2, J Balvin x2, Bad Bunny, Sidhu Moose Wala x2, Manu Chao. | 0 of 1 |
| Boyfriend | Antidote at 1, Admirin' You at 9. | 2 of 4 |

TUNE recall@10 = 4 of 81, recall@20 = 5 of 81. The engine is a scene-similarity engine with the feel
axes drowned out. Three root causes, each measured:

**Root cause A: the candidate pool is CLAP-only, so the positives are unreachable.**
`recommender.py:706-707` builds the pool as FAISS top-500 over raw CLAP (plus 100 from the side
index). The rerank only reorders those 500. CLAP rank of the owner positives among all 20,000:

| Seed | Positive | CLAP rank | Rank in the DSP axes alone (standardised, equal weights) |
|---|---|---|---|
| Simple Man | When the Levee Breaks | 2,439 | 2,893 (568 once the seed's tempo octave is corrected) |
| Simple Man | Stairway to Heaven | 4,067 | 13,159 |
| Simple Man | Rooster | 4,535 | 16,469 |
| Waiting Around to Die | No Surprises | 954 | 311 |
| Waiting Around to Die | Me Gustas Tu | 4,417 | 188 |
| Waiting Around to Die | Black (Danger Mouse) | 5,997 | 693 |
| Bang Data | Tidal Wave | 9,490 | 8,761 |
| Boyfriend | Antidote | 5 | 257 |
| Boyfriend | For a Reason | 3 | 6 |

Nothing with a CLAP rank above 500 can ever appear, whatever the weights. In the stillness world the
DSP axes already rank the positives three to twenty times better than CLAP does; they never get the
chance.

**Root cause B: the felt-tempo octave is wrong on a large part of the catalogue.**
Stored `felt_bpm`: Simple Man 123 (the song is felt at about 61; its own posterior is
`{60: 0.36, 123: 0.64}`), Waiting Around to Die 184, Black 172, No Surprises 152, Me Gustas Tu 173.
Rooster 71, Levee 72 and Stairway 84 are right. Across the catalogue 28.3% of rows are above 150 BPM,
median 123, and the posterior's top probability is below 0.7 on 85.5% of rows. The scoring at
`features.py:166-173` is
`0.30 clarity + 0.15 low_clarity + 0.25 backbeat + 0.15 density + 0.10 bar_flux + 0.05 prior`
with the prior centred at 110 BPM at weight 0.05 and the density term (`features.py:144-150`) giving
1.0 for 0.7 to 3.2 onsets per beat, which rewards the tatum on fingerpicked or busy material. There is
no term that looks at the backbeat period, which is what a listener taps to.

Measured on the seven local iTunes previews (librosa raw tempo T; autocorrelation of the mean-removed
full-band onset envelope at lags of 1, 2 and 4 raw beats):

| Preview | Raw T | acf 1 beat | acf 2 beats | acf 4 beats | Felt (truth) |
|---|---|---|---|---|---|
| Simple Man | 123.0 | 0.37 | 0.13 | 0.42 | 61 |
| Rooster | 136.0 | 0.21 | 0.22 | 0.43 | 68 |
| When the Levee Breaks | 143.6 | 0.51 | 0.44 | 0.39 | 72 |
| Start Me Up | 123.0 | 0.47 | 0.46 | 0.39 | 123 |
| Boyfriend | 95.7 | 0.52 | 0.54 | 0.47 | 96 |
| Waiting Around to Die | 89.1 | 0.31 | 0.25 | 0.21 | 89 |
| Peaches | 99.4 | 0.10 | 0.22 | 0.21 | 99 |

The 4-beat lag winning is the half-time signature (Simple Man, Rooster); the 1-beat lag winning at
123 is a true 123 (Start Me Up). Levee is not separated by this term alone (its mid-band backbeat
term already halves it), and Waiting Around to Die is drumless, so the term must be one vote among
several, not a rule. Kick-band onset rate and onsets per second do not separate Simple Man from
Start Me Up (3.64 vs 3.84 per second); do not use them for the octave.

**Root cause C: three axes are dead and one is mislabelled.**
- Theme: `harvest_v2.py:827` encodes the whole lyric as one string through MiniLM, whose default
  `max_seq_length` is 128 tokens, so only the first eight or so lines are read; `theme_vec` is the raw
  cosine to each prompt and the displayed `theme` is the raw argmax (`harvest_v2.py:830`), which the
  highest-mean prompt wins (resignation mean 0.378 vs 0.143 to 0.32 for the rest). `build_index_v2.py:222-231`
  standardises `theme_vec` but never rewrites `theme`. Also `harvest_v2.py:684` loads
  `theme_prompts.json` as `{name: prompt}` while `build_index_v2.py:139-146` rewrites that file as a
  bundle `{prompts, means, stds, keys}`, so a re-run of harvest would embed the words "prompts",
  "means", "stds", "keys".
- Language: `features.py:613-650` uses only title, artist and album script plus a genre map; Latin
  script with no mapped genre gives None, which is 91.8% of rows. Lyrics exist for 19,993 rows and are
  cached on disk (`harvest_v2.py:761,772`) and are never used for language.
- Kick class: the absolute thresholds at `features.py:268-275` put 18,644 of 20,000 rows (93.2%) in
  `syncopated_808`. `build_index_v2.py:197-213` then overwrites catalogue rows with a quantile rule,
  but cold seeds described at request time still get the absolute rule, and `recommender.py:521`
  compares the two labelings as if they were one.
- The response still carries constants from the deleted synthesiser: `valence 0.5`, `stereoWidth 0.5`,
  `phaseCorrelation 0.95`, `crestFactor 10.0`, `vocalPresence 0.5`, `vocalPattern "Dynamic Melodic
  Phrasing"`, `lyricalIntent`, `spatialStaging`, `instrumentPace`, `instruments []`. Both clients render
  some of them.

**What reweighting can and cannot reach.** With all 22 DSP axes plus CLAP cosine, the seed's tempo
corrected, and 400 random non-negative weightings over the full 20,000 rows, the best attainable
ranks were: Levee 163, Stairway 5,335, Rooster 8,730; No Surprises 27, Me Gustas Tu 378, Black 3,875;
Tidal Wave 7,230; For a Reason 1, Antidote 11. The adapter (`adapter.npz`, 48,769 playlist pairs)
applied to CLAP moves Rooster to 1,883 and Stairway to 982 but pushes No Surprises to 2,947, Me
Gustas Tu to 8,054, Black to 12,077 and Tidal Wave to 17,679: it learns scene, as 11.8 warned, and
must stay out of serving. Rooster, Tidal Wave and Black share feel with their seeds that the current
descriptors do not measure (Simple Man's preview is the bright chorus; darkness 0.20 vs Rooster 0.39,
mode polarity +0.25 vs -0.41, lag 3.8 standard deviations apart). The acceptance in section 5 therefore
names the pairs the descriptors can connect, and no fix may be tuned on any title (rule 3, rule 14).

## 2. Rules, in addition to rules 1 to 11 of FEEL_ALGORITHM.md section 12.0

12. A progress-log entry contains the exact command that produced each number and the path of the
    artifact it wrote. A number without a command is deleted from the log, not kept.
13. "Gate passed" is written only by code: the script that evaluates the gate writes the before and
    after values and the verdict into the manifest. A manifest field no script writes is a lie.
14. No fix references a benchmark title, artist or language anywhere in `backend/` or `ml/`. Truth
    fixtures for tests live under `tests/fixtures/` and are whitelisted by path in
    `tests/test_no_benchmark_leakage.py`; the leakage test keeps scanning everything else.
15. Every step in section 3 ends with a deploy, then `tests/feel_benchmark.py --split tune --base-url <space>`
    and `tests/test_live_overlap.py --base-url <space>`, with both outputs pasted into the log entry.
16. Timing claims are server-side: `X-Process-Time` set by middleware, never a client round trip.

## 3. Fixes

Each fix states the problem, the exact change, and what "done" means. Artifacts are listed so the
next audit can check them without trusting the log.

**S1. Candidate pool: union of a CLAP pool and a feel-space pool.**
- Problem: section 1 root cause A.
- Change: in `build_index_v2.py`, build `ml/inference/feel_matrix.npy` (20000 x 17, float32) as the
  weighted standardised feel vector of each row, and `ml/inference/feel_index.faiss`
  (`IndexFlatL2`) over it. Axes, in this order, each standardised with the catalogue mean and std
  stored in the metadata header, then multiplied by the axis weight:
  `log2(felt_bpm)` 1.5, `pulse_rate` 0.8, `hits_per_bar` 0.8, `darkness` 1.0, `brightness` 0.6,
  `low_share` 0.5, `grit` 0.8, `roughness` 0.8, `intimacy` 1.2, `reverb_tail` 0.5, `loudness_range` 0.6,
  `crest_factor` 0.6, `lag_ms` 0.5, `swing` 0.4, `syllable_rate` 0.6, `sustain_ratio` 0.6,
  `mode_polarity` 0.5 (0 when `mode_undefined`). These are starting weights; S7 tunes them.
  In `recommender.py` replace the single search at lines 706-707 with: CLAP top `CANDIDATES_CLAP`
  (500) plus feel-space top `CANDIDATES_FEEL` (500) plus the side index as today; union by catalogue
  index; rerank the union with `_compute_pair_score` unchanged. Both constants go in `config.py`.
  The seed's feel vector comes from its stored row, or from `describe` for a cold seed using the
  header mean and std. Record `pool_source` ("clap", "feel", "both") per candidate for S6.
- Done when: `tests/test_candidate_pool.py --base-url <space>` (new; measurement only, may name
  benchmark titles) reports, for each TUNE seed, which benchmark positives are in the union pool, and
  at least 6 of the 11 owner positives are in it across the four seeds (today 2: Antidote, For a
  Reason). Server-side p95 of `/api/recommendations` for an in-catalogue seed stays under 300 ms.
- Artifacts: `feel_matrix.npy`, `feel_index.faiss`, header fields `feel_axes`, `feel_mean`, `feel_std`,
  `feel_weights`, the test's printed table in the log entry.

**S2. Felt tempo: add the backbeat-period vote and stop rewarding the tatum.**
- Problem: section 1 root cause B.
- Change, all in `backend/app/audio/features.py` around lines 67-182:
  1. Candidates: keep `[tatum, tatum/2, tatum/4]`; add `tatum*2` only when `tatum < 90` (today 120).
  2. New term `backbeat_acf_c`: autocorrelation of the mean-removed, lag-0-normalised full-band onset
     envelope, evaluated as the max within plus or minus 3 frames of the lag equal to 2 beats of `c`
     (hop 512 at 22,050 Hz). Divide by the max over the candidates so the term is 0 to 1.
  3. `density_c` becomes: 1.0 for 1.0 to 2.5 onsets per beat at `c`, 0.6 for 2.5 to 4.0, 0.2 above 4.0
     or below 0.5.
  4. `prior_c = exp(-0.5 * (log2(c / 100) / 0.5) ** 2)`.
  5. Candidates above 150 BPM are multiplied by 0.5 unless both `clarity_c >= 0.9` and `backbeat_acf_c`
     is the maximum over the candidates.
  6. Weights: `0.25 clarity + 0.10 low_clarity + 0.20 backbeat + 0.20 backbeat_acf + 0.10 density + 0.15 prior`.
     Drop `bar_flux`. Softmax temperature stays 0.08. Store the posterior as today.
  Then re-describe all 20,000 rows from the cached previews (the harvest caches previews under its
  data dir; if a preview is missing, re-fetch from iTunes US) and rebuild the index. S3, S4 and S5
  change the same pass; do them together, one re-describe, one rebuild.
- Done when: (a) `tests/fixtures/octave_truth.json` (new; the seven previews above as title, artist,
  iTunes US lookup, truth range) passes: Simple Man 55 to 66, Rooster 62 to 76, Levee 66 to 78,
  Start Me Up 115 to 130, Boyfriend 90 to 102, Waiting Around to Die 82 to 96, Peaches 93 to 105;
  (b) all golden fixtures in `tests/test_features_golden.py` still pass; (c) on the rebuilt catalogue
  the share of rows above 150 BPM is at most 10% (today 28.3%), below 60 BPM at most 5%, the median
  is between 95 and 115, and the posterior's top probability is at least 0.7 on at least 50% of rows
  (today 14.5%); (d) the implementation reproduces the acf table in section 1 within 0.05 on the seven
  previews, printed by the test.
- Artifacts: the fixture file, the test output, `HARVEST_MANIFEST.json` with the new distribution
  numbers, rebuilt `track_metadata.json` and `music_index.faiss`.

**S3. Theme: read the whole lyric, standardise before choosing, show nothing when unsure.**
- Problem: section 1 root cause C, theme.
- Change: in `harvest_v2.py` (a) load `theme_prompts.json` through the same bundle-aware reader
  `build_index_v2.py:123-126` uses; (b) split the lyric into chunks of at most 40 words on line
  boundaries, encode every chunk, average, L2-normalise, then take cosines to the ten prompts
  (`theme_vec` raw stays the stored form); (c) do not set `theme` at harvest. In `build_index_v2.py`
  after standardisation set `theme = keys[argmax(z)]` when `max(z) >= 0.5`, else `null`. Clients show
  no theme badge for null. `theme_prompts.json` keeps the ten prompts; the standardisation stats are
  recomputed on the rebuilt catalogue.
- Done when: no single theme is the argmax on more than 30% of rows and at least 6 of the 10 themes
  each hold at least 3%; `tests/test_theme_sanity.py --base-url <space>` (new; measurement only)
  reports the displayed theme for these catalogue tracks and at least 5 of 7 match: Boyfriend
  swagger, Mask Off swagger, Get Lucky celebration, Tum Hi Ho devotion or longing, Hurt grief or
  resignation, Rooster defiance or grief, Simple Man devotion or resignation. Live results for the
  four seeds no longer all show one theme.
- Artifacts: the test output, the theme histogram in `HARVEST_MANIFEST.json`.

**S4. Language from the lyrics, with fallbacks, and a source field.**
- Problem: section 1 root cause C, language.
- Change: add `language_from_lyrics(text)` in `features.py` using `lingua-language-detector`
  (Apache-2.0, pure Python, CPU) restricted to the languages present in
  `language_affinity.py`; accept when confidence is at least 0.6. Order of resolution per row:
  lyrics (if `has_lyrics`), then the existing script detection, then the genre map, then a majority
  vote over the same artist's already-resolved rows, else None. Store `language`, `language_conf`
  and `language_source` in {"lyrics", "script", "genre", "artist_vote", "none"}. Romanised Punjabi
  and Hindi lyrics are the known weak spot: when lingua returns "en" with confidence under 0.8 and the
  genre or script path says an Indic language, the Indic answer wins.
- Done when: `language = null` on at most 10% of rows (today 91.8%); the 14 benchmark seeds in
  `feel_benchmark_v2.json` resolve to their declared language (measurement test, may name titles);
  the log entry prints the language histogram and the fraction of scored pairs where
  `s_language != 0.75` (must exceed 80%).
- Artifacts: the histogram in `HARVEST_MANIFEST.json`, the test output.

**S5. One labelling for kick class and restraint.**
- Problem: section 1 root cause C, kick class.
- Change: the quantile rule at `build_index_v2.py:197-213` is the only rule. Move it into a function
  in `features.py` that takes the header cut points, call it from `build_index_v2.py` for catalogue
  rows and from `recommender.describe` for cold seeds (the header is loaded at startup). Delete the
  absolute thresholds at `features.py:268-275` and `547-552`; keep the raw inputs (`beat_low_ratio`,
  `off_beat_ratio`, `low_share`, `hits_per_bar`, `loudness_range`, `crest_factor`) on the row.
- Done when: `tests/test_describe_parity.py` (new) describes 20 random catalogue previews through
  the cold path and compares `kick_class`, `restraint_bucket` and `felt_bpm` (within one octave bin)
  to the stored row: at least 18 of 20 equal on each.
- Artifacts: the test output.

**S6. Debug output, placeholder fields, and the KEY stat.**
- Change: `GET /api/recommendations/{id}?key=...&debug=1` returns, per result, `axes` (every
  `s_*` value from `_compute_pair_score`), `weights` (the effective weights after redistribution)
  and `pool_source`. Without `debug=1` the response shape is unchanged. `matchReasons` stays.
  Remove from the response every field the pipeline does not measure: `valence`, `stereoWidth`,
  `phaseCorrelation`, `crestFactor` (the constant one; the measured `crest_factor` stays),
  `vocalPresence`, `vocalPattern`, `vocalComplementation`, `lyricalIntent`, `spatialStaging`,
  `instrumentPace`, `instruments`, `instrumentOrder`, `bpm` (0 everywhere; `felt_bpm` is the value),
  `camelotKey`. Remove the KEY stat at `SongCard.tsx:63-64` and `static/index.html:1582-1584`, and the
  duplicate `key` declaration in `api.ts` (keep one, optional). Clients render only fields present.
- Done when: over 50 random in-catalogue tracks no response field has the same value on all 50
  (test `tests/test_no_constant_fields.py --base-url <space>`); the debug payload for Simple Man
  lists an `s_tempo` for each result; both clients build and show no "KEY" stat.
- Artifacts: the test output, a screenshot or DOM dump of the hero card from each client.

**S7. Weight search that leaves a trail, run only after S1 to S5.**
- Change: `search_weights.py` writes every evaluation (weights, recall@10, recall@20, MRR,
  negatives in top 10, objective) to `ml/training/weight_search_log.jsonl`; runs 10 random restarts
  in the simplex plus coordinate ascent from each, step 0.02, keeps the best; writes
  `ml/inference/WEIGHTS_REPORT.md` with the best weights, the trajectory length, and an ablation table
  (objective with each axis zeroed, one row per axis). The feel-pool weights of S1 are tuned in the
  same run as a second block. HOLDOUT is evaluated once at the end and reported; it is never used to
  choose.
- Done when: the log has at least 200 evaluations; the report exists; the chosen weights do not
  reduce TUNE recall@20 below the pre-search value.
- Artifacts: the log, the report, `rerank_weights.json`.

**S8. Adapter: out of the plan.**
- Change: nothing is wired. `adapter.npz` and `ADAPTER_MANIFEST.json` stay as a record. Remove the
  "TUNE before/after" promise from the `train_adapter.py` docstring. The progress log gets a
  corrected R7 entry stating that the adapter was never served, that no gate was computed, and the
  measured ranks from section 1. The `deezer_id` requirement of 12.4 is withdrawn with it, since its only
  consumer was the adapter's pair join.

**S9. Tests and CI first, so every later step can be measured.**
- Change: `tests/feel_benchmark.py:181` reads `resp.json()` as the list it is. `tests.yml` installs
  ffmpeg (`sudo apt-get update && sudo apt-get install -y ffmpeg`) before the golden tests.
  `tests/test_live_api.py` reads `X-Process-Time` (new middleware in `backend/app/main.py`) and
  asserts on that, not on the round trip. Add the new tests from S1 to S6 to the workflow. Whitelist
  `tests/fixtures/` by path in `test_no_benchmark_leakage.py`.
- Done when: a green run of `tests.yml` on main, with its URL in the log entry; `feel_benchmark.py
  --base-url` runs to completion against the Space and prints TUNE recall@10, recall@20, MRR,
  negatives.

**S10. Correct the progress log.**
- Change: add dated correction entries (do not edit the old ones) for R2 (timing), R3 and R4
  (`deezer_id`, language coverage), R7 (adapter, affinity, weight search) and R8 (CI), each stating
  what section 1 measured. From then on rule 12 applies to every entry.

## 4. Order and gates

1. S9 (tests runnable, CI green on the current code).
2. S2 + S3 + S4 + S5 together: one change set in `features.py`, `harvest_v2.py` and
   `build_index_v2.py`; one re-describe of the 20,000 cached previews; one index rebuild; one deploy.
   Gate: all four "done when" blocks, plus rule 15.
3. S1 (pool). Gate: the pool test, server-side p95, rule 15.
4. S6 (debug, fields, KEY). Gate: the constant-fields test, both clients.
5. S7 (weights). Gate: the report, no TUNE regression.
6. S8 and S10 (log corrections).
7. section 5 measured on the live Space, with the HOLDOUT split reported once.

A step that fails its gate is not deployed and the failure is logged with the command output.
Weights are never touched before step 5, because weights tuned on an unreachable pool or a wrong
octave are meaningless.

## 5. Acceptance (live Space, measured, pasted into the progress log)

| Check | Today | Required |
|---|---|---|
| TUNE recall@20 (`feel_benchmark.py --split tune --base-url`) | 5 of 81 (6.2%) | at least 10 of 81 (12.3%) |
| TUNE recall@10 | 4 of 81 (4.9%) | at least 7 of 81 (8.6%) |
| Negatives in top 10, all TUNE seeds | 0 | 0 |
| `test_live_overlap.py` | 0/20 | still passes |
| Simple Man top 20 | none | contains When the Levee Breaks or Stairway to Heaven |
| Waiting Around to Die top 20 | none | contains No Surprises, Me Gustas Tu or Black |
| Boyfriend top 10 | Antidote, Admirin' You | at least 2 of Antidote, For a Reason, Baller, Roll With Me |
| Bang Data top 10 | reggaeton | every result's `felt_bpm` within 0.5x to 2x of the seed's, mean `grit` at least the seed's minus 0.1 (Tidal Wave itself is not required; section 1 shows the descriptors cannot reach it) |
| Catalogue `felt_bpm` above 150 | 28.3% | at most 10% |
| Catalogue `language = null` | 91.8% | at most 10% |
| Largest single theme share | 72% | at most 30% |
| Response fields constant across 50 tracks | 10 fields | 0 |
| CI on main | red | green, URL in the log |
| HOLDOUT recall@20 | not reported | reported once, after everything else, not used for any choice |

Rooster, Tidal Wave and Black are not in this table because section 1 measured that no weighting of the
current descriptors reaches them. Reaching them needs new descriptors (tension slope and peak-to-floor
ratio inside the window, vocal strain, timbral warmth), which is a separate, measured experiment
after section 5 passes, not a reason to tune on their names.

## 6. Cleanup

Everything below is dead, stale or misleading at `ed7af17`. Each item says what to delete or
change and why. Git history keeps the old versions; nothing is archived in the tree. Do the cleanup
in the step named in section 4 so each deletion is covered by a passing run.

**C1. Dead services and config.**
- `backend/app/services/lastfm.py`: no caller in `backend/app`. Delete it.
- `backend/app/services/deezer.py`: `get_related_artists` and `get_genre_and_language_candidates`
  have no callers. Delete the functions; keep `search_track`, `get_track`, `get_album_genres`,
  `fetch_preview_audio`.
- `backend/app/config.py:11` `lastfm_api_key`: delete, and remove the key from the Space secrets and
  from `SETUP.md:49`, which still describes a "Genre & Language Gate" built on Last.fm tags and the
  Deezer related-artist graph. Rewrite that paragraph to describe the section 5 pipeline of
  `FEEL_ALGORITHM.md` (no gates, soft language prior).
- Do this in step 4 (with S6), then run `tests/test_no_gates.py`.

**C2. Placeholder fields and their defaults.**
- `backend/app/models/schemas.py:22-36` declares constants the pipeline never measures:
  `stereoWidth 0.5`, `phaseCorrelation 0.95`, `crestFactor 10.0`, `spatialStaging`, `lyricalIntent`,
  `valence 0.5`, `instruments`, `instrumentOrder`, `instrumentPace`, `vocalPresence 0.5`,
  `vocalPattern`, `vocalComplementation`; `routes.py:81` onwards copies them into every response.
  Delete the fields (this is the schema half of S6). `energy` stays only if it is computed from
  audio; if it is a constant too, it goes. `bpm` goes; `felt_bpm` is the value.
- Both clients: delete the renderers for those fields (the hero card stats, the vocal and
  instrument rows if any remain) so nothing reads a field that no longer exists.

**C3. The obfuscated method name.**
- `backend/app/api/routes.py:524` reads `create_f = getattr(loop, "create_" + "fut" + "ure")`.
  Splitting a method name into string fragments hides it from grep and reads as evasion of a check.
  Replace with `loop.create_future()`. If some test rejected the word, fix the test, not the code.

**C4. Stale test artifacts and runners from the pre-section-11 engine.**
- Delete: `tests/feel_benchmark.json` (v1, replaced by `feel_benchmark_v2.json`),
  `tests/feel_results_baseline.json`, `feel_results_phase3.json`, `feel_results_phase3_local.json`,
  `feel_results_phase4.json`, `feel_results_phase4_local.json` (results of an engine that no longer
  exists), `tests/benchmark_1000_dataset.json` and `benchmark_1000_results.json`.
- `tests/test_recommendation_benchmark.py`, `tests/evaluator.py`, `tests/benchmark_dataset.py`:
  they score "archetype" vectors and genre gates (`evaluator.py:102,117`,
  `test_recommendation_benchmark.py:83`) that were deleted in R2. Delete all three.
- `tests/test_identify_debug.py`: a debug script, not a test. Delete or move under
  `ml/experiments/` if the hum-identify path still needs it.
- `tests/backbone_previews.json`, `tests/fetch_backbone_previews.py`, `tests/previews/`: keep, they
  feed `ml/experiments/backbone_check.py` and the new `tests/fixtures/octave_truth.json` (S2).
- Do this in step 1 (S9) so the workflow only lists tests that exist.

**C5. Visualisation leftovers from the ONNX encoder.**
- `ml/inference/visualize.py` ("Visualise trained CnnMusicEncoder embeddings", runs after
  `export_onnx`), `ml/inference/plots/` (`tsne.png`, `training_curves.png`, `genre_similarity.png`)
  and `ml/inference/visualizations/`: the encoder and the index builder they describe were deleted
  in R2. Delete all of them.

**C6. Experiments with the known HF-port bug.**
- `ml/experiments/clap_sanity.py` calls the HF `ClapProcessor` with `audio=` (the bug 12.0 recorded)
  and tests a port 12.1 rejected. Delete it and `clap_sanity_results.json`. Keep
  `backbone_check.py` and `backbone_check_results.json`; they are the record of the 12.1 decision.

**C7. The adapter manifest's unwritten fields.**
- `ml/inference/ADAPTER_MANIFEST.json:11-13` contains `tune_recall_before`, `tune_recall_after`
  and `gate_passed: true`. No script writes those keys (`train_adapter.py:223-230` writes six other
  fields). Delete the three keys and add `"served": false` and
  `"note": "never loaded by the server; measured harmful outside rock, see FIX_PLAN.md section 1"`.
  Remove the "TUNE before/after" and "Gate" lines from the `train_adapter.py` docstring (S8).

**C8. Large binaries in two places.**
- `ml/inference/music_index.faiss`, `ml/inference/track_metadata.json`, `ml/data/clap.npy`,
  `ml/data/catalog_features.jsonl`, `ml/data/playlist_membership.jsonl` are Git LFS pointers in the
  repo, while the Space loads from the HF dataset `shreyash616/echo-ml-artifacts`, which holds only
  the `ml/inference` files. Two sources of truth, and GitHub's free LFS bandwidth (1 GB a month) is
  exhausted by a handful of pushes of these sizes.
- Make the HF dataset the only home: upload the three `ml/data` files there too (same paths), add
  the five paths to `.gitignore`, remove them from LFS tracking (`.gitattributes`) and from the
  tree, and make `tests/deploy_hf.py` and `build_index_v2.py` read and write the dataset paths. The
  `HARVEST_MANIFEST.json` with the SHA256 of each file stays in git as the record.

**C9. Client leftovers.**
- `app/src/services/api.ts:5` and `:19` declare `key` twice (optional and required). Keep one,
  optional, documented as the `artist:title` catalogue key, never shown.
- The KEY stat (S6) at `SongCard.tsx:63-64` and `static/index.html:1582-1584`.
- `previewUrl` on `TrackResult` is declared and never used by the app; leave it, it is real data and
  the next feature (in-app preview playback) needs it.

**C10. Documents.**
- `VISION_ALIGNMENT.md`: add a two-line note at the top that it is superseded by
  `FEEL_ALGORITHM.md` and this file, and that its phase list and line numbers are historical.
- `FEEL_ALGORITHM.md`: its header already points here. Its progress log gets the S10 correction
  entries; nothing else in it changes.
- `SETUP.md`: C1 above; also remove any mention of `music_encoder.onnx`, `archetypes.json` or
  Last.fm keys.

**C11. Workflow.**
- `.github/workflows/tests.yml` after S9 lists exactly: golden features, no-gates, no-benchmark-
  leakage, candidate pool, theme sanity, describe parity, no-constant-fields, live overlap, and the
  benchmark runner in report-only mode. Nothing that was deleted in C4 may remain in the list.
