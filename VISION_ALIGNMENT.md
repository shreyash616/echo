# Echo: aligning the engine with the "feel" vision

> This document is superseded by FEEL_ALGORITHM.md and FIX_PLAN.md. Its phase list and line numbers are historical.

Status: plan, written 2026-10-26 against commit `ee863aa` on `main`. All `file:line`
references are from that commit. Re-grep before editing; do not trust a line number blindly.

This document is written for an agent (or a person) who has not seen the codebase. It
states the product vision, what the code does today, the evidence that the two disagree,
and an ordered work plan with acceptance criteria. Read it top to bottom once, then work
phase by phase. Do not skip Phase 0.

---

## 1. The vision

> I should be able to find songs that make me feel the way song A does.

The owner's reference examples (these are the benchmark, see section 3):

| Seed | Should recommend |
|---|---|
| Simple Man - Lynyrd Skynyrd | When the Levee Breaks - Led Zeppelin; Stairway to Heaven - Led Zeppelin; Rooster - Alice in Chains |
| Waiting Around to Die - Townes Van Zandt | Black - Danger Mouse & Daniele Luppi; No Surprises - Radiohead; Me Gustas Tu - Manu Chao |
| Bang Data - Bang Data | Tidal Wave - Thee Oh Sees (and songs like it) |
| Boyfriend - Karan Aujla | Antidote - Karan Aujla; For A Reason - Karan Aujla; Baller - Shubh; Roll With Me - Sukha |

What these share is **feel**, which decomposes into four measurable axes:

| Axis | Simple Man set | Waiting Around set | Bang Data set | Boyfriend set |
|---|---|---|---|---|
| Felt tempo | slow, 60 to 75 BPM half-time | slow to mid, drifting | fast, driving | mid, 85 to 95 |
| Weight / darkness | heavy, brooding, minor | sparse, resigned, warm | raw, loud, fuzzy | bright, melodic |
| Dynamics | quiet start, builds to loud | flat, intimate | relentless | flat groove |
| Texture | guitar-led, big | acoustic, lo-fi | garage distortion | trap drums, Auto-Tuned melody |

What is **not** an axis:

- **Era.** Simple Man (1973) to Rooster (1992). Townes (1968) to Black (2011).
- **Genre label.** Country to Radiohead to Manu Chao. Southern rock to grunge.
- **Artist graph / popularity.** Thee Oh Sees are not a "related artist" of Bang Data anywhere.
- **Language**, with one refinement from the owner: *same language should be prioritised, with
  some leeway, but not too far, and the distance between languages should be encoded.* Language
  is a soft prior on ranking, never a gate on candidates. "Not too far" is a per-pair affinity
  (section 5, task 3.3), not a binary. Me Gustas Tu (Spanish/French) for an English seed, and
  Thee Oh Sees (English) for a Spanish seed, must both be reachable.

The Boyfriend example is the special case where feel and scene coincide (contemporary Punjabi
trap-pop: same producers, same tempo band, same vocal treatment). The engine must get it right
*because of the feel*, not because it walked the artist graph. Today it gets it right for the
wrong reason, and that masks the failure on the other three.

---

## 2. What the code does today (as built, not as documented)

`SETUP.md` is stale. It describes a 65% neural engine. The code is a metadata-gated
Deezer + Last.fm recommender with a 22% neural term. Pipeline for `GET /api/recommendations/{id}`
and `POST /api/identify`, both ending in `_find_and_rank_similar_tracks` (`backend/app/api/routes.py:445`):

| Step | What happens | Where |
|---|---|---|
| Context | Source language from Deezer genre + Last.fm artist tags + Unicode script; genre family from a fixed keyword list (`_GENRE_FAMILIES`, "pop" matched last) | `routes.py:118-157`, `routes.py:204-288` |
| Candidates (three sources, merged) | (a) Deezer: 2 of the source artist's own top tracks + top 3 tracks of each of 12 `/artist/{id}/related` artists, every one stamped with the SOURCE's genre and flagged `_trusted_scene`; then text searches `"{artist} {genre}"`, `"{lang} {genre}"` etc. (b) Last.fm `track.getSimilar` (co-listening). (c) FAISS neighbours over a 1,656-track index. | `routes.py:474-492`, `backend/app/services/deezer.py:387-493` (stamp at `:433-436`), `backend/app/services/recommender.py:1265-1403` |
| Resolve + merge | Last.fm and preview-less FAISS titles are text-resolved through `search_track`, 16-slot budget, Last.fm first; merge order FAISS, Deezer, resolved; dedup on id and `artist:title`; cap 30 | `routes.py:500-540` |
| Language wall | Candidate rejected if its tags/script yield any language different from the source; for an English source, rejected if ANY language is detected (so "latin" or "french" artist tags kill an English track) | `routes.py:566-578`; FAISS equivalent `recommender.py:1293-1324` (`_GENRE_TO_LANG`) |
| Per-candidate scoring | Download 30 s preview, encode with ONNX, compute DSP metrics, fetch LRCLIB lyrics, composite score, +0.03 if genre matched | `routes.py:549-664`, `recommender.py:865-948` |
| Weights | neural 0.22, lyric theme 0.17, mood 0.13, vocal 0.13, key+bpm 0.10, instruments 0.09, pace 0.06, spatial 0.04, complement 0.03, era 0.03, all times a lyric "clash" multiplier (0.35 / 0.45 / 0.50) | `recommender.py:921-923`, `:947`; `backend/app/services/lyrics.py:57-61` |
| Genre partition | If 20+ candidates are genre-matched, every other candidate is dropped regardless of score | `routes.py:668-672` |
| Rank | MMR with lambda 0.72, max 2 per artist, then sort by score | `routes.py:348`, `:674-680` |

Supporting facts:

- The ML artifacts in `ml/inference/` are Git LFS pointers (132 and 133 bytes). Real files are on
  HF Hub dataset `shreyash616/echo-ml-artifacts`, fetched at container start by `backend/startup.sh`.
  If that fetch fails the encoder silently returns a zero vector (`recommender.py:1162-1169`, `:1233`).
- The index is 1,656 tracks: punjabi 158, rock 151, hindi 142, hip-hop 117, tamil 77, indie 77,
  latin 76, acoustic 70. Fourteen KMeans archetypes, named by their members' top genre tags
  (`ml/inference/build_index.py:342-365`). No English-pop or classic-rock archetype exists.
- The "vibes" shown to the user are those archetype names ("Rock / Telugu", "Punjabi / Afrobeats"),
  assigned at `recommender.py:1226-1229`. They are genre labels, not feel.
- Index tracks never have measured instruments / vocal / pace / spatial features. They are
  synthesised from the genre string (`recommender.py:469-543`): any "rock" track becomes
  Guitar & Strings + Drums + Bass. That is 35 points of the composite.
- The encoder is an ~11.7M-param ResNet trained with SimCLR NT-Xent (temp 0.1, `ml/training/losses.py:17-34`)
  where the positive pair is two energy-matched crops of the SAME track (`ml/training/dataset.py:8-21, 48+`)
  with near-identity augmentations (gain 0.95 to 1.05, +-1 mel bin, <=8-frame masks, sigma 0.005 noise,
  `dataset.py:22-30`). It learns "same recording" invariance, not cross-song feel.
- Tempo estimation applies a log-Gaussian prior centred on 120 BPM (`recommender.py:146`,
  `tempo_prior = exp(-0.5 * (log2(bpm/120) / 0.55)**2)`), range clipped to 65..185 (`:132`, `:164`).
  A 60 BPM half-time ballad is therefore read as 120 by design.
- Two clients exist. `app/` (React Native, last touched 2026-03-24) reads 13 fields and ignores
  `matchReasons`, `year`, `lyricalIntent`, `camelotKey`, never plays `previewUrl`.
  `backend/static/index.html` (1,988 lines, commits 2026-10-24..26) is the live product: Spotify PKCE,
  playlist export, diversity slider, era and reason badges.

---

## 3. Evidence: the four examples on the live backend (2026-10-26)

Backend: `https://shreyash616-echo-backend.hf.space`. Procedure: `GET /api/search?q=<seed>`,
pick the original (not a cover), `GET /api/recommendations/{id}`.

| Seed (id used) | Targets found in top 20 | Top 5 returned | Round trip |
|---|---|---|---|
| Simple Man (Deezer `1559695`) | 1 of 3: Rooster at #19 (0.83) | Free Bird, Start Me Up, Sharp Dressed Man, Gimme All Your Lovin', Highway Star | 31 s |
| Waiting Around to Die (iTunes `it_1232323565`) | 0 of 3 | **itself** (#1), "Acoustic" - Olivier Deriviere, Loretta - Townes, "Acoustic" - Billy Raffoul, ceilings - Lizzy McAlpine | 11 s |
| Bang Data (Deezer `731189182`) | 0 of 1 | Drivin' Steady - Bang Data, Orkesta Mendoza x2, J Balvin, Wisin & Yandel | 11 s |
| Boyfriend (Deezer `3495188951`) | 1 of 4: For A Reason at #2 | Dil Dardeh - Sukha, For A Reason, Sweet Flower - AP Dhillon, HEY LUV - Sukha, 295 - Sidhu Moose Wala | 10 s |

**Recall@20 across the four seeds: 2 of 11.** The existing benchmark
(`tests/benchmark_1000_results.json`) reports 100% pass on the same engine. See finding 9.

What the engine measured on the seeds themselves (`GET /api/track/{id}`):

- Simple Man: bpm **120** (felt ~60), energy 0.56, valence 0.52, instrumentPace
  "Upbeat & Groovy (8th-Note Subdivision)", instruments include "Synthesizer & Keys" and "Bass & 808".
  Only `vocalPattern: Melodic Legato (Sustained Ballad / Soulful)` is right.
- Waiting Around to Die: bpm **178** (felt ~90).
- Across all 60 recommendations in the three rock/folk runs, energy stayed in 0.37..0.70 and
  valence in 0.40..0.64. Start Me Up scored energy 0.61 / valence 0.50; Rooster 0.54 / 0.51.
  The mood features cannot separate them.
- Every rock recommendation carried the reason "Shared Guitar & Strings, Drums & Percussion"
  (the genre synthesis), and `year` was `None` on all 60, so era logic never actually fired.

Why the specific targets were unreachable:

- Me Gustas Tu: Manu Chao's Last.fm tags include "latin"/"french", so for an English seed
  `routes.py:577-578` returns None. Structurally impossible under the current gate.
- Tidal Wave - Thee Oh Sees: Bang Data is detected as Spanish; an English-tagged candidate
  fails `routes.py:572-575`. Structurally impossible.
- Stairway / Levee: Led Zeppelin is in Skynyrd's Deezer related list (10th of 12), but every
  Zeppelin hit through the backend's own `/api/search` has an `it_` (iTunes) id, so the Deezer
  `/artist/{id}/top` walk at `deezer.py:448-458` yields nothing for them. Only the 16-slot
  Last.fm text-resolve budget could have brought them in, and Last.fm queries fill it first.
- Black - Danger Mouse: not in the artist graph of Townes, not a Last.fm co-listen neighbour.
  No feel-based path exists to reach it.

---

## 4. Findings that drive the plan

1. **Candidates are chosen by metadata, scored by feel.** The acoustic model never decides what
   is in the pool. Fix: Phase 4.
2. **Language is a hard wall in two places.** `routes.py:566-578`, `recommender.py:1316-1324`.
   Owner wants a soft prior. Fix: Phase 3, task 3.3.
3. **Source perception is wrong before ranking starts.** Tempo prior at 120 BPM
   (`recommender.py:146`), energy/valence pinned near 0.5. Fix: Phase 2.
4. **35 points of the score compare real DSP on the seed against genre-string fabrications on
   the candidate** (`recommender.py:469-543`). Fix: Phase 3, task 3.1.
5. **The lyric-theme multiplier can cut a perfect acoustic match to 35%** (`recommender.py:947`,
   `lyrics.py:57-61`), and the theme is a keyword count over title+artist+lyrics with lists that
   fire on "baby", "night", "top", "body". Fix: Phase 3, task 3.2.
6. **Trusted candidates are genre-matched by construction** (`deezer.py:433-436` stamps the
   source's genre; `routes.py:572-577` then matches on it), and the partition at
   `routes.py:668-672` drops everything else once 20 exist. Fix: Phase 3, task 3.4.
7. **The index is too small and skewed for the vision.** 1,656 tracks, no English rock cluster.
   Fix: Phase 4, task 4.2.
8. **The training objective learns recording identity, not feel.** Fix: Phase 5.
9. **The benchmark cannot see any of this.** `tests/benchmark_1000_results.json` has
   `total_evaluated: 10`, all Hindi, all 1950s. Local mode uses `np.random.randn(1, 512)` as the
   embedding (`tests/test_recommendation_benchmark.py:79-81`) and key fixed to "C Major"
   (`:86`). The evaluator compares every rec's key to a hardcoded "C Major"
   (`tests/evaluator.py:246`), counts a missing year as MAE 12.0 / score 0.75 / **pass**
   (`:235-238`), and its pass rule is language purity + era MAE + genre tag precision
   (`:258`). No acoustic similarity is measured anywhere. Fix: Phase 0.
10. **Small live bugs**: the seed recommends itself when the seed title carries "(feat. ...)"
    (`routes.py:530` keys on raw title); covers outrank originals in search
    ("Waiting Around to Die (Townes Van Zandt) - The Black River Gypsies" #1,
    "Stairway to Heaven (led zeppelin) - Scott D. Davis" #1, two Simple Man covers above the
    original; `deezer.py:88` `score_track_originality`); the `"{lang} {genre}"` text query at
    `deezer.py:462-478` pulls in any track literally titled "Acoustic". Fix: Phase 1.

---

## 5. Work plan

Phases are ordered by dependency and payoff. Within a phase, tasks are independent unless
stated. Every task has an acceptance criterion; do not mark a task done without meeting it.

### Phase 0: build the feel benchmark first (nothing else is measurable without it)

**0.1 Seed file.** Create `tests/feel_benchmark.json`:

```json
[
  {
    "seed": {"title": "Simple Man", "artist": "Lynyrd Skynyrd", "language": "en"},
    "targets": [
      {"title": "When the Levee Breaks", "artist": "Led Zeppelin"},
      {"title": "Stairway to Heaven", "artist": "Led Zeppelin"},
      {"title": "Rooster", "artist": "Alice in Chains"}
    ]
  },
  {
    "seed": {"title": "Waiting Around to Die", "artist": "Townes Van Zandt", "language": "en"},
    "targets": [
      {"title": "Black", "artist": "Danger Mouse"},
      {"title": "No Surprises", "artist": "Radiohead"},
      {"title": "Me Gustas Tu", "artist": "Manu Chao"}
    ]
  },
  {
    "seed": {"title": "Bang Data", "artist": "Bang Data", "language": "es"},
    "targets": [
      {"title": "Tidal Wave", "artist": "Thee Oh Sees"}
    ]
  },
  {
    "seed": {"title": "Boyfriend", "artist": "Karan Aujla", "language": "pa"},
    "targets": [
      {"title": "Antidote", "artist": "Karan Aujla"},
      {"title": "For A Reason", "artist": "Karan Aujla"},
      {"title": "Baller", "artist": "Shubh"},
      {"title": "Roll With Me", "artist": "Sukha"}
    ]
  }
]
```

Grow this to 20 seeds over time, 3 to 5 human-picked targets each. Do not pad with variants
or synthetic titles. Every target must be a song a human chose for feel.

**0.2 Runner.** Create `tests/feel_benchmark.py`:

- For each seed: `GET /api/search?q="{title} {artist}"`, choose the first result whose artist
  matches the seed artist case-insensitively and whose title does not contain "(", "cover",
  "karaoke", "tribute" (prefer a Deezer numeric id over an `it_` id when both match).
- `GET /api/recommendations/{id}`. Match targets on normalised `title` + `artist` substring
  (strip parentheses, lowercase, collapse whitespace).
- Report per seed: `k_returned`, `hits`, `recall@20`, rank of each hit, and the top 5 returned.
  Report overall `recall@20` = total hits / total targets, and `self_recommended` count
  (any returned track whose normalised title+artist equals the seed).
- Flag `--base-url` (default the HF space) and `--out results.json`.
- Also record for each seed what `/api/track/{id}` measured: `bpm`, `key`, `energy`,
  `valence`, `instrumentPace`. These are the perception numbers Phase 2 must move.

**Acceptance:** runner executes against the live backend and prints `recall@20 = 2/11` (or
whatever the current number is) with Rooster at rank 19 for Simple Man and the self-recommend
flagged for Townes. Commit the first `results.json` as the baseline.

**0.3 Retire or quarantine the old benchmark.** Leave `tests/evaluator.py`,
`tests/benchmark_dataset.py`, `tests/test_recommendation_benchmark.py` in place but add a
module docstring at the top of each stating they measure language purity / era / genre tag
precision, not feel, and that `feel_benchmark.py` is the acceptance test. Fix the two outright
bugs since they are cheap: `evaluator.py:246` must compare to the seed's measured key, not
"C Major"; `evaluator.py:235-238` must not count a missing year as a pass (append `None` and
exclude from the mean, do not increment `era_window_passes`). Delete or fix
`tests/test_identify_debug.py:10-12`, which asserts that `b"fake audio data 1234567890"`
identifies a track.

### Phase 1: correctness fixes that show up in every run

**1.1 Self-recommendation.** `routes.py:528-540`. Normalise the title before keying: strip
anything in parentheses or after " - ", " (feat", "[", lowercase, collapse whitespace. Apply the
same normaliser to candidates. Add the normalised `artist:title` of the seed to `seen_keys`.
Acceptance: Townes seed no longer returns itself; `feel_benchmark.py` reports
`self_recommended = 0`.

**1.2 Covers above originals in search.** `deezer.py:88` `score_track_originality`. The current
demotions (remix/cover/karaoke) are not catching "(Townes Van Zandt)" in a title by a different
artist, or "(led zeppelin)" in a piano cover. Add: if the query's artist tokens appear in the
candidate TITLE but not in the candidate ARTIST, treat as a cover (-160). If the candidate artist
exactly matches the query artist, add a large bonus (+60) that outweighs the popularity term
(`deezer.py:136-138`, up to +25). Acceptance: `/api/search?q=Simple Man Lynyrd Skynyrd` and
`q=Waiting Around to Die Townes Van Zandt` and `q=Stairway to Heaven Led Zeppelin` all return
the original artist first.

**1.3 Text-query noise.** `deezer.py:462-478`. Drop the `"{lang} {genre}"` and bare
`"{genre}"` queries entirely; they return tracks named after the genre word ("Acoustic" x2 in
the Townes run). Keep `"{artist} {decade}"` only if Phase 4 has not yet landed; delete it after.

**1.4 Operational.** One shared `httpx.AsyncClient` per service module instead of a new client
per call (`deezer.py:154, 237, 277, 319, 344, 372, 510`; `lastfm.py:23`; `lyrics.py:75`;
`routes.py:268`). Add an overall deadline to `_find_and_rank_similar_tracks` (target 15 s:
`asyncio.wait_for` around candidate evaluation, return what has finished). Log at WARNING, not
DEBUG, when `_evaluate_candidate` swallows an exception (`routes.py:662-664`). Make
`startup.sh` fail the container if an artifact download fails rather than letting the encoder
return zeros. Remove `langdetect` and `pyacoustid` from `backend/requirements.txt`; neither is
imported.

### Phase 2: fix source perception (everything downstream inherits these numbers)

All of this is in `backend/app/services/recommender.py` and mirrored in
`ml/inference/audio_features.py` (`estimate_tempo_from_mel` at `audio_features.py:92`). Keep
the two in sync or make the backend import the ml function.

**2.1 Felt tempo.** `recommender.py:132-166`. The prior at `:146` is centred on 120 with
sigma 0.55 octaves. Replace with: compute the autocorrelation peak, then explicitly evaluate the
half-time and double-time candidates and pick using `tempo_fast_ratio` (already one of the 16
`EmotionFeatureExtractor` features in `ml/models/music_encoder.py`; expose it from the DSP path)
plus onset density: low onset density and low `tempo_fast_ratio` means take the slower octave.
Widen the clip range to 50..200. Return BOTH `bpm` and `felt_bpm`; match on `felt_bpm`.
Acceptance: Simple Man reports felt_bpm in 55..70; Waiting Around to Die in 80..100; Boyfriend
stays in 85..95; Bang Data stays above 120. Add these as assertions in `feel_benchmark.py`
using the `/api/track` numbers.

**2.2 Energy and valence.** `recommender.py:206-262` (`tempo_score` at `:242` is a linear
map of bpm; that is why double-time reads also inflate energy). After 2.1, recompute energy
from RMS mean, spectral flux and felt tempo. Then calibrate: take 40 tracks you can label by
hand (10 each: slow-dark, slow-bright, fast-dark, fast-bright), run the DSP, and fit a
per-feature affine map so that the labelled quadrants separate. Store the fitted constants in
code with the 40-track list in `ml/data/calibration_set.json`. Acceptance: on the 40, energy
separates fast from slow with AUC >= 0.9 and valence separates bright from dark with AUC >= 0.8.
Start Me Up must score energy >= 0.75 and Rooster energy <= 0.45.

**2.3 Dynamics / build.** New feature. On whatever audio window is available compute:
RMS slope (linear fit of frame RMS over time), loudness range (90th minus 10th percentile RMS,
already `dynamic_range` in the 16 features), and position of the loudest 10% of frames
(early / mid / late). Expose as `buildScore` in `[0, 1]` and `loudnessRange`. Note the
limitation in a comment: 30 s previews usually sit on the chorus, so this is a proxy. Acceptance:
feature is returned by `/api/track` and used in Phase 3 scoring.

**2.4 Mode / darkness.** The Krumhansl key detection assigned 4 keys to 20 rock songs in the
Simple Man run. Do not spend time on exact key; expose `modeMinorProb` (minor-profile correlation
minus major-profile correlation, squashed to `[0, 1]`) and `brightness` (normalised spectral
centroid, already `centroid_mean`). Acceptance: Rooster, Levee, Simple Man all report
`modeMinorProb >= 0.6`; Boyfriend and Start Me Up `<= 0.4`.

### Phase 3: scoring that reflects feel

All in `recommender.py:865-948` (`compute_composite_match_score`) and the duplicated inline
version at `:1326-1382` (FAISS fast path). **First collapse the duplication**: make the FAISS
path call `compute_composite_match_score` so there is one scoring function.

**3.1 Remove fabricated features.** Delete `_synthesize_structural_meta_if_missing`'s
genre-to-instrument branch (`recommender.py:486-543`). Candidates without measured structural
features get those terms set to a neutral 0.5 with weight redistributed, never a genre guess.
Long term, measure them at index build time (`ml/inference/build_index.py`) so index tracks
carry real numbers.

**3.2 Remove the lyric-theme multiplier.** `recommender.py:947` and `:1381`: drop
`* clash_mult`. Reduce `w_lyric` from 0.17 to 0.05 and keep it only as a tiebreaker. The
keyword lists in `lyrics.py:17-55` are too coarse to carry 17% of the score. Delete the dead
"Instrumental" branch at `:925-935` (`classify_lyrical_intent` can never return
"Instrumental", see `lyrics.py:89-110`).

**3.3 Language as a soft prior (owner's rule: same language first, some leeway, not too far).**

- Delete the wall at `routes.py:566-578` and the FAISS skip at `recommender.py:1316-1324`.
  Keep `_detect_track_context` and `_detect_language_from_tags`; they now only feed the prior.
- Add `s_language` to the composite, weight `w_language = 0.08`, where
  `s_language = language_affinity(seed_lang, cand_lang)`. (0.08 beats the current 0.03 genre
  bonus and sits below every acoustic term, so same-language wins ties but a strong
  cross-language feel match can still rank.)
- **Language distance is encoded** (owner decision, 2026-10-26). Create
  `backend/app/services/language_affinity.py` with a symmetric lookup and a default:

  ```python
  # Affinity in [0, 1]. 1.0 = same language. Values are starting points; tune against
  # tests/feel_benchmark.py and record changes in docs/VISION_ALIGNMENT.md.
  _GROUPS = {
      "indic_north": {"hi", "pa", "ur", "bn", "mr", "gu"},
      "indic_south": {"ta", "te", "ml", "kn"},
      "romance":     {"es", "pt", "it", "fr", "ca"},
      "east_asia":   {"ko", "ja", "zh"},
      "germanic":    {"en", "de", "nl", "sv", "no", "da"},
      "arabic":      {"ar", "fa", "tr"},
  }
  _CROSS_GROUP = {              # listener-plausible neighbours across groups
      frozenset({"indic_north", "indic_south"}): 0.70,
      frozenset({"germanic", "romance"}):        0.70,   # en <-> es/fr/pt: Townes -> Manu Chao, Bang Data -> Thee Oh Sees
      frozenset({"germanic", "indic_north"}):    0.60,   # en <-> hi/pa: Punjabi trap-pop borrows from US hip-hop
      frozenset({"germanic", "east_asia"}):      0.60,
      frozenset({"romance", "arabic"}):          0.60,
  }
  SAME = 1.0; SAME_GROUP = 0.85; DEFAULT_CROSS = 0.50; UNKNOWN = 0.60

  def language_affinity(seed: str | None, cand: str | None) -> float:
      if not seed or not cand: return UNKNOWN
      if seed == cand: return SAME
      gs, gc = _group_of(seed), _group_of(cand)
      if gs and gs == gc: return SAME_GROUP
      return _CROSS_GROUP.get(frozenset({gs, gc}), DEFAULT_CROSS)
  ```

  Seed and candidate language codes come from the existing `_detect_track_context` /
  `_detect_language_from_tags` (`routes.py:118-186`); extend `_GENRE_LANG_RULES` with the codes
  above that it does not yet emit (ur, bn, mr, gu, ml, kn, pt, it, ca, zh, de, nl, fa, tr).
  English detection is currently "no language found" (`routes.py:163-186` only maps non-Latin
  scripts); treat Latin-script-with-no-tag as `en` for the affinity lookup, and log it so the
  rate of unknowns is visible.
- In `_apply_mmr` (`routes.py:348`) replace the idea of a flat cross-language cap with an
  affinity budget: `max_cross_language: int = 6` counts candidates with affinity < 1.0, and
  `min_affinity: float = 0.50` drops anything below it outright (this is the only place a
  language threshold is allowed, and it is a presentation rule applied after scoring, not a
  candidate gate). Both become query params on `/api/recommendations`
  (`?cross_language_max=&min_language_affinity=`) so both clients can expose them.
- Return `languageAffinity` on each `TrackResult` so the clients can badge it
  ("Same language", "Nearby language", "Further afield").
- Acceptance: Townes seed can return Me Gustas Tu; Bang Data seed can return English garage
  tracks; Boyfriend seed still returns >= 14 of 20 Punjabi tracks, and any Hindi results for it
  rank above any English results of equal acoustic score (affinity 0.85 vs 0.60).

**3.4 Genre and era stop being gates.**

- `deezer.py:433-436`: stop stamping `genre_hint` onto candidates and stop setting
  `_trusted_scene`. Candidates carry their own Deezer genre or nothing.
- `routes.py:572-577`: `genre_matched` becomes a 0.02 tiebreaker, not a partition key.
- `routes.py:668-672`: delete the `genre_strict` partition. Rank the whole `valid` pool.
- `routes.py:586-590`: delete the era multiplier on the FAISS fast path. Keep `w_era` at 0.03
  or drop it. If the owner wants era filtering, it is a query param, not a default.

**3.5 New weights.** Starting point, to be tuned against `feel_benchmark.py`:

| Term | Weight | Source |
|---|---|---|
| neural (embedding cosine) | 0.30 | ONNX, after Phase 5 retrain; 0.22 until then with the remainder on felt tempo |
| felt tempo (octave-aware) | 0.15 | Phase 2.1 |
| energy + loudness range | 0.12 | Phase 2.2, 2.3 |
| mode / darkness + brightness | 0.12 | Phase 2.4 |
| build / dynamics | 0.08 | Phase 2.3 |
| vocal pattern + presence | 0.08 | existing `vocalPattern` is the one feature that read Simple Man correctly |
| language prior | 0.08 | Phase 3.3 |
| lyric theme | 0.05 | tiebreaker only |
| genre tiebreaker | 0.02 | metadata |

Remove the per-component floors (`s_inst >= 0.4`, `key 0.7 when unknown`, `bpm >= 0.15`,
`pace >= 0.35`, `mood >= 0.1`, `spatial >= 0.1`, `lyric 0.70`) or the score spread stays inside
0.15 and ordering is decided by the tiebreakers. Replace the neural squash
`(cos + 0.1) / 1.1` at `:882` with a plain clip of cosine to `[0, 1]`.

Acceptance for Phase 3 as a whole: `feel_benchmark.py` recall@20 >= 4/11 on the current index
and candidate sources (Rooster should move into the top 5 for Simple Man; For A Reason stays
top 3 for Boyfriend). Latency must not regress beyond Phase 1.4's deadline.

### Phase 4: invert candidate generation (feel first, metadata second)

**4.1 Pipeline order.** In `_find_and_rank_similar_tracks`:

1. FAISS neighbours of the seed embedding, k = 200, no language or genre filter.
2. Resolve previews/metadata for those (most index rows have `previewUrl` from the Deezer
   harvest; FMA rows do not, resolve them via `search_track`, budget 32, in parallel).
3. Deezer related-artist tracks and Last.fm similar tracks become supplementary sources,
   capped at 10 each, appended after the FAISS set.
4. Score everything with the Phase 3 function. Rank with MMR + the language cap.

**4.2 Index growth.** The vision cannot work on 1,656 tracks. Target 30k to 50k previews.
`ml/data/harvest_deezer_previews.py:63-100` has 46 seed queries, mostly Indian + hip-hop; add
at least: classic rock, southern rock, blues rock, grunge, garage rock, psych rock, folk,
alt-country, indie folk, dream pop, shoegaze, post-punk, britpop, trip-hop, downtempo, Latin alt,
cumbia, reggae, Afrobeats, J-rock, city pop, K-indie, Bollywood 60s/70s, ghazal, Sufi, Tamil
film 90s, Telugu film 2010s. Raise `--tracks-per-query` and add artist-seeded queries from the
benchmark targets (Led Zeppelin, Alice in Chains, Radiohead, Manu Chao, Thee Oh Sees, Shubh,
Sukha). Rebuild with `ml/inference/build_index.py`, re-upload via `ml/upload_artifacts.py`.
Note Led Zeppelin previews come from iTunes, not Deezer, in this pipeline; the harvester must
fall back to the iTunes lookup for artists Deezer lacks.

**4.3 Archetype names.** `build_index.py:342-352` names clusters by top genre tags. Replace
with names derived from the cluster's mean felt tempo / energy / mode / build (e.g.
"Slow, Heavy, Builds", "Fast, Raw, Bright") so the "vibes" the UI shows are feel words.
Update `app/src/components/VibeTag.tsx:25` colour map to the new vocabulary.

Acceptance: `feel_benchmark.py` recall@20 >= 7/11; Stairway or Levee appears for Simple Man;
Tidal Wave or another Thee Oh Sees / Ty Segall track appears for Bang Data.

### Phase 5: train for feel, not identity

`ml/training/dataset.py`, `ml/training/losses.py`, `ml/training/train.py`.

- Positive pairs: two DIFFERENT tracks whose Phase 2 features are close (felt tempo within 10%,
  `modeMinorProb` within 0.2, energy within 0.15, same `buildScore` tercile), sampled from the
  harvested catalogue. Keep same-track crops as a second, lower-weight positive. Hard negatives:
  same artist or same genre tag but far apart on the feel features (Free Bird vs Simple Man is
  the canonical hard negative).
- Keep NT-Xent + the VICReg terms; raise augmentation strength modestly (time masks up to 24
  frames, gain 0.8..1.2) since the objective is no longer identity.
- Train on the Phase 4.2 catalogue. Export with `ml/inference/export_onnx.py`, rebuild the
  index, upload.
- Commit a training manifest (`ml/checkpoints/MANIFEST.json`: catalogue hash, track count,
  epochs, final loss, date) so the shipped ONNX is traceable. Today nothing in the repo says
  what the ONNX was trained on.

Acceptance: on a held-out set of feel-labelled pairs, embedding cosine for same-feel pairs
exceeds different-feel pairs with AUC >= 0.85; `feel_benchmark.py` recall@20 >= 8/11 with
`w_neural = 0.30`.

### Phase 6: clients and documentation

- **Both clients are the product and must be similar in nature** (owner decision, 2026-10-26).
  The web client (`backend/static/index.html`, commits 2026-10-24..26) is ahead; the RN app
  (`app/`, last commit 2026-03-24) must be brought to parity, and from then on every
  user-facing feature lands in both in the same change. Parity checklist, with today's state:

  | Feature | Web (`index.html`) | RN (`app/src`) | Action |
  |---|---|---|---|
  | Mic identify | yes (`MediaRecorder`, `:1331-1352`) | yes (`RecordButton`, `HomeScreen.tsx:103-149`) | keep |
  | Text search with top-5 suggestions | yes | yes (`SearchBar.tsx`) | keep |
  | Hero card: score, feel vibes, felt tempo, energy, mode | partial | partial (`SongCard.tsx:42-67`) | both render the Phase 2 fields: `felt_bpm`, `modeMinorProb`, `buildScore`, feel-word vibes |
  | `matchReasons` badges | yes (`:1577-1579`) | no | add to `SongCard.tsx` |
  | `year` / era badge | yes (`:1498-1502`) | no | add, display only, never a filter by default |
  | `languageAffinity` badge | no | no | add to both (Phase 3.3) |
  | Preview playback (`previewUrl`) | yes | no (`api.ts:9` typed, never read) | add with `expo-av` |
  | Diversity slider (`?diversity=`) | yes (`:1072, :1190`) | no | add |
  | Cross-language controls (`?cross_language_max=&min_language_affinity=`) | no | no | add to both |
  | Spotify PKCE login + Create Playlist | yes (`:1682-1696`, `:1923`) | no | add via `expo-auth-session` PKCE against `/api/config`; same scopes |
  | Deep link to Spotify | web search URL | `spotify:search:` (`ResultsScreen.tsx:67-73`) | keep both |
  | `/api/track/{id}` detail view | yes (`:1467`) | no | add |

  Mechanics for keeping them similar: generate the client `TrackResult` type from
  `backend/app/models/schemas.py` (one script under `tests/` that emits `app/src/services/types.ts`
  and a JSDoc block for `index.html`, run in CI) so a schema field cannot exist in one client
  and not the other. Add a parity test that loads `feel_results_<phase>.json` and asserts both
  clients render every field the backend returns.
  `app/app/index.tsx` is the untouched Expo Router scaffold and `expo-router` is unused; delete
  both. Fix the GitHub description to say the product is web + iOS/Android.
- Rewrite `SETUP.md` section 1 to describe the Phase 3 scoring and the Phase 4 pipeline. It
  currently documents a 65/15/10/10 split that has not existed for many commits.
- Add `docs/FEEL_BENCHMARK.md` describing how to add a seed and how to read `results.json`.

---

## 6. Do not

- Do not re-add a language, genre or era gate "temporarily". Every one of them is a candidate
  filter that the benchmark targets fail.
- Do not synthesise features from metadata strings. A missing feature is neutral, never guessed.
- Do not tune weights by hand without running `feel_benchmark.py` before and after and committing
  both result files.
- Do not trust `SETUP.md`, docstrings or the comments in `recommender.py` for what the code
  does; several are already wrong (e.g. the 65% neural claim, the "Instrumental" branch).
- Do not use the old `benchmark_1000_results.json` numbers as evidence of anything. They are
  10 Hindi 1950s seeds scored with a random embedding.
- Do not commit the real `.onnx` / `.faiss` / `track_metadata.json`; they are LFS pointers in
  git and live on HF Hub. Use `ml/upload_artifacts.py`.

---

## 7. Verification checklist (run after every phase)

```
# 1. live feel benchmark against the deployed backend
python tests/feel_benchmark.py --base-url https://shreyash616-echo-backend.hf.space --out tests/feel_results_<phase>.json

# 2. perception sanity on the four seeds (printed by the runner):
#    Simple Man felt_bpm 55-70, modeMinorProb >= 0.6
#    Waiting Around to Die felt_bpm 80-100
#    Boyfriend felt_bpm 85-95, >= 14/20 Punjabi results
#    Bang Data felt_bpm >= 120

# 3. no self recommendations, no covers above originals (runner flags both)

# 4. latency: every seed under the Phase 1.4 deadline
```

Targets per phase: baseline 2/11; after Phase 3, >= 4/11; after Phase 4, >= 7/11; after Phase 5,
>= 8/11. Record the actual numbers in this file under a "Progress" heading as phases land.

---

## 8. Appendix: running the backend locally

```
cd backend
cp .env.example .env            # fill ACOUSTID_API_KEY, Spotify keys optional for recs
pip install -r requirements.txt
bash startup.sh                 # pulls the real artifacts from HF Hub into ml/inference/
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`ffmpeg` and `fpcalc` (Chromaprint) must be on PATH for `/api/identify`; recommendations by
track id work without them. The HF space is `shreyash616/echo-backend`; deploys go through
`.github/workflows/deploy.yml` (`upload_folder` of `backend/`).

## Progress

- 2026-10-26: baseline measured, recall@20 = 2/11. No phases started.
