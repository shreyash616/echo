# Echo feel algorithm: the panel's design

Status: design, written 2026-10-26 against commit `d8cb36e`. Supersedes the scoring, perception and
retrieval parts of `VISION_ALIGNMENT.md`; the benchmark-first rule, the language-as-soft-prior
rule and the "do not" list there still apply. Read this file top to bottom before touching code.

**Implementing agent: section 11 is the operative plan.** It is the zero-cost, no-human-labelling
path and it says exactly what to build, in what order, with what checks. Sections 1 to 10 are the
reasoning behind it; read them, but execute section 11. Anything in sections 3, 6 or 9 that needs
money, a non-commercial licence, or the owner's labelling hours is deferred and must not be started.

How this was produced: six independent reviews (a musicologist / MIR specialist, a classic-rock and
grunge producer, a folk and Americana songwriter, a garage-psych bandleader, a Punjabi hip-hop
producer, and an audio-ML / recommender-systems engineer) each read the vision, the live results
and the plan, and answered a different question. Section 10 lists where they disagreed and which
way this document goes. Every code claim below was re-verified against the tree at `d8cb36e`.

---

## 1. What "feels the same" actually is

The owner's sentence: *find songs that make me feel the way song A does.*

The panel's consensus on what the listener is hearing, set by set:

| Seed set | The shared thing, in a musician's words | Explicitly NOT shared (never match on it) |
|---|---|---|
| Simple Man -> Levee, Stairway, Rooster | Slow boot-stomp you count at 60 to 75; snare on 2 and 4 a hair behind, few hits per bar, air between them; chord colour that aches (major chords sung as a lament, or an undefined power-chord riff); chest-voice restraint that ends in a wail; weight. Build is the bonus (Levee grinds flat and still belongs). | Key, detector BPM, mode as such (Simple Man is C major leaning on vi; Levee and Rooster have no stable third), distortion level, decade, lyric topic |
| Waiting Around to Die -> Black, No Surprises, Me Gustas Tu | One voice close to the mic, singing slightly behind an unhurried pulse, not trying to convince you; a short figure that goes round; air in the arrangement; resignation with warmth; small room. | Mode (No Surprises and Me Gustas Tu are major), tempo in BPM, instruments, language, rhythm template (Me Gustas Tu is a reggae skank), decade |
| Bang Data -> Tidal Wave | Relentless straight eighths with no lay-back and no arc; fuzz into an already-hot input, gain two stages deep, top end turned to hash; loud from saturation, not from a limiter; shouted vocal inside the guitars. The push, not the count. | BPM (an 86 stomp vs a fast garage count), room vs box, harmony, language |
| Boyfriend -> Antidote, For A Reason, Baller, Roll With Me | Same chassis: 90 to 100 BPM lean-back bounce, snare on 2 and 4, 16th hats with triplet rolls, long-decay sliding 808, short hard-tuned pentatonic hook with octave doubles on the hook only, flat loop arrangement. | Brightness and mood (Baller is dark), key, lyric theme. Scene and feel coincide here, which is why the artist graph fakes it |

So the four axes in VISION_ALIGNMENT.md section 1 were close but wrong in two places:

- **"Weight / darkness" is not mode.** Three of the four sets contain major-key targets. The axis is
  a darkness scalar built from brightness, low-register share and roughness, with mode polarity as a
  small term that is switched off when the track has no stable third.
- **"Dynamics" cannot be measured on a 30 s chorus preview**, and three reviewers independently said
  the current `buildScore` is a line fitted to a plateau. It is dropped until full-track audio or a
  second window exists. What a preview CAN show is restraint vs saturation: loudness range and
  crest factor.

The corrected axis set, each measurable on a 30 s preview unless marked:

| Axis | What it is | Seeds it separates |
|---|---|---|
| Felt tempo | the count a drummer gives, as a distribution over octaves, plus pulse rate (eighth-note onset density) as a separate number | Simple Man 60-75 from Start Me Up 125; Bang Data's drive from its 86 count |
| Groove | kick pattern (syncopated vs four-on-the-floor), backbeat presence, hits per bar, swing vs straight, behind-the-beat lag | Boyfriend from Tauba Tauba; Simple Man from Sharp Dressed Man (shuffle) |
| Weight / darkness | low-register share, brightness, roughness, mode polarity when defined | Simple Man from Miss a Thing (slow and light); Rooster from Enter Sandman (heavy and fast is a tempo miss, not a darkness miss) |
| Texture / grit | whole-mix saturation, 4-10 kHz non-harmonic hash, drum transient-to-sustain separation, mid/side width | Bang Data from Fall Out Boy (both low-crest; only one is saturated) |
| Intimacy / room | vocal-to-mix ratio, reverb tail, stereo width, timing looseness, pitch steadiness, breath and string noise | Townes from a Nashville ballad |
| Vocal delivery | syllable rate, sustain ratio, pitch range used, vibrato, tuning depth, stacking at the hook | Townes (talk-sung) from Adele (belted); Boyfriend (hard-tuned stack) from 295 (rapped) |
| Restraint vs saturation | short-term loudness range, crest factor, bucketed | acoustic intimacy from brickwalled garage |
| Harmonic rhythm | chord changes per bar, loop length | Simple Man (one chord per bar) from Miss a Thing (every two beats) |
| Language | a prior on ranking, never a feature of feel | see section 5 |

---

## 2. Corrections to VISION_ALIGNMENT.md

These acceptance criteria and assumptions in the earlier plan are wrong and must not be met:

| Plan line | Says | Why it is wrong | Replace with |
|---|---|---|---|
| 2.4 acceptance | Simple Man, Rooster, Levee `modeMinorProb >= 0.6` | Simple Man is major; Levee and Rooster have no stable third. The implementer met it by adding a tempo bias inside the mode feature (`recommender.py:259`). | darkness scalar high (>= 0.6) for all three; `mode_undefined` flag true for Levee and Rooster |
| 2.1 acceptance | Bang Data felt tempo >= 120 | The song is an 86 stomp with eighths pumping; forcing it up pulls in pop-punk at 160. | felt tempo 80-95 with pulse rate high (>= 0.7 normalised); Tidal Wave matches on pulse rate, loosely on tempo |
| 3.3 acceptance | Boyfriend returns >= 14 of 20 Punjabi | That is a scene metric; it rewards the Honey Singh bhangra remake. | >= 14 of 20 are half-time trap with tuned vocals, any language; report the Punjabi share separately |
| 2.3 | `buildScore` from RMS slope, weight 0.08 | Previews sit on the chorus; the slope is within-chorus noise. | drop; use loudness range and crest factor as a restraint bucket |
| 2.2 | energy from RMS mean, flux and tempo | mel is `power_to_db(ref=np.max)` per clip (`recommender.py:1121`), so RMS is relative to the clip's own peak and has no cross-track meaning; tempo-in-energy propagates every octave error | integrated loudness (LUFS) after normalisation, band-split onset rates (sub-200 Hz for groove, 5 kHz+ for hats), flux |
| 5 Phase 5 | positives = pairs close on tempo, mode, energy, build | distils the DSP (bugs included) into the embedding, which then adds nothing the reranker lacks | positives from human pairs and playlist co-occurrence; DSP only for hard-negative mining |
| 3.3 table | en <-> Punjabi 0.60, Punjabi <-> Tamil 0.70, unknown 0.60, `min_affinity` 0.50 with `DEFAULT_CROSS` 0.50 | Punjabi trap borrows from Atlanta and Toronto, not Tamil film; unknown-language tracks are the old and foreign ones this feature exists to surface; default-on-threshold is a wall by another name | section 5 |
| 3.5 | `w_mode 0.12` | drops Me Gustas Tu from the Townes set every time | mode polarity 0.04, compatible-match, undefined matches anything |

---

## 3. The algorithm

### 3.1 Offline: catalogue and features (everything expensive happens here)

1. **Catalogue.** Deezer search, chart and playlist endpoints give a public 30 s `preview`; iTunes
   Search is the fallback for artists Deezer lacks (Led Zeppelin resolves only as `it_` ids today).
   Target 50k tracks by week 2, 200k by week 8. Store features and embeddings, never audio; previews
   are fetched transiently at index time. Harvest seeds must include every world the benchmark
   touches (section 7 lists them) and must not be limited to the benchmark artists.
2. **One decoder, pinned.** Delete the soundfile and PyAV paths (`recommender.py:984-1017`); keep one
   ffmpeg call with fixed flags (`-ac 1 -ar 22050 -f f32le`) in a pinned Docker image. Normalise to
   -23 LUFS (pyloudnorm) before any feature. Ten golden WAV fixtures with expected feature JSON,
   tolerance 1e-3, run in CI on the same image the Space runs. A number that cannot be reproduced in
   CI does not go in the index. (Today Simple Man reads felt 62 locally and 120 live from the same
   code; that is this bug.)
3. **Per-track feel descriptors** (section 3.2), computed once.
4. **Embedding** (section 3.3), computed once.
5. **Index.** `IndexFlatIP` over the 128-d adapter output: 50k x 128 float32 is 25 MB and answers
   k=500 in milliseconds on CPU. Do not reach for IVF or HNSW below 500k vectors.
6. **Per-track metadata**: language code with confidence (section 5), popularity, artist id,
   Deezer genre string (feature only), Last.fm tag vector where available.

### 3.2 Feel descriptors (the DSP layer)

All of these are inputs to the reranker (3.4). None has a fixed weight in code once the reranker
is learned; until then the hand weights in 3.4 apply.

**Felt tempo: a posterior, not an int.** Delete the four octave rules (`recommender.py:199-213`,
`ml/inference/audio_features.py:189-195`).
- Onset envelopes in three bands: low (< 150 Hz, kick), mid (150 Hz-2 kHz, snare and body), high
  (> 4 kHz, hats and strums). Tempogram per band and fused, 40-240 BPM.
- Tatum = fastest strong periodicity in the fused tempogram. Tactus candidates = tatum x {1, 2, 3, 4, 6, 8}.
- Evidence per candidate: pulse clarity (fused and low band); backbeat alternation (mid band peaks
  every 2 or 4 candidate beats means tactus; peaks on every candidate beat means you are at the
  tatum); onset density per candidate beat (1 to 3 is tactus-like); flux periodicity at 4x the
  candidate (chord changes land on bars); a weak Moelants-style resonance prior around 100-120
  applied to the POSTERIOR, not to the raw estimate (the deleted 120 prior was fine, its placement
  was wrong).
- Half-time templates: rock ballad = snare every other candidate beat with eighths in the high band;
  trap = hats at 2x or 3x with snare every two kick cycles. In both the tactus is the snare cycle.
- Output `felt_bpm_dist` (probabilities over candidates) and `felt_bpm` (argmax), plus `pulse_rate`
  (eighth-note onset density, normalised) as a SEPARATE feature. Compare tracks with an
  octave-folded distance: sum over candidate pairs p_i p_j d(bpm_i, bpm_j), where d is |log2 ratio|
  folded so ratio 1 costs 0, ratio 2 costs 0.4, ratio 1.5 costs 0.7.
- Once 200 seeds have an owner-tapped tempo, replace the evidence rules with a logistic regression
  over the same evidence (tempogram energy at the three octaves, onset density, flux, LUFS, CLAP
  "slow"/"fast" anchor cosines). Keep the posterior; the reranker consumes `P(felt = T/2)`.

**Groove.** From the band-split onset envelopes at the chosen tactus: kick pattern class (syncopated
808 kick vs four-on-the-floor vs sparse), backbeat strength (mid-band energy on beats 2 and 4 vs 1
and 3), hits per bar at a given loudness, swing ratio (2:1 vs 1:1 subdivision), lag (mean onset
offset behind the grid). The kick-pattern binary alone removes Tauba Tauba, Bachke Bachke,
Millionaire and the Honey Singh remake from a Boyfriend query.

**Weight / darkness.** HPSS, keep the harmonic part. CQT, 36 bins per octave, C1 to C7, tuning
estimated, folded to 12-bin chroma, median over beat-synchronous frames (the 128-bin mel is wider
than a semitone below 500 Hz, which is why Krumhansl assigned 4 keys to 20 rock songs).
- `brightness` = loudness-normalised spectral centroid.
- `low_share` = energy below 200 Hz over total.
- `roughness` = Sethares/Vassilakis dissonance over the top partials, or spectral flatness fallback.
- `mode_polarity` = chroma[tonic+4] - chroma[tonic+3] over their sum, plus a small +9 vs +8 term;
  `mode_undefined = |polarity| < 0.15`.
- `darkness = 0.30 (1 - brightness) + 0.25 low_share + 0.25 roughness + 0.20 minor_polarity`,
  with the last term zero and its weight redistributed when `mode_undefined`.

**Texture / grit.** Whole-mix saturation estimate (harmonic distortion of the residual after HPSS, or
level-dependent clipping statistics), non-harmonic energy 4-10 kHz, drum transient-to-sustain
separation, mid/side ratio. This is the axis the earlier plan had no feature for; without it The
Sonics and Fall Out Boy score the same energy. Do not try to detect "produced to sound raw": the
Bang Data seed is a box record and belongs with Tidal Wave anyway.

**Intimacy / room.** Vocal-to-mix ratio (stem split; intimate records are vocal-dominant by 6 dB or
more), reverb decay after transients, stereo width, onset deviation from the grid, f0 steadiness
(natural drift vs stepped Auto-Tune), high-frequency transient energy where nothing is sounding
(breath, string noise). Items 1 to 3 are reliable in 30 s; the rest are reported with lower
confidence.

**Vocal delivery.** From the vocal stem or vocal band: syllable rate, sustain ratio, pitch range
used, vibrato, tuning depth (how tightly f0 snaps to semitones), harmonic-to-noise jumps at the
hook (stacking). Keep the existing `vocalPattern` label as a feature; it is the one thing that read
Simple Man correctly.

**Restraint vs saturation.** Short-term loudness range (EBU R128 style, 90th minus 10th percentile of
3 s windows) and crest factor, bucketed into restrained / moderate / saturated.

**Harmonic rhythm.** Cosine distance between successive beat-synchronous chroma frames: changes per
bar and loop length. Riff songs with no chords are flagged, not scored.

**Lyrics.** Topic is out. If any lyric feature is kept it is delivery: words per minute, repetition
ratio, syllables per note, line-end direction. These come from the vocal features above without
reading the words. Delete the keyword classifier from scoring (`backend/app/services/lyrics.py`
stays only if a client wants to display a theme badge).

### 3.3 Embedding: pretrained backbone plus a learned feel adapter

Stop training the 11.7M ResNet. With 1,656 same-track-crop positives it learned recording identity;
its 14 clusters collapse to 6 names because it has about six usable directions and most are genre.

- **Backbone:** LAION CLAP `larger_clap_music` (Apache 2.0, ~30M audio tower, about 0.4 s per 30 s
  on 2 vCPU). Frozen. 512-d per track, computed once offline. MERT-95M as an optional second input
  only if the owner accepts a non-commercial licence (section 9).
- **Text anchors:** 20 to 40 prompts across the axes in section 1, for example "slow heavy brooding
  guitar ballad that builds", "fast raw fuzzy garage rock", "bright melodic trap pop with
  auto-tuned vocals", "sparse intimate acoustic lament", "relentless distorted punk with shouted
  vocals". Cosine to each anchor is a label-free, interpretable feel vector. The user-facing
  "vibes" are the top anchor names; delete the KMeans archetypes (`ml/inference/archetypes.json`,
  `build_index.py:354-385`).
- **Adapter:** Linear 512 -> 128 (or a 2-layer MLP), trained with InfoNCE or triplet loss on the
  pairs in section 6. Training runs in minutes over cached embeddings. The index stores the 128-d
  output. Raw CLAP cosine is dominated by instrumentation and genre; never ship it as the score.

### 3.4 Online: retrieve wide, rerank, present

1. **Seed lookup** by id in the catalogue. Only if absent: one preview download, one CLAP call, one
   DSP pass (about 1.5 s), written back so it is absent only once. Mic identify stays as it is
   (Chromaprint + AcoustID) and then enters here.
2. **ANN k=500** over the adapter space. No filter of any kind: no language, genre, era, artist or
   popularity gate. Deezer related-artist walks and Last.fm similar-tracks leave the request path
   entirely; if the artist graph has value it becomes an offline feature ("graph distance") and the
   reranker decides whether it earns weight.
3. **Rerank** each of the 500 with a small model over these features:

   | Feature | Starting hand weight (until the reranker is learned) | Match type |
   |---|---|---|
   | adapter cosine | 0.22 | close |
   | felt tempo posterior distance (octave-folded) | 0.16 | compatible |
   | pulse rate distance | 0.06 | close |
   | groove: kick class match, backbeat, hits per bar, swing, lag | 0.12 | class equality + close |
   | darkness distance | 0.10 | close |
   | texture / grit distance | 0.10 | close |
   | vocal delivery distance | 0.10 | close |
   | intimacy / room distance | 0.06 | close |
   | restraint bucket | 0.04 | same bucket |
   | harmonic rhythm | 0.03 | same bucket |
   | mode polarity | 0.03 | compatible; undefined matches anything; major vs minor costs 0.5 |
   | language affinity (section 5) | 0.08 | prior |
   | same-artist flag, popularity prior, Deezer genre equality | features with 0 weight at start | the reranker may learn them; they are not allowed to become gates |

   No per-component floors. Every distance is a plain clip to [0, 1]. Replace the hand weights with
   logistic regression on the tune split once the gold set has 100 seeds, and with a 50-tree
   LightGBM once it has 200. A DSP feature whose learned weight is zero was never measuring feel;
   delete it then, not before.
4. **MMR** with lambda about 0.7, per-artist cap 2, and the language budget in section 5.
5. **Latency budget:** in-catalogue seed under 300 ms, cold seed under 3 s. Everything that makes
   today's request take 10-30 s (30 preview downloads, 30 ONNX runs, 30 LRCLIB calls, Last.fm tag
   calls, Deezer graph walks) is gone from the request path.

### 3.5 Why this gets the four sets

- Simple Man: tempo posterior puts mass at 60-70; groove says sparse backbeat, straight, behind;
  darkness high via low_share and roughness with mode undefined or weakly major; vocal delivery
  "restrained chest, sustained"; restraint bucket moderate. Levee, Stairway (from the drums) and
  Rooster sit in the same cell; Start Me Up (fast, light, on top of the beat), Miss a Thing (slow,
  light, pillow drums, chords every two beats), Enter Sandman (heavy, fast, tight) each miss on a
  different axis.
- Townes: intimacy high, vocal delivery talk-sung and sparse, restraint bucket restrained, harmonic
  rhythm slow loop. Mode is irrelevant so Me Gustas Tu survives; language costs it 0.70 x 0.08 of the
  score, which a clear feel match overcomes. Doc Watson will also come back, and should.
- Bang Data: pulse rate high, grit high, restraint bucket saturated, vocal buried; tempo is matched
  loosely because the posterior has mass at both 86 and 172. Tidal Wave shares all of that;
  Nickelback and Fall Out Boy fail on grit, Rush on groove steadiness, reggaeton on everything
  acoustic.
- Boyfriend: tempo 90-100 with the trap half-time template, kick class syncopated-808, hats at 16ths
  with rolls, vocal hard-tuned with hook stacking. Tauba Tauba fails the kick class, 295 fails
  vocal delivery, the bhangra remake fails low end and groove. Maan Meri Jaan, Mood, Peaches and
  Sunflower pass, and the language prior orders Punjabi first among equals, which is what the owner
  asked for.

---

## 4. Vibes shown to the user

Anchor-prompt names, not cluster names. Each track carries its top two anchors by CLAP cosine, with
names drawn from the axis vocabulary ("Slow and heavy, builds", "Sparse and intimate", "Raw and
driving", "Lean-back trap bounce"). Both clients render them with the colour map keyed on the axis
word (heavy, intimate, raw, bright, driving, warm). The `languageAffinity` badge stays.

---

## 5. Language: same first, leeway, not too far, as a prior

Language is not a feature of feel; it is a product prior on ranking. Three rules:

**Affinity table v2** (starting values; learned later, see section 6):

| Pair | Affinity |
|---|---|
| same language | 1.00 |
| Punjabi <-> Hindi <-> Urdu | 0.90 (shared producers, shared audience) |
| same group otherwise (indic_north, indic_south, romance, east_asia, germanic, arabic) | 0.85 |
| English <-> Punjabi/Hindi/Urdu | 0.70 (the drums came from Atlanta and Toronto) |
| English <-> Romance | 0.70 |
| English <-> East Asian | 0.60 |
| Romance <-> Arabic/Turkish/Persian | 0.60 |
| indic_north <-> indic_south | 0.55 |
| default cross-group | 0.55 |
| unknown on either side | 0.75 |

Unknown is high on purpose: untagged tracks are the old and foreign ones this feature exists to
surface. "Latin script with no tag" must NOT default to English; it stays unknown until a vocal
language classifier (section 6) labels it. Log the unknown rate.

**Budget, not gate.** Delete `min_language_affinity` (a drop is a gate by another name). Keep a
cross-language budget in MMR, but:
- it scales with supply: if fewer than 14 same-language candidates cleared the feel threshold, the
  budget opens rather than padding with same-language scene filler;
- it never evicts a cross-language track that beats the next same-language track by a clear margin
  (0.05 in rerank score); the budget breaks ties within a band, it does not override feel;
- a bilingual seed opens the budget by default.

**Listener-aware later.** The table encodes one listener's map. When listening history exists,
languages in the user's recent plays get a bump; this beats any hand-set number.

Implementation detail: two reranker features, `lang_affinity` in [0, 1] and `same_language` as a
binary, so the model can learn "same first" separately from "how far is too far".

---

## 6. Training signal and what gets learned

| Signal | Use |
|---|---|
| Gold set: owner's seeds grown to 200, 3-5 targets each, plus per-seed negatives | the only ground truth; adapter positives, reranker labels, octave labels (owner taps the tempo), evaluation. About 10 owner-hours with a pairwise picker UI |
| Deezer playlists: `/search/playlist?q=<mood word>` then `/playlist/{id}/tracks`, ~300 mood queries in en, es, hi, pa, ta | about 1M co-occurrence positives at scale; also the PMI source for learning the language affinity table |
| Spotify Million Playlist Dataset (research licence, 2018) | strong for English rock and folk, mapped by title+artist to Deezer ids; useless for 2023 Punjabi |
| LLM weak labels from title+artist (four axes on 1-5 plus confidence, two models must agree, only above a popularity threshold) | per-track weak feel vectors to filter co-playlist pairs and mine hard negatives; validated on the gold set before trust; about $30-50 for 50k tracks |
| Last.fm tags | a feature vector for English catalogue, never the objective |
| DSP descriptors | hard-negative mining only (same artist or same genre, far apart on feel), never positives |
| Vocal language classifier (e.g. a small Whisper language-id pass on the vocal stem) | language code with confidence, replacing tag and script guessing |

Adapter positives: co-playlist pairs whose weak vectors agree plus gold pairs. Hard negatives:
same-artist or same-genre pairs whose weak vectors disagree (Free Bird vs Simple Man is the
canonical one). Reranker: logistic regression on the tune split at 100 seeds, LightGBM at 200.
Octave classifier: logistic regression on tapped tempos. Language affinity: PMI from playlist
co-occurrence, regularised toward the table in section 5.

---

## 7. Evaluation protocol and benchmark v2

**Protocol**
- Gold set split by seed: 70% tune, 30% holdout. Holdout is read only at phase gates and the number
  is committed with the commit hash.
- Metrics: recall@10, recall@20, MRR, nDCG@20, each with Wilson intervals, so nobody tunes on noise
  (today one hit is 9 points of recall).
- Slices: "feel but not scene" (targets sharing no artist, decade or language with the seed:
  Townes -> Manu Chao, Bang Data -> Thee Oh Sees) and "scene but not feel" negatives (same artist or
  Deezer-related artist with a different feel: Free Bird, Tauba Tauba). The negative slice must stay
  out of the top 10. One Start Me Up in a Simple Man top 10 is a fail regardless of recall.
- Perception unit tests on golden audio fixtures, not on live API responses: felt tempo octave, LUFS
  band, darkness band, kick class.
- CI rule: a test greps every non-test source directory for every title and artist in the benchmark
  file and fails on any hit. The tree at `d8cb36e` fails this in `recommender.py:202-208`,
  `audio_features.py:189-195`, `language_affinity.py:20` and `harvest_deezer_previews.py:65-68`.
- Human checks a musician would respect, run at phase gates with 5 to 10 listeners, blind: tap test
  (seed and candidate tempos within 15%, never an octave apart); odd-one-out triads (seed, machine
  pick, human pick; machine pick called odd more than 3 in 10 is a fail); set-list test (would a
  band play the top 10 in a row without the room changing); cover test (the Shinedown and an
  acoustic Simple Man must land near the top, or the engine is matching production); "same record"
  test (would a human put this on the same record; this is how Doc Watson for Townes gets credited
  instead of punished).
- Operational: self-recommendation count 0, covers above originals 0, p95 latency.

**Benchmark v2 content.** Keep the owner's four seeds and eleven targets. Add, per seed, the
negatives and cross-world positives below (all human-picked by the panel; the owner should veto any
they disagree with). Sets 6-10 of the musicologist's additions are HOLDOUT ONLY.

Simple Man. Positives: Since I've Been Loving You, Tuesday's Gone, Comfortably Numb, The Thrill Is
Gone, Brothers in Arms, Fade to Black, Black (Pearl Jam), Creep, Like a Stone, Hurt (Johnny Cash),
Ohne Dich (Rammstein, de), Lamento Boliviano (Enanitos Verdes, es). Negatives: Free Bird, Sweet Home
Alabama, Start Me Up, Sharp Dressed Man, I Don't Want to Miss a Thing, Enter Sandman, Knockin' on
Heaven's Door, Let It Be, Every Breath You Take, Don't Stop Believin', Smoke on the Water.

Waiting Around to Die. Positives: Rising Sun Blues (Doc Watson), Los Ejes de Mi Carreta (Atahualpa
Yupanqui, es), Tous les garcons et les filles (Francoise Hardy, fr), Clandestino (Manu Chao, es),
Tum Itna Jo Muskura Rahe Ho (Jagjit Singh, hi/ur), Con Toda Palabra (Lhasa de Sela, es), Sodade
(Cesaria Evora), Something on Your Mind (Karen Dalton), Place to Be (Nick Drake), Between the Bars
(Elliott Smith), The Stranger Song (Leonard Cohen), Too Many Birds (Bill Callahan). Near miss,
texture right and stance wrong: Shake the Frost (Tyler Childers). Negatives: Don't Take the Girl,
Need You Now, I Don't Want to Miss a Thing, Someone Like You, Photograph (Ed Sheeran), Let Her Go,
Tum Hi Ho, any lo-fi study beat.

Bang Data. Positives: The Witch (The Sonics), Demolicion (Los Saicos, es), Les Cactus (Jacques
Dutronc, fr), A Minha Menina (Os Mutantes, pt), Jet Generation (Guitar Wolf, ja), Smokin' Billy
(Thee Michelle Gun Elephant, ja), Mr. P. Mosh (Plastilina Mosh), Mas (Kinky, es), Fuego (Bomba
Estereo, es), Kumbia Dark (Kumbia Queers, es), Girlfriend (Ty Segall), Afrique Victime (Mdou
Moctar). Negatives: All the Small Things, American Idiot, The Pretender, Enter Sandman, Sex on
Fire, Believer, Thunderstruck, Figure It Out (Royal Blood), How You Remind Me, Centuries, Tom
Sawyer, any reggaeton.

Boyfriend. Positives outside Punjabi: Maan Meri Jaan (King, hi), Mood (24kGoldn), Peaches (Justin
Bieber), Sunflower (Post Malone, Swae Lee), First Class (Jack Harlow), Hawai (Maluma, es), All I
Wanna Do (Jay Park, ko), Last Last (Burna Boy), Location (Khalid), Taste (Tyga), Yonaguni (Bad
Bunny, es), Fall (Davido). Positives inside: Softly, Admirin' You, Wavy (Karan Aujla), Lover
(Diljit Dosanjh), Tere Te (AP Dhillon), Still Rollin (Shubh), Summer High (AP Dhillon), Dil Dardeh
(Sukha). Negatives from the same scene: Tauba Tauba, Chitta Kurta, Bachke Bachke (Karan Aujla),
Elevated (Shubh), Excuses (AP Dhillon), 295 (Sidhu Moose Wala), G.O.A.T. (Diljit Dosanjh),
Millionaire (Yo Yo Honey Singh), Gur Nalo Ishq Mitha remake. The test: if the engine ranks the
outside positives above the same-scene negatives, it has learned feel; if the reverse, it is still
walking the graph.

Additional seed sets (musicologist), 1-5 for tuning, 6-10 holdout:
1. Hurt (Johnny Cash) -> Mad World (Gary Jules), The Night We Met (Lord Huron), Ne Me Quitte Pas (Brel, fr)
2. Tum Hi Ho (Arijit Singh, hi) -> All of Me (John Legend), Channa Mereya, Je Te Laisserai Des Mots (Patrick Watson, fr)
3. Mask Off (Future) -> XO Tour Llif3, Goosebumps, Brown Munde (AP Dhillon, pa)
4. Get Lucky (Daft Punk) -> Treasure, September, Plastic Love (Mariya Takeuchi, ja)
5. Tomboy (Hyukoh, ko) -> Chamber of Reflection, Ode to Viceroy (Mac DeMarco)
6. Seven Nation Army -> Song 2, Are You Gonna Be My Girl; NOT Pumped Up Kicks (same tempo, wrong weight)
7. Teardrop (Massive Attack) -> Glory Box, Breathe Me, Lovely
8. Skinny Love (Bon Iver) -> Flightless Bird American Mouth, Youth (Daughter), Poison Tree (Grouper)
9. Nothing Else Matters -> November Rain, Wind of Change, Still Loving You
10. Video Games (Lana Del Rey) -> Apocalypse (Cigarettes After Sex), Fade Into You, Space Song

Harvest seeds to add so these are even in the catalogue: Nuggets, Back From the Grave, In The Red,
Castle Face, Goner, Burger, Japanese garage, Tropicalia, Peruvian 60s, Argentine cumbia punk,
Americana, old-time, chanson, ghazal, Cape Verdean morna, Tuareg guitar, melodic trap in en/es/ko/
afrobeats, plus every artist named above.

---

## 8. What to delete from the tree at `d8cb36e`, in order

| Delete | Where | Replaced by |
|---|---|---|
| soundfile and PyAV decode paths | `recommender.py:984-1017` | one pinned ffmpeg call + LUFS normalisation |
| per-clip `ref=np.max` mel normalisation as the basis for loudness features | `recommender.py:1121` | LUFS-normalised audio; mel only feeds CLAP/tempogram |
| four octave rules and the seed-name comments | `recommender.py:199-213`, `audio_features.py:189-195` | tempo posterior (3.2) |
| `tempo_minor_bias` inside mode | `recommender.py:259` | darkness scalar with `mode_undefined` |
| energy = 50% felt tempo | `recommender.py:301-304` | LUFS, band-split onset rates, flux |
| `buildScore` and its 0.08 weight | `recommender.py:320-334`, `:958` | restraint bucket |
| `_synthesize_structural_meta_if_missing` and its non-neutral vocal defaults | `recommender.py:545-562` | measured features or feature absent; the reranker handles missing values |
| `genre_override` stamping and `_trusted_scene` | `deezer.py:434`, `routes.py:555` | nothing; genre string is a zero-weight feature |
| per-request preview download, ONNX, LRCLIB and Last.fm calls for candidates | `routes.py` evaluation block (`~:610-690`) and `_resolve_query` | offline features, ANN, reranker |
| Deezer related-artist and Last.fm candidate sources in the request path | `routes.py:530-540`, `deezer.py:387+`, `lastfm.py` | offline graph-distance feature, zero initial weight |
| `min_language_affinity` drop | `routes.py:395-400`, `:504` | budget rules in section 5 |
| lyric keyword classifier in scoring | `recommender.py:945-948`, `:961` | delivery features; `lyrics.py` becomes display-only or is removed |
| KMeans archetypes | `archetypes.json`, `build_index.py:354-385`, `recommender.py:1226-1229` | anchor-prompt vibes |
| own encoder training | `ml/training/`, `ml/models/music_encoder.py`, `export_onnx.py` | archive; CLAP + adapter |
| the "Phase 2.1 perception acceptance" numbers that were wrong | `VISION_ALIGNMENT.md` 2.1, 2.4, 3.3 | section 2 of this file |

---

## 9. Delivery and the decisions only the owner can make

**Weeks 1-2: a stable floor.** Gate: features reproducible in CI (golden fixtures), all 11 original
targets present in the catalogue, p95 latency under 3 s. Pinned decoder, LUFS, delete the request-
path candidate work, precompute CLAP + descriptors for the existing 1,656 and harvest to 20k via
Deezer playlist queries, ship ANN k=500 + hand-weighted rerank + MMR with the language budget.
Recall on 11 targets is still noise at this gate; do not chase it.

**Weeks 3-4: a signal.** Gate: holdout recall@20 and MRR on a 100-seed gold set, with intervals,
and the Boyfriend "outside positives above same-scene negatives" test passing. Pairwise picker UI,
gold set to 100 with tapped tempos, Deezer co-occurrence and LLM weak labels for the popular half,
train the adapter and the octave classifier, logistic-regression reranker, anchor-prompt vibes.

**Weeks 5-8: scale and learn the rest.** Gate: holdout recall@20 up on the "feel but not scene"
slice, negative slice out of top 10 on every tuning seed, human set-list test passed on five seeds.
Catalogue to 200k, learned affinity table, LightGBM reranker, MERT if licensed, both clients at
parity on the new fields (VISION_ALIGNMENT.md Phase 6 still applies).

**Owner decisions, needed before week 3:**
1. Licensing: permissive weights only (CLAP, MS-CLAP, beat_this) or allow non-commercial (MERT,
   Essentia, BeatNet)? Is caching Deezer and iTunes preview-derived features at index time acceptable
   under their terms?
2. Hosting ceiling: the free CPU Space covers 50k tracks and sub-300 ms in-catalogue queries; 200k+
   plus periodic re-embedding needs a paid Space or small VM plus about $10 of GPU per embedding pass.
3. Label budget: about 10 owner-hours for 200 seeds with tapped tempos, plus $30-50 of LLM weak
   labels. Without the gold set nothing in sections 3.4, 5 or 6 can be tuned honestly.

---

## 10. Where the panel disagreed, and which way this document goes

| Question | Positions | Resolution here |
|---|---|---|
| Is minor mode part of the feel? | Rock producer: chord colour matters, but concedes Simple Man is major. Musicologist, folk, Punjabi: mode is undefined or irrelevant on three of four sets. | Darkness scalar (brightness, low share, roughness) carries the weight; mode polarity 0.03, compatible match, off when undefined. |
| Dynamics / build | Rock producer: the whole-song loudness trajectory is the second most important feature. Musicologist, folk, ML: unmeasurable on a chorus preview, the current proxy is noise. | Dropped on previews; restraint bucket instead. Reintroduce only with a second window or full-track audio, as loudness-envelope shape, and then give it the rock producer's weight. |
| Felt tempo | Musicologist: evidence-based posterior. ML: learned octave classifier with posterior. Garage: pulse rate separate from BPM. Rock and Punjabi: the count is the kick-snare period. | All four, combined: band-split evidence, posterior output, pulse rate as a separate feature, classifier once tapped labels exist. |
| Language numbers | Folk: unknown to 0.75, listener-aware. Musicologist: en <-> pa 0.75, pa <-> ta 0.60. Punjabi: pa <-> hi 0.90 with Urdu, en <-> pa 0.70, pa <-> ta 0.55, cap scales with supply. Garage: default 0.50 on a 0.50 threshold is a wall. ML: no drop at all, learn from playlist PMI. | Table v2 in section 5 with no `min_affinity` drop, supply-scaled budget, margin rule, and PMI learning once playlist data exists. |
| Keep hand-weighted DSP in the score? | Musicologist supplies a weight table. ML: every DSP term is a reranker input whose learned weight is the evidence for keeping it. | Hand weights only until 100 gold seeds; then learned; a zero-weight feature is deleted. |
| Is Doc Watson for Townes a hit or a miss? | Folk: a hit, closer than two owner picks. The benchmark as written counts it as nothing. | Added as a positive in benchmark v2, and the "same record" human test exists so the engine is never trained to avoid the best answer. |

---

## 11. Zero-cost, no-labelling execution plan (the operative plan)

### 11.0 Scope and hard rules

Constraints from the owner: no paid services, no paid hosting, no licensing exercise, and no
labelling work by the owner. Everything below satisfies all four. The owner's only optional act is
reading the benchmark lists in 11.9 and deleting any song they disagree with.

Allowed dependencies (all permissive licences): `transformers` + `torch` (Apache 2.0 / BSD) for
LAION CLAP, `librosa` (ISC), `pyloudnorm` (MIT), `faiss-cpu` (MIT), `scikit-learn` (BSD), `numpy`,
`scipy`, `httpx`, `soundfile` is NOT used for decoding (ffmpeg only). Model weights: only
`laion/larger_clap_music` (Apache 2.0). Not allowed: MERT, Essentia models, BeatNet, madmom (any
non-commercial or unmaintained package), any paid API, any LLM labelling call, Spotify APIs.

Compute: all offline work (harvest, feature extraction, embedding, training, index build) runs on
the owner's Mac. Serving runs on the existing free Hugging Face CPU Space `shreyash616/echo-backend`
with artifacts on the existing HF dataset repo `shreyash616/echo-ml-artifacts` (free; the owner's
`HF_TOKEN` already exists for deploys).

Rules the agent must follow, in addition to VISION_ALIGNMENT.md section 6:
1. No benchmark title or artist may appear anywhere outside `tests/`. 11.9 adds a CI test for this.
2. No rule, threshold or weight may be set by looking at a specific benchmark song. If you find
   yourself writing a comment like "(Simple Man)" next to a number, stop.
3. No candidate gate of any kind: not language, genre, era, artist, popularity or "trusted".
4. No audio download, encoder call, lyrics call, Last.fm call or Deezer graph call inside the
   request path, except the single cold-seed path in 11.7.
5. Every feature number must be reproducible in CI on synthetic fixtures (11.2) before it enters
   the index.
6. Gold data (11.9) is for measurement only. Nothing in 11.8 trains on it.
7. Commit the holdout number together with the commit hash every time it is read, under
   "Progress" at the end of this file.

Order of execution: 11.1, 11.2, 11.3 (decision point), 11.4, 11.5, 11.6, 11.7, 11.9, 11.8, then
iterate 11.8 and 11.9. Do not reorder. Do not start 11.4 before 11.3 has produced its number.

### 11.1 Environment

Create `ml/requirements-offline.txt` (Mac, offline work) and extend `backend/requirements.txt`
(Space, serving). Pin versions; the versions below are known-good as of 2026-10; if a pin fails to
resolve, pick the nearest newer patch release and record it.

```
# ml/requirements-offline.txt
torch==2.4.1
transformers==4.45.2
librosa==0.10.2.post1
pyloudnorm==0.1.1
faiss-cpu==1.9.0
scikit-learn==1.5.2
numpy==1.26.4
scipy==1.14.1
httpx==0.27.2
huggingface_hub==0.25.2
tqdm==4.66.5
```

Serving adds `transformers`, `torch` (CPU wheel), `pyloudnorm`, `faiss-cpu` to
`backend/requirements.txt` and removes `onnxruntime`, `av`, `soundfile` and anything only used by
the deleted code paths. `ffmpeg` is pinned in `backend/Dockerfile` (install the distro package and
record `ffmpeg -version` output in `backend/FFMPEG_VERSION.txt`); the same ffmpeg major version must
be installed on the Mac (`brew install ffmpeg`).

Create `backend/app/audio/decode.py` with ONE decoder used by both the Mac scripts and the Space:

```python
FFMPEG_ARGS = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0",
               "-ac", "1", "-ar", "22050", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]

def decode_preview(audio_bytes: bytes) -> np.ndarray:
    # returns mono float32 at 22050 Hz, loudness-normalised to -23 LUFS with pyloudnorm,
    # peak-limited to 0.99, trimmed to the first 30.0 s, zero-padded if shorter
```

`ml/` imports this module (add `backend` to `sys.path` in the ml scripts, as
`tests/feel_benchmark.py` already does). CLAP needs 48 kHz input; resample from the 22050 Hz
normalised signal with `librosa.resample(..., res_type="soxr_hq")` inside the embedding function
only, so that every DSP feature and the embedding start from the same normalised audio.

### 11.2 Step 0: freeze measurement (deletions and golden fixtures)

Deletions, all at `d8cb36e` line numbers (re-grep before editing):

| Delete | Where |
|---|---|
| soundfile + PyAV + ffmpeg fallback chain | `backend/app/services/recommender.py:984-1042`; replace the call site with `decode_preview` |
| four octave rules and the seed-name comments | `recommender.py:189-216` and `ml/inference/audio_features.py:92-203` (the whole `estimate_tempo_from_mel`); replaced by 11.5.1 |
| `tempo_minor_bias` and the mode computation | `recommender.py:255-261`, `audio_features.py:245-248`; replaced by 11.5.3 |
| energy formula with felt tempo in it | `recommender.py:298-306`, `audio_features.py:303-318`; replaced by 11.5.2 and 11.5.6 |
| `buildScore` / `loudnessRange` block | `recommender.py:320-334`, `audio_features.py:321-335`; `loudnessRange` is recomputed in 11.5.6 from LUFS windows |
| `_synthesize_structural_meta_if_missing` | `recommender.py:545-562` entire function and its callers |
| `genre_override` stamping | `deezer.py:434` (pass no override), `:68`, `:73` |
| `_trusted_scene` | `routes.py:555`, `:627-630`, `:641` |
| `min_language_affinity` and its drop | `routes.py:385-400`, `:504`, `:866-867` |
| lyric term in the composite and the LRCLIB call in the request path | `recommender.py:945-948`, `:961`; `routes.py` wherever `fetch_lyrics` / `classify_lyrical_intent` is awaited |
| per-candidate preview download + encode + Last.fm tags in the request path | the `_evaluate_candidate` body in `routes.py` (~`:610-690`); the whole function goes in 11.7 |
| Deezer related-artist and Last.fm candidate sources in the request path | `routes.py:530-540`; `get_genre_and_language_candidates`, `get_similar_tracks` are no longer called online |
| KMeans archetypes | `ml/inference/build_index.py:335-385`, `recommender.py:1184-1193`, `:1224-1229`, `ml/inference/archetypes.json` |
| ONNX encoder load and `ml/training/`, `ml/models/`, `ml/inference/export_onnx.py` | `recommender.py:1150-1170`; move the ml training code to `ml/archive/` with a README line saying why |

The composite score function, `recommend()`, and `_find_and_rank_similar_tracks` are rewritten in
11.7, so do not try to patch them in place; delete and replace.

Golden fixtures (no copyrighted audio; everything is synthesised so the expected values are known):
create `tests/fixtures/make_fixtures.py` that writes ten 30 s WAVs at 22050 Hz into
`tests/fixtures/audio/` and an `expected.json`:

| Fixture | Signal | Expected (tolerance in brackets) |
|---|---|---|
| `click_72_backbeat.wav` | kick (60 Hz sine burst, 80 ms) on every beat at 72 BPM, snare (white-noise burst, 60 ms, 2 kHz band-passed) on beats 2 and 4, hats (8 kHz noise, 20 ms) on eighths | `felt_bpm` argmax 72 (+-2), `P(felt in {70..74}) >= 0.6`, `pulse_rate` mid, `kick_class = "sparse_backbeat"` |
| `click_144_fourfloor.wav` | kick on every beat at 144, hats on eighths, no snare | `felt_bpm` 144 (+-3), `kick_class = "four_on_floor"` |
| `trap_96.wav` | kick on 1 and the "and" of 2, snare on 3, hats at 16ths with a triplet roll every 2 bars, 50 Hz 808 with 400 ms decay | `felt_bpm` 96 (+-2), `kick_class = "syncopated_808"`, `low_share >= 0.35`, `hat_subdivision = 16` |
| `major_triad_pad.wav` | C major triad (C4 E4 G4 sines) sustained | `mode_polarity > 0.3`, `mode_undefined = false`, `harmonic_changes_per_bar = 0` |
| `minor_triad_pad.wav` | C minor triad | `mode_polarity < -0.3` |
| `power_chord_pad.wav` | C3 + G3 + C4 sines | `mode_undefined = true` |
| `pink_quiet.wav` | pink noise at -30 LUFS | after normalisation `lufs = -23 (+-0.5)`, `crest_factor` high, `restraint_bucket = "restrained"` |
| `pink_clipped.wav` | pink noise hard-clipped at 0.3 then normalised | `saturation >= 0.5`, `restraint_bucket = "saturated"` |
| `bright_vs_dark` pair | band-limited noise 2 to 8 kHz vs 60 to 400 Hz | `brightness` high vs low, `darkness` low vs high |
| `chord_cycle_2s.wav` | C, F, G, C triads changing every 2 s at 120 BPM (one change per bar) | `harmonic_changes_per_bar = 1 (+-0.2)` |

`tests/test_features_golden.py` decodes each fixture through `decode_preview` and asserts every
expected value. CI runs it inside the same Docker image the Space uses. 11.5 is not done until this
test passes on the Mac AND in CI with identical numbers (store the Mac numbers in `expected.json`
to 1e-3 and assert equality in CI, not just the ranges).

### 11.3 Step 1: the CLAP decision experiment

Create `ml/experiments/clap_sanity.py`. Purpose: decide, before harvesting anything, whether the
frozen CLAP space already orders the panel's positives above the panel's negatives.

1. Load `tests/feel_benchmark_v2.json` (11.9). For every seed, positive and negative: resolve a
   preview by calling the backend's existing `search_track` (Deezer then iTunes) with
   `"{title} {artist}"`, pick the first result whose normalised artist contains the expected artist
   and whose title has no "cover", "karaoke", "tribute", "live", "remix", "(". Log the ones that do
   not resolve; they are excluded, not substituted.
2. Download the preview, `decode_preview`, resample to 48 kHz, embed with
   `ClapModel.from_pretrained("laion/larger_clap_music")` + `ClapProcessor` (`get_audio_features`),
   L2-normalise. Cache embeddings in `ml/experiments/cache/clap_<deezer_or_itunes_id>.npy`.
3. Embed the anchor prompts in 11.6.2 with `get_text_features`.
4. For every seed report: mean cosine to its positives, mean cosine to its negatives, and the
   pairwise win rate (fraction of (positive, negative) pairs where cos(seed, pos) > cos(seed, neg)).
   Also report, per seed, the top 3 anchor prompts for the seed and for each positive.
5. Write `ml/experiments/clap_sanity_results.json` and print a table.

Decision rule, applied by the agent without asking:
- Overall win rate >= 0.70 and every seed >= 0.55: CLAP is the backbone. Proceed to 11.4.
- Overall win rate in [0.55, 0.70): proceed, but set the initial `clap_cosine` weight in 11.7 to
  0.12 instead of 0.22 and move the difference onto `groove` and `grit`.
- Overall win rate < 0.55: stop after 11.4 and 11.5 (catalogue and DSP are still needed) and write
  under "Progress" that the embedding must be replaced; the DSP-only rerank is then shipped in
  11.7 with `clap_cosine` weight 0 and `adapter_cosine` weight 0.

### 11.4 Step 2: harvest 20k to 30k tracks on the Mac

Create `ml/data/harvest_v2.py`. It replaces `harvest_deezer_previews.py` (keep the old file until
the new one has produced a full catalogue, then delete it).

Sources, in this order, until `--target` (default 25000) unique tracks:

1. **Deezer mood playlists.** For each (mood word, language) pair: `GET https://api.deezer.com/search/playlist?q=<word>&limit=25`,
   then for each playlist `GET /playlist/{id}/tracks?limit=100` (follow `next`). Record every
   (playlist_id, track_id) pair in `ml/data/playlist_membership.jsonl` even for tracks you do not
   keep, because 11.8 needs the co-occurrence. Mood words (use all, in every listed language;
   translate literally, the search is lexical):
   - en: sad, melancholy, heartbreak, chill, mellow, calm, acoustic, late night, rainy day, slow,
     intimate, lonely, nostalgic, dark, heavy, brooding, epic, driving, workout, hype, party,
     garage, punk, psych, fuzz, lo-fi, dreamy, ambient, folk, blues, soul, groove, bounce, trap,
     drill, afrobeats, reggaeton, cumbia, indie, shoegaze, post punk, classic rock, southern rock,
     grunge, metal ballad, power ballad, road trip, sunset, morning coffee, study, sleep
   - es: triste, melancolia, desamor, relajado, tranquilo, acustico, noche, lento, oscuro, pesado,
     fiesta, garage, punk, psicodelico, cumbia, reggaeton, trap, indie, rock en espanol, balada,
     nostalgia, carretera
   - hi: udaas (sad), dard, ishq, romantic, sufi, ghazal, chill, lofi, bollywood sad, bollywood
     romantic, old hindi, retro, night drive, breakup, slow, soulful
   - pa: sad punjabi, punjabi romantic, punjabi trap, punjabi hip hop, punjabi chill, bhangra,
     punjabi folk, punjabi lofi, late night punjabi
   - ta/te: tamil melody, tamil sad, tamil lofi, tamil love, telugu melody, telugu sad, ilaiyaraaja,
     ar rahman melody
   - ko/ja/pt/fr: k-indie, korean rnb, korean hip hop, city pop, j-rock, japanese garage, mpb,
     bossa nova, tropicalia, chanson, french pop, french indie
2. **Deezer charts**: `GET /chart/0/tracks?limit=100` plus `/chart/{country_id}/tracks` for the
   Deezer editorial ids of SG, MY, PH, IN, US, GB, ES, MX, BR, FR, JP, KR (look them up with
   `GET /editorial`).
3. **Artist seeds**: for each artist in the list below, `GET /search/artist?q=` then
   `/artist/{id}/top?limit=50` and `/artist/{id}/related?limit=20` one level deep. Artists:
   every artist named in section 7 of this file, plus the harvest seeds listed at the end of
   section 7. (These are discovery seeds; they add the worlds, they do not add labels.)
4. **iTunes fallback**: for any artist in step 3 with no Deezer results or no previews (Led
   Zeppelin), `https://itunes.apple.com/search?term=<artist>&entity=song&limit=200`, keep
   `previewUrl` entries, id prefixed `it_` as the backend already does.

Rules: at most 50 Deezer requests per 5 s (sleep to comply; Deezer returns 4 "Quota limit
exceeded" when you exceed it; back off 10 s on that). Dedup on normalised `artist:title` (same
normaliser as `routes.py:79-85`). Keep only tracks with a preview URL. Store per track:
`id, title, artist, album, year (Deezer release_date year if present), deezer_genre (album genre
name if present), language_hint (script-based only: Devanagari hi, Gurmukhi pa, Tamil ta, Telugu
te, Hangul ko, Kana/Kanji ja, Arabic ar, Cyrillic ru; Latin script = unknown), preview_url,
popularity (Deezer rank), source (playlist|chart|artist|itunes)`.

Feature extraction is in the same script with `--extract`: for each kept track download the
preview to a temp dir, `decode_preview`, compute 11.5 descriptors and the CLAP embedding, append
one row to `ml/data/catalog_features.jsonl` (one JSON object per line, fields in 11.5.9), delete
the audio. Checkpoint every 500 tracks; `--resume` skips ids already present. Expected throughput on
an M-series Mac: 2 to 4 tracks per second for DSP, 2 to 3 per second for CLAP; 25k tracks is a few
hours. Expected transient download: about 12 GB total, never more than a few hundred MB on disk at
once.

Done when: `catalog_features.jsonl` has >= 20000 rows, every title in `tests/feel_benchmark_v2.json`
that resolved in 11.3 is present (check by id), and `playlist_membership.jsonl` has >= 200k rows.

### 11.5 Feel descriptors: exact specification

All functions live in `backend/app/audio/features.py` (new) and take the `decode_preview` output
(mono float32, 22050 Hz, -23 LUFS, 30 s). The ml scripts import the same module. Frame settings:
`n_fft=2048`, `hop=512`, `n_mels=128` for mel; CQT `fmin=C1 (32.7 Hz)`, `bins_per_octave=36`,
`n_bins=216` (6 octaves). Every function is pure and deterministic (no randomness, fixed
`librosa` resampling type).

**11.5.1 Felt tempo posterior** `felt_tempo(y) -> dict`
1. Onset envelopes: `librosa.onset.onset_strength_multi` with three band groups on the mel:
   low = mel bins covering < 150 Hz, mid = 150 Hz to 2 kHz, high = > 4 kHz; plus fused = all bins.
2. Autocorrelation tempogram per envelope (`librosa.feature.tempogram`, `win_length=384`),
   averaged over time to one curve per band, converted to a BPM axis 40 to 240.
3. Tatum = BPM of the highest peak of the fused curve above 120 (if none above 120, the highest
   peak overall). Tactus candidates `C = {tatum / k for k in [1, 2, 3, 4, 6, 8]}` kept within
   [40, 240].
4. Evidence per candidate `c` (each in [0, 1], computed from the curves and envelopes):
   - `clarity_c` = fused curve value at `c` divided by the curve maximum.
   - `low_clarity_c` = low-band curve value at `c` divided by its maximum.
   - `backbeat_c` = from the mid-band envelope sampled at beat positions of `c` (phase chosen by
     maximum mean energy): the ratio of mean energy on even beats vs odd beats, mapped so that a
     strong 2-and-4 pattern gives ~1.0, a flat pattern gives 0.5. Compute also `backbeat_half_c`
     for a snare every four beats; use `max(backbeat_c, backbeat_half_c)`.
   - `density_c` = mean number of fused onsets (peaks above 0.3 of max) per beat of `c`, mapped:
     1 to 3 onsets per beat -> 1.0, 4 to 6 -> 0.5, > 6 -> 0.2, < 1 -> 0.4.
   - `bar_flux_c` = fused-curve value at `c / 4` divided by the maximum (chord changes on bars).
   - `prior_c` = `exp(-0.5 * (log2(c / 110) / 0.6)**2)` (weak resonance prior).
5. Score `s_c = 0.30 clarity + 0.15 low_clarity + 0.25 backbeat + 0.15 density + 0.10 bar_flux +
   0.05 prior`, posterior `p_c = softmax(s_c / 0.08)` over `C`.
6. Return `{"felt_bpm": argmax, "felt_bpm_dist": {str(round(c)): p_c}, "tatum_bpm": tatum,
   "pulse_rate": fused onsets per second normalised by 12 (clip to [0,1]),
   "hat_subdivision": 8 or 16 (16 if the high-band curve peaks at >= 3.5x the felt bpm, else 8),
   "pulse_clarity": clarity at argmax}`.
   Tempo distance between two tracks (used in 11.7):
   `d = sum_i sum_j p_i q_j * fold(|log2(bpm_i / bpm_j)|)` with `fold(x) = min(x, 0.4) if x
   rounds to 1.0 else (0.7 if x rounds to log2(1.5) else min(1, x))`. `s_tempo = 1 - d`.

**11.5.2 Groove** `groove(y, felt) -> dict`
- `kick_class`: from the low-band onset pattern at the felt tactus over the 30 s: "four_on_floor"
  if >= 85% of beats have a low onset; "sparse_backbeat" if <= 55% of beats have a low onset and
  `backbeat` >= 0.7; "syncopated_808" if low onsets fall off-beat >= 30% of the time and
  `low_share` (11.5.3) >= 0.3; else "other".
- `backbeat` (from 11.5.1), `hits_per_bar` = fused onsets per 4 beats, `swing` = ratio of the
  second-eighth offset to the beat period (0.5 straight, ~0.67 shuffle), `lag_ms` = mean signed
  offset of fused onsets from the nearest grid beat.

**11.5.3 Weight / darkness** `darkness(y) -> dict`
- HPSS (`librosa.effects.hpss`), keep harmonic `yh`.
- CQT on `yh`, tuning estimated with `librosa.estimate_tuning`, fold to 12-bin chroma, median over
  beat-synchronous frames (beats from 11.5.1 argmax).
- Tonic: Krumhansl-Schmuckler on the 12-bin chroma (major and minor profiles), `tonal_stability` =
  best correlation minus second-best.
- `mode_polarity = (chroma[t+4] - chroma[t+3]) / (chroma[t+4] + chroma[t+3] + 1e-6)` plus
  `0.3 * (chroma[t+9] - chroma[t+8]) / (...)`; `mode_undefined = abs(mode_polarity) < 0.15 or
  tonal_stability < 0.05`.
- `brightness` = spectral centroid of `y` divided by 4000 Hz, clipped to [0, 1].
- `low_share` = energy below 200 Hz over total energy (from the STFT power).
- `roughness` = spectral flatness of `yh` in the 200 Hz to 2 kHz band (0 pure, 1 noisy), as the
  cheap stand-in for a dissonance model; document it as such.
- `darkness = 0.30 (1 - brightness) + 0.25 low_share + 0.25 roughness + 0.20 minor_term`, where
  `minor_term = clip(-mode_polarity, 0, 1)` if not undefined else redistribute (divide the first
  three weights by 0.8).

**11.5.4 Texture / grit** `grit(y) -> dict`
- `saturation`: fraction of 20 ms windows whose sample distribution kurtosis is below 2.0 (flat-
  topped) combined with the fraction of windows with |sample| > 0.95, `saturation = 0.5 * flat +
  0.5 * clip_frac * 10` clipped to [0, 1].
- `hf_hash` = energy 4 to 10 kHz of the percussive part `yp` divided by total energy of `yp`.
- `transient_sep` = mean (peak / mean) ratio of the percussive envelope at low-band onsets; low
  means drums inside the guitars.
- `width` = 0.0 on mono previews (document that Deezer previews are stereo MP3; if the decoder
  is extended to stereo later, use mid/side energy ratio).
- `grit = 0.45 saturation + 0.35 hf_hash + 0.20 (1 - transient_sep_norm)`.

**11.5.5 Intimacy / room and vocal delivery proxies (no stem separation)** `vocal(y) -> dict`
- Vocal band proxy: harmonic part band-passed 200 Hz to 3.5 kHz; `vocal_ratio` = its energy over
  the energy of everything else, in dB, mapped to [0, 1] over [-12, +12] dB.
- `reverb_tail`: mean decay time (ms) after fused onsets for the energy to fall 20 dB, clipped at
  800 ms, normalised.
- `timing_loose` = std of `lag_ms` from 11.5.2 normalised by the beat period.
- `pitch_steady`: `librosa.pyin` on the vocal band (`fmin=80, fmax=800`); `tuning_depth` = fraction
  of voiced frames within 15 cents of a semitone; `vibrato` = std of f0 in cents within voiced
  runs; `pitch_range` = 90th minus 10th percentile of voiced f0 in semitones.
- `syllable_rate` = voiced-run onsets per second (a run starts when voicing flag turns on after
  >= 60 ms off); `sustain_ratio` = fraction of voiced frames in runs longer than 400 ms.
- `breath_noise` = high-frequency (> 6 kHz) energy in frames where the fused onset envelope is
  below 0.1 of max, over the track's total HF energy.
- `intimacy = 0.35 vocal_ratio + 0.25 (1 - reverb_tail) + 0.15 timing_loose + 0.15 (1 -
  tuning_depth) + 0.10 breath_noise`.

**11.5.6 Restraint vs saturation** `restraint(y) -> dict`
- `lufs` (integrated, pyloudnorm, after normalisation this is -23; store the PRE-normalisation
  value as `lufs_raw` for popularity-era effects), `loudness_range` = 90th minus 10th percentile
  of 3 s short-term loudness, `crest_factor` = peak over RMS in dB over the whole clip.
- `restraint_bucket` = "restrained" if `loudness_range >= 8 LU and crest_factor >= 14 dB`,
  "saturated" if `loudness_range <= 4 LU and crest_factor <= 9 dB`, else "moderate".
- `energy` = `0.40 pulse_rate + 0.30 clip((flux_mean - 0.02) / 0.03, 0, 1) + 0.30 (1 -
  crest_norm)`, where `flux_mean` is the mean spectral flux of the normalised signal and
  `crest_norm` maps 6..20 dB to 0..1. Tempo is NOT in this formula.

**11.5.7 Harmonic rhythm** `harmonic_rhythm(y, beats) -> dict`
- Beat-synchronous chroma from 11.5.3; `changes_per_bar` = number of frames where cosine distance
  to the previous frame > 0.35, per 4 beats; `loop_bars` = lag (in bars, 1 to 8) of the maximum
  autocorrelation of the chroma sequence. `riff_flag = tonal_stability < 0.05`.

**11.5.8 Language code** `language(meta) -> (code | None, confidence)`
- Script-based only (as in 11.4). Latin script with no other evidence returns `None` with
  confidence 0. Do not default to English. Deezer `album.genre` names that are language names
  ("Bollywood", "Punjabi", "Tamil", "K-Pop", "J-Pop", "Latin") map to codes with confidence 0.6.
  Last.fm is not called.

**11.5.9 Per-track record** (one JSON object per line in `catalog_features.jsonl`, and the same
fields in `track_metadata.json` for the Space):
`id, title, artist, album, year, popularity, source, preview_url, language, language_conf,
felt_bpm, felt_bpm_dist, tatum_bpm, pulse_rate, hat_subdivision, pulse_clarity, kick_class,
backbeat, hits_per_bar, swing, lag_ms, mode_polarity, mode_undefined, tonal_stability,
brightness, low_share, roughness, darkness, saturation, hf_hash, transient_sep, grit, vocal_ratio,
reverb_tail, timing_loose, tuning_depth, vibrato, pitch_range, syllable_rate, sustain_ratio,
breath_noise, intimacy, lufs_raw, loudness_range, crest_factor, restraint_bucket, energy,
changes_per_bar, loop_bars, riff_flag, clap (512 floats, stored in a separate .npy keyed by row
order to keep the jsonl small), anchors (dict prompt -> cosine), vibes (top 2 anchor names)`.

### 11.6 Build and publish the index

**11.6.1** `ml/inference/build_index_v2.py`: read `catalog_features.jsonl` + `clap.npy`;
L2-normalise; `faiss.IndexFlatIP(512)` (or 128 after 11.8); write `music_index.faiss`,
`track_metadata.json` (all 11.5.9 fields except the raw clap vector), `anchors.json` (prompt ->
text embedding). Upload with `ml/upload_artifacts.py` (extend its file list; remove
`archetypes.json` and `music_encoder.onnx`). `backend/startup.sh` downloads the new set and fails
the container if any is missing.

**11.6.2** Anchor prompts (`ml/inference/anchor_prompts.json`); the display name is the key:
```
"Slow and heavy, builds":        "a slow heavy brooding rock ballad with electric guitars that starts quiet and builds"
"Slow and heavy, grinding":      "a slow heavy grinding blues rock groove with a huge drum sound"
"Sparse and intimate":           "a sparse intimate acoustic song, one voice close to the microphone, resigned and warm"
"Hushed and melancholic":        "a hushed melancholic song with gentle repetitive accompaniment"
"Laid-back and sunny":           "a laid back sunny acoustic groove with a relaxed vocal"
"Raw and driving":               "fast raw fuzzy garage rock with shouted vocals and distorted guitars"
"Psychedelic and driving":       "driving psychedelic rock with reverb drenched vocals and fuzz guitar"
"Punk energy":                   "loud fast punk rock with aggressive vocals"
"Lean-back trap bounce":         "mid tempo melodic trap with auto tuned vocals, 808 bass and rolling hi hats"
"Dark trap flex":                "dark minor key trap with heavy 808s and confident rapped vocals"
"Bright pop bounce":             "bright upbeat pop song with a catchy tuned vocal hook"
"Dance floor":                   "four on the floor dance track with a driving kick drum"
"Bhangra lift":                  "energetic bhangra with dhol drums and a soaring vocal"
"Soulful ballad":                "a soulful slow ballad with a powerful belted vocal and strings"
"Dreamy haze":                   "dreamy hazy shoegaze or dream pop with washed out guitars"
"Lo-fi drift":                   "mellow lo-fi hip hop beat with dusty drums and no vocals"
"Blues ache":                    "slow aching electric blues with expressive guitar"
"Folk story":                    "a folk song telling a story over fingerpicked guitar"
"Cinematic and dark":            "dark cinematic song with twangy guitar and a sultry vocal"
"Reggae skank":                  "laid back reggae groove with offbeat guitar skank"
"Epic metal ballad":             "a metal ballad with clean arpeggios building to heavy distorted guitars"
"Afro swing":                    "afrobeats groove with syncopated percussion and a melodic vocal"
"Reggaeton":                     "reggaeton with a dembow rhythm and tuned vocals"
"Cumbia punk":                   "cumbia rhythm played loud and raw by a punk band"
```
`vibes` for a track = the two highest-cosine keys. These names replace the archetype names in
both clients; the colour map keys on the first word.

### 11.7 Step 3: the online path

Rewrite `backend/app/services/recommender.py` around three functions and delete the rest:

```python
def load() -> None                        # faiss index, metadata, anchors, CLAP model (CPU)
def describe(audio_bytes) -> dict         # decode_preview -> 11.5 descriptors + clap + anchors
def recommend(seed: dict, k: int = 20, cross_language_max: int = 6,
              diversity: float = 0.7) -> list[dict]
```

`routes.py` keeps the same endpoints and the same `TrackResult` schema (add the new descriptor
fields; keep `vibes`, `matchScore`, `matchReasons`, `languageAffinity`), so neither client breaks.

`recommend(seed)`:
1. **Seed record.** If `seed["id"]` is in the catalogue, use its stored record. Otherwise
   (`/identify` on a track not in the index, or a search hit not in the index): download that one
   preview, `describe()` it (about 1.5 to 2.5 s on the Space CPU), and append it to an in-memory
   `extra_records` dict plus a `faiss.IndexFlatIP` side index, so the same seed is never described
   twice in a process lifetime. This is the only network and audio work allowed online.
2. **Retrieve.** `index.search(seed_clap, 500)` (or the adapter vector after 11.8). No filtering.
   Also search the side index. Union, drop the seed id and any record whose normalised
   `artist:title` equals the seed's.
3. **Rerank.** For each candidate compute the features below and the score; initial weights from
   the table (sum 1.00). After 11.8 these come from `ml/inference/rerank_weights.json`, loaded at
   startup, so changing weights never needs a code change.

   | Feature | How | Initial weight |
   |---|---|---|
   | `clap_cosine` | dot of L2-normalised clap vectors, clipped [0, 1] | 0.22 (see 11.3 rule) |
   | `s_tempo` | 11.5.1 distance on the two posteriors | 0.16 |
   | `s_pulse` | `1 - |pulse_rate_a - pulse_rate_b|` | 0.06 |
   | `s_groove` | `0.4 [kick_class equal] + 0.2 (1 - |backbeat diff|) + 0.2 (1 - |hits_per_bar diff| / 8) + 0.1 (1 - |swing diff| / 0.2) + 0.1 (1 - |lag diff| / 40ms)` | 0.12 |
   | `s_dark` | `1 - |darkness diff|` | 0.10 |
   | `s_grit` | `1 - |grit diff|` | 0.10 |
   | `s_vocal` | mean of `1 - |diff|` over `syllable_rate/8, sustain_ratio, pitch_range/24, tuning_depth, vibrato_norm` | 0.10 |
   | `s_intimacy` | `1 - |intimacy diff|` | 0.06 |
   | `s_restraint` | 1.0 same bucket, 0.5 adjacent, 0.0 restrained vs saturated | 0.04 |
   | `s_harm` | `0.5 (1 - |changes_per_bar diff| / 4) + 0.5 [loop_bars equal]`; 0.5 if either `riff_flag` | 0.03 |
   | `s_mode` | 1.0 if either `mode_undefined`; else `1 - 0.5 * |polarity_a - polarity_b|` | 0.03 |
   | `s_language` | `language_affinity(seed_lang, cand_lang)` from 11.7.1 | 0.08 |
   | `same_artist`, `popularity`, `deezer_genre_equal` | computed and logged, weight 0.0 | 0.00 |

   `score = sum(w_i * s_i)`, clipped to [0.05, 0.99]. No floors anywhere. `matchReasons` = the
   three features with the highest `w_i * s_i`, rendered as short strings ("Felt tempo", "Groove",
   "Grit"...). Keep the `matchScore` field.
4. **MMR + language budget.** Standard MMR with `lambda = diversity` over the rerank score and the
   clap cosine between picked items; per-artist cap 2 (same normaliser). Language budget:
   - `same_count` = number of candidates in the top 60 by score with affinity 1.0.
   - `budget = cross_language_max` if `same_count >= 14` else `k` (budget opens when supply is thin).
   - When the budget is spent, a cross-language candidate is still picked if its score exceeds the
     best remaining same-language candidate by >= 0.05.
   - There is no minimum affinity and no drop.
5. Return `k` records with `matchScore`, `matchReasons`, `vibes`, `languageAffinity` and all
   descriptor fields.

**11.7.1 Language affinity v2**: replace the tables in `backend/app/services/language_affinity.py`
with section 5 of this file (same 1.00; pa/hi/ur 0.90; same group 0.85; en<->indic_north 0.70;
en<->romance 0.70; en<->east_asia 0.60; romance<->arabic 0.60; indic_north<->indic_south 0.55;
default 0.55; unknown on either side 0.75). Delete the benchmark-pair comment. After 11.8.3 the
values are loaded from `ml/inference/language_affinity.json` if present.

**Latency gate**: `/api/recommendations/{id}` for an in-catalogue id under 300 ms p95 on the
Space; cold seed under 4 s. Measure with `tests/test_live_api.py` extended to time 20 calls.

### 11.8 Learning without labels (all automatic, all on the Mac)

**11.8.1 Adapter from playlist co-occurrence.** `ml/training/train_adapter.py`:
- Positives: pairs of tracks that share at least one Deezer playlist in `playlist_membership.jsonl`,
  restricted to tracks present in the catalogue; weight a pair by `1 / sqrt(playlist_size)` so
  5,000-track playlists do not dominate. Drop playlists with > 2,000 tracks.
- Model: `Linear(512, 128)` on the frozen clap vector, L2-normalised output. Loss: InfoNCE,
  temperature 0.07, in-batch negatives, batch 1024, AdamW lr 1e-3, 30 epochs, early stop on a
  held-out 5% of playlists (not tracks). Minutes on CPU.
- Hard negatives: within a batch, up-weight (x3) pairs with the same artist or the same
  `deezer_genre` whose `darkness`, `pulse_rate` and `kick_class` disagree (DSP used for negatives
  only, never for positives).
- Output `ml/inference/adapter.npz` (W, b). Rebuild the index on the 128-d output; the Space
  applies the adapter to cold seeds. `adapter_cosine` replaces `clap_cosine` in 11.7 with the same
  weight. Keep `clap_cosine` as a zero-weight logged feature.
- Gate: on the 11.9 TUNE split, recall@20 with the adapter must be >= recall@20 without it. If
  not, ship without the adapter and record it.

**11.8.2 Weight search.** `ml/training/search_weights.py`:
- Coordinate ascent over the 12 non-zero weights in 11.7, step 0.02, weights constrained to
  [0, 0.4] and renormalised to sum 1.0, objective = mean over TUNE seeds of
  `0.6 * recall@20 + 0.3 * MRR - 0.1 * (negatives in top 10 / 10)`.
- Run the ranking offline against the catalogue (same code path as the Space, imported), never
  against the live API. Stop when no single step improves the objective by >= 0.002.
- Write `ml/inference/rerank_weights.json` and print TUNE and HOLDOUT numbers. The HOLDOUT number
  is recorded under "Progress" with the commit hash; it is not used to choose weights.

**11.8.3 Language affinity from PMI.** `ml/training/learn_affinity.py`: for every language pair
(a, b), PMI = log(P(a, b) / (P(a) P(b))) over playlist co-membership, with P computed from
`playlist_membership.jsonl` and the language codes of the catalogue (unknown excluded). Map PMI to
[0.5, 0.95] by rank within the pairs, blend 50/50 with the section 5 table, write
`ml/inference/language_affinity.json`. Same-language stays 1.0, unknown stays 0.75.

### 11.9 Benchmark v2 and the tests

**`tests/feel_benchmark_v2.json`** (replace v1; keep the four owner seeds first). Schema:

```json
{
  "seed": {"title": "...", "artist": "...", "language": "en"},
  "split": "tune" | "holdout",
  "positives": [{"title": "...", "artist": "...", "source": "owner" | "panel"}],
  "negatives": [{"title": "...", "artist": "..."}],
  "near_misses": [{"title": "...", "artist": "...", "note": "..."}],
  "scene_test": {"inside_negatives": [...], "outside_positives": [...]}   // Boyfriend only
}
```

Populate it exactly from section 7 of this file: the four owner seeds with owner targets marked
`"source": "owner"` and panel lists marked `"source": "panel"`; the musicologist's sets 1 to 5 as
`tune`, 6 to 10 as `holdout`. The owner may delete entries; the agent may not add any.

**`tests/feel_benchmark.py`** is rewritten to:
- run offline against the catalogue (default) or live (`--base-url`);
- resolve each title to a catalogue id with the 11.3 rule, and FAIL LOUDLY listing any benchmark
  title absent from the catalogue (11.4 must then be extended; do not silently skip);
- report per seed and overall: recall@10, recall@20, MRR, nDCG@20, negatives in top 10,
  near-miss rank, and for Boyfriend the scene test (mean rank of outside positives vs mean rank of
  inside negatives; pass if outside < inside);
- report Wilson 95% intervals on recall;
- report TUNE and HOLDOUT separately and never mix them;
- record the seed's `felt_bpm`, `darkness`, `kick_class`, `restraint_bucket` as information, with
  no per-song assertions.

**`tests/test_no_benchmark_leakage.py`**: load every title and artist from
`tests/feel_benchmark_v2.json`; grep (case-insensitive, whole-word) every `.py`, `.json`, `.sh`,
`.html`, `.ts`, `.tsx` file outside `tests/` and `ml/archive/`; fail on any hit. Allowed
exception: `ml/data/harvest_v2.py` artist seed list, which is discovery, not labels; put those
artists in `ml/data/harvest_seed_artists.json` and whitelist only that file.

**`tests/test_features_golden.py`**: 11.2.

**`tests/test_no_gates.py`**: static test that greps `backend/app` for `return None` / `continue`
inside `recommend` and the rerank loop on any of `language`, `genre`, `year`, `decade`, `artist`,
`popularity`, `_trusted`; fail on any hit.

### 11.10 Gates and progress

| Gate | Must be true | Record |
|---|---|---|
| G0 after 11.2 | golden test green on Mac and CI with identical numbers; Space and Mac describe the same preview to 1e-3 | Progress line |
| G1 after 11.3 | `clap_sanity_results.json` committed; decision rule applied and stated | Progress line with the win rate |
| G2 after 11.4 and 11.6 | catalogue >= 20k, every benchmark title resolvable, artifacts on HF Hub, Space boots from them | Progress line with counts |
| G3 after 11.7 | p95 < 300 ms in-catalogue, cold < 4 s; `feel_benchmark.py` TUNE numbers; Boyfriend scene test result | Progress line with TUNE recall@20 and intervals |
| G4 after 11.8 | adapter gate passed or recorded as skipped; `rerank_weights.json` committed; HOLDOUT read once and recorded with hash | Progress line |
| Iterate | each later change: TUNE before/after committed; HOLDOUT only at a declared gate | Progress line per gate |

Success for "working properly" at zero cost: HOLDOUT recall@20 >= 0.5 with the lower Wilson bound
above 0.3, zero negatives in any top 10 on TUNE, Boyfriend scene test passing, p95 < 300 ms. When
that is reached, the paid tiers (catalogue to 200k, MERT, listener-aware language) become worth
discussing; not before.

## Progress

- 2026-10-26: design written; live recall@20 at `d8cb36e` is 0/11 on the owner's 11 targets. No
  section 11 step started.
