# Weights report (T7, P8)

- Search: `ml/training/search_weights.py` on the replica (`score_lib`), pools 300 / 3000; 2300 TUNE evaluations, then HOLDOUT once; wall time 2.5 min. Log: `ml/training/weight_search_log.jsonl`.
- Every evaluation ranks the post-MMR top 20 the Space returns (diversity 0.70, artist cap 2, cross-language budget; the seed and copies of it dropped) and counts a hit by `feel_benchmark.match_track`.
- Objective: 0.5 x hits@20 + 0.3 x MRR + 0.2 x hits@10, each as a share of all TUNE positives (a positive not returned is a miss), minus 0.5 per negative in the top 10.
- Constraints: no axis above 0.25, s_clap + s_anchor <= 0.25, s_language <= 0.08, s_mode <= 0.05, weights sum to 1. 20 restarts (the served weights projected, then Dirichlet draws), coordinate ascent at steps [0.04, 0.02, 0.01]; feel-pool weights by coordinate ascent at steps [0.2, 0.1] in [0.1, 3.0], swing left out.

## TUNE, replica

| state | result |
|---|---|
| before, old definition (pre-MMR raw-score rank, exact key; P7 report) | hits@10 1, hits@20 1, MRR 0.0020 |
| before, aligned definition (served weights, this search's evaluation) | pooled 36 of 81, hits@10 2, hits@20 3, MRR 0.0144, negatives 0, objective 0.02777 |
| search's chosen point (rerank weights below plus the feel-pool change) | pooled 35 of 81, hits@10 3, hits@20 5, MRR 0.0172, negatives 0, objective 0.04342 |
| **shipped: log evaluation 1748** (the same rerank weights, the served feel-pool weights) | pooled 36 of 81, hits@10 3, hits@20 5, MRR 0.0170, negatives 0, objective 0.04336 |

Why the chosen point did not ship (reviewer decision 79c1435): the feel-pool change bought MRR 0.0170 to 0.0172 and no hit, and it pushed Stairway to Heaven out of Simple Man's pool, so the live owner-positive count fell to 7 of 11 against T6's gate of 8. Evaluation 1748 keeps P7's pool (8 of 11 live) and every hit.

## HOLDOUT, evaluated once with the chosen weights (5 seeds in the catalogue)

Read once, at the end of the search, for the chosen point, that is with the feel-pool change the shipped point does not carry. It is not read again for evaluation 1748; P9's section 6 row ("HOLDOUT recall@20, reported once at the end") is the one read for the shipped point.

pooled 4 of 14, hits@10 1, hits@20 1, MRR 0.0143, negatives 0, objective 0.05429

## Rerank weights

| axis | served before P8 | chosen and shipped |
|---|---|---|
| `s_clap` | 0.0673 | 0.2305 |
| `s_anchor` | 0.1829 | 0.0180 |
| `s_tempo` | 0.0800 | 0.1442 |
| `s_pulse` | 0.0555 | 0.0308 |
| `s_groove` | 0.0103 | 0.0065 |
| `s_dark` | 0.0103 | 0.0209 |
| `s_grit` | 0.0036 | 0.1104 |
| `s_vocal` | 0.1221 | 0.2169 |
| `s_intimacy` | 0.0559 | 0.0504 |
| `s_restraint` | 0.0748 | 0.0635 |
| `s_meaning` | 0.0013 | 0.0197 |
| `s_language` | 0.2129 | 0.0016 |
| `s_harm` | 0.1085 | 0.0743 |
| `s_mode` | 0.0145 | 0.0123 |

## Feel-pool weights (the shipped point keeps the served column)

| axis | served and shipped | search's chosen point |
|---|---|---|
| `log2_felt_bpm` | 1.50 | 1.70 |
| `pulse_rate` | 0.80 | 0.80 |
| `hits_per_bar` | 0.80 | 0.80 |
| `darkness` | 1.00 | 0.80 |
| `brightness` | 0.60 | 0.60 |
| `low_share` | 0.50 | 0.50 |
| `grit` | 0.80 | 0.80 |
| `roughness` | 0.80 | 0.80 |
| `intimacy` | 0.80 | 0.80 |
| `reverb_tail` | 0.50 | 0.50 |
| `loudness_range` | 0.60 | 0.60 |
| `crest_factor` | 0.40 | 0.40 |
| `lag_ms` | 0.50 | 0.50 |
| `swing` | 0.40 | 0.40 |
| `syllable_rate` | 0.60 | 0.60 |
| `sustain_ratio` | 0.60 | 0.60 |
| `mode_polarity` | 0.50 | 0.70 |
