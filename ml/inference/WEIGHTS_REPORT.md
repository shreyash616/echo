# Weights Report (S7)

- Date: 2026-10-28
- Total evaluations logged: 387 (Requirement: >= 200)
- Restarts in simplex: 10
- Step size: 0.02

## Best Rerank Weights

```json
{
  "s_clap": 0.0673,
  "s_anchor": 0.1829,
  "s_tempo": 0.08,
  "s_pulse": 0.0555,
  "s_groove": 0.0103,
  "s_dark": 0.0103,
  "s_grit": 0.0036,
  "s_vocal": 0.1221,
  "s_intimacy": 0.0559,
  "s_restraint": 0.0748,
  "s_meaning": 0.0013,
  "s_language": 0.2129,
  "s_harm": 0.1085,
  "s_mode": 0.0145
}
```

## Performance Summary

| Split | Metric | Baseline (Pre-Search) | Optimized (Post-Search) |
|---|---|---|---|
| TUNE | Objective | 0.0096 | 0.0318 |
| TUNE | Recall@20 | 0.0123 | 0.0370 |
| TUNE | Recall@10 | 0.0000 | 0.0123 |
| TUNE | MRR | 0.0074 | 0.0320 |
| TUNE | Negatives@10 | 0 | 0 |
| HOLDOUT (eval once) | Objective | — | 0.0937 |
| HOLDOUT (eval once) | Recall@20 | — | 0.1429 |
| HOLDOUT (eval once) | Recall@10 | — | 0.0000 |
| HOLDOUT (eval once) | MRR | — | 0.0265 |

## Ablation Table (Objective with Each Axis Zeroed)

| Axis | Best Weight | Objective without Axis | Recall@20 without Axis | Delta Objective |
|---|---|---|---|---|
| `s_clap` | 0.0673 | 0.0234 | 0.0247 | -0.0084 |
| `s_anchor` | 0.1829 | 0.0407 | 0.0123 | +0.0089 |
| `s_tempo` | 0.0800 | 0.0314 | 0.0370 | -0.0004 |
| `s_pulse` | 0.0555 | 0.0200 | 0.0247 | -0.0119 |
| `s_groove` | 0.0103 | 0.0228 | 0.0247 | -0.0091 |
| `s_dark` | 0.0103 | 0.0318 | 0.0370 | +0.0000 |
| `s_grit` | 0.0036 | 0.0318 | 0.0370 | +0.0000 |
| `s_vocal` | 0.1221 | 0.0107 | 0.0123 | -0.0211 |
| `s_intimacy` | 0.0559 | 0.0220 | 0.0247 | -0.0099 |
| `s_restraint` | 0.0748 | 0.0141 | 0.0123 | -0.0178 |
| `s_meaning` | 0.0013 | 0.0318 | 0.0370 | +0.0000 |
| `s_language` | 0.2129 | 0.0157 | 0.0123 | -0.0161 |
| `s_harm` | 0.1085 | 0.0000 | 0.0000 | -0.0318 |
| `s_mode` | 0.0145 | 0.0318 | 0.0370 | +0.0000 |
