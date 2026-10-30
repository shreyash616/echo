"""Rank the benchmark positives offline, under any weights, without a deploy.

Usage:
  ECHO_ARTIFACT_DIR=<dir with track_metadata.json, music_index.faiss, feel_matrix.npy,
                     feel_index.faiss, rerank_weights.json> \
  python ml/experiments/offline_rerank/report.py --split tune [--weights w.json]
         [--k-clap 500 --k-feel 600] [--validate-live https://<space>]

Prints, per seed: pool size, every positive with its pool rank (or "not pooled") and its axes
next to the top-3 axes, and its rank in the post-MMR top 20 the Space returns (score_lib.mmr_select,
matched by feel_benchmark.match_track); then hits@10, hits@20, MRR over ALL positives on that top 20
(not returned = miss), negatives in its top 10, the
per-axis standard deviation over the pool (an axis under 0.02 on a seed is inert there), and,
with --validate-live, the max deviation between these axes and the Space's debug=1 axes.
"""
import argparse, json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("tests"))
from score_lib import AXES, Catalogue, make_track_key, score_matrix  # noqa: E402
from feel_benchmark import match_track  # noqa: E402  (P8: the live benchmark's hit rule, imported)


def load_benchmark(path, split):
    out = []
    for e in json.load(open(path)):
        if split != "all" and e.get("split") != split:
            continue
        seed = make_track_key(e["seed"]["artist"], e["seed"]["title"])
        pos = [(make_track_key(p["artist"], p["title"]), p.get("source", "?"), p) for p in e.get("positives", [])]
        neg = [(make_track_key(p["artist"], p["title"]), p) for p in e.get("negatives", [])]
        out.append((seed, f'{e["seed"]["artist"]} - {e["seed"]["title"]}', pos, neg))
    return out


def fmt(row):
    return " ".join(f"{v:.2f}" for v in row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="tests/feel_benchmark_v2.json")
    ap.add_argument("--split", default="tune", choices=["tune", "holdout", "all"])
    ap.add_argument("--weights", default=None, help="weights json; default: the served rerank_weights.json")
    ap.add_argument("--k-clap", type=int, default=None)
    ap.add_argument("--k-feel", type=int, default=None)
    ap.add_argument("--validate-live", default=None, help="base URL of the Space")
    args = ap.parse_args()

    cat = Catalogue()
    if args.k_clap:
        cat.k_clap = args.k_clap
    if args.k_feel:
        cat.k_feel = args.k_feel
    w = json.load(open(args.weights)) if args.weights else cat.weights_live
    print("axes:", " ".join(a.replace("s_", "") for a in AXES))
    print("weights:", {k: round(v, 3) for k, v in w.items()})

    tot_pos = pooled = h10 = h20 = 0
    rr = []
    neg10 = 0
    max_dev = 0.0
    for seed_key, label, positives, negatives in load_benchmark(args.benchmark, args.split):
        if seed_key not in cat.id_map:
            print(f"\n== {label}: SEED NOT IN CATALOGUE ({seed_key})")
            continue
        si, idxs, srcs, F, flags = cat.feature_table(seed_key)
        scores = score_matrix(F, flags, w)
        order = np.argsort(-scores)
        rank_of = {int(idxs[j]): r + 1 for r, j in enumerate(order)}
        # P8: hits, MRR and negatives on the post-MMR top 20 the Space would return, matched as
        # feel_benchmark matches (key, else title-and-artist containment)
        top = [cat.meta[i] for i in cat.top20(idxs, F, flags, w)]
        post_rank = lambda t: next((r + 1 for r, m in enumerate(top) if match_track(m, t)), None)
        print(f"\n== {label} | pool {len(idxs)} (clap {int((srcs=='clap').sum())} feel {int((srcs=='feel').sum())} both {int((srcs=='both').sum())})")
        for r in range(3):
            j = order[r]
            m = cat.meta[int(idxs[j])]
            print(f"  top{r+1} {m.get('key','')[:34]:34} {scores[j]:.3f} {srcs[j]:4} | {fmt(F[j])}")
        for pk, src, pd in positives:
            tot_pos += 1
            pr = post_rank(pd)
            if pr is not None:
                h10 += pr <= 10
                h20 += pr <= 20
                rr.append(1.0 / pr)
            shown = f"top20 rank {pr}" if pr is not None else "not in top 20"
            pi = cat.id_map.get(pk)
            if pi is None:
                print(f"  POS {pk[:34]:34} NOT IN CATALOGUE ({src}) | {shown}")
                continue
            if pi not in rank_of:
                print(f"  POS {pk[:34]:34} not pooled ({src}) | {shown}")
                continue
            pooled += 1
            j = int(np.where(idxs == pi)[0][0])
            print(f"  POS {pk[:34]:34} pool rank {rank_of[pi]:4d} {srcs[j]:4} ({src}) | {shown} | {fmt(F[j])}")
        for nk, nd in negatives:
            nr = post_rank(nd)
            if nr is not None and nr <= 10:
                neg10 += 1
                print(f"  NEG {nk[:34]:34} top20 rank {nr} IN TOP 10")
        std = F.std(axis=0)
        inert = [AXES[i].replace("s_", "") for i in range(len(AXES)) if std[i] < 0.02]
        print(f"  axis std over pool: {fmt(std)}" + (f"  INERT: {inert}" if inert else ""))
        if args.validate_live:
            import httpx
            r = httpx.get(f"{args.validate_live}/api/recommendations/x",
                          params={"key": seed_key, "debug": 1, "diversity": 1.0}, timeout=180).json()
            res = r["results"] if isinstance(r, dict) else r
            for x in res:
                xi = cat.id_map.get(x.get("key") or "")
                if xi is None or xi not in rank_of:
                    continue
                j = int(np.where(idxs == xi)[0][0])
                for a_i, a in enumerate(AXES):
                    if a in x.get("axes", {}):
                        max_dev = max(max_dev, abs(float(x["axes"][a]) - F[j, a_i]))
    n_missing = tot_pos - pooled
    mrr = (sum(rr) + 0.0) / tot_pos if tot_pos else 0.0
    print(f"\nSUMMARY split={args.split}: positives {tot_pos}, pooled {pooled}, not pooled {n_missing}, "
          f"hits@10 {h10}, hits@20 {h20}, MRR {mrr:.4f}, negatives in top10 {neg10}")
    if args.validate_live:
        print(f"LIVE VALIDATION: max axis deviation vs {args.validate_live} = {max_dev:.4f} (must be < 0.001)")


if __name__ == "__main__":
    main()
