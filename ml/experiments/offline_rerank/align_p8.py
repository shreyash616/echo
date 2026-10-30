"""P8 item 1 (reviewer c63b103): does the replica return the Space's 20 tracks?

  ECHO_ARTIFACT_DIR=ml/inference python ml/experiments/offline_rerank/align_p8.py [--base-url URL]

For every TUNE seed in the catalogue, at the served weights: the replica's post-MMR top 20
(score_lib.Catalogue.top20, diversity 0.70, artist cap, cross-language budget) against the live
/api/recommendations top 20, counted by catalogue id; the differing tracks named. Side index:
the Space's recommend adds min(100, side_index.ntotal) extra records to every pool, so the
live debug=1 candidate_pool ids that are not catalogue rows count them exactly.
"""
import argparse, json, os, sys
import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_lib import Catalogue, make_track_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://shreyash616-echo-backend.hf.space")
    args = ap.parse_args()
    cat = Catalogue()
    client = httpx.Client(base_url=args.base_url, timeout=180)
    print(f"pools {cat.k_clap} / {cat.k_feel}; served weights from {os.environ.get('ECHO_ARTIFACT_DIR', 'ml/inference')}/rerank_weights.json")
    print("| seed | shared of 20 | live pool | side records in live pool | replica only | live only |")
    print("|---|---|---|---|---|---|")
    worst = 20
    for e in json.load(open("tests/feel_benchmark_v2.json", encoding="utf-8")):
        if e.get("split") != "tune":
            continue
        sk = make_track_key(e["seed"]["artist"], e["seed"]["title"])
        if sk not in cat.id_map:
            continue
        si, idxs, srcs, F, flags = cat.feature_table(sk)
        rep = [str(cat.meta[i]["id"]) for i in cat.top20(idxs, F, flags, cat.weights_live)]
        live_res = client.get(f"/api/recommendations/{cat.meta[si]['id']}", params={"key": sk}).json()
        live_res = live_res["results"] if isinstance(live_res, dict) else live_res
        live = [str(x.get("id")) for x in live_res]
        dbg = client.get(f"/api/recommendations/{cat.meta[si]['id']}", params={"key": sk, "debug": 1}).json()
        side = sum(1 for c in dbg["candidate_pool"] if c.get("id") not in cat.id_map)
        shared = len(set(rep) & set(live))
        worst = min(worst, shared)
        name = lambda i: (cat.meta[cat.id_map[i]].get("key") if i in cat.id_map else i)
        r_only = [name(i) for i in rep if i not in live]
        l_only = [name(i) for i in live if i not in rep]
        same_order = "same order" if rep == live else "order differs"
        print(f"| {e['seed']['title']} | {shared} ({same_order}) | {len(dbg['candidate_pool'])} | {side} | "
              f"{', '.join(r_only) or '-'} | {', '.join(l_only) or '-'} |")
    print(f"\nminimum shared over the TUNE seeds: {worst} of 20 (gate: at least 18 on every seed)")


if __name__ == "__main__":
    main()
