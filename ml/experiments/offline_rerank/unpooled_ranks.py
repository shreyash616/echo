"""Where the unpooled TUNE positives sit (P7 diagnostic, reviewer 29a1dfd).

  ECHO_ARTIFACT_DIR=ml/inference python ml/experiments/offline_rerank/unpooled_ranks.py

For every TUNE positive outside the replica pool: whether the catalogue has it (exact key, else
feel_benchmark.match_track's title-and-artist containment), and its rank among all rows by CLAP
cosine and by feel L2 distance to the seed, the two lists the pool is drawn from. Ranks count the
seed itself as rank 1, as the FAISS searches do, so "rank <= k" means "inside a k pool".
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath("tests"))
sys.path.insert(0, os.path.abspath("backend"))
from score_lib import Catalogue, make_track_key  # noqa: E402
from feel_benchmark import match_track  # noqa: E402

OWNER = {
    "Simple Man": ["When the Levee Breaks", "Stairway to Heaven", "Rooster"],
    "Waiting Around to Die": ["No Surprises", "Me Gustas Tu", "Black"],
    "Bang Data": ["Tidal Wave"],
    "Boyfriend": ["Antidote", "For A Reason", "Baller", "Roll With Me"],
}


def ranks_of(scores, ascending):
    order = np.argsort(scores if ascending else -scores, kind="stable")
    r = np.empty(len(order), dtype=np.int64)
    r[order] = np.arange(1, len(order) + 1)
    return r


def main():
    cat = Catalogue()
    rows = []
    for e in json.load(open("tests/feel_benchmark_v2.json", encoding="utf-8")):
        if e.get("split") != "tune":
            continue
        seed_key = make_track_key(e["seed"]["artist"], e["seed"]["title"])
        if seed_key not in cat.id_map:
            continue
        si, pool = cat.pool(seed_key)
        pooled = {i for i, _ in pool}
        clap_rank = ranks_of(cat.clap @ cat.clap[si], ascending=False)
        feel_rank = ranks_of(np.sum((cat.feel_matrix - cat.feel_matrix[si]) ** 2, axis=1), ascending=True)
        owner = OWNER.get(e["seed"]["title"], [])
        for p in e.get("positives", []):
            idx, how = cat.id_map.get(make_track_key(p["artist"], p["title"])), "exact"
            if idx is None:
                hits = [i for i, m in enumerate(cat.meta) if match_track(m, p)]
                idx, how = (hits[0], f"containment ({cat.meta[hits[0]].get('key')})") if hits else (None, "absent")
            if idx is not None and idx in pooled:
                continue
            rows.append({
                "seed": e["seed"]["title"], "positive": f'{p["artist"]} - {p["title"]}',
                "owner": p["title"] in owner, "match": how,
                "clap_rank": int(clap_rank[idx]) if idx is not None else None,
                "feel_rank": int(feel_rank[idx]) if idx is not None else None,
            })

    print(f"| seed | positive | owner | catalogue row | CLAP rank | feel rank |")
    print(f"|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['seed']} | {r['positive']} | {'yes' if r['owner'] else ''} | {r['match']} | "
              f"{r['clap_rank'] or '-'} | {r['feel_rank'] or '-'} |")
    present = [r for r in rows if r["clap_rank"] is not None]
    owner_rows = [r for r in rows if r["owner"]]
    print(f"\nunpooled: {len(rows)} (owner {len(owner_rows)}); absent from the catalogue: {len(rows) - len(present)} "
          f"(owner {sum(1 for r in owner_rows if r['clap_rank'] is None)}); present: {len(present)}")
    for label, test in (("feel <= 3000", lambda r: r["feel_rank"] <= 3000),
                        ("feel <= 5000", lambda r: r["feel_rank"] <= 5000),
                        ("CLAP <= 500", lambda r: r["clap_rank"] <= 500)):
        n = [r for r in present if test(r)]
        print(f"present and inside {label}: {len(n)} (owner {sum(r['owner'] for r in n)})")


if __name__ == "__main__":
    main()
