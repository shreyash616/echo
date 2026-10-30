"""P9 (reviewer 97e244f): the section 6 rows no existing script measures, against the live Space.

  ECHO_ARTIFACT_DIR=ml/inference python ml/experiments/offline_rerank/acceptance_p9.py [--base-url URL]
      [--part seeds|languages|latency|all]

seeds:     the live top 20 for Simple Man, Waiting Around to Die, Boyfriend (top 10) and Bang Data
           (top 10, felt_bpm within 0.5x to 2x of the seed, mean grit at least the seed's minus
           0.1), tracks named; and whether Rooster, Tidal Wave or Black reached their seed's top 20.
           Matching is feel_benchmark.match_track's rule.
languages: the 14 benchmark seeds' language on the live /api/track against the declared one.
latency:   100 in-catalogue /api/recommendations calls, 10 seeds x 2 passes x 5 calls in a row,
           every call's seed, position in its run of 5 and X-Process-Time; then the same 100
           calls in-process on the box with gc.callbacks timing every garbage collection, to see
           whether a slow call lines up with a generation-2 collection. Reports only.
"""
import argparse, gc, json, os, sys, time
import httpx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_lib import Catalogue, make_track_key  # noqa: E402
from feel_benchmark import match_track  # noqa: E402 (score_lib puts tests/ on the path)

BENCH = "tests/feel_benchmark_v2.json"


def live_top(client, cat, seed_key):
    r = client.get(f"/api/recommendations/{cat.meta[cat.id_map[seed_key]]['id']}", params={"key": seed_key}).json()
    return r["results"] if isinstance(r, dict) else r


def part_seeds(client, cat):
    def named(top, wanted):
        return [(w, next((i + 1 for i, t in enumerate(top) if match_track(t, {"artist": a, "title": w})), None)) for a, w in wanted]

    rows = [
        ("lynyrd skynyrd:simple man", 20, [("Led Zeppelin", "When the Levee Breaks"), ("Led Zeppelin", "Stairway to Heaven")], 1),
        ("townes van zandt:waiting around to die", 20, [("Radiohead", "No Surprises"), ("Manu Chao", "Me Gustas Tu"), ("Danger Mouse", "Black")], 1),
        ("karan aujla:boyfriend", 10, [("Karan Aujla", "Antidote"), ("Karan Aujla", "For A Reason"), ("Shubh", "Baller"), ("Sukha", "Roll With Me")], 2),
    ]
    for sk, k, wanted, need in rows:
        top = live_top(client, cat, sk)
        found = named(top[:k], wanted)
        got = [f"{w} (rank {r})" for w, r in found if r]
        print(f"\n{sk} live top {k}: requires at least {need} of {[w for _, w in wanted]}; found {len(got)}: {got or 'none'} -> {'pass' if len(got) >= need else 'fail'}")
        for i, t in enumerate(top[:k], 1):
            print(f"  {i:2d}. {t.get('artist')} - {t.get('title')}  ({t.get('matchScore'):.3f})")

    sk = "bang data:bang data"
    seed = cat.meta[cat.id_map[sk]]
    top = live_top(client, cat, sk)[:10]
    sb, sg = float(seed["felt_bpm"]), float(seed["grit"])
    bpm_ok = [0.5 * sb <= float(t["felt_bpm"]) <= 2.0 * sb for t in top]
    mean_grit = float(np.mean([float(t["grit"]) for t in top]))
    ok = all(bpm_ok) and mean_grit >= sg - 0.1
    print(f"\n{sk} live top 10: seed felt_bpm {sb:.0f}, grit {sg:.3f}; felt_bpm within {0.5 * sb:.0f} to {2 * sb:.0f} on "
          f"{sum(bpm_ok)} of 10; mean grit {mean_grit:.3f} against at least {sg - 0.1:.3f} -> {'pass' if ok else 'fail'}")
    for i, t in enumerate(top, 1):
        print(f"  {i:2d}. {t.get('artist')} - {t.get('title')}  felt_bpm {t['felt_bpm']} grit {float(t['grit']):.3f}")

    print("\nOutside the required set (section 6's last paragraph), live top 20 on their seed:")
    for sk, a, w in (("lynyrd skynyrd:simple man", "Alice in Chains", "Rooster"),
                     ("bang data:bang data", "Thee Oh Sees", "Tidal Wave"),
                     ("townes van zandt:waiting around to die", "Danger Mouse", "Black")):
        top = live_top(client, cat, sk)
        r = next((i + 1 for i, t in enumerate(top) if match_track(t, {"artist": a, "title": w})), None)
        print(f"  {a} - {w} on {sk}: {'rank ' + str(r) if r else 'not in the top 20'}")


def part_languages(client, cat):
    ok = 0
    entries = json.load(open(BENCH, encoding="utf-8"))
    print(f"\n{len(entries)} benchmark seeds, declared language against live /api/track:")
    for e in entries:
        s = e["seed"]
        sk = make_track_key(s["artist"], s["title"])
        if sk not in cat.id_map:
            print(f"  {s['artist']} - {s['title']}: not in the catalogue")
            continue
        t = client.get(f"/api/track/{cat.meta[cat.id_map[sk]]['id']}", params={"key": sk}).json()
        match = (t.get("language") or "") == s.get("language")
        ok += match
        print(f"  {e['split']:7} {s['artist']} - {s['title']}: declared {s.get('language')}, live {t.get('language')} "
              f"({t.get('language_source')}) -> {'ok' if match else 'MISMATCH'}")
    print(f"on their declared language: {ok} of {len(entries)}")


def seeds_for_latency(cat):
    keys = [make_track_key(e["seed"]["artist"], e["seed"]["title"]) for e in json.load(open(BENCH, encoding="utf-8"))]
    return [k for k in keys if k in cat.id_map][:10]


def part_latency(client, cat):
    keys = seeds_for_latency(cat)
    plan = [(p, k, j) for p in range(2) for k in keys for j in range(5)]
    print(f"\nlive: {len(plan)} plain /api/recommendations calls, seeds {len(keys)} x passes 2 x 5 in a row")
    rows = []
    for n, (p, k, j) in enumerate(plan, 1):
        t0 = time.perf_counter()
        r = client.get(f"/api/recommendations/{cat.meta[cat.id_map[k]]['id']}", params={"key": k})
        wall = time.perf_counter() - t0
        rows.append((n, p, k, j + 1, float(r.headers.get("X-Process-Time", "nan")), wall))
    pt = np.array([x[4] for x in rows])
    print(f"X-Process-Time p50 {np.percentile(pt, 50):.3f} s, p95 {np.percentile(pt, 95):.3f} s, max {pt.max():.3f} s")
    for n, p, k, pos, x, wall in rows:
        flag = "  <-- over 0.300 s" if x > 0.300 else ""
        print(f"  {n:3d} pass {p + 1} {k[:38]:38} call {pos} of 5  {x:.3f} s (client {wall:.3f} s){flag}")
    slow = [r for r in rows if r[4] > 0.300]
    print(f"over 0.300 s: {len(slow)}; positions in their run of 5: {[r[3] for r in slow]}; seeds: {[r[2] for r in slow]}")

    # the same plan in-process, every garbage collection timed
    sys.path.insert(0, os.path.abspath("backend"))
    from app.services.recommender import recommender  # noqa: E402
    recommender.load()
    events, starts = [], {}

    def cb(phase, info):
        if phase == "start":
            starts[info["generation"]] = time.perf_counter()
        else:
            events.append((info["generation"], starts.pop(info["generation"], time.perf_counter()), time.perf_counter()))

    for k in keys:  # warm each seed once, as the Space has been since boot
        recommender.recommend(seed=recommender.get_track_record(key=k), k=20, key=k)
    gc.callbacks.append(cb)
    local = []
    for n, (p, k, j) in enumerate(plan, 1):
        t0 = time.perf_counter()
        recommender.recommend(seed=recommender.get_track_record(key=k), k=20, key=k)
        t1 = time.perf_counter()
        g2 = [e for e in events if e[0] == 2 and e[1] >= t0 and e[2] <= t1]
        local.append((n, k, j + 1, t1 - t0, len(g2), sum(e[2] - e[1] for e in g2)))
    gc.callbacks.remove(cb)
    lt = np.array([x[3] for x in local])
    print(f"\nin-process on the box: p50 {np.percentile(lt, 50):.3f} s, p95 {np.percentile(lt, 95):.3f} s, max {lt.max():.3f} s")
    for n, k, pos, t, ng2, g2s in local:
        if ng2 or t > 2 * np.percentile(lt, 50):
            print(f"  {n:3d} {k[:38]:38} call {pos} of 5  {t:.3f} s; generation-2 collections {ng2}, {g2s:.3f} s")
    g2_calls = [x for x in local if x[4]]
    print(f"calls with a generation-2 collection: {len(g2_calls)} of {len(local)}; their mean {np.mean([x[3] for x in g2_calls]) if g2_calls else 0:.3f} s "
          f"against {np.mean([x[3] for x in local if not x[4]]):.3f} s without; gc thresholds {gc.get_threshold()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://shreyash616-echo-backend.hf.space")
    ap.add_argument("--part", default="all", choices=["seeds", "languages", "latency", "all"])
    args = ap.parse_args()
    cat = Catalogue()
    client = httpx.Client(base_url=args.base_url, timeout=180)
    if args.part in ("seeds", "all"):
        part_seeds(client, cat)
    if args.part in ("languages", "all"):
        part_languages(client, cat)
    if args.part in ("latency", "all"):
        part_latency(client, cat)


if __name__ == "__main__":
    main()
