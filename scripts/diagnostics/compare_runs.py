"""
Paired per-fold comparison of CatBoost runs against a reference run.

Every run in this study uses `KFold(n, shuffle=True, random_state=seed)` on the
same label table, so fold k holds the same structures in every run and the
per-fold R^2 values are paired. A paired t-test over folds is then the right
test, and it is far more sensitive than comparing the two means against their
own across-fold spreads -- the fold-to-fold variance is large and common to
both runs, so it cancels.

Refuses to compare runs whose seed, fold count or row count differ, because the
pairing silently stops being valid.

    python scripts/diagnostics/compare_runs.py --ref RESULTS/catboost/k12__baselines \
        RESULTS/connect/k12__* RESULTS/void/k12__*
"""
import argparse
import json
import os

import numpy as np
from scipy import stats


def load(spec):
    """
    One run per experiment in a metrics.json, as `PATH` or `PATH:group`.

    train_catboost.py writes one experiment per (group, target) into a single
    output directory, so a run that swept five groups has five experiments in
    one metrics.json. This used to read `experiments[0]` only, which silently
    compared the first group in the sweep and ignored the rest -- fine while
    every recorded run had exactly one group per directory, wrong the moment
    one does not. `PATH:group` selects one; a bare PATH takes all of them.
    """
    path, _, want = spec.partition(":")
    m = json.load(open(os.path.join(path, "metrics.json")))
    runs = []
    for e in m["experiments"]:
        if want and e["group"] != want:
            continue
        name = os.path.basename(path.rstrip("/"))
        if len(m["experiments"]) > 1:
            name = f"{name}:{e['group']}"
        runs.append({"dir": path, "name": name, "group": e["group"],
                     "n_features": e["n_features"],
                     "r2": np.array([f["r2"] for f in e["folds"]]),
                     "seed": m["seed"], "folds": m["folds"],
                     "n_rows": m["n_rows"],
                     "ceiling": m.get("koff_ceiling")})
    if not runs:
        raise SystemExit(f"no experiment matching {spec!r}")
    return runs


def load_one(spec):
    runs = load(spec)
    if len(runs) > 1:
        raise SystemExit(
            f"{spec} holds {len(runs)} experiments "
            f"({', '.join(r['group'] for r in runs)}) -- name one as "
            f"{spec}:<group>")
    return runs[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()

    ref = load_one(a.ref)
    print(f"reference: {ref['name']}  ({ref['group']}, {ref['n_features']} feat)  "
          f"R2 = {ref['r2'].mean():.4f}   seed {ref['seed']}, "
          f"{ref['folds']} folds, {ref['n_rows']} rows")
    if ref["ceiling"]:
        print(f"label-noise ceiling: {ref['ceiling']:.4f}")
    print(f"\n{'run':34s} {'feat':>6s} {'R2':>8s} {'delta':>8s} {'folds':>6s} "
          f"{'t':>7s} {'p':>8s}")
    expanded = []
    for d in sorted(a.runs):
        if not os.path.exists(os.path.join(d.partition(":")[0], "metrics.json")):
            continue
        expanded.extend(load(d))

    for r in expanded:
        name = r["name"]
        if (r["seed"], r["folds"], r["n_rows"]) != (ref["seed"], ref["folds"], ref["n_rows"]):
            print(f"{name:34s}  NOT PAIRED with the reference "
                  f"(seed/folds/rows {r['seed']}/{r['folds']}/{r['n_rows']})")
            continue
        d_r2 = r["r2"] - ref["r2"]
        won = int((d_r2 > 0).sum())
        if np.allclose(d_r2, 0):
            t, pv = 0.0, 1.0
        else:
            t, pv = stats.ttest_rel(r["r2"], ref["r2"])
        print(f"{name:34s} {r['n_features']:6d} {r['r2'].mean():8.4f} "
              f"{d_r2.mean():+8.4f} {won:3d}/{ref['folds']:<2d} "
              f"{t:7.2f} {pv:8.4f}")


if __name__ == "__main__":
    main()
