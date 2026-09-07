"""
Merge every finished experiment into one table.

Safe to run at any time, including mid-grid: `run_experiments.sh` writes one
`results.csv` per (target, feature-group) unit, so a partial run collects
cleanly and the headline tier is readable long before the rest finishes.

    python scripts/collect_results.py --results RESULTS
    python scripts/collect_results.py --results RESULTS --target k_off

Output: `RESULTS/summary.csv`, a pivot of R^2 by model and feature group, and
the k_off ranking — the comparison the study turns on, and the one where NOTES
5.6 predicted trouble (7 fabric features beat all 1124 TDA features at
recovering the generator's orientation, before any labels existed).
"""

import argparse
import glob
import os
import sys

import pandas as pd

# TDA first, then baselines, then the nulls — so the printed table reads in the
# order the argument is made rather than alphabetically.
GROUP_ORDER = ["tda", "ecp", "ph", "all", "baselines", "fab", "tpc", "por", "porosity"]


def load(results):
    rows = []
    for f in sorted(glob.glob(os.path.join(results, "*", "*", "results.csv"))):
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"  ! unreadable, skipped: {f} ({e})", file=sys.stderr)
            continue
        if d.empty:
            continue
        d["run"] = os.path.basename(os.path.dirname(f))
        d["kind"] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        rows.append(d)
    if not rows:
        return None
    r = pd.concat(rows, ignore_index=True)
    if "mode" not in r.columns:
        r["mode"] = "single"
    r["mode"] = r["mode"].fillna("single")
    return r


def order_groups(idx):
    known = [g for g in GROUP_ORDER if g in set(idx)]
    return known + sorted(set(idx) - set(known))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="RESULTS")
    ap.add_argument("--target", default="k_off",
                    help="which target to rank in the closing table")
    ap.add_argument("--quiet-if-empty", action="store_true",
                    help="say nothing when no results exist yet (for mid-run calls)")
    a = ap.parse_args()

    r = load(a.results)
    if r is None:
        if not a.quiet_if_empty:
            print(f"no results under {a.results}")
        return 0

    dest = os.path.join(a.results, "summary.csv")
    r.to_csv(dest, index=False)

    # `mode` stays in the index: a joint fit and a single fit of the same
    # component are different experiments, and averaging over them hides which
    # one won — which is the whole point of running full_tensor separately.
    piv = r.pivot_table(index=["kind", "group", "mode"], columns="target",
                        values="r2_mean", aggfunc="max").round(4)
    piv = piv.reindex(sorted(piv.index, key=lambda t: (t[0], order_groups([t[1]]) and
                                                       GROUP_ORDER.index(t[1])
                                                       if t[1] in GROUP_ORDER else 99,
                                                       t[2])))
    print("=== R^2 by model, feature group and target ===")
    print(piv.to_string())
    print(f"\n-> {dest}   ({len(r)} rows, "
          f"{r['run'].nunique()} units, {r['target'].nunique()} targets)")

    if a.target in set(r.target):
        sub = r[r.target == a.target].sort_values("r2_mean", ascending=False)
        cols = [c for c in ("kind", "group", "mode", "r2_mean", "r2_std", "mae")
                if c in sub.columns]
        print(f"\n=== ranked on {a.target} ===")
        print(sub[cols].head(12).to_string(index=False))

        # The comparison NOTES 5.6 flagged as the one that decides the framing.
        tda = sub[(sub.group == "tda")]
        fab = sub[(sub.group == "fab")]
        if len(tda) and len(fab):
            t, f = tda.r2_mean.max(), fab.r2_mean.max()
            print(f"\nTDA (1124f) {t:.4f}   vs   fabric (7f) {f:.4f}   "
                  f"delta {t - f:+.4f}")
            print("NOTES 5.6: fabric beat TDA on the generator's orientation "
                  "(0.994 vs 0.983) before labels existed.")

    if "ceiling" in r.columns and r["ceiling"].notna().any():
        print(f"\nlabel-noise ceiling on k_off: R^2 <= {r['ceiling'].max():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
