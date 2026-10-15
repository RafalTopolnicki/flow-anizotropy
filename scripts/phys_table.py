"""
R^2 table for the physics-baseline comparison, plus paired fold differences.

Every run in a results root shares KFold(5, shuffle, seed 0), so fold i of one
group is the same test set as fold i of another and the difference is paired.
`wins` counts folds where the row beats the reference group.

    python scripts/phys_table.py RESULTS/rock_berea/catboost --targets K_xx k_xy k_xz k_yz
    python scripts/phys_table.py RESULTS/aniso3d_all --ref classical
"""
import argparse
import json
import os

import numpy as np

GROUPS = ["por", "tpc", "fab", "fabdir", "baselines", "kc", "cp", "phys", "classical",
          "tda", "thr", "all", "baselines+thr", "phys+thr", "classical+thr",
          "classical+tda", "classical+tda+thr"]


def folds(root, t, g):
    p = os.path.join(root, f"{t}__{g}", "metrics.json")
    if not os.path.exists(p):
        return None
    e = json.load(open(p))["experiments"][0]
    return np.array([f["r2"] for f in sorted(e["folds"], key=lambda f: f["fold"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--targets", nargs="+",
                    default=["K_xx", "K_yy", "K_zz", "k_xy", "k_xz", "k_yz"])
    ap.add_argument("--ref", default="thr", help="group the paired column compares against")
    a = ap.parse_args()

    print("| group | " + " | ".join(a.targets) + " |")
    print("|---|" + "---|" * len(a.targets))
    for g in GROUPS:
        cells, any_ = [], False
        for t in a.targets:
            r, ref = folds(a.root, t, g), folds(a.root, t, a.ref)
            if r is None:
                cells.append("—")
                continue
            any_ = True
            c = f"{r.mean():.3f}"
            if ref is not None and g != a.ref:
                d = r - ref
                c += f" ({d.mean():+.3f}, {int((d > 0).sum())}/5)"
            cells.append(c)
        if any_:
            print(f"| {g} | " + " | ".join(cells) + " |")
    print(f"\n(Δ vs `{a.ref}`, folds won of 5; same folds throughout)")


if __name__ == "__main__":
    main()
