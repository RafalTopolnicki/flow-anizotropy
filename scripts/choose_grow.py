"""
Per-rock grain shave g for the 11 Sandstones dataset, from the s=3 pilots.

Rule (user's decision 2026-10-15): for each rock pick the tested g whose mean cube
porosity is closest to the target (.60, rock_berea's level), among the g values at
which EVERY pilot cube percolates on all three torus axes. Rocks are then alike in
porosity and pore width, and differ in pore shape and topology -- which is what
leave-one-rock-out should test.

    python scripts/choose_grow.py DATA/rock_pilot_sst11_s3 DATA/rock_pilot_sst11_s3b \
        --out DATA/rock_sst11_grow.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="pilot roots holding <rock>/pilot.csv")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--target", type=float, default=0.60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = pd.concat([pd.read_csv(p) for r in a.roots for p in Path(r).glob("*/pilot.csv")])
    d = d[d.s == a.scale]
    d["perc"] = d[["perc_x", "perc_y", "perc_z"]].all(axis=1)
    t = d.groupby(["rock", "g"]).agg(n=("porosity", "size"), por=("porosity", "mean"),
                                     dt50=("dt_p50", "mean"), perc=("perc", "mean"))
    print(t.round(3).to_string())

    choice = {}
    for rock, g in t.groupby(level=0):
        ok = g[g.perc == 1.0]
        assert len(ok), f"{rock}: no g at which every cube percolates"
        best = (ok.por - a.target).abs().idxmin()[1]
        r = t.loc[(rock, best)]
        choice[rock] = dict(grow=float(best), por=round(float(r.por), 3),
                            dt50=round(float(r.dt50), 2))
    print("\nchosen:")
    for k, v in choice.items():
        print(f"  {k:14s} g={v['grow']:g}  por={v['por']:.3f}  dt50={v['dt50']:.2f}")
    Path(a.out).write_text(json.dumps(dict(scale=a.scale, target=a.target, rocks=choice),
                                      indent=2))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
