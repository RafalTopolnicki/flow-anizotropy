"""
Non-additive connectivity features, and whether TPC already knows them.

The Euler characteristic is additive, so for excursion sets it is fixed by the
covariance -- which is why ECP tracks TPC at ~.85 regardless of porosity. These
features are deliberately NOT additive: they ask which pore clusters wrap the
torus, which is a global property no local sum can see.

Tiling and 4-connectivity follow gen_aniso_structures.percolates() exactly, so
"percolating" means here what it means to the generator and the solver.
"""
import numpy as np, pandas as pd, sys, os
from scipy.ndimage import label
sys.path.insert(0, "scripts")
from structure_io import load_structure


def features(solid):
    pore = ~solid
    H, W = pore.shape
    lab, _ = label(np.tile(pore, (3, 3)))        # 4-connectivity, as the generator
    mid = lab[H:2 * H, W:2 * W]
    span_x = (set(lab[:, 0].flat) & set(lab[:, -1].flat)) - {0}   # axis 1 = solver x
    span_y = (set(lab[0, :].flat) & set(lab[-1, :].flat)) - {0}   # axis 0 = solver y

    ids, counts = np.unique(mid[mid > 0], return_counts=True)
    tot = counts.sum()
    in_x = np.array([i in span_x for i in ids])
    in_y = np.array([i in span_y for i in ids])
    fx, fy = counts[in_x].sum() / tot, counts[in_y].sum() / tot
    return {
        "beta0_pore":   len(ids),                       # clusters on the torus
        "backbone_x":   fx,                             # pore volume that conducts in x
        "backbone_y":   fy,
        "backbone_iso": counts[in_x & in_y].sum() / tot,
        "dead_frac":    1.0 - counts[in_x | in_y].sum() / tot,
        "n_perc_x":     int(in_x.sum()),
        "n_perc_y":     int(in_y.sum()),
        "backbone_aniso": fx - fy,                      # directional, signed
        "largest_frac": counts.max() / tot,
    }


COLS = ["beta0_pore", "backbone_x", "backbone_y", "backbone_iso", "dead_frac",
        "n_perc_x", "n_perc_y", "backbone_aniso", "largest_frac"]

out = {}
for name, ds in (("probe   por .45-.55", "DATA/aniso_lowpor_probe"),
                 ("existing por .65-.90", "DATA/aniso")):
    df = pd.read_csv(f"{ds}/structures.csv")
    rng = np.random.default_rng(0)
    sub = df.iloc[rng.choice(len(df), 300, replace=False)] if len(df) > 300 else df
    rows = [dict(sample_id=r.sample_id,
                 **features(load_structure(os.path.join(ds, "structures", r.filename)) > 0))
            for r in sub.itertuples()]
    out[name] = pd.DataFrame(rows)
    print(f"=== {name} (n={len(rows)}) ===")
    print(out[name][COLS].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())
    print()

out["probe   por .45-.55"].to_csv(
    "/tmp/claude-607000063/-home-rtopolnicki-flow-flow-anizotropy/"
    "5633e86f-9ce7-4241-ac35-82bc10d45869/scratchpad/probe_connect.csv", index=False)
