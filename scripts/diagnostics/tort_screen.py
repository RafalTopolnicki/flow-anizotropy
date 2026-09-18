"""
Graded directional connectivity, and whether TPC already knows it.

Binary percolation carries no variance here -- the generator REJECTS anything
that does not percolate in both x and y, so every accepted structure spans both
ways and n_perc_x = n_perc_y = 1 identically. The graded version is geodesic
tortuosity: the length of the shortest pore path across the cell, divided by the
straight-line distance. It is directional, non-additive, and it varies even when
every structure percolates.
"""
import numpy as np, pandas as pd, sys, os, catboost as cb
from scipy.ndimage import label
from skimage.graph import MCP_Geometric
from sklearn.model_selection import KFold
sys.path.insert(0, "scripts")
from structure_io import load_structure


def tortuosity(pore, axis):
    """
    Shortest pore path across `axis`, in units of the straight distance.

    The cell is tiled 3x in the TRANSVERSE direction so a path may wander
    through the periodic wrap, which is what the solver allows. Without that,
    a structure percolating only via the wrap scores NaN rather than long.
    """
    pore = np.tile(pore, (3, 1)) if axis == 1 else np.tile(pore, (1, 3))
    cost = np.where(pore, 1.0, np.inf)
    if axis == 0:                                  # across rows = solver y
        starts = [(0, j) for j in np.flatnonzero(pore[0, :])]
        ends = np.flatnonzero(pore[-1, :]); L = pore.shape[0]
        end_idx = [(pore.shape[0] - 1, j) for j in ends]
    else:                                          # across cols = solver x
        starts = [(i, 0) for i in np.flatnonzero(pore[:, 0])]
        ends = np.flatnonzero(pore[:, -1]); L = pore.shape[1]
        end_idx = [(i, pore.shape[1] - 1) for i in ends]
    if not starts or not end_idx:
        return np.nan
    m = MCP_Geometric(cost)
    cum, _ = m.find_costs(starts)
    best = min(cum[i, j] for i, j in end_idx)
    return best / (L - 1) if np.isfinite(best) else np.nan


rows = []
ds = "DATA/aniso_lowpor_probe"
for r in pd.read_csv(f"{ds}/structures.csv").itertuples():
    pore = ~(load_structure(os.path.join(ds, "structures", r.filename)) > 0)
    lab, _ = label(np.tile(pore, (3, 3)))
    H, W = pore.shape
    mid = lab[H:2*H, W:2*W]
    ids, counts = np.unique(mid[mid > 0], return_counts=True)
    rows.append({"sample_id": r.sample_id,
                 "beta0_pore": len(ids),
                 "dead_frac": 1.0 - counts.max() / counts.sum(),
                 "tort_x": tortuosity(pore, 1),
                 "tort_y": tortuosity(pore, 0)})
c = pd.DataFrame(rows)
c["tort_aniso"] = c.tort_x - c.tort_y
FEATS = ["beta0_pore", "dead_frac", "tort_x", "tort_y", "tort_aniso"]
print("=== connectivity features on the probe (n=%d) ===" % len(c))
print(c[FEATS].describe().T[["mean", "std", "min", "50%", "max"]].round(4).to_string())

d = pd.read_csv("DESC/aniso_lowpor_probe/descriptors.csv")
b = pd.read_csv("DESC/aniso_lowpor_probe/baselines.csv")
df = d.merge(b, on="sample_id").merge(c, on="sample_id")
tpc = [x for x in df.columns if x.startswith("tpc_")]
X = df[tpc + ["porosity"]].to_numpy(float)
cv = KFold(4, shuffle=True, random_state=0)
print("\n=== can TPC + porosity predict them? ===")
for f in FEATS:
    y = df[f].to_numpy(float)
    ok = np.isfinite(y)
    if y[ok].std() == 0:
        print(f"  {f:15s}  no variance"); continue
    Xo, yo = X[ok], y[ok]
    oof = np.zeros(len(yo))
    for tr, te in cv.split(Xo):
        m = cb.CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                 verbose=0, allow_writing_files=False)
        m.fit(Xo[tr], yo[tr]); oof[te] = m.predict(Xo[te])
    r2 = 1 - ((yo - oof) ** 2).sum() / ((yo - yo.mean()) ** 2).sum()
    print(f"  {f:15s}  R^2 = {r2:7.4f}   (std {yo.std():.4f}, n={ok.sum()})")
