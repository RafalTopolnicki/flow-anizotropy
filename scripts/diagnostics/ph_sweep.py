"""
Sweep the persistence-image parameters on 200 structures (direction 0).

Diagrams are computed once and re-imaged for every combination, so the grid
costs almost nothing. Two scores, and both matter:

  rank  -- dims for 99% of variance. 3-4 means the image is one fixed shape
           scaled by a number, which is the degeneracy being chased.
  R^2   -- predictability from TPC + porosity of the top 3 components. Low is
           good: it means the features are not a re-encoding of second-order
           statistics.

A combination that raises rank but keeps R^2 high has only added noise
dimensions, so neither number is sufficient alone.
"""
import numpy as np, pandas as pd, glob, os, re, gudhi as gd, catboost as cb
from gudhi.representations import DiagramSelector, PersistenceImage
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FILT = "/tmp/claude-607000063/-home-rtopolnicki-flow-flow-anizotropy/5633e86f-9ce7-4241-ac35-82bc10d45869/scratchpad/filt_a0"
sel = DiagramSelector(use=True, limit=np.inf, point_type="finite")

files = sorted(glob.glob(f"{FILT}/*.npy"))
sids, diags = [], {0: [], 1: []}
for f in files:
    filt = np.load(f)
    if filt.ndim == 3: filt = filt[:, :, 0]
    cx = gd.PeriodicCubicalComplex(top_dimensional_cells=filt.astype(np.float64),
                                   periodic_dimensions=[True, True])
    cx.compute_persistence()
    sids.append(int(re.search(r"sample_(\d+)", os.path.basename(f)).group(1)))
    for d in (0, 1):
        diags[d].append(sel(cx.persistence_intervals_in_dimension(d)))
print(f"{len(files)} diagrams computed")

b = pd.read_csv("DESC/aniso/baselines.csv")
dsc = pd.read_csv("DESC/aniso/descriptors.csv", usecols=["sample_id", "porosity"])
tab = b.merge(dsc, on="sample_id").set_index("sample_id").loc[sids]
tpc = [c for c in tab.columns if c.startswith("tpc_")]
X = tab[tpc + ["porosity"]].to_numpy(float)
cv = KFold(4, shuffle=True, random_state=0)

WEIGHTS = {"const": lambda x: 1.0, "linear": lambda x: x[1]}
# persistence-axis range per dimension: H1 bars are p99 = 0.083, so the shared
# 0-1.25 puts 99% of them inside one pixel row.
RANGES = {
    "shared 1.25":  {0: [0., 1.25, 0., 1.25], 1: [0., 1.25, 0., 1.25]},
    "perdim":       {0: [0., 1.25, 0., 1.25], 1: [0., 1.0,  0., 0.10]},
    "perdim tight": {0: [0., 1.0,  0., 1.25], 1: [0., 1.0,  0., 0.10]},
}

def score(V):
    Y = StandardScaler().fit_transform(V)
    keep = ~np.isnan(Y).any(0)
    Y = Y[:, keep]
    if Y.shape[1] < 3: return np.nan, np.nan, np.nan
    p = PCA(min(60, Y.shape[1], len(Y) - 1)).fit(Y)
    ev = np.cumsum(p.explained_variance_ratio_)
    rank = int(np.searchsorted(ev, 0.99)) + 1
    Z = p.transform(Y)[:, :3]
    r2 = []
    for k in range(3):
        oof = np.zeros(len(Z))
        for tr, te in cv.split(X):
            m = cb.CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                                     verbose=0, allow_writing_files=False, thread_count=6)
            m.fit(X[tr], Z[tr, k]); oof[te] = m.predict(X[te])
        r2.append(1 - ((Z[:, k]-oof)**2).sum() / ((Z[:, k]-Z[:, k].mean())**2).sum())
    return rank, float(p.explained_variance_ratio_[0]), float(np.mean(r2))

print(f"\n{'weight':7s} {'bw':>7s} {'bw/px':>6s} {'range':13s} {'res':>3s} | "
      f"{'H0 rank':>7s} {'PC0':>5s} {'R^2':>6s} | {'H1 rank':>7s} {'PC0':>5s} {'R^2':>6s}")
rows = []
for res in (10, 20):
    for wname, wfn in WEIGHTS.items():
        for rname, rng in RANGES.items():
            for bw in (1.0, 0.25, 0.125, 0.05, 0.025):
                out = {}
                for d in (0, 1):
                    im = PersistenceImage(bandwidth=bw, weight=wfn,
                                          resolution=[res, res], im_range=rng[d])
                    V = np.array([np.asarray(im(dg)).ravel() if len(dg)
                                  else np.zeros(res*res) for dg in diags[d]])
                    out[d] = score(V)
                px = (rng[0][3]-rng[0][2]) / res
                print(f"{wname:7s} {bw:7.3f} {bw/px:6.1f} {rname:13s} {res:3d} | "
                      f"{out[0][0]:7d} {out[0][1]:5.2f} {out[0][2]:6.3f} | "
                      f"{out[1][0]:7d} {out[1][1]:5.2f} {out[1][2]:6.3f}", flush=True)
                rows.append(dict(weight=wname, bandwidth=bw, rng=rname, res=res,
                                 h0_rank=out[0][0], h0_r2=out[0][2],
                                 h1_rank=out[1][0], h1_r2=out[1][2]))
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(FILT), "ph_sweep.csv"), index=False)
