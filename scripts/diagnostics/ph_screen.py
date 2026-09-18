"""
Is PH also a re-encoding of TPC -- and does H0 (connectivity) differ from H1?

Persistent homology is NOT an additive functional, so unlike the Euler
characteristic it is not forced to be a function of the covariance. H0 in
particular tracks components merging, which is connectivity. If PH still comes
out TPC-predictable, the cause is the REPRESENTATION, not the topology.

Two representation defects were found this way and both are now fixed in
ph2d.py: gudhi's default `weight` is constant, so a bar of length 1e-3 counted
as much as one of length 1; and its default `bandwidth` of 1.0 is eight pixels
wide at the old range and resolution, which flattened the image to rank 1.
See NOTES 11.5a and 11.5c -- and note that fixing the second raised k_off by
.097 while a further H1 correction raised effective rank fourfold and moved
k_off by nothing, so read this screen alongside a real target, never alone.
"""
import argparse
import os
import numpy as np, pandas as pd, catboost as cb, re
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--descroot", default="DESC/aniso",
               help="descriptor root holding descriptors.csv; baselines.csv and "
                    "porosity always come from DESC/aniso, so PH computed with "
                    "different imaging parameters is scored against one X")
p.add_argument("--prefix", default="ph_",
               help="column prefix of the persistence-image block to screen. "
                    "`thr_` screens the throat-scale PH of "
                    "filtration_throat2d.py; the H0/H1 split below assumes "
                    "ph2d.py's layout, which both use")
p.add_argument("--limit", type=int, default=0,
               help="use only the first N rows. R^2 moves with sample size "
                    "(NOTES 11.6), so a comparison against a recorded number "
                    "must match the n it was measured at")
a = p.parse_args()

d = pd.read_csv(f"{a.descroot}/descriptors.csv")
# The descroot's OWN baselines when it has them. The hardcoded DESC/aniso path
# this used to have is right for comparing imaging parameters on one dataset --
# the original job -- and silently wrong for a different dataset, where it joins
# a candidate's PH to unrelated structures' TPC on sample_id and returns R^2 ~ 0
# that reads as "PH escapes the covariance". That cost a retracted result on
# 2026-09-17 (NOTES 11.11).
_own = f"{a.descroot}/baselines.csv"
b = pd.read_csv(_own if os.path.exists(_own) else "DESC/aniso/baselines.csv")
if not os.path.exists(_own):
    print(f"  ! {a.descroot} has no baselines.csv -- falling back to DESC/aniso's."
          f"\n    Valid ONLY if this descroot is the same structures re-imaged.")
if "porosity" not in d.columns:
    d = d.merge(pd.read_csv("DESC/aniso/descriptors.csv",
                            usecols=["sample_id", "porosity"]), on="sample_id")
df = d.merge(b, on="sample_id").reset_index(drop=True)
if a.limit:
    df = df.iloc[:a.limit].reset_index(drop=True)
tpc = [c for c in df.columns if c.startswith("tpc_")]
X = df[tpc + ["porosity"]].to_numpy(float)

# ph2d.py writes H0 then H1, so per direction the first half of the columns is
# H0. Inferred rather than hardcoded: resolution 10 gives 200 per direction,
# resolution 20 gives 800.
ph = [c for c in df.columns if c.startswith(a.prefix)]
if not ph:
    raise SystemExit(f"no columns with prefix {a.prefix!r} in {a.descroot}")
idx = {c: int(re.search(r"_(\d+)$", c).group(1)) for c in ph}
tag = {c: re.match(rf"{re.escape(a.prefix)}(a[^_]+)_", c).group(1) for c in ph}
half = {t: (sum(1 for c in ph if tag[c] == t)) // 2 for t in set(tag.values())}
lbl = a.prefix.rstrip("_").upper()
blocks = {
    f"{lbl} H0 (components)": [c for c in ph if idx[c] < half[tag[c]]],
    f"{lbl} H1 (loops)":      [c for c in ph if idx[c] >= half[tag[c]]],
    f"{lbl} all":             ph,
    "ECP (for reference)": [c for c in df.columns if c.startswith("ecp_")],
}
blocks = {k: v for k, v in blocks.items() if v}
print(f"  descroot: {a.descroot}   n = {len(df)}   "
      f"per-direction {lbl} columns: {2 * list(half.values())[0]}")

cv = KFold(5, shuffle=True, random_state=0)
print(f"{'block':22s} {'cols':>5s}  " + "  ".join(f"PC{k}" for k in range(5)) + "   var-wtd")
for name, cols in blocks.items():
    Y = StandardScaler().fit_transform(df[cols].to_numpy(float))
    p = PCA(5).fit(Y); Z = p.transform(Y)
    r2s = []
    for k in range(5):
        y, oof = Z[:, k], np.zeros(len(df))
        for tr, te in cv.split(X):
            m = cb.CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                     verbose=0, allow_writing_files=False)
            m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
        r2s.append(1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    w = p.explained_variance_ratio_[:5]
    print(f"{name:22s} {len(cols):5d}  " + "  ".join(f"{v:.3f}" for v in r2s)
          + f"   {np.average(r2s, weights=w):.4f}   (top5 = {w.sum()*100:.0f}% of var)")
