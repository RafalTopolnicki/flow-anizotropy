"""
The NOTES 11.7 screen: is a candidate structure family's ECP still a re-encoding
of its two-point correlation?

The criterion, checkable before a single LBM solve: **ECP-from-TPC must come in
materially below .83**, measured against sample-size-matched controls. It exists
because on DATA/aniso the answer is .834 -- the structures are excursion sets of
a near-Gaussian field, the Euler characteristic is additive, and for Gaussian
excursion sets additive functionals have closed forms in the covariance and the
threshold. A family that cannot beat the screen cannot repay an LBM campaign:
three separate descriptor fixes each gained on their own descriptor and nothing
on `all` (NOTES 11.8), so the binding constraint is the structure family.

Controls are the point, not a formality. R^2 here moves with sample size -- it
fell .85 -> .67 going from 1250 to 300 rows on the SAME data (NOTES 11.6), which
would have read as a result. Every subset is therefore scored at the same n, and
each gets its own PCA basis: the question is how much of *this* ensemble's ECP
variation TPC accounts for, not a shared coordinate system.

    python scripts/diagnostics/screen_family.py --descroot DESC/aniso_chan
"""
import argparse
import os

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--descroot", required=True, help="the candidate family")
p.add_argument("--controls", nargs="*",
               default=["DESC/aniso", "DESC/aniso_lowpor_probe"])
p.add_argument("--n", type=int, default=0, help="rows per subset (default: the candidate's)")
p.add_argument("--folds", type=int, default=4)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()


def load(root):
    d = pd.read_csv(f"{root}/descriptors.csv")
    b = pd.read_csv(f"{root}/baselines.csv")
    return d.merge(b, on="sample_id", how="inner").reset_index(drop=True)


cand = load(a.descroot)
n = a.n or len(cand)
ecp = [c for c in cand.columns if c.startswith("ecp_")]
tpc = [c for c in cand.columns if c.startswith("tpc_")]
ph = [c for c in cand.columns if c.startswith("ph_")]

subsets = {f"{a.descroot}  (candidate)": cand}
rng = np.random.default_rng(a.seed)
for c in a.controls:
    if not os.path.exists(f"{c}/descriptors.csv"):
        print(f"  ! control {c} not found, skipping")
        continue
    df = load(c)
    # every subset scored at the same n -- R^2 moves with sample size
    if len(df) > n:
        df = df.iloc[rng.choice(len(df), n, replace=False)].reset_index(drop=True)
    subsets[f"{c}  (control)"] = df

cv = KFold(a.folds, shuffle=True, random_state=a.seed)


def predictability(df, block):
    """Variance-weighted R^2 of the block's top-5 PCs from TPC + porosity."""
    cols = [c for c in block if c in df.columns]
    if not cols or len(df) < 40:
        return np.nan, np.nan
    Y = StandardScaler().fit_transform(df[cols].to_numpy(float))
    pca = PCA(5).fit(Y)
    Z = pca.transform(Y)
    X = df[tpc + ["porosity"]].to_numpy(float)
    r2s = []
    for k in range(5):
        y, oof = Z[:, k], np.zeros(len(df))
        for tr, te in cv.split(X):
            m = cb.CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                     verbose=0, allow_writing_files=False)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        r2s.append(1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    w = pca.explained_variance_ratio_[:5]
    return float(np.average(r2s, weights=w)), float(w.sum())


print(f"\nECP- and PH-from-TPC, {a.folds}-fold, n={n} per subset, "
      f"variance-weighted mean R^2 over the top 5 PCs\n")
print(f"{'subset':44s} {'n':>5s} {'porosity':>16s} {'ECP':>8s} {'PH':>8s}")
for name, df in subsets.items():
    e, _ = predictability(df, ecp)
    h, _ = predictability(df, ph)
    por = f"{df.porosity.min():.2f}-{df.porosity.max():.2f}"
    print(f"{name:44s} {len(df):5d} {por:>16s} {e:8.4f} "
          + (f"{h:8.4f}" if np.isfinite(h) else "      --"))

print("\nCriterion: the candidate's ECP must be MATERIALLY BELOW the controls,"
      "\nand below .83 in absolute terms, to be worth an LBM campaign.")
