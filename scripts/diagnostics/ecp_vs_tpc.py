"""
Is ECP a function of the two-point correlation on the 2D dataset?

If the structures are thresholded (near-)Gaussian fields, the Euler
characteristic of the excursion set is analytically determined by the field's
covariance and the threshold -- so ECP should be predictable from TPC +
porosity, and would then carry no information the baseline does not already
have. This measures that directly, with no solver and no labels.
"""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

d = pd.read_csv("DESC/aniso/descriptors.csv")
b = pd.read_csv("DESC/aniso/baselines.csv")
df = d.merge(b, on="sample_id", how="inner")
print(f"{len(df)} structures")

ecp = [c for c in df.columns if c.startswith("ecp_")]
tpc = [c for c in df.columns if c.startswith("tpc_")]
por = [c for c in df.columns if c.startswith("por_")]

Y = df[ecp].to_numpy(float)
sets = {
    "porosity only  ":       df[["porosity"]].to_numpy(float),
    "TPC            ":       df[tpc].to_numpy(float),
    "TPC + porosity ":       df[tpc + ["porosity"]].to_numpy(float),
    "TPC + por + porosity":  df[tpc + por + ["porosity"]].to_numpy(float),
}

cv = KFold(5, shuffle=True, random_state=0)

def pooled_r2(X, Y):
    """Variance-weighted R^2 over all ECP features, out of fold."""
    oof = np.zeros_like(Y)
    for tr, te in cv.split(X):
        sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(Y[tr])
        m = RidgeCV(alphas=np.logspace(-3, 4, 15))
        m.fit(sx.transform(X[tr]), sy.transform(Y[tr]))
        oof[te] = sy.inverse_transform(m.predict(sx.transform(X[te])))
    ss_res = ((Y - oof) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    per = 1 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    return 1 - ss_res.sum() / ss_tot.sum(), per

print("\n=== linear (ridge): predicting all 324 ECP features ===")
print(f"{'predictors':22s} {'pooled R^2':>11s}   {'median':>7s} {'p10':>7s} {'p90':>7s}")
for name, X in sets.items():
    pooled, per = pooled_r2(X, Y)
    print(f"{name:22s} {pooled:11.4f}   {np.nanmedian(per):7.4f} "
          f"{np.nanpercentile(per,10):7.4f} {np.nanpercentile(per,90):7.4f}")

# Nonlinearity matters: the Gaussian-excursion formulae are nonlinear in the
# threshold and the spectral moments. Check the leading ECP directions with a
# model that can bend.
print("\n=== nonlinear (CatBoost) on the leading ECP principal components ===")
import catboost as cb
p = PCA(20).fit(StandardScaler().fit_transform(Y))
Z = p.transform(StandardScaler().fit_transform(Y))
X = sets["TPC + porosity "]
print(f"  top 20 PCs carry {p.explained_variance_ratio_.sum()*100:.1f}% of ECP variance")
print(f"  {'PC':>3s} {'var%':>6s} {'R^2 from TPC+porosity':>22s}")
tot = 0.0
for k in range(8):
    oof = np.zeros(len(Z))
    for tr, te in cv.split(X):
        m = cb.CatBoostRegressor(iterations=400, depth=6, learning_rate=0.05,
                                 verbose=0, allow_writing_files=False, thread_count=4)
        m.fit(X[tr], Z[tr, k])
        oof[te] = m.predict(X[te])
    r2 = 1 - ((Z[:, k] - oof) ** 2).sum() / ((Z[:, k] - Z[:, k].mean()) ** 2).sum()
    v = p.explained_variance_ratio_[k] * 100
    tot += v * max(r2, 0)
    print(f"  {k:3d} {v:6.2f} {r2:22.4f}")
print(f"  variance-weighted over these 8 PCs ({p.explained_variance_ratio_[:8].sum()*100:.1f}% of ECP): "
      f"{tot/p.explained_variance_ratio_[:8].sum()/100:.4f}")
