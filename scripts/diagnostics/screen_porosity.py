"""
Does ECP stop being a re-encoding of TPC as porosity falls toward percolation?

If yes, generating at low porosity is validated with no new structures. Two
confounds are controlled:

  * restricting the porosity range shrinks the variance of everything, which
    lowers R^2 mechanically -- so the PCA basis is fit ONCE on all 5000 and the
    same fixed components are predicted in every subset, and the residual is
    also reported in units of the FULL-dataset component std;
  * a quartile has 1250 rows, not 5000, and CatBoost fits worse on less data --
    so a random 1250-row subset spanning all porosities is the control.
"""
import numpy as np, pandas as pd, catboost as cb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

d = pd.read_csv("DESC/aniso/descriptors.csv")
b = pd.read_csv("DESC/aniso/baselines.csv")
df = d.merge(b, on="sample_id", how="inner").reset_index(drop=True)
ecp = [c for c in df.columns if c.startswith("ecp_")]
tpc = [c for c in df.columns if c.startswith("tpc_")]

# one fixed basis for every subset
Y = StandardScaler().fit_transform(df[ecp].to_numpy(float))
p = PCA(5).fit(Y)
Z = p.transform(Y)
sig_full = Z.std(0)
X = df[tpc + ["porosity"]].to_numpy(float)
por = df["porosity"].to_numpy()

q = np.quantile(por, [0.25, 0.5, 0.75])
subsets = {
    f"Q1 por {por.min():.2f}-{q[0]:.2f}": np.where(por <= q[0])[0],
    f"Q2 por {q[0]:.2f}-{q[1]:.2f}":      np.where((por > q[0]) & (por <= q[1]))[0],
    f"Q3 por {q[1]:.2f}-{q[2]:.2f}":      np.where((por > q[1]) & (por <= q[2]))[0],
    f"Q4 por {q[2]:.2f}-{por.max():.2f}": np.where(por > q[2])[0],
}
rng = np.random.default_rng(0)
subsets["control: random 1250, all por"] = rng.choice(len(df), 1250, replace=False)

cv = KFold(4, shuffle=True, random_state=0)
print(f"{'subset':32s} {'n':>5s}  " + "  ".join(f"PC{k}" for k in range(5))
      + "   var-wtd   resid/sigma_full")
for name, idx in subsets.items():
    r2s, res = [], []
    for k in range(5):
        y = Z[idx, k]
        oof = np.zeros(len(idx))
        for tr, te in cv.split(idx):
            m = cb.CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                     verbose=0, allow_writing_files=False)
            m.fit(X[idx][tr], y[tr])
            oof[te] = m.predict(X[idx][te])
        r2s.append(1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        res.append(np.sqrt(((y - oof) ** 2).mean()) / sig_full[k])
    w = p.explained_variance_ratio_[:5]
    print(f"{name:32s} {len(idx):5d}  " + "  ".join(f"{v:.3f}" for v in r2s)
          + f"   {np.average(r2s, weights=w):.4f}   {np.average(res, weights=w):.4f}")
