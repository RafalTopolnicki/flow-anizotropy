"""ECP-from-TPC on the low-porosity probe, against matched n=300 controls."""
import numpy as np, pandas as pd, catboost as cb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load(descroot):
    d = pd.read_csv(f"{descroot}/descriptors.csv")
    b = pd.read_csv(f"{descroot}/baselines.csv")
    return d.merge(b, on="sample_id", how="inner").reset_index(drop=True)

old, new = load("DESC/aniso"), load("DESC/aniso_lowpor_probe")
ecp = [c for c in old.columns if c.startswith("ecp_")]
tpc = [c for c in old.columns if c.startswith("tpc_")]
assert ecp == [c for c in new.columns if c.startswith("ecp_")]
assert tpc == [c for c in new.columns if c.startswith("tpc_")]

rng = np.random.default_rng(0)
lo = old.porosity <= np.quantile(old.porosity, 0.25)
subsets = {
    "probe        por .45-.55": new,
    "existing Q1  por .65-.72": old[lo].iloc[rng.choice(int(lo.sum()), 300, replace=False)],
    "existing all por .65-.90": old.iloc[rng.choice(len(old), 300, replace=False)],
}

cv = KFold(4, shuffle=True, random_state=0)
print(f"{'subset':26s} {'n':>4s}  " + "  ".join(f"PC{k}" for k in range(5)) + "   var-wtd")
for name, df in subsets.items():
    # each subset gets its own basis: the question is how much of THIS
    # ensemble's ECP variation TPC accounts for, not a shared coordinate system
    Y = StandardScaler().fit_transform(df[ecp].to_numpy(float))
    p = PCA(5).fit(Y); Z = p.transform(Y)
    X = df[tpc + ["porosity"]].to_numpy(float)
    r2s = []
    for k in range(5):
        y, oof = Z[:, k], np.zeros(len(df))
        for tr, te in cv.split(X):
            m = cb.CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                     verbose=0, allow_writing_files=False)
            m.fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
        r2s.append(1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    w = p.explained_variance_ratio_[:5]
    print(f"{name:26s} {len(df):4d}  " + "  ".join(f"{v:.3f}" for v in r2s)
          + f"   {np.average(r2s, weights=w):.4f}   (top5 = {w.sum()*100:.0f}% of ECP var)")
