"""How much does pore-phase CONNECTIVITY actually vary across the 2D dataset?"""
import numpy as np, pandas as pd, sys, os
from scipy.ndimage import label
sys.path.insert(0, "scripts")
from structure_io import load_structure

df = pd.read_csv("DATA/aniso/permeability.csv")
rng = np.random.default_rng(0)
sub = df.iloc[rng.choice(len(df), 300, replace=False)]

rows = []
for r in sub.itertuples():
    s = load_structure(os.path.join("DATA/aniso", "structures", r.filename)) > 0
    pore = ~s
    lab, n = label(pore)                       # 4-connectivity, same as the generator
    if n == 0:
        continue
    sizes = np.bincount(lab.ravel())[1:]
    big = sizes.max()
    # Euler characteristic of the pore phase = components - holes (2D)
    _, n_solid = label(s)
    rows.append({"por": r.porosity, "k_mean": r.k_mean, "k_off": abs(r.k_off),
                 "n_pore_clusters": n, "largest_frac": big / sizes.sum(),
                 "n_solid_clusters": n_solid, "euler": n - n_solid})

d = pd.DataFrame(rows)
print(f"{len(d)} structures sampled\n")
print(d[["por", "n_pore_clusters", "largest_frac", "n_solid_clusters", "euler"]]
      .describe().T.round(4).to_string())
print(f"\nbackbone is one cluster (>99% of pore in the largest): "
      f"{(d.largest_frac > 0.99).mean()*100:.1f}% of samples")
print(f"largest_frac min = {d.largest_frac.min():.4f}")
print("\ncorrelation with k_mean:")
print(d[["por", "largest_frac", "n_pore_clusters", "n_solid_clusters", "euler"]]
      .corrwith(d.k_mean).round(3).to_string())
print("\ncorrelation with |k_off|:")
print(d[["por", "largest_frac", "n_pore_clusters", "n_solid_clusters", "euler"]]
      .corrwith(d.k_off).round(3).to_string())
