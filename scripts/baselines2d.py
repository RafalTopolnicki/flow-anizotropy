r"""
Non-topological directional baselines: porosity profile, two-point correlation,
fabric tensor.

2D ports of `direction-aware-tda/scripts/compute_directional_*.py`, described in
that repo's METHODS.md. These exist to make the TDA comparison honest — in the
elasticity paper the interesting result was not "TDA works" but "TDA beats
strong baselines specifically on the coupling terms", and that only means
something if the baselines are properly implemented.

Descriptors
-----------
**Directional porosity profile** — mean solid fraction in slabs perpendicular
to d, `por_a{θ}_{i}`. Slabs are indexed by the *lattice plane* family, i.e.
`p = (h·x + k·y) mod N` for the integer normal (h, k). That is not a detail: on
a torus, binning a real-valued projection `x cosθ + y sinθ` breaks periodicity
for diagonal directions (the range is 2^0.5 N, and opposite ends of the profile
are actually adjacent). The integer-normal form wraps exactly, giving N genuine
slabs for every direction.

**Directional two-point correlation** — S2(r) = <g(x) g(x + r·d)> by cumulative
periodic rolls, `tpc_a{θ}_{i}`. S2(0) = solid fraction; the decay length is the
feature scale along d.

**Fabric tensor** — F_ij = <n_i n_j> over solid/void interface pixels, with n
the normalised gradient of the solid indicator (periodic central differences).
In 2D this is a symmetric 2x2: 3 components, 2 eigenvalues, and an orientation.
The orientation is emitted as (cos 2θ, sin 2θ), never as the angle, since it is
defined mod 180 degrees.

Expect the fabric tensor to be the hardest baseline to beat — it is close to a
direct measurement of structural anisotropy orientation, which is exactly what
the off-diagonal permeability encodes.

Usage
-----
    python scripts/baselines2d.py --self-test
    python scripts/baselines2d.py --dataset DATA/aniso --output DESC/aniso/baselines.csv
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io import load_structure          # noqa: E402

# direction angle (deg CCW from +x) -> integer step/normal (dx, dy)
DIRECTIONS = {0: (1, 0), 45: (1, 1), 90: (0, 1), 135: (-1, 1)}


def porosity_profile(solid, dxy, n_out):
    """Mean solid fraction per lattice plane perpendicular to dxy."""
    h, k = dxy
    H, W = solid.shape
    y, x = np.mgrid[0:H, 0:W]
    p = (h * x + k * y) % W                       # exact on the torus
    tot = np.bincount(p.ravel(), weights=solid.ravel().astype(float), minlength=W)
    cnt = np.bincount(p.ravel(), minlength=W)
    prof = tot / np.maximum(cnt, 1)
    idx = np.linspace(0, len(prof) - 1, n_out)
    return np.interp(idx, np.arange(len(prof)), prof)


def two_point_correlation(solid, dxy, n_out, max_steps=128):
    """S2(r) along dxy by cumulative periodic rolls."""
    g = solid.astype(np.float64)
    dx, dy = dxy
    vals = [float((g * g).mean())]
    rolled = g
    for _ in range(max_steps):
        rolled = np.roll(rolled, shift=(dy, dx), axis=(0, 1))   # (row, col)
        vals.append(float((g * rolled).mean()))
    vals = np.asarray(vals)
    idx = np.linspace(0, len(vals) - 1, n_out)
    return np.interp(idx, np.arange(len(vals)), vals)


def fabric_tensor(solid):
    """
    F_ij = <n_i n_j> over interface pixels, plus eigenvalues and orientation.

    Returns (fab_xx, fab_xy, fab_yy, eig_min, eig_max, cos2t, sin2t).
    """
    g = solid.astype(np.float64)
    # periodic central differences; axis 0 is y, axis 1 is x
    gy = 0.5 * (np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0))
    gx = 0.5 * (np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1))
    mag = np.hypot(gx, gy)
    m = mag > 1e-10
    if not m.any():
        return (np.nan,) * 7

    nx, ny = gx[m] / mag[m], gy[m] / mag[m]
    fxx = float((nx * nx).mean())
    fxy = float((nx * ny).mean())
    fyy = float((ny * ny).mean())

    tr, det = fxx + fyy, fxx * fyy - fxy * fxy
    disc = max(0.25 * tr * tr - det, 0.0) ** 0.5
    e_max, e_min = 0.5 * tr + disc, 0.5 * tr - disc
    t = 0.5 * np.arctan2(2 * fxy, fxx - fyy)      # principal axis, mod 180 deg
    return fxx, fxy, fyy, e_min, e_max, float(np.cos(2 * t)), float(np.sin(2 * t))


def _one(args):
    sample_id, path, n_por, n_tpc = args
    s = load_structure(path)
    row = {"sample_id": sample_id}
    for th, dxy in DIRECTIONS.items():
        for i, v in enumerate(porosity_profile(s, dxy, n_por)):
            row[f"por_a{th}_{i}"] = v
        for i, v in enumerate(two_point_correlation(s, dxy, n_tpc)):
            row[f"tpc_a{th}_{i}"] = v
    names = ["fab_xx", "fab_xy", "fab_yy", "fab_eig_min", "fab_eig_max",
             "fab_cos2t", "fab_sin2t"]
    row.update(dict(zip(names, fabric_tensor(s))))
    return row


def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    N = 64
    # horizontal bars: solid rows -> structure varies along y, uniform along x
    bars = np.zeros((N, N), dtype=np.uint8)
    for k in range(0, N, 16):
        bars[k:k + 8] = 1

    p0 = porosity_profile(bars, DIRECTIONS[0], 64)     # slabs perpendicular to x
    p90 = porosity_profile(bars, DIRECTIONS[90], 64)   # slabs perpendicular to y
    check("porosity profile flat along the bars, structured across",
          p0.std() < 1e-9 < p90.std(), f"std ∥={p0.std():.2e} ⊥={p90.std():.3f}")

    s2_0 = two_point_correlation(bars, DIRECTIONS[0], 65)
    s2_90 = two_point_correlation(bars, DIRECTIONS[90], 65)
    check("S2(0) equals the solid fraction",
          abs(s2_0[0] - bars.mean()) < 1e-12, f"{s2_0[0]:.4f} vs {bars.mean():.4f}")
    check("S2 constant along the bars, decaying across",
          s2_0.std() < 1e-9 < s2_90.std(), f"std ∥={s2_0.std():.2e} ⊥={s2_90.std():.3f}")

    fxx, fxy, fyy, *_ , c2, s2 = fabric_tensor(bars)
    check("fabric of x-running bars has normals along y",
          fyy > 0.9 > fxx and abs(fxy) < 1e-9,
          f"fxx={fxx:.3f} fyy={fyy:.3f} fxy={fxy:.2e}")
    check("fabric orientation of x-bars is 90° (cos2θ=-1)",
          abs(c2 + 1) < 1e-6 and abs(s2) < 1e-6, f"cos2θ={c2:.4f} sin2θ={s2:.4f}")

    # rotating the structure must rotate the fabric tensor the same way
    rot = np.rot90(bars)
    f2 = fabric_tensor(rot)
    check("rot90 swaps fxx and fyy", abs(f2[0] - fyy) < 1e-9 and abs(f2[2] - fxx) < 1e-9,
          f"rot: fxx={f2[0]:.3f} fyy={f2[2]:.3f}")

    # periodicity: a diagonal profile must wrap exactly (N slabs, not 2N)
    rng = np.random.default_rng(0)
    r = (rng.random((N, N)) > 0.5).astype(np.uint8)
    pr = porosity_profile(r, DIRECTIONS[45], N)
    check("diagonal profile mean equals the solid fraction",
          abs(pr.mean() - r.mean()) < 0.02, f"{pr.mean():.4f} vs {r.mean():.4f}")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--output")
    p.add_argument("--n_por", type=int, default=128)
    p.add_argument("--n_tpc", type=int, default=65)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.output):
        p.error("--dataset and --output are required (or use --self-test)")

    df = pd.read_csv(os.path.join(a.dataset, "structures.csv"))
    if a.limit:
        df = df.head(a.limit)
    tasks = [(r.sample_id, os.path.join(a.dataset, "structures", r.filename),
              a.n_por, a.n_tpc) for r in df.itertuples()]

    from tqdm import tqdm
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="baselines"):
            rows.append(f.result())

    out = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    out.to_csv(a.output, index=False)
    n_feat = out.shape[1] - 1
    print(f"  {len(out)} rows x {n_feat} features -> {a.output}")
    print(f"  NaNs: {int(out.isna().sum().sum())}")


if __name__ == "__main__":
    main()
