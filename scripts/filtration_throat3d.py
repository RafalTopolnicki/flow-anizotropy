r"""
Throat-scale 3D filtration: the port of `filtration_throat2d.py`.

Why this exists, in one paragraph
---------------------------------
`filtration3d.py` measures *occupancy* — the solid fraction of a double cone.
NOTES 11.2 is the argument against that: for a Gaussian excursion set the
Minkowski functionals of occupancy fields have closed forms in the same
parameters that fix the covariance, so ECP re-encodes the two-point correlation
and PH inherits most of it.  Critical path analysis says permeability is set by
the narrowest constriction on the widest spanning path, which is a *radius*.
The Euclidean distance transform of the pore space is the inscribed-sphere
radius, so in a sublevel filtration of `-dt` **an H0 death time is a throat
radius**.  In 2D this took the best TDA descriptor on k_off from .6607 to
.7060 and made it the first one that does not lose to the baselines
(NOTES 11.12).  This is the 3D arm of that experiment.

Two reasons to expect more here than in 2D
------------------------------------------
**In 2D the pore and solid phases cannot both percolate** — planar duality
forbids it — so connectivity is geometrically constrained and much of what H0
and H1 could express is unavailable.  In 3D both phases can percolate at once,
so connectivity genuinely varies.  And **H2 exists**: enclosed cavities are a
feature class with no 2D analogue, and `ph3d.py` already computes all three
dimensions.

The dataset is also tighter, which favours a throat descriptor: `DATA/aniso3d`
has solid fraction ~0.30 against 2D's ~0.15, and the pore-space dt measures
p50 **3.16**, p75 4.69, p90 6.16, p99 **8.66**, p99.9 10.34 voxels, per-
structure max 6.6-13.3 (12 structures).  Median inscribed radius of 3.16 voxels
in an 80^3 cell is a far narrower pore space, relative to the cell, than 2D's
6.4 px in 256^2.

The two modes — identical in meaning to 2D
------------------------------------------
`--mode iso` — non-directional.  `f = 1 - min(dt, RCAP)/RCAP` on the phase the
descriptor is about, 1.25 sentinel elsewhere, so sublevel sets are erosions of
the pore space and `{f <= t}` holds the voxels with inscribed radius
>= RCAP(1-t).

`--mode dir` — walk ±d at most `--horizon` steps, stopping when the walk leaves
the phase, and take the **minimum** dt encountered: the narrowest constriction
met travelling along d.  The 2D docstring records why the obvious alternative —
an anisotropic metric, cheap along d and expensive across — is wrong: it reads
a channel *blocked* along d as "infinitely wide".  The min-along-the-walk form
gets both limiting cases right and the self-test pins them.

RCAP is a fixed global constant
-------------------------------
`--rcap` defaults to **10** voxels, which covers this dataset's dt p99.9.  It
is not a per-structure `dt.max()`: k ~ r^2 makes the absolute throat scale the
entire point, and normalising per structure would map two structures whose
throats differ twofold onto the same filtration values.  Descriptors built with
different caps are not comparable; the value is recorded in `config.txt`.

One consequence of the tight 3D pore space is worth stating plainly: with
RCAP = 10 and a median dt of 3.16, the filtration takes **few distinct values**
compared with the 2D case, so births and deaths pile up on a coarse grid.  That
is a property of an 80^3 cell with solid fraction .30, not of the cap, and it
cannot be tuned away here.

Frame
-----
Arrays are `[z, y, x] = [solver_z, solver_y, solver_x]`, as
`structure_io3d.load_structure` returns.  `--direction H K L` is (dx, dy, dz) in
the **solver** frame, matching `structures.csv`'s e1_x/e1_y/e1_z and the
solver's force axes; `compute()` converts once, `compute_arr()` takes the
direction already in array order.  Same split as `filtration3d.py`.  Do not
transpose the array.

Output
------
One `(Z, Y, X)` float32 array per structure — a single channel, where
`filtration3d.py` writes three.  `ph3d.py` reads 3D input unchanged.  There are
no PCA orientation channels because ECP is not the target of this experiment;
the 2D arm established that the gain is in H0, and ECP cannot see it.

Disk, and the quota
-------------------
One filtration is 80^3 x float32 = **2.05 MB**, a third of `filtration3d.py`'s
6.1 MB, but nine directions x 5000 structures is still ~92 GB if kept.  /home
carries a **150 GiB ceph quota that `df` does not report**, and as of this
writing 135.7 GB of it is already used — about 25 GB free.  `run_throat3d.sh`
therefore chunks and deletes, exactly as `run_descriptors3d.sh` does.

    getfattr -n ceph.quota.max_bytes ~
    getfattr -n ceph.dir.rbytes      ~

Usage
-----
    python scripts/filtration_throat3d.py --self-test
    python scripts/filtration_throat3d.py --dataset DATA/aniso3d \
        --outputdir DESC/aniso3d_throat/filt/a1_-1_0 --mode dir \
        --direction 1 -1 0 [--rcap 10] [--horizon 6] [--phase void]
        [--workers 24] [--start 0] [--limit N]

Note on argparse: `--direction 1 -1 0` parses because no option string of this
parser looks like a negative number.  Do not add a short option like -1.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numba import njit, prange
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io3d import load_structure          # noqa: E402

VOID = np.float32(1.25)
PHASES = {"solid": 1, "void": 0}


# ---------------------------------------------------------------------------
# Periodic distance transform
# ---------------------------------------------------------------------------

def periodic_edt(grid: np.ndarray, fg: int, rcap: float) -> np.ndarray:
    """
    Distance from every `fg` voxel to the nearest non-`fg` voxel, on the 3-torus.

    Wrap-padding by rcap+2 makes this exact wherever the result is <= rcap,
    which is all the caller keeps: a voxel whose nearest other-phase voxel lies
    beyond the pad has true dt > rcap and is capped anyway.  Padding costs
    (80+2p)^3 rather than the 3^3 = 27x of the tiling trick used elsewhere in
    the repo, which matters at this size.
    """
    pad = int(np.ceil(rcap)) + 2
    mask = np.pad(grid == fg, pad, mode="wrap")
    dt = distance_transform_edt(mask)
    return np.ascontiguousarray(dt[pad:-pad, pad:-pad, pad:-pad],
                                dtype=np.float32)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _iso(grid, dt, rcap, fg):
    """f = 1 - min(dt, rcap)/rcap on the fg phase, 1.25 elsewhere."""
    Z, Y, X = grid.shape
    out = np.empty((Z, Y, X), dtype=np.float32)
    for k in prange(Z):
        for j in range(Y):
            for i in range(X):
                if grid[k, j, i] != fg:
                    out[k, j, i] = 1.25
                    continue
                d = dt[k, j, i]
                if d > rcap:
                    d = rcap
                out[k, j, i] = 1.0 - d / rcap
    return out


@njit(parallel=True, cache=True)
def _dirmin(grid, dt, d_z, d_y, d_x, horizon, rcap, fg):
    """
    f = 1 - min(dt along ±d)/rcap, the walk stopping when it leaves the phase.

    Unit steps along d, rounded to the nearest voxel, so every direction is
    treated identically and no grid rotation is needed -- departure 1 of
    filtration3d.py, for the same reason.  Stopping at the phase boundary
    rather than injecting a zero keeps the field smooth: the pore voxel next to
    a wall already carries dt ~ 1, which is the signal that the path is blocked.
    """
    Z, Y, X = grid.shape
    out = np.empty((Z, Y, X), dtype=np.float32)
    for k in prange(Z):
        for j in range(Y):
            for i in range(X):
                if grid[k, j, i] != fg:
                    out[k, j, i] = 1.25
                    continue
                m = dt[k, j, i]
                for sgn in range(-1, 2, 2):
                    for s in range(1, horizon + 1):
                        kk = int(np.floor(k + sgn * s * d_z + 0.5)) % Z
                        jj = int(np.floor(j + sgn * s * d_y + 0.5)) % Y
                        ii = int(np.floor(i + sgn * s * d_x + 0.5)) % X
                        if grid[kk, jj, ii] != fg:
                            break
                        v = dt[kk, jj, ii]
                        if v < m:
                            m = v
                if m > rcap:
                    m = rcap
                out[k, j, i] = 1.0 - m / rcap
    return out


def compute_arr(grid, d_arr, rcap: float = 10.0, horizon: int = 6,
                mode: str = "dir", phase: str = "void") -> np.ndarray:
    """Throat filtration with the direction already in array order (dz, dy, dx)."""
    fg = np.uint8(PHASES[phase])
    g = np.ascontiguousarray(grid, dtype=np.uint8)
    dt = periodic_edt(g, int(fg), rcap)
    if mode == "iso":
        return _iso(g, dt, np.float32(rcap), fg)
    d = np.asarray(d_arr, dtype=np.float64)
    n = np.linalg.norm(d)
    if n == 0.0:
        raise ValueError("direction must be non-zero")
    d = d / n
    return _dirmin(g, dt, float(d[0]), float(d[1]), float(d[2]),
                   int(horizon), np.float32(rcap), fg)


def compute(grid, direction=(1, 0, 0), rcap: float = 10.0, horizon: int = 6,
            mode: str = "dir", phase: str = "void") -> np.ndarray:
    """Throat filtration along (dx, dy, dz) in the solver frame."""
    dx, dy, dz = direction
    return compute_arr(grid, (dz, dy, dx), rcap, horizon, mode, phase)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _one(args):
    path, out_path, direction, rcap, horizon, mode, phase = args
    if os.path.exists(out_path):
        return "cached"
    filt = compute(load_structure(path), direction, rcap, horizon, mode, phase)
    np.save(out_path, filt)
    return "ok"


def run(dataset, outputdir, direction, rcap, horizon, mode, phase,
        workers, start, limit):
    import pandas as pd
    df = pd.read_csv(os.path.join(dataset, "structures.csv"))
    df = df.iloc[start:start + limit] if limit else df.iloc[start:]
    os.makedirs(outputdir, exist_ok=True)

    tasks = [(os.path.join(dataset, "structures", r.filename),
              os.path.join(outputdir, r.filename.replace(".raw", ".npy")),
              direction, rcap, horizon, mode, phase)
             for r in df.itertuples()]

    from tqdm import tqdm
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"throat3d {mode} {phase} d={direction}"):
            done += f.result() == "ok"
    print(f"  {done} computed, {len(tasks) - done} cached, mode={mode} "
          f"rcap={rcap:g} horizon={horizon} -> {outputdir}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    RCAP = 10.0

    # --- the transform -----------------------------------------------------
    # One solid voxel: dt must be the distance to it measured on the 3-torus,
    # i.e. the minimum over all 27 translates.
    N = 32
    g = np.zeros((N, N, N), dtype=np.uint8)
    g[10, 14, 20] = 1
    dt = periodic_edt(g, 0, RCAP)
    kk, jj, ii = np.mgrid[0:N, 0:N, 0:N]
    best = np.full((N, N, N), np.inf)
    for sk in (-N, 0, N):
        for sj in (-N, 0, N):
            for si in (-N, 0, N):
                best = np.minimum(best, np.sqrt((kk - (10 + sk)) ** 2
                                                + (jj - (14 + sj)) ** 2
                                                + (ii - (20 + si)) ** 2))
    near = best <= RCAP
    check("periodic EDT matches the min over the 27 translates",
          np.allclose(dt[near], best[near], atol=1e-5),
          f"max diff {np.abs(dt[near] - best[near]).max():.2e}")

    # The wrap must be used: a voxel at z=0 is 1 from a solid slab at z=31.
    g = np.zeros((N, N, N), dtype=np.uint8)
    g[31, :, :] = 1
    dt = periodic_edt(g, 0, RCAP)
    check("EDT wraps across the seam", np.isclose(dt[0, 5, 5], 1.0),
          f"dt[0,5,5]={dt[0, 5, 5]:.3f} (non-periodic would give 31)")

    # --- iso ---------------------------------------------------------------
    f = compute(g, (1, 0, 0), RCAP, mode="iso")
    check("solid voxels get the 1.25 sentinel", np.all(f[31] == 1.25))
    check("iso: pore next to a wall is near 1",
          np.isclose(f[0, 5, 5], 1.0 - 1.0 / RCAP), f"f={f[0, 5, 5]:.4f}")
    check("iso: filtration is in [0, 1] on the pore phase",
          f[g == 0].min() >= 0.0 and f[g == 0].max() <= 1.0,
          f"[{f[g == 0].min():.3f}, {f[g == 0].max():.3f}]")
    empty = np.zeros((40, 40, 40), dtype=np.uint8)
    check("iso: capped flat when no solid is within rcap",
          np.allclose(compute(empty, (1, 0, 0), RCAP, mode="iso"), 0.0))

    rng = np.random.default_rng(0)
    s = (rng.random((32, 32, 32)) > 0.75).astype(np.uint8)
    check("iso is direction-independent",
          np.array_equal(compute(s, (1, 0, 0), RCAP, mode="iso"),
                         compute(s, (1, -2, 3), RCAP, mode="iso")))

    # --- phase symmetry ----------------------------------------------------
    # void(g) == solid(1-g) exactly, in both modes: the check that catches a
    # half-applied phase swap.
    g = (rng.random((32, 32, 32)) > 0.5).astype(np.uint8)
    for mode in ("iso", "dir"):
        for d in ((1, 0, 0), (1, -1, 0)):
            v = compute(g, d, RCAP, 6, mode, "void")
            c = compute((1 - g).astype(np.uint8), d, RCAP, 6, mode, "solid")
            check(f"{mode}: void(g) == solid(1-g) along {d}",
                  np.array_equal(v, c), f"max diff {np.abs(v - c).max():.2e}")

    # --- dir: the two limiting cases ---------------------------------------
    # Slab channels normal to z: 8 voxels of pore between 8-voxel solid slabs.
    # In-plane the walk stays inside and reads the half-width; along z it
    # reaches the slab and reads a radius near 1.  This pair is what rejected
    # the anisotropic-metric design.
    walls = np.zeros((32, 32, 32), dtype=np.uint8)
    for k0 in range(0, 32, 16):
        walls[k0:k0 + 8] = 1
    pore = walls == 0
    d_in = compute(walls, (1, 0, 0), RCAP, 6, "dir", "void")   # in-plane
    d_z = compute(walls, (0, 0, 1), RCAP, 6, "dir", "void")    # across slabs
    check("dir: slab channel reads open in-plane and blocked across it",
          d_in[pore].mean() < d_z[pore].mean() - 0.05,
          f"mean f in-plane={d_in[pore].mean():.4f} across={d_z[pore].mean():.4f}")
    ck = 12                                     # centre plane of the pore slab
    check("dir: channel centre keeps its half-width in-plane",
          np.isclose(d_in[ck, 16, 16], 1.0 - 4.0 / RCAP, atol=1e-6),
          f"f={d_in[ck, 16, 16]:.4f} expected {1.0 - 4.0 / RCAP:.4f}")
    check("dir: channel centre is pulled to the wall radius across it",
          np.isclose(d_z[ck, 16, 16], 1.0 - 1.0 / RCAP, atol=1e-6),
          f"f={d_z[ck, 16, 16]:.4f} expected {1.0 - 1.0 / RCAP:.4f}")

    s = (rng.random((32, 32, 32)) > 0.7).astype(np.uint8)
    fi = compute(s, (1, 1, 0), RCAP, 6, "iso", "void")
    fd = compute(s, (1, 1, 0), RCAP, 6, "dir", "void")
    p = s == 0
    check("dir >= iso on the pore phase (the walk can only narrow)",
          np.all(fd[p] >= fi[p] - 1e-6),
          f"worst violation {np.min(fd[p] - fi[p]):.2e}")
    check("dir with horizon 0 reduces to iso",
          np.array_equal(compute(s, (1, 1, 0), RCAP, 0, "dir", "void"), fi))

    # --- frame and equivariance -------------------------------------------
    # compute() takes (dx,dy,dz) and compute_arr() takes (dz,dy,dx); a swap
    # between them is invisible on an axis direction and wrong on every
    # diagonal, so it is checked on a diagonal.
    check("compute((dx,dy,dz)) == compute_arr((dz,dy,dx))",
          np.array_equal(compute(s, (1, -1, 0), RCAP, 6, "dir", "void"),
                         compute_arr(s, (0, -1, 1), RCAP, 6, "dir", "void")))

    # Periodicity: rolling the structure rolls the filtration.
    r = np.roll(s, (3, -5, 7), axis=(0, 1, 2))
    check("roll(structure) equals roll(filtration)",
          np.allclose(np.roll(compute(s, (1, -1, 0), RCAP, 6, "dir", "void"),
                              (3, -5, 7), axis=(0, 1, 2)),
                      compute(r, (1, -1, 0), RCAP, 6, "dir", "void"), atol=0))

    # Axis permutation: transposing the grid and permuting the array-order
    # direction the same way must transpose the filtration.
    d_arr = np.array([0.0, 1.0, -1.0])
    ref = compute_arr(s, d_arr, RCAP, 6, "dir", "void")
    for t in ((1, 0, 2), (2, 1, 0), (0, 2, 1)):
        got = compute_arr(np.transpose(s, t), d_arr[list(t)],
                          RCAP, 6, "dir", "void")
        check(f"transpose{t} of the structure transposes the filtration",
              np.allclose(np.transpose(ref, t), got, atol=0),
              f"max diff {np.abs(np.transpose(ref, t) - got).max():.2e}")

    # --- the property the descriptor rests on ------------------------------
    # Two spherical chambers joined by a cylindrical throat of known radius w:
    # the long-lived finite H0 bar must die at the level with RCAP(1-t) = w.
    # Checked across a range of w, because one radius would also pass under a
    # wrong but monotone calibration.
    try:
        import gudhi as gd
        from gudhi.representations import DiagramSelector
    except ImportError:
        print("[skip] H0 death times recover throat radii (gudhi not importable)")
    else:
        sel = DiagramSelector(use=True, limit=np.inf, point_type="finite")
        worst, detail = 0.0, []
        for w in (2, 3, 4, 5):
            M, R, sep = 48, 9, 22
            g = np.ones((M, M, M), dtype=np.uint8)
            kk, jj, ii = np.mgrid[0:M, 0:M, 0:M]
            c1, c2 = M // 2 - sep // 2, M // 2 + sep // 2
            for ci in (c1, c2):
                g[(kk - M // 2) ** 2 + (jj - M // 2) ** 2
                  + (ii - ci) ** 2 <= R * R] = 0
            # cylinder along x joining the two chambers. Built as one 3D mask:
            # a (z,y) disc mask plus an x slice indexes 4 axes and throws.
            cyl = (((kk - M // 2) ** 2 + (jj - M // 2) ** 2 <= w * w)
                   & (ii >= c1) & (ii <= c2))
            g[cyl] = 0
            f = compute(g, (1, 0, 0), RCAP, 6, "iso", "void")
            cx = gd.PeriodicCubicalComplex(
                top_dimensional_cells=f.astype(np.float64),
                periodic_dimensions=[True, True, True])
            cx.compute_persistence()
            bars = np.asarray(sel(cx.persistence_intervals_in_dimension(0)))
            b = int(np.argmax(bars[:, 1] - bars[:, 0]))
            r_meas = RCAP * (1.0 - bars[b, 1])
            worst = max(worst, abs(r_meas - w))
            detail.append(f"w={w}->{r_meas:.2f}")
        # The measured radii come out at exactly sqrt(w^2 + 1) -- 2.236, 3.162,
        # 4.123, 5.099 for w = 2..5 -- because the nearest solid voxel to a
        # digitised cylinder's axis sits one voxel off the radial direction.
        # So the calibration is exact and the offset is pure voxelisation,
        # which is why the tolerance is 0.25 here and 0.05 in 2D.
        check("H0 death times recover known throat radii", worst < 0.25,
              f"max error {worst:.3f} vox  ({', '.join(detail)})")

    # --- coverage ----------------------------------------------------------
    sparse = (rng.random((80, 80, 80)) > 0.70).astype(np.uint8)
    cov = (compute(sparse, (1, 0, 0), RCAP, 6, "dir", "void") < 1.25).mean()
    check("void phase carries the majority of a porous cell", cov > 0.6,
          f"non-sentinel fraction {cov:.3f}")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--outputdir")
    p.add_argument("--mode", choices=("iso", "dir"), default="dir")
    p.add_argument("--direction", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                   help="(dx, dy, dz) in the solver frame; ignored by --mode iso")
    p.add_argument("--rcap", type=float, default=10.0,
                   help="distance cap in voxels, a FIXED global scale; the "
                        "default covers DATA/aniso3d's dt p99.9")
    p.add_argument("--horizon", type=int, default=6,
                   help="steps along ±d for --mode dir; matches the cone "
                        "height in filtration3d.py")
    p.add_argument("--phase", choices=sorted(PHASES), default="void")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.outputdir):
        p.error("--dataset and --outputdir are required (or use --self-test)")
    run(a.dataset, a.outputdir, tuple(a.direction), a.rcap, a.horizon,
        a.mode, a.phase, a.workers, a.start, a.limit)


if __name__ == "__main__":
    main()
