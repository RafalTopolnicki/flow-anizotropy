r"""
Throat-scale 2D filtration: the distance transform the repo never had.

Motivation — why a distance transform and not another occupancy stencil
-----------------------------------------------------------------------
`filtration2d.py` measures *occupancy*: the fraction of a wedge that is solid.
It is a local texture statistic, and NOTES 11.2 shows why that is a losing
bet — for a Boolean model or a Gaussian excursion set the Minkowski functionals
of occupancy fields have closed forms in the same parameters that fix the
two-point correlation, so ECP re-encodes TPC and PH inherits most of it.

Critical path analysis says permeability is set by something occupancy cannot
express: the **narrowest constriction on the widest spanning path**.  For a
pore space that quantity is a *radius*, not a fraction — the inscribed-disc
radius at the throat.  The Euclidean distance transform of the pore space,

    dt(x) = distance from pore pixel x to the nearest solid pixel,

is exactly the inscribed radius at x, and a sublevel filtration of `-dt` grows
the pore space from the widest regions outward, so an **H0 death time is a
throat radius**: the radius at which two wide pore bodies first merge.  That is
the critical-path quantity in the persistence diagram, directly.

NOTES 11.8 records that no distance transform exists anywhere in this repo:
the elasticity original's `distance_transfrom_nondirect.py` has
`material`/`void`/`both` options but is non-directional, no run script there
uses it, and it was never ported.  This is that port, plus a directional mode.

The two modes
-------------
`--mode iso` — the non-directional transform, i.e. the original's `void`
option.  The filtration value on a pore pixel is

    f = 1 - min(dt, RCAP) / RCAP        in [0, 1]

so 0 is the middle of the widest pore body and f -> 1 at the walls.  Solid
pixels get the 1.25 sentinel, as everywhere else in this pipeline, so they
enter every sublevel set last and the filtration is *about* the pore space.
Because sublevel sets of f are erosions of the pore space, `{f <= t}` is the
set of pixels with inscribed radius >= RCAP(1-t), and an H0 merge at level t
means two bodies connect through a throat of radius r = RCAP(1-t).

`--mode dir` — directional, and it is the mode that carries the flow direction.
Walk along ±d from each pore pixel, at most `--horizon` steps, stopping at
solid, and take the *minimum* dt encountered:

    dt_d(x) = min { dt(x + s d) : |s| <= horizon, path not yet blocked }

This is "the narrowest constriction you meet travelling along d from here" —
the localised critical-path quantity.  It gets the two limiting cases right,
which the obvious alternative does not: in a channel running along x, at θ=0
the walk stays inside the channel and reads the half-width (open), while at
θ=90 it reaches the wall in one or two steps and reads a radius near 1
(blocked).  An anisotropic *metric* — cheap displacements along d, expensive
across — reads that same blocked channel as "infinitely wide in x" and is
therefore wrong; it was tried and rejected before this was written.

RCAP is a fixed global constant, not a per-structure maximum
------------------------------------------------------------
Normalising each structure by its own `dt.max()` would destroy the absolute
throat scale, and the absolute scale is the whole point: k ~ r^2 for a throat
of radius r, so two structures whose throats differ by a factor of two must not
map to the same filtration values.  `--rcap` defaults to 28 px, the p99 of dt
over `DATA/aniso` (measured: p50 6.4, p75 11.0, p90 16.1, p99 27.7, per-
structure max 13.9-53.7).  Re-measure it for a different family — a
low-porosity dataset will want a much smaller cap — and record it, because
descriptors built with different caps are not comparable.

Periodicity
-----------
The cells are toroidal.  scipy's `distance_transform_edt` is not periodic, so
the pore mask is wrap-padded by RCAP+2 before the transform and cropped after.
That is exact for every distance the cap keeps: a pixel whose nearest solid
lies beyond the pad has true dt > RCAP and is capped anyway.  The 3x3-tiling
trick used elsewhere in the repo would also work and costs 9x the area.

Frame
-----
Arrays are [row, col] = [solver_y, solver_x], as `load_structure` returns; θ is
CCW from +x, directly comparable with the generator's ψ and the solver's
`theta_deg`.  Do not transpose.

Output
------
One `(H, W)` float32 array per structure — a single channel, unlike
`filtration2d.py`'s two.  `ph2d.py` reads 2D input unchanged.  There is no PCA
orientation channel here because this filtration has no use for one: ECP is not
the target of this experiment (NOTES: step 0 tests whether PH on a throat-scale
filtration can clear the `tpc` bar of .6842 on k_off).

Usage
-----
    python scripts/filtration_throat2d.py --self-test
    python scripts/filtration_throat2d.py --dataset DATA/aniso \
        --outputdir DESC/aniso_throat/filt/a45 --mode dir --direction 45 \
        [--rcap 28] [--horizon 6] [--phase void|solid] [--workers 14]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numba import njit, prange
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io import load_structure          # noqa: E402

VOID = np.float32(1.25)
PHASES = {"solid": 1, "void": 0}


# ---------------------------------------------------------------------------
# Periodic distance transform
# ---------------------------------------------------------------------------

def periodic_edt(grid: np.ndarray, fg: int, rcap: float) -> np.ndarray:
    """
    Distance from every `fg` pixel to the nearest non-`fg` pixel, on the torus.

    Wrap-padding by rcap+2 makes this exact wherever the result is <= rcap,
    which is all the caller keeps.  Non-`fg` pixels come back 0.
    """
    pad = int(np.ceil(rcap)) + 2
    mask = np.pad(grid == fg, pad, mode="wrap")
    dt = distance_transform_edt(mask)
    return np.ascontiguousarray(dt[pad:-pad, pad:-pad], dtype=np.float32)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _iso(grid, dt, rcap, fg):
    """f = 1 - min(dt, rcap)/rcap on the fg phase, 1.25 elsewhere."""
    H, W = grid.shape
    out = np.empty((H, W), dtype=np.float32)
    for i in prange(H):
        for j in range(W):
            if grid[i, j] != fg:
                out[i, j] = 1.25
                continue
            d = dt[i, j]
            if d > rcap:
                d = rcap
            out[i, j] = 1.0 - d / rcap
    return out


@njit(parallel=True, cache=True)
def _dirmin(grid, dt, d_i, d_j, horizon, rcap, fg):
    """
    f = 1 - min(dt along ±d)/rcap, the walk stopping when it leaves the phase.

    The walk samples at unit steps along d and rounds to the nearest pixel, so
    every direction is treated identically (no grid rotation, matching
    departure 1 in filtration2d.py).  Stopping at the phase boundary rather
    than injecting a zero there keeps the field smooth: the pore pixel next to
    a wall already carries dt ~ 1, which is the signal that the path is
    blocked.
    """
    H, W = grid.shape
    out = np.empty((H, W), dtype=np.float32)
    for i in prange(H):
        for j in range(W):
            if grid[i, j] != fg:
                out[i, j] = 1.25
                continue
            m = dt[i, j]
            for sgn in range(-1, 2, 2):
                for s in range(1, horizon + 1):
                    ii = int(np.floor(i + sgn * s * d_i + 0.5)) % H
                    jj = int(np.floor(j + sgn * s * d_j + 0.5)) % W
                    if grid[ii, jj] != fg:
                        break
                    v = dt[ii, jj]
                    if v < m:
                        m = v
            if m > rcap:
                m = rcap
            out[i, j] = 1.0 - m / rcap
    return out


def compute(grid: np.ndarray, theta_deg: float = 0.0, rcap: float = 28.0,
            horizon: int = 6, mode: str = "dir",
            phase: str = "void") -> np.ndarray:
    """Throat-scale filtration of a binary structure (1 = solid)."""
    fg = np.uint8(PHASES[phase])
    g = np.ascontiguousarray(grid, dtype=np.uint8)
    dt = periodic_edt(g, int(fg), rcap)
    if mode == "iso":
        return _iso(g, dt, np.float32(rcap), fg)
    t = np.radians(theta_deg)
    return _dirmin(g, dt, float(np.sin(t)), float(np.cos(t)),
                   int(horizon), np.float32(rcap), fg)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _one(args):
    path, out_path, theta, rcap, horizon, mode, phase = args
    if os.path.exists(out_path):
        return "cached"
    filt = compute(load_structure(path), theta, rcap, horizon, mode, phase)
    np.save(out_path, filt)
    return "ok"


def run(dataset, outputdir, theta, rcap, horizon, mode, phase, workers, limit):
    import pandas as pd
    df = pd.read_csv(os.path.join(dataset, "structures.csv"))
    if limit:
        df = df.head(limit)
    os.makedirs(outputdir, exist_ok=True)

    tasks = [(os.path.join(dataset, "structures", r.filename),
              os.path.join(outputdir, r.filename.replace(".gif", ".npy")),
              theta, rcap, horizon, mode, phase)
             for r in df.itertuples()]

    from tqdm import tqdm
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"throat {mode} {phase} θ={theta:g}°"):
            done += f.result() == "ok"
    print(f"  {done} computed, {len(tasks) - done} cached, "
          f"mode={mode} rcap={rcap:g} horizon={horizon} -> {outputdir}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    RCAP = 28.0

    # --- the transform itself ---------------------------------------------
    # One solid pixel in an otherwise empty cell: dt is the distance to it,
    # measured on the torus.  Checked against an explicit minimum over the
    # nine translates, which is what periodicity means here.
    g = np.zeros((64, 64), dtype=np.uint8)
    g[20, 30] = 1
    dt = periodic_edt(g, 0, RCAP)
    ii, jj = np.mgrid[0:64, 0:64]
    best = np.full((64, 64), np.inf)
    for si in (-64, 0, 64):
        for sj in (-64, 0, 64):
            best = np.minimum(best, np.hypot(ii - (20 + si), jj - (30 + sj)))
    near = best <= RCAP
    check("periodic EDT matches the min over the 9 translates",
          np.allclose(dt[near], best[near], atol=1e-5),
          f"max diff {np.abs(dt[near] - best[near]).max():.2e}")

    # The wrap must actually be used: a pixel at row 0 is 1 px from a solid
    # wall at row 63, not 63 px from one.
    g = np.zeros((32, 32), dtype=np.uint8)
    g[31, :] = 1
    dt = periodic_edt(g, 0, RCAP)
    check("EDT wraps across the seam", np.isclose(dt[0, 5], 1.0),
          f"dt[0,5]={dt[0,5]:.3f} (non-periodic would give 31)")

    # --- iso mode ----------------------------------------------------------
    f = compute(g, 0.0, RCAP, mode="iso")
    check("solid pixels get the 1.25 sentinel", np.all(f[31, :] == 1.25))
    check("iso: pore next to a wall is near 1",
          np.isclose(f[0, 5], 1.0 - 1.0 / RCAP),
          f"f={f[0, 5]:.4f}")
    # The widest point of a 31 px slab is 16 px from a wall, so f = 1 - 16/28;
    # nothing in this cell can reach the cap.
    check("iso: filtration is in [0, 1] on the pore phase",
          f[g == 0].min() >= 0.0 and f[g == 0].max() <= 1.0,
          f"[{f[g == 0].min():.3f}, {f[g == 0].max():.3f}]")

    # A cell with no solid at all: dt exceeds the cap everywhere, so the whole
    # pore space sits at 0 and the filtration is flat.  This is the case that
    # a per-structure normalisation would silently turn into noise.
    empty = np.zeros((80, 80), dtype=np.uint8)
    check("iso: capped flat when no solid is within rcap",
          np.allclose(compute(empty, 0.0, RCAP, mode="iso"), 0.0))

    # iso must not depend on the direction argument at all.
    rng = np.random.default_rng(0)
    s = (rng.random((64, 64)) > 0.8).astype(np.uint8)
    a0 = compute(s, 0.0, RCAP, mode="iso")
    a37 = compute(s, 37.0, RCAP, mode="iso")
    check("iso is direction-independent", np.array_equal(a0, a37))

    # --- phase symmetry ----------------------------------------------------
    # void(g) == solid(1-g) exactly, in both modes.  This is the check that
    # catches a half-applied phase swap -- the EDT following `fg` while the
    # walk does not, or vice versa.
    g = (rng.random((64, 64)) > 0.55).astype(np.uint8)
    for mode in ("iso", "dir"):
        for th in (0.0, 45.0):
            v = compute(g, th, RCAP, 6, mode, "void")
            c = compute((1 - g).astype(np.uint8), th, RCAP, 6, mode, "solid")
            check(f"{mode}: void(g) == solid(1-g) at θ={th:g}°",
                  np.array_equal(v, c),
                  f"max diff {np.abs(v - c).max():.2e}")

    # --- dir mode: the two limiting cases ----------------------------------
    # Channels running along x, 8 px of pore between 8 px walls.  Along the
    # channel the walk stays inside and reads the half-width; across it the
    # walk hits a wall and reads a radius near 1.  This is the pair of cases
    # that rejected the anisotropic-metric design, which reports the blocked
    # direction as wide open.
    walls = np.zeros((64, 64), dtype=np.uint8)
    for k in range(0, 64, 16):
        walls[k:k + 8] = 1
    pore = walls == 0
    d0 = compute(walls, 0.0, RCAP, 6, "dir", "void")
    d90 = compute(walls, 90.0, RCAP, 6, "dir", "void")
    m0, m90 = d0[pore].mean(), d90[pore].mean()
    check("dir: channel reads open along it and blocked across it",
          m0 < m90 - 0.05, f"mean f along={m0:.4f} across={m90:.4f}")
    # Sharper: at the centre of a channel, along-channel the whole horizon is
    # 4 px from a wall, while across it the walk reaches the wall.
    ci = 12                                    # centre row of the pore band
    check("dir: channel centre keeps its half-width along the channel",
          np.isclose(d0[ci, 32], 1.0 - 4.0 / RCAP, atol=1e-6),
          f"f={d0[ci, 32]:.4f} expected {1.0 - 4.0 / RCAP:.4f}")
    check("dir: channel centre is pulled to the wall radius across it",
          np.isclose(d90[ci, 32], 1.0 - 1.0 / RCAP, atol=1e-6),
          f"f={d90[ci, 32]:.4f} expected {1.0 - 1.0 / RCAP:.4f}")

    # dir <= iso pointwise in f-terms means dt_d <= dt, i.e. a minimum over a
    # walk that includes s=0 can only shrink.  A sign slip in the walk shows up
    # here immediately.
    s = (rng.random((64, 64)) > 0.75).astype(np.uint8)
    fi = compute(s, 30.0, RCAP, 6, "iso", "void")
    fd = compute(s, 30.0, RCAP, 6, "dir", "void")
    p = s == 0
    check("dir >= iso on the pore phase (the walk can only narrow)",
          np.all(fd[p] >= fi[p] - 1e-6),
          f"worst violation {np.min(fd[p] - fi[p]):.2e}")

    # horizon 0 collapses the walk, so dir must reduce to iso exactly.
    f0 = compute(s, 30.0, RCAP, 0, "dir", "void")
    check("dir with horizon 0 reduces to iso", np.array_equal(f0, fi))

    # --- equivariance ------------------------------------------------------
    # rot90 of the structure at θ+90 must be rot90 of the filtration.  The 3D
    # code pays a lossy affine_transform for this; here the walk is built at θ
    # so it should hold to the floating-point rounding of the step positions.
    a = compute(s, 0.0, RCAP, 6, "dir", "void")
    b = compute(np.rot90(s), 90.0, RCAP, 6, "dir", "void")
    check("rot90(structure) at θ+90° equals rot90(filtration)",
          np.allclose(np.rot90(a), b, atol=1e-6),
          f"max diff {np.abs(np.rot90(a) - b).max():.2e}")

    # --- the property the whole descriptor rests on ------------------------
    # An H0 death time must BE a throat radius.  Two circular chambers joined
    # by a straight throat of known half-width w: the long-lived finite H0 bar
    # is the second chamber merging into the first, and it must die at the
    # level t with RCAP(1 - t) = w.  This is the claim that distinguishes this
    # filtration from an occupancy stencil, so it is checked rather than
    # asserted -- and it is checked across a 4x range of w, because a single
    # width would also pass under a wrong but monotone calibration.
    try:
        import gudhi as gd
        from gudhi.representations import DiagramSelector
    except ImportError:
        print("[skip] H0 death times recover throat radii (gudhi not importable)")
    else:
        sel = DiagramSelector(use=True, limit=np.inf, point_type="finite")
        worst, detail = 0.0, []
        for w in (2, 3, 4, 6, 8):
            N, R, sep = 160, 18, 60
            g = np.ones((N, N), dtype=np.uint8)
            ii, jj = np.mgrid[0:N, 0:N]
            c1j, c2j = N // 2 - sep // 2, N // 2 + sep // 2
            for cj in (c1j, c2j):
                g[(ii - N // 2) ** 2 + (jj - cj) ** 2 <= R * R] = 0
            g[N // 2 - w: N // 2 + w, c1j:c2j] = 0
            f = compute(g, 0.0, RCAP, 6, "iso", "void")
            cx = gd.PeriodicCubicalComplex(
                top_dimensional_cells=f.astype(np.float64),
                periodic_dimensions=[True, True])
            cx.compute_persistence()
            bars = np.asarray(sel(cx.persistence_intervals_in_dimension(0)))
            k = int(np.argmax(bars[:, 1] - bars[:, 0]))
            r = RCAP * (1.0 - bars[k, 1])
            worst = max(worst, abs(r - w))
            detail.append(f"w={w}->{r:.2f}")
        check("H0 death times recover known throat radii", worst < 0.05,
              f"max error {worst:.3f} px  ({', '.join(detail)})")

    # --- coverage ----------------------------------------------------------
    # The point of the void phase: the descriptor must carry the majority of a
    # porous cell, not a seventh of it (NOTES 11.8).
    sparse = (rng.random((256, 256)) > 0.85).astype(np.uint8)
    cov = (compute(sparse, 0.0, RCAP, 6, "dir", "void") < 1.25).mean()
    check("void phase carries the majority of a porous cell", cov > 0.8,
          f"non-sentinel fraction {cov:.3f}")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--outputdir")
    p.add_argument("--mode", choices=("iso", "dir"), default="dir",
                   help="iso: the non-directional transform (the elasticity "
                        "original's `void` option). dir: minimum dt along ±d")
    p.add_argument("--direction", type=float, default=0.0,
                   help="degrees CCW from +x (same frame as theta_deg); "
                        "ignored by --mode iso")
    p.add_argument("--rcap", type=float, default=28.0,
                   help="distance cap in pixels, a FIXED global scale; the "
                        "default is the p99 of dt over DATA/aniso")
    p.add_argument("--horizon", type=int, default=6,
                   help="steps along ±d for --mode dir; matches the wedge "
                        "height in filtration2d.py")
    p.add_argument("--phase", choices=sorted(PHASES), default="void",
                   help="phase the descriptor is about; void (the pore space) "
                        "is the flow-relevant one and the default here")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.outputdir):
        p.error("--dataset and --outputdir are required (or use --self-test)")
    run(a.dataset, a.outputdir, a.direction, a.rcap, a.horizon, a.mode,
        a.phase, a.workers, a.limit)


if __name__ == "__main__":
    main()
