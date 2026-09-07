r"""
Direction-aware 2D filtration: the wedge + PCA analogue of the 3D cone + PCA.

Port of `direction-aware-tda-for-porous-materials/scripts/src/filtrations/`
to 2D, with three deliberate departures from the 3D original — see below.

Channels, per pixel, for a chosen direction d = (cos θ, sin θ)
-------------------------------------------------------------
0. **wedge** — the 2D double cone. Offsets (dx, dy) with

       along = dx cosθ + dy sinθ ,   perp = -dx sinθ + dy cosθ

   lie inside when |along| <= height and |perp| <= radius * |along| / height,
   i.e. a bow-tie with its apex on the pixel, opening along ±d.

       wedge = 1 - (solid pixels inside) / (wedge area)

   0 means the whole wedge along the flow direction is solid; near 1 means it
   is open. This is the channel that carries directional connectivity.

1. **dir** — local orientation. PCA over solid pixels in a disc of `radius`,
   top eigenvector v, then `dir = 1 - |v·d|`. Small means the local solid
   structure runs along d.

Void pixels get the sentinel 1.25 in both channels, as in 3D, so they enter
every sublevel set only after all solid pixels.

Departures from the 3D code, all intentional
--------------------------------------------
1. **No grid rotation.** `pipeline.py: rotate_grid_to_z` transposes for
   axis-aligned directions and falls back to a lossy nearest-neighbour
   `affine_transform` for diagonals. In 2D there is no reason to pay that: the
   wedge offsets are built directly at angle θ, so every direction is exact and
   all directions are treated identically. Diagonals are no longer second-class.
2. **One PCA channel, not two.** In 3D, `dir_y` and `dir_z` are independent. In
   2D the perpendicular component carries no extra information, since
   |v·d|² + |v·d_perp|² = 1 — a second channel would be a deterministic
   function of the first and would only inflate the ECP grid.
3. **Periodic in both axes.** The cells are toroidal, unlike the 3D structures
   where only z wrapped.

Frame
-----
Arrays are indexed [row, col] = [solver_y, solver_x], the convention
`structure_io.load_structure` returns. θ is measured counter-clockwise from +x,
so it is directly comparable with the generator's ψ and the solver's
`theta_deg`. Do not transpose the array.

Usage
-----
    python scripts/filtration2d.py --self-test
    python scripts/filtration2d.py --dataset DATA/aniso --outputdir DESC/filt \
        --direction 45 [--radius 4] [--wedge-radius 3] [--wedge-height 6] \
        [--workers 14] [--limit N]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io import load_structure          # noqa: E402

VOID = np.float32(1.25)


# ---------------------------------------------------------------------------
# Offset stencils
# ---------------------------------------------------------------------------

def wedge_offsets(theta_deg: float, radius: float, height: float) -> np.ndarray:
    """(di, dj) offsets inside the double wedge pointing along ±θ."""
    t = np.radians(theta_deg)
    ct, st = np.cos(t), np.sin(t)
    lim = int(np.ceil(height + radius)) + 1

    # Tolerance, not cosmetic: cos(90 deg) is 6.1e-17 rather than 0, which
    # perturbs offsets sitting exactly on the wedge boundary.  Without it,
    # (di=2, dj=1) is admitted at theta=90 but (di=-2, dj=1) is not, so the
    # stencil loses central symmetry and theta=90 gets 42 offsets where
    # theta=0 gets 48 -- different directions would then be measured with
    # different-sized wedges, and rotation equivariance breaks.
    EPS = 1e-9

    offs = []
    for di in range(-lim, lim + 1):          # di = dy
        for dj in range(-lim, lim + 1):      # dj = dx
            if di == 0 and dj == 0:
                continue
            along = dj * ct + di * st
            perp = -dj * st + di * ct
            if (abs(along) <= height + EPS
                    and abs(perp) <= radius * abs(along) / height + EPS):
                offs.append((di, dj))
    return np.asarray(offs, dtype=np.int16).reshape(-1, 2)


def disc_offsets(radius: int) -> np.ndarray:
    """(di, dj) offsets with di² + dj² <= radius²."""
    r2 = radius * radius
    offs = [(di, dj)
            for di in range(-radius, radius + 1)
            for dj in range(-radius, radius + 1)
            if di * di + dj * dj <= r2]
    return np.asarray(offs, dtype=np.int16).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@njit(inline="always")
def _at(grid, i, j):
    return grid[i % grid.shape[0], j % grid.shape[1]]


@njit(inline="always")
def _top_evec_2x2(a, b, c):
    """
    Top eigenvector of [[a, b], [b, c]], closed form.

    The 3D code needs 12 rounds of power iteration for a 3x3; in 2D the
    eigenproblem is a quadratic, so this is exact and much cheaper.
    """
    if b == 0.0:
        return (1.0, 0.0) if a >= c else (0.0, 1.0)
    lam = 0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b * b)
    vi, vj = lam - c, b
    n = np.sqrt(vi * vi + vj * vj)
    if n == 0.0:
        return 1.0, 0.0
    return vi / n, vj / n


@njit(parallel=True, cache=True)
def _filtration(grid, wedge_off, disc_off, d_i, d_j):
    """grid: uint8 (H, W), 1 = solid. Returns (H, W, 2) float32."""
    H, W = grid.shape
    out = np.empty((H, W, 2), dtype=np.float32)
    n_wedge = wedge_off.shape[0]

    for i in prange(H):
        for j in range(W):
            if grid[i, j] == 0:
                out[i, j, 0] = 1.25
                out[i, j, 1] = 1.25
                continue

            # --- channel 0: wedge occupancy -------------------------------
            filled = 0
            for k in range(n_wedge):
                if _at(grid, i + wedge_off[k, 0], j + wedge_off[k, 1]) != 0:
                    filled += 1
            out[i, j, 0] = 1.0 - filled / n_wedge if n_wedge > 0 else 1.0

            # --- channel 1: local orientation vs d ------------------------
            n = 0.0
            si = sj = sii = sij = sjj = 0.0
            for k in range(disc_off.shape[0]):
                di = disc_off[k, 0]
                dj = disc_off[k, 1]
                if _at(grid, i + di, j + dj) == 0:
                    continue
                n += 1.0
                fi = float(di)
                fj = float(dj)
                si += fi
                sj += fj
                sii += fi * fi
                sij += fi * fj
                sjj += fj * fj

            if n < 2.0:
                out[i, j, 1] = 1.0
                continue

            mi, mj = si / n, sj / n
            cii = sii / n - mi * mi
            cij = sij / n - mi * mj
            cjj = sjj / n - mj * mj
            vi, vj = _top_evec_2x2(cii, cij, cjj)
            # v is (v_row, v_col) = (v_y, v_x); d is (d_i, d_j) = (sinθ, cosθ)
            out[i, j, 1] = 1.0 - abs(vi * d_i + vj * d_j)

    return out


def compute(grid: np.ndarray, theta_deg: float, radius: int = 4,
            wedge_radius: float = 3.0, wedge_height: float = 6.0) -> np.ndarray:
    """Filtration of a binary structure (1 = solid) along θ degrees from +x."""
    t = np.radians(theta_deg)
    return _filtration(np.ascontiguousarray(grid, dtype=np.uint8),
                       wedge_offsets(theta_deg, wedge_radius, wedge_height),
                       disc_offsets(radius),
                       float(np.sin(t)), float(np.cos(t)))


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _one(args):
    path, out_path, theta, radius, wr, wh = args
    if os.path.exists(out_path):
        return "cached"
    filt = compute(load_structure(path), theta, radius, wr, wh)
    np.save(out_path, filt)
    return "ok"


def run(dataset, outputdir, theta, radius, wr, wh, workers, limit):
    import pandas as pd
    df = pd.read_csv(os.path.join(dataset, "structures.csv"))
    if limit:
        df = df.head(limit)
    os.makedirs(outputdir, exist_ok=True)

    tasks = [(os.path.join(dataset, "structures", r.filename),
              os.path.join(outputdir, r.filename.replace(".gif", ".npy")),
              theta, radius, wr, wh)
             for r in df.itertuples()]

    from tqdm import tqdm
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs),
                      desc=f"filtration θ={theta:g}°"):
            done += f.result() == "ok"
    print(f"  {done} computed, {len(tasks) - done} cached -> {outputdir}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    # wedge geometry: symmetric, opens along θ
    for th in (0.0, 45.0, 90.0):
        w = wedge_offsets(th, 3.0, 6.0)
        sym = {(a, b) for a, b in w} == {(-a, -b) for a, b in w}
        check(f"wedge θ={th:g}° is centrally symmetric (double-ended)", sym,
              f"{len(w)} offsets")

    w0 = wedge_offsets(0.0, 3.0, 6.0)
    spread_j = np.abs(w0[:, 1]).max()      # along x
    spread_i = np.abs(w0[:, 0]).max()      # across
    check("θ=0° wedge extends along x, not y", spread_j > spread_i,
          f"|dx|max={spread_j} |dy|max={spread_i}")

    # all-solid -> wedge 0 ; void -> sentinel
    solid = np.ones((32, 32), dtype=np.uint8)
    f = compute(solid, 0.0)
    check("all-solid gives wedge = 0", np.allclose(f[..., 0], 0.0),
          f"max {f[..., 0].max():.3g}")
    g = np.ones((32, 32), dtype=np.uint8)
    g[16, 16] = 0
    f = compute(g, 0.0)
    check("void pixel gets the 1.25 sentinel", f[16, 16, 0] == 1.25)

    # equal-size stencils across directions (the FP-boundary bug above)
    counts = {th: len(wedge_offsets(th, 3.0, 6.0)) for th in (0, 45, 90, 135)}
    check("wedge stencil size is direction-independent for θ and θ+90",
          counts[0] == counts[90] and counts[45] == counts[135], str(counts))

    # direction sensitivity: horizontal bars are open along x, blocked across.
    # Bars must be thicker than the wedge half-width (3), or even the aligned
    # wedge sticks out into void and the contrast washes out.
    stripes = np.zeros((64, 64), dtype=np.uint8)
    for k in range(0, 64, 16):
        stripes[k:k + 8] = 1                # 8-thick bars running along x
    f0 = compute(stripes, 0.0)              # along the bars
    f90 = compute(stripes, 90.0)            # across them
    m0 = f0[..., 0][stripes == 1].mean()
    m90 = f90[..., 0][stripes == 1].mean()
    check("wedge is direction-sensitive on bars", m0 < m90 - 0.3,
          f"mean wedge along={m0:.3f} across={m90:.3f}")

    # PCA channel: on bars running along x, dir ~ 0 at θ=0 and ~1 at θ=90
    d0 = f0[..., 1][stripes == 1].mean()
    d90 = f90[..., 1][stripes == 1].mean()
    check("PCA channel aligns with the bar direction", d0 < 0.1 and d90 > 0.9,
          f"dir(θ=0)={d0:.3f} dir(θ=90)={d90:.3f}")

    # rotating the structure by 90° must rotate the filtration the same way
    rng = np.random.default_rng(0)
    s = (rng.random((64, 64)) > 0.6).astype(np.uint8)
    a = compute(s, 0.0)[..., 0]
    b = compute(np.rot90(s), 90.0)[..., 0]
    check("rot90(structure) at θ+90° equals rot90(filtration)",
          np.allclose(np.rot90(a), b, atol=1e-6),
          f"max diff {np.abs(np.rot90(a) - b).max():.2e}")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--outputdir")
    p.add_argument("--direction", type=float, default=0.0,
                   help="degrees CCW from +x (same frame as theta_deg)")
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--wedge-radius", type=float, default=3.0)
    p.add_argument("--wedge-height", type=float, default=6.0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.outputdir):
        p.error("--dataset and --outputdir are required (or use --self-test)")
    run(a.dataset, a.outputdir, a.direction, a.radius,
        a.wedge_radius, a.wedge_height, a.workers, a.limit)


if __name__ == "__main__":
    main()
