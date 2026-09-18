r"""
3D persistent homology of the cone channel, vectorised as persistence images.

The 3D counterpart of `ph2d.py`, and a port of the elasticity repo's
`ph_calc.py` with two changes:

  * **Periodic in all three axes.** The original passes
    `periodic_dimensions=[False, False, True]` because only z wrapped in that
    dataset — and its anisotropic stratum was not periodic at all (NOTES 8).
    Our cells are 3-torus periodic by construction and verified
    (`mean(wrap - adjacent agreement) = -0.00006`, NOTES 8.2), and that torus is
    what the LBM solves, so all three wrap.
  * **Failures are surfaced, not swallowed.** `ph_calc.process_directory`
    submits its futures and never calls `.result()`, so a per-structure
    exception silently produces a missing file instead of an error. Here every
    result is collected and counted, as in 2D.

Dimensions 0, 1 and 2 — unlike 2D, H2 exists, and NOTES 10.3 registers the
prediction that this is what makes PH worth computing in 3D after it was dead
weight in 2D. That is only testable if it is computed, so it is.

Channel 0 only, as in 2D and 3D alike: PH runs on the cone (scalar) channel and
the orientation channels feed ECP. A consequence worth remembering is that
`--radius` cannot change the PH features at all, so there is no point scanning
it for PH (the elasticity repo's `notes/why_ph_insensitive_to_radius.md`).

Essential (infinite) bars are dropped by `DiagramSelector(point_type="finite")`.

The weight and the bandwidth — and why this script grew two flags
-----------------------------------------------------------------
Until 2026-09-17 this script called `PersistenceImage(resolution, im_range)`
with **both of gudhi's defaults**, and both are wrong for porous structures:

  * `weight` defaults to **constant**, so a bar of length 1e-3 contributes as
    much as one of length 1.  A porous cell yields thousands of near-diagonal
    noise bars and a handful of structural ones, so the image becomes a smeared
    bar histogram.
  * `bandwidth` defaults to **1.0**, in persistence units, not pixels.  At
    `--im_range 0 1.25` and `--resolution 10` a pixel is 0.125 wide, so the
    kernel is **eight pixels across a ten-pixel axis**: every diagram paints
    the same blob scaled by one number.

Both were found and fixed in `ph2d.py` in session 7 (NOTES 11.5, 11.5a) and the
fix was never ported here.  Measured on `DESC/aniso3d`, direction [1,0,0], the
100-column blocks come out at **top-1 variance .9960 (H0), .9991 (H1), .9997
(H2)** — one effective degree of freedom each, the same signature 2D showed.
Every recorded 3D `ph` and `tda` number therefore rests on rank-1 images; in 2D
repairing this gained **+.097** on the `ph` group.

`--weight linear` (the default now) weights each bar by its persistence.
**gudhi applies `BirthPersistenceTransform` before calling the weight**, so the
function receives `(birth, persistence)` and the linear weight is `x[1]`.

Bandwidth is only meaningful relative to the pixel size, which `--im_range` and
`--resolution` jointly set, so this script prints the ratio per dimension at
startup and flags a kernel wider than a third of the shorter extent as
`SMEARED`.  Because H0, H1 and H2 live on different scales, `--im_range` and
`--bandwidth` each take a shared value plus per-dimension overrides
`--im_range_h{0,1,2}` and `--bandwidth_h{0,1,2}`.  A per-dimension range
*without* a matching per-dimension bandwidth silently reintroduces the original
bug on the tighter dimension.

`DESC/aniso3d/ph` was built with the old defaults.  Anything compared against
it must use `--weight constant --bandwidth 1.0`, or that comparison is
confounded — see NOTES 11.11 for what that costs.

Cost, measured on a real 80^3 structure: 4.05 s and 420 MB peak RSS per
structure. This is the most expensive stage of the pipeline.

Usage
-----
    python scripts/ph3d.py --inputdir DESC/aniso3d/filt/a1_1_0 \
        --outputdir DESC/aniso3d/ph/a1_1_0 \
        [--weight linear] [--resolution 10 10] \
        [--im_range 0 1.25 0 1.25] [--bandwidth 0.125] \
        [--im_range_h1 0 1 0 0.3] [--bandwidth_h1 0.03] [--workers 24]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gudhi as gd
from gudhi.representations import DiagramSelector, PersistenceImage
from tqdm import tqdm

DIMS = (0, 1, 2)

# Built inside the worker: a lambda would not survive being pickled to the pool.
# The argument is (birth, persistence) -- see the docstring.
WEIGHTS = {
    "constant":  lambda x: 1.0,
    "linear":    lambda x: x[1],
    "quadratic": lambda x: x[1] * x[1],
}


def _one(args):
    fname, inputdir, outputdir, resolution, params, weight = args
    out = os.path.join(outputdir, fname)
    if os.path.exists(out):
        return "cached"

    filt = np.load(os.path.join(inputdir, fname))
    if filt.ndim == 4:
        filt = filt[:, :, :, 0]                    # cone channel

    cx = gd.PeriodicCubicalComplex(
        top_dimensional_cells=filt.astype(np.float64),
        periodic_dimensions=[True, True, True])
    cx.compute_persistence()

    sel = DiagramSelector(use=True, limit=np.inf, point_type="finite")

    vecs = []
    for d in DIMS:
        bandwidth, im_range = params[d]
        bars = sel(cx.persistence_intervals_in_dimension(d))
        if len(bars) == 0:
            vecs.append(np.zeros(resolution[0] * resolution[1]))
        else:
            imager = PersistenceImage(bandwidth=bandwidth,
                                      resolution=list(resolution),
                                      im_range=list(im_range),
                                      weight=WEIGHTS[weight])
            vecs.append(np.asarray(imager(bars)).ravel())
    np.save(out, np.concatenate(vecs))
    return "ok"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputdir", required=True)
    p.add_argument("--outputdir", required=True)
    p.add_argument("--weight", choices=sorted(WEIGHTS), default="linear",
                   help="per-bar weight in the persistence image. gudhi's own "
                        "default is 'constant', which is what DESC/aniso3d/ph "
                        "was built with; see the docstring")
    p.add_argument("--resolution", type=int, nargs=2, default=[10, 10])
    p.add_argument("--im_range", type=float, nargs=4, default=[0.0, 1.25, 0.0, 1.25],
                   metavar=("BMIN", "BMAX", "PMIN", "PMAX"),
                   help="birth/persistence extent, shared by all dimensions "
                        "unless overridden below")
    p.add_argument("--bandwidth", type=float, default=1.0,
                   help="Gaussian sigma in PERSISTENCE UNITS, not pixels; "
                        "gudhi's default of 1.0 is ~8 px at the default range "
                        "and resolution, which flattens the image to rank 1")
    for d in DIMS:
        p.add_argument(f"--im_range_h{d}", type=float, nargs=4, default=None,
                       metavar=("BMIN", "BMAX", "PMIN", "PMAX"),
                       help=f"override --im_range for H{d}")
        p.add_argument(f"--bandwidth_h{d}", type=float, default=None,
                       help=f"override --bandwidth for H{d}")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    # Per-dimension (bandwidth, im_range): the shared flag unless overridden.
    params = {}
    for d in DIMS:
        rng = getattr(a, f"im_range_h{d}") or a.im_range
        bw = getattr(a, f"bandwidth_h{d}")
        params[d] = (a.bandwidth if bw is None else bw, tuple(rng))

    os.makedirs(a.outputdir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.inputdir) if f.endswith(".npy"))

    print(f"  weight: {a.weight}   resolution: {a.resolution[0]}x{a.resolution[1]}")
    for d in DIMS:
        bw, rng = params[d]
        px_b = (rng[1] - rng[0]) / a.resolution[0]
        px_p = (rng[3] - rng[2]) / a.resolution[1]
        # Degeneracy is the kernel covering a large share of an AXIS, not of a
        # pixel: on a strongly anisotropic grid any bandwidth sane on the birth
        # axis is several pixels on the persistence axis, so a pixel-based test
        # flags good configurations too (NOTES 11.5c).
        span = min(rng[1] - rng[0], rng[3] - rng[2])
        flag = "  <-- SMEARED" if bw > span / 3 else ""
        print(f"  H{d}: range b[{rng[0]:g},{rng[1]:g}] p[{rng[2]:g},{rng[3]:g}]  "
              f"bandwidth {bw:g} = {bw / px_b:.1f} px birth, "
              f"{bw / px_p:.1f} px persistence{flag}")

    tasks = [(f, a.inputdir, a.outputdir, tuple(a.resolution), params, a.weight)
             for f in files]

    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="PH"):
            done += f.result() == "ok"
    n_feat = 3 * a.resolution[0] * a.resolution[1]
    print(f"  {done} computed, {len(tasks) - done} cached, "
          f"{n_feat} features each, weight={a.weight} -> {a.outputdir}")


if __name__ == "__main__":
    main()
