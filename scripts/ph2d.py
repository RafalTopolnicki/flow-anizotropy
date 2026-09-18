r"""
2D persistent homology of the wedge channel, vectorised as persistence images.

The 2D counterpart of `direction-aware-tda/scripts/ph_calc.py`. Differences,
all forced by the dimension or by the periodic cell:

  * **Dimensions 0 and 1 only.** In 2D there are no enclosed cavities, so H2
    does not exist: 2 x resolution^2 features instead of 3 x.
  * **Periodic in both axes.** The 3D code passes
    `periodic_dimensions=[False, False, True]` because only z wrapped there.
    Our cells are toroidal, so both axes wrap — matching what the LBM solves.
  * **Channel 0 only**, as in 3D: PH runs on the wedge (scalar) channel. The
    orientation channel feeds ECP alone. A consequence worth remembering is
    that `--radius` cannot change the PH features at all, so there is no point
    scanning it for PH (the same fact is recorded in the 3D repo's
    `notes/why_ph_insensitive_to_radius.md`).

Essential (infinite) bars — one in H0, two in H1 for a torus — are dropped by
`DiagramSelector(point_type="finite")`, as in the 3D pipeline.

The persistence weight — `--weight`, and the reason this script has a flag
--------------------------------------------------------------------------
gudhi's `PersistenceImage` defaults to a **constant** weight, so a bar of length
1e-3 contributes as much as a bar of length 1. A porous structure yields
thousands of near-diagonal noise bars and a handful of long structural ones, so
under that default the image is a smeared bar histogram — a local texture
statistic. Measured consequence on the 2D dataset (NOTES 11.5): the 400 H0
columns carried **three** effective degrees of freedom and were **91%**
predictable from the two-point correlation, i.e. the representation discarded
the topology before any model saw it.

`--weight linear` (the default here) weights each bar by its persistence.
**gudhi applies `BirthPersistenceTransform` before calling the weight**, so the
function receives `(birth, persistence)` and the linear weight is `x[1]`, not
`x[1] - x[0]` — the latter would be persistence minus birth, which is not a
weighting anyone wants.

`DESC/aniso/ph/` and `DESC/aniso3d/ph/` were computed with `constant`, before
this flag existed. Anything compared against them must use the same setting.

The bandwidth — the second, larger degeneracy
---------------------------------------------
gudhi's `PersistenceImage` defaults to `bandwidth=1.0`: the standard deviation
of the Gaussian dropped on each point, in *persistence units*, not pixels. With
the old `--im_range 0 1.25` at `--resolution 10` a pixel is 0.125 wide, so the
default kernel is **eight pixels** across a ten-pixel axis. Every diagram then
paints the same blob scaled by one number, and the 400 H0 columns measured
**rank 1** (NOTES 11.5a). The weight above was not the cause of that; this was.

Bandwidth is therefore only meaningful *relative to the pixel size*, which
`--im_range` and `--resolution` jointly set. This script prints the ratio per
dimension at startup — keep it near 1-2, and treat a large value as the bug it
is.

Because H0 and H1 live on different scales (H1 bars here have p99 = 0.083
against H0's 1.25), one shared range puts 99% of H1 inside a single pixel row.
So `--im_range` and `--bandwidth` each take a shared value plus optional
per-dimension overrides `--im_range_h0/h1` and `--bandwidth_h0/h1`. A per-dim
range *without* a matching per-dim bandwidth silently reintroduces the original
bug on the tighter dimension.

Usage
-----
    python scripts/ph2d.py --inputdir DESC/filt/a45 --outputdir DESC/ph/a45 \
        [--weight linear] [--resolution 20 20] \
        [--bandwidth 0.125] [--bandwidth_h1 0.00625] \
        [--im_range 0 1.25 0 1.25] [--im_range_h1 0 1.0 0 0.10] \
        [--workers 14]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gudhi as gd
from gudhi.representations import DiagramSelector, PersistenceImage
from tqdm import tqdm

DIMS = (0, 1)

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
    if filt.ndim == 3:
        filt = filt[:, :, 0]                       # wedge channel

    cx = gd.PeriodicCubicalComplex(
        top_dimensional_cells=filt.astype(np.float64),
        periodic_dimensions=[True, True])
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
                        "default is 'constant', which is what DESC/aniso/ph "
                        "and DESC/aniso3d/ph were built with; see the docstring")
    p.add_argument("--resolution", type=int, nargs=2, default=[10, 10])
    p.add_argument("--im_range", type=float, nargs=4, default=[0.0, 1.25, 0.0, 1.25],
                   metavar=("BMIN", "BMAX", "PMIN", "PMAX"),
                   help="birth/persistence extent of the image, shared by all "
                        "dimensions unless overridden below")
    p.add_argument("--bandwidth", type=float, default=1.0,
                   help="Gaussian sigma in PERSISTENCE UNITS, not pixels; gudhi's "
                        "default of 1.0 is ~8 px at the default range and "
                        "resolution, which flattens the image to rank 1. See the "
                        "docstring")
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
        # pixel: on a 10:1 anisotropic grid any bandwidth sane on the birth
        # axis is several pixels on the persistence axis, so a pixel-based test
        # flags good configurations too. A third of the shorter extent catches
        # every case seen so far and no others (NOTES 11.5c).
        span = min(rng[1] - rng[0], rng[3] - rng[2])
        flag = "  <-- SMEARED" if bw > span / 3 else ""
        print(f"  H{d}: range b[{rng[0]:g},{rng[1]:g}] p[{rng[2]:g},{rng[3]:g}]  "
              f"bandwidth {bw:g} = {bw / px_b:.1f} px birth, "
              f"{bw / px_p:.1f} px persistence{flag}")

    tasks = [(f, a.inputdir, a.outputdir, tuple(a.resolution), params,
              a.weight) for f in files]

    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="PH"):
            done += f.result() == "ok"
    n_feat = 2 * a.resolution[0] * a.resolution[1]
    print(f"  {done} computed, {len(tasks) - done} cached, "
          f"{n_feat} features each, weight={a.weight} -> {a.outputdir}")


if __name__ == "__main__":
    main()
