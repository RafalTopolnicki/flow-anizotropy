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

Usage
-----
    python scripts/ph2d.py --inputdir DESC/filt/a45 --outputdir DESC/ph/a45 \
        [--resolution 10 10] [--im_range 0 1.25 0 1.25] [--workers 14]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gudhi as gd
from gudhi.representations import DiagramSelector, PersistenceImage
from tqdm import tqdm

DIMS = (0, 1)


def _one(args):
    fname, inputdir, outputdir, resolution, im_range = args
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
    imager = PersistenceImage(resolution=list(resolution), im_range=list(im_range))

    vecs = []
    for d in DIMS:
        bars = sel(cx.persistence_intervals_in_dimension(d))
        if len(bars) == 0:
            vecs.append(np.zeros(resolution[0] * resolution[1]))
        else:
            vecs.append(np.asarray(imager(bars)).ravel())
    np.save(out, np.concatenate(vecs))
    return "ok"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputdir", required=True)
    p.add_argument("--outputdir", required=True)
    p.add_argument("--resolution", type=int, nargs=2, default=[10, 10])
    p.add_argument("--im_range", type=float, nargs=4, default=[0.0, 1.25, 0.0, 1.25])
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    os.makedirs(a.outputdir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.inputdir) if f.endswith(".npy"))
    tasks = [(f, a.inputdir, a.outputdir, tuple(a.resolution), tuple(a.im_range))
             for f in files]

    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_one, t) for t in tasks]
        for f in tqdm(as_completed(futs), total=len(futs), desc="PH"):
            done += f.result() == "ok"
    n_feat = 2 * a.resolution[0] * a.resolution[1]
    print(f"  {done} computed, {len(tasks) - done} cached, "
          f"{n_feat} features each -> {a.outputdir}")


if __name__ == "__main__":
    main()
