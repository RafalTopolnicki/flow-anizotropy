r"""
Non-additive pore-connectivity descriptors for the labelled 2D dataset.

Why these exist
---------------
NOTES 11.2: the Euler characteristic is *additive*, so on thresholded Gaussian
fields ECP is a closed-form function of the covariance -- measured at .834
predictable from TPC.  NOTES 11.4: the quantities that escape TPC are the
*non-additive* ones, because no local sum can see whether a pore cluster wraps
the torus.  This script computes them on all 5000 labelled structures so they
can be scored against the real permeability labels rather than against a proxy.

Two families:

  cluster  beta0_pore, backbone_*, dead_frac, largest_frac -- which pore
           clusters span the cell, and how much pore volume is stranded.
           Periodic labelling on a 3x3 tiling, exactly as the generator's
           percolates() does it, because a non-periodic split overstates
           fragmentation by ~6x (NOTES 11.6).

  tort     geodesic tortuosity in FOUR directions, 0/45/90/135 degrees.

Why four directions and not two
-------------------------------
`tort_x - tort_y` is the k_xx - k_yy contrast; it says nothing about k_off.
The off-diagonal is the contrast between the [1,1] and [1,-1] diagonals, and a
feature set holding only the axes cannot express its sign -- the same fact that
forced 9 directions rather than 7 in the 3D filtration.  `tort_d1 - tort_d2` is
the k_off-shaped feature, and it is the reason this script is not just
tort_screen.py pointed at DATA/aniso.

Getting the diagonals right
---------------------------
The obvious construction -- shear the lattice so the diagonal becomes an axis,
then measure a slab crossing as tort_screen.py does -- is WRONG, and the
self-test catches it.  A shear is an exact lattice automorphism, so it does not
change *which* paths cross a slab; a crossing with free transverse displacement
is therefore the same quantity for [1,1] as for [1,0], and the two diagonals
come out bitwise equal.  Slab crossing can only ever express two directions.

What actually distinguishes a direction is the **net displacement**, so the
directional quantity is the shortest pore path from a point to its own periodic
image one cell away along d:

    tort(d) = d_geo(p, p + L*d) / |L*d|

i.e. the shortest pore cycle winding once around the torus in direction d.  A
straight channel along d gives exactly 1; a structure with no pore connectivity
along d gives infinity.  It is directional, non-additive, signed under the
[1,1] / [1,-1] exchange, and 1 is a hard floor.

Computed on a 3x3 periodic tiling with 8-neighbour true Euclidean weights, so
all four images p + (L,0), (0,L), (L,L), (L,-L) lie inside the cover.  Paths
that would stray beyond the 3x3 window are cut off, which makes the result an
upper bound on the true universal-cover distance -- tight here, since at
porosity ~0.85 the pore space is richly connected and geodesics are near
straight.  `tort_screen.py` tiles 3x for the same reason.

The source point p is averaged over a few deterministic anchors rather than
minimised over all p: the exact minimum is a shortest-cycle problem needing one
Dijkstra per source, and the anchored mean is reproducible, ~L cheaper, and
adequate for a feature.  Both the mean and the min over anchors are reported.

Usage
-----
    python scripts/connect2d.py --self-test
    python scripts/connect2d.py --dataset DATA/aniso \
        --output DESC/aniso/connect.csv [--workers 24] [--limit N]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io import load_structure          # noqa: E402

S2 = np.sqrt(2.0)

# (dy, dx) of the 8 neighbours with their true Euclidean lengths.
NEIGH = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
         (1, 1, S2), (1, -1, S2), (-1, 1, S2), (-1, -1, S2)]

# direction -> (winding (wy, wx), |d|).  x and y are the two axes; d1 = [1,1]
# and d2 = [1,-1] are the pair whose contrast is k_off.
DIRS = {"x": ((0, 1), 1.0), "y": ((1, 0), 1.0),
        "d1": ((1, 1), S2), "d2": ((1, -1), S2)}

def _grid_graph(pore):
    """CSR over the pore pixels of a tiled array, 8-neighbour, true lengths."""
    H, W = pore.shape
    idx = np.full((H, W), -1, np.int64)
    flat = np.flatnonzero(pore)
    idx.ravel()[flat] = np.arange(flat.size)
    yy, xx = np.divmod(flat, W)

    rows, cols, data = [], [], []
    src = np.arange(flat.size)
    for dy, dx, w in NEIGH:
        ny, nx = yy + dy, xx + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        tgt = np.where(ok, idx[np.clip(ny, 0, H - 1), np.clip(nx, 0, W - 1)], -1)
        ok &= tgt >= 0
        rows.append(src[ok])
        cols.append(tgt[ok])
        data.append(np.full(int(ok.sum()), w))
    n = flat.size
    g = csr_matrix((np.concatenate(data),
                    (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return g, idx


def _anchors(pore):
    """
    Anchor pixels, chosen so the feature is EXACTLY equivariant under the
    lattice symmetries that matter.

    The torus mirror that exchanges [1,1] and [1,-1] is y -> -y (mod H), and
    the one that exchanges the axes is the transpose.  Both map the sublattice
    {y = 0 mod H/2, x = 0 mod W/2} to itself and pore to pore, so taking every
    sublattice site that happens to be pore -- and simply skipping the ones
    that are solid -- commutes with them.  Picking the *nearest* pore pixel to
    a target instead does not: it silently broke the mirror symmetry of
    tort_off, which is the whole point of the feature, and the self-test caught
    it.

    With solid fraction ~0.15 all four sites are solid about 5 times in 10000;
    those fall back to the first pore pixel, losing exactness on those rows
    only.
    """
    H, W = pore.shape
    sites = [(a * H // 2, b * W // 2) for a in (0, 1) for b in (0, 1)]
    out = [(y, x) for y, x in sites if pore[y, x]]
    if out:
        return out
    py, px = np.nonzero(pore)
    return [(int(py[0]), int(px[0]))] if py.size else []


def directional_tortuosity(pore):
    """{dir: (mean, min)} over anchors of d_geo(p, p + L*d) / |L*d|."""
    H, W = pore.shape
    tiled = np.tile(pore, (3, 3))
    g, idx = _grid_graph(tiled)
    anchors = _anchors(pore)
    if g.shape[0] == 0 or not anchors:
        return {d: (np.nan, np.nan) for d in DIRS}

    src = [idx[H + y, W + x] for y, x in anchors]
    dist = dijkstra(g, directed=False, indices=src)

    vals = {d: [] for d in DIRS}
    for row, (y, x) in zip(dist, anchors):
        for d, ((wy, wx), norm) in DIRS.items():
            j = idx[H + y + wy * H, W + x + wx * W]
            v = row[j] if j >= 0 else np.inf
            vals[d].append(v / (norm * H))
    out = {}
    for d, v in vals.items():
        a = np.asarray(v, float)
        fin = a[np.isfinite(a)]
        out[d] = (float(fin.mean()), float(fin.min())) if fin.size else (np.nan, np.nan)
    return out


def cluster_features(pore):
    """Which pore clusters span the torus, and how much pore is stranded."""
    H, W = pore.shape
    lab, _ = label(np.tile(pore, (3, 3)))          # 4-connectivity, as generator
    mid = lab[H:2 * H, W:2 * W]
    span_x = (set(lab[:, 0].flat) & set(lab[:, -1].flat)) - {0}   # axis 1 = x
    span_y = (set(lab[0, :].flat) & set(lab[-1, :].flat)) - {0}   # axis 0 = y

    ids, counts = np.unique(mid[mid > 0], return_counts=True)
    tot = counts.sum()
    in_x = np.array([i in span_x for i in ids], bool)
    in_y = np.array([i in span_y for i in ids], bool)
    fx, fy = counts[in_x].sum() / tot, counts[in_y].sum() / tot
    return {
        "beta0_pore": float(len(ids)),
        "backbone_x": fx,
        "backbone_y": fy,
        "backbone_iso": counts[in_x & in_y].sum() / tot,
        "backbone_aniso": fx - fy,
        "dead_frac": 1.0 - counts[in_x | in_y].sum() / tot,
        "largest_frac": counts.max() / tot,
    }


def features(solid):
    pore = ~solid
    f = cluster_features(pore)
    t = directional_tortuosity(pore)
    for d in DIRS:
        f[f"tort_{d}"] = t[d][0]
        f[f"tortmin_{d}"] = t[d][1]
    f["tort_aniso"] = t["x"][0] - t["y"][0]      # the k_xx - k_yy contrast
    f["tort_off"] = t["d1"][0] - t["d2"][0]      # the k_off contrast
    f["tort_mean"] = float(np.nanmean([t[d][0] for d in DIRS]))
    return f


COLS = ["beta0_pore", "backbone_x", "backbone_y", "backbone_iso",
        "backbone_aniso", "dead_frac", "largest_frac",
        "tort_x", "tort_y", "tort_d1", "tort_d2",
        "tortmin_x", "tortmin_y", "tortmin_d1", "tortmin_d2",
        "tort_aniso", "tort_off", "tort_mean"]


def _one(args):
    sid, path = args
    return sid, features(load_structure(path) > 0)


def run(dataset, output, workers, limit):
    df = pd.read_csv(os.path.join(dataset, "structures.csv"))
    if limit:
        df = df.head(limit)
    tasks = [(r.sample_id, os.path.join(dataset, "structures", r.filename))
             for r in df.itertuples()]

    from tqdm import tqdm
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sid, f in tqdm(pool.map(_one, tasks, chunksize=4), total=len(tasks),
                           desc="connectivity"):
            rows.append({"sample_id": sid, **{f"conn_{k}": v for k, v in f.items()}})

    out = pd.DataFrame(rows).sort_values("sample_id")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    out.to_csv(output, index=False)
    n_nan = out.isna().sum().sum()
    print(f"\n{len(out)} rows x {out.shape[1] - 1} features -> {output}"
          f"   ({n_nan} NaN)")
    print(out[[f"conn_{c}" for c in COLS]].describe().T[
        ["mean", "std", "min", "50%", "max"]].round(4).to_string())


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    rng = np.random.default_rng(0)
    N = 64
    yy, xx = np.mgrid[0:N, 0:N]

    # an empty cell is tortuosity 1 in every direction
    t = directional_tortuosity(np.ones((N, N), bool))
    check("open cell gives tortuosity 1 in all four directions",
          all(abs(t[d][0] - 1.0) < 1e-9 for d in DIRS),
          str({d: round(t[d][0], 6) for d in DIRS}))

    # a straight channel conducts along itself and nowhere else
    cases = {
        "x": (yy % 16) < 5,                 # rows of pore
        "y": (xx % 16) < 5,                 # columns of pore
        "d1": ((xx - yy) % N) < 5,          # band along [1, 1]
        "d2": ((xx + yy) % N) < 5,          # band along [1, -1]
    }
    for d, pore in cases.items():
        t = directional_tortuosity(pore)
        others = [o for o in DIRS if o != d]
        check(f"straight channel along {d}: tortuosity 1 along, blocked across",
              abs(t[d][0] - 1.0) < 1e-9
              and all(not np.isfinite(t[o][0]) or t[o][0] > 1.2 for o in others),
              str({k: (round(v[0], 4) if np.isfinite(v[0]) else "inf")
                   for k, v in t.items()}))

    # the two diagonals must exchange exactly under a y -> -y mirror,
    # which is what makes tort_off a signed, k_off-shaped feature
    q = rng.random((N, N)) > 0.45
    mirror = np.roll(np.flipud(q), 1, axis=0)        # y -> -y (mod H)
    a, b = directional_tortuosity(q), directional_tortuosity(mirror)
    check("the y -> -y mirror exchanges tort_d1 and tort_d2",
          abs(a["d1"][0] - b["d2"][0]) < 1e-9 and abs(a["d2"][0] - b["d1"][0]) < 1e-9,
          f"d1={a['d1'][0]:.4f} d2={a['d2'][0]:.4f}")
    c = directional_tortuosity(q.T)
    check("transpose exchanges tort_x and tort_y",
          abs(a["x"][0] - c["y"][0]) < 1e-9 and abs(a["y"][0] - c["x"][0]) < 1e-9)

    # it is a ratio to the straight line, so 1 is a hard floor
    vals = []
    for _ in range(4):
        t = directional_tortuosity(rng.random((N, N)) > 0.4)
        vals += [v[0] for v in t.values()] + [v[1] for v in t.values()]
    fin = [v for v in vals if np.isfinite(v)]
    check("tortuosity >= 1 on random structures", min(fin) >= 1.0 - 1e-9,
          f"min {min(fin):.6f}, {len(fin)}/{len(vals)} finite")
    check("min over anchors <= mean over anchors",
          all(t[d][1] <= t[d][0] + 1e-12 for d in DIRS if np.isfinite(t[d][0])))

    # anchors are deterministic: the feature must not depend on a seed
    r = rng.random((N, N)) > 0.45
    check("features are reproducible", features(~r) == features(~r))

    # cluster features: one straight channel conducts in x only
    f = cluster_features((yy % 16) < 5)
    check("cluster features see a one-way backbone",
          f["backbone_x"] == 1.0 and f["backbone_y"] == 0.0
          and f["dead_frac"] == 0.0,
          str({k: round(v, 3) for k, v in f.items()}))

    # a real structure from the dataset: every accepted one percolates both ways,
    # so all four directions must be finite and the two contrasts small but real
    p = "DATA/aniso/structures"
    if os.path.isdir(p):
        import pandas as pd
        d0 = pd.read_csv("DATA/aniso/structures.csv").iloc[0]
        f = features(load_structure(os.path.join(p, d0.filename)) > 0)
        check("a real structure gives finite tortuosity in all four directions",
              all(np.isfinite(f[f"tort_{d}"]) for d in DIRS),
              str({d: round(f[f"tort_{d}"], 4) for d in DIRS}))

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--output")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()
    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.output):
        p.error("--dataset and --output are required (or use --self-test)")
    run(a.dataset, a.output, a.workers, a.limit)


if __name__ == "__main__":
    main()
