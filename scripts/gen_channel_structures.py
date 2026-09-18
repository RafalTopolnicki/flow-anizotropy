r"""
Generate 2D periodic CHANNEL-NETWORK structures with controlled anisotropy.

Why a second generator
----------------------
`gen_aniso_structures.py` thresholds a random trigonometric field, so its
structures are excursion sets of a (near-)Gaussian field.  For those, additive
functionals -- the Euler characteristic, hence ECP -- have closed forms in the
covariance and the threshold (Tomita/Adler).  Measured consequence: ECP is .834
predictable from TPC (NOTES 11.2), and three separate descriptor fixes each
gained on their own descriptor and nothing on `all` (NOTES 11.8).  The ceiling
is the structure family, not the descriptors.

Lowering the porosity does **not** escape it -- tested twice, NOTES 11.3, no
movement.  The family has to change.

The model
---------
A stick network.  `n_seg` finite segments are thrown on the torus with

  * orientation from a von Mises on the DOUBLED angle, mean 2*psi and
    concentration kappa, so the distribution is pi-periodic and concentrated
    about psi.  kappa = 0 is isotropic; large kappa is a near-parallel bundle.
  * length around a per-structure mean, width an integer per segment.

Pore is the union of the dilated segments; solid is everything else.

Junction blocking -- and why the plain network is not enough
-----------------------------------------------------------
A union of **independently placed** sticks is a Boolean model (Poisson germs,
random grains), and for Boolean models the Minkowski functionals -- Euler
characteristic included -- have closed forms in the intensity and the mean grain
measures (Miles/Davy), with the two-point correlation a function of the same
parameters.  So ECP and TPC are again analytically tied, exactly as for Gaussian
excursion sets, by a different closed form.  Measured (NOTES 11.10): the plain
stick network scored ECP-from-TPC **.779** against DATA/aniso's **.629** at
matched n and matched porosity spread -- *more* re-encoded, not less.  Changing
the grain from a blob to a stick changed nothing that mattered.

**Independent placement is what has to go.**  `--block_max` puts small solid
plugs at a random subset of the channel intersections.  A plug of 2-4 px barely
moves S2 -- it removes a handful of pixels -- but it splits a loop and can sever
a conducting path outright, so connectivity and permeability decouple from the
second-order statistics almost by construction.  The plugged pixels are also no
longer an independently placed grain process: their positions are conditioned on
where the sticks crossed.  Physically this is cementation at grain contacts.

`block_frac = 0` reproduces the plain network exactly, so the two families sit on
one continuum and the screen can attribute any movement to the blocking alone.
The realised counts (`n_junctions`, `n_blocked`) are recorded per sample, never
assumed, because percolation rejection biases the accepted blocking fraction.

Anisotropy enters as an orientation distribution rather than a squashed
wavevector cloud, so the principal axis is psi by construction and the ensemble
still satisfies the label-noise constraint of NOTES 3.3: it needs a real
anisotropy ratio AND a principal axis rotated off the lattice axes.

Periodicity
-----------
Exact, and load-bearing -- K is only defined on a periodic cell, and K = K^T is
the free per-sample error bar the whole label-quality story rests on (NOTES 3.3).
Segments wrap by taking their rasterised pixel coordinates modulo `size`, and
the widening is a roll-based dilation, which is periodic by construction with no
tiling and no cropping.  `--self-test` checks continuity across the wrap with the
same column-agreement statistic that caught the non-periodic K0 generator
(NOTES 2): inside 0.979 vs across-the-wrap 0.693 at a chance level of 0.704.

Note for anyone tempted to periodise a non-periodic structure by mirroring:
mirror padding imposes a reflection symmetry, and a structure symmetric under
x -> -x has k_xy = 0 exactly.  It would zero the target.

Frame convention
----------------
Identical to `gen_aniso_structures.py`: `grid[i, j]` is `(x, y) = (j, i)`, so
array axis 1 is the solver's x and axis 0 is the solver's y.  A segment at angle
theta CCW from +x advances by `(dx, dy) = (cos theta, sin theta)`, written to
`array[y, x]`.  psi is therefore directly comparable with the solver's
`theta_deg` and with the other generator's psi.

Usage
-----
    python scripts/gen_channel_structures.py --self-test
    python scripts/gen_channel_structures.py --output DATA/aniso_chan \
        --n_samples 600 [--size 256] [--seed 12000] [--workers 24]
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from array2gif import write_gif

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_aniso_structures import percolates          # noqa: E402

# (name, kappa_lo, kappa_hi) -- concentration of the orientation distribution.
# Cyclic by index as in the other generator, so any prefix of the dataset is
# balanced and an unshuffled KFold sees every stratum in every fold.
STRATA = [
    ("iso",    0.0,  0.3),
    ("mod",    1.0,  3.0),
    ("strong", 4.0, 12.0),
]


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def _disc_offsets(radius):
    r = int(np.ceil(radius))
    return [(dy, dx)
            for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= radius * radius]


def _dilate_periodic(mask, radius):
    """
    Binary dilation by a disc, wrapping in both axes.

    Done with rolls rather than `scipy.ndimage.binary_dilation` on a 3x3 tiling:
    rolls are periodic by construction, so there is no tile-and-crop step to get
    subtly wrong at the seam -- which is the one property this generator must
    not lose.
    """
    out = mask.copy()
    for dy, dx in _disc_offsets(radius):
        if dy or dx:
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def _draw_segments(size, centres, thetas, lengths):
    """1-pixel periodic lines; returns a bool mask."""
    mask = np.zeros((size, size), bool)
    for (cy, cx), th, ell in zip(centres, thetas, lengths):
        n = max(2, int(np.ceil(ell * 2)))            # 0.5 px steps, no gaps
        t = np.linspace(-ell / 2.0, ell / 2.0, n)
        xs = np.floor(cx + t * np.cos(th)).astype(np.int64) % size
        ys = np.floor(cy + t * np.sin(th)).astype(np.int64) % size
        mask[ys, xs] = True
    return mask


def _junction_mask(size, centres, thetas, lengths):
    """
    Pixels covered by two or more DISTINCT segments.

    Each segment is rasterised, widened by one pixel so that two lines crossing
    at a shallow angle still share a pixel, and reduced to unique indices before
    counting -- otherwise a single segment doubling back on its own rasterised
    pixels would register as a crossing with itself.
    """
    count = np.zeros(size * size, np.int16)
    for (cy, cx), th, ell in zip(centres, thetas, lengths):
        n = max(2, int(np.ceil(ell * 2)))
        t = np.linspace(-ell / 2.0, ell / 2.0, n)
        xs = np.floor(cx + t * np.cos(th)).astype(np.int64)
        ys = np.floor(cy + t * np.sin(th)).astype(np.int64)
        # widen by the 4-neighbourhood
        ys = np.concatenate([ys, ys + 1, ys - 1, ys, ys])
        xs = np.concatenate([xs, xs, xs, xs + 1, xs - 1])
        flat = np.unique((ys % size) * size + (xs % size))
        count[flat] += 1
    return (count >= 2).reshape(size, size)


def _label_periodic(mask):
    """
    Connected components with wrap-around adjacency, and their circular means.

    Not the 3x3-tiling trick used elsewhere in this project: that one is right
    for large pore clusters, which connect to their own translates through the
    bulk, but wrong for a small blob sitting on the seam -- its copies land near
    different corners of the tiling, get different labels, and are counted two
    or three times.  Worse, a centroid averaged in tile coordinates across a
    seam lands in the middle of the cell, so the plug would be placed nowhere
    near the junction.  Union-find over the two wrap edges is exact and the
    blobs here are tiny.

    Returns a list of (y, x) component centres, each a circular mean so that a
    blob spanning the seam resolves to the seam rather than to the far side.
    """
    from scipy.ndimage import label as _label

    H, W = mask.shape
    lab, n = _label(mask)
    if n == 0:
        return []

    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(H):                                   # left <-> right edge
        if mask[i, 0] and mask[i, -1]:
            union(lab[i, 0], lab[i, -1])
    for j in range(W):                                   # top <-> bottom edge
        if mask[0, j] and mask[-1, j]:
            union(lab[0, j], lab[-1, j])

    ys, xs = np.nonzero(mask)
    roots = np.array([find(lab[y, x]) for y, x in zip(ys, xs)])
    out = []
    for r in np.unique(roots):
        sel = roots == r
        # circular mean: a blob on the seam must resolve to the seam
        cy = np.angle(np.exp(2j * np.pi * ys[sel] / H).mean()) / (2 * np.pi) * H
        cx = np.angle(np.exp(2j * np.pi * xs[sel] / W).mean()) / (2 * np.pi) * W
        out.append((int(round(cy)) % H, int(round(cx)) % W))
    return out


def block_junctions(rng, pore, junc, frac, radius):
    """Plug a random `frac` of junctions with solid discs of `radius`."""
    if frac <= 0:
        return pore, 0, 0
    centres = _label_periodic(junc)
    if not centres:
        return pore, 0, 0

    k = int(round(frac * len(centres)))
    if k <= 0:
        return pore, len(centres), 0
    pick = rng.choice(len(centres), size=min(k, len(centres)), replace=False)

    plugs = np.zeros_like(pore)
    for i in pick:
        y, x = centres[i]
        plugs[y, x] = True
    return pore & ~_dilate_periodic(plugs, radius), len(centres), int(len(pick))


def build_network(rng, size, psi, kappa, seg_len, width_mean, porosity,
                  block_frac=0.0, plug_radius=2.0):
    """Pore mask for one stick network, and the realised segment count."""
    # Boolean-model coverage estimate: a target pore fraction phi needs
    # n = -ln(1-phi) * area / (per-stick area).  Overlaps are what make this
    # an estimate rather than an identity, so the realised porosity is measured
    # afterwards and recorded, never assumed.
    per_stick = seg_len * max(width_mean, 1.0)
    n_seg = int(max(4, round(-np.log(max(1e-3, 1.0 - porosity)) * size * size / per_stick)))

    # pi-periodic orientation: draw on the doubled angle, then halve
    mu2 = float(np.angle(np.exp(2j * psi)))
    thetas = rng.vonmises(mu2, kappa, n_seg) / 2.0
    lengths = seg_len * rng.uniform(0.5, 1.5, n_seg)
    centres = rng.uniform(0, size, (n_seg, 2))
    widths = np.clip(np.round(width_mean * rng.uniform(0.6, 1.4, n_seg)), 2, 9).astype(int)

    pore = np.zeros((size, size), bool)
    for w in np.unique(widths):
        sel = widths == w
        line = _draw_segments(size, centres[sel], thetas[sel], lengths[sel])
        pore |= _dilate_periodic(line, w / 2.0)

    n_junc = n_blocked = 0
    if block_frac > 0:
        junc = _junction_mask(size, centres, thetas, lengths) & pore
        pore, n_junc, n_blocked = block_junctions(rng, pore, junc,
                                                  block_frac, plug_radius)
    return pore, n_seg, n_junc, n_blocked


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------

def gen_one(sample_id, cfg):
    rng = np.random.default_rng(cfg["seed"] + sample_id)
    name, k_lo, k_hi = STRATA[sample_id % len(STRATA)]
    size = cfg["size"]

    for attempt in range(cfg["max_attempts"]):
        kappa = float(rng.uniform(k_lo, k_hi))
        psi = float(rng.uniform(0.0, np.pi))
        seg_len = float(rng.uniform(cfg["len_min"], cfg["len_max"]) * size)
        width_mean = float(rng.uniform(cfg["w_min"], cfg["w_max"]))
        porosity = float(rng.uniform(cfg["por_min"], cfg["por_max"]))
        block_frac = float(rng.uniform(cfg["block_min"], cfg["block_max"]))
        plug_radius = float(rng.uniform(cfg["plug_min"], cfg["plug_max"]))

        pore, n_seg, n_junc, n_blocked = build_network(
            rng, size, psi, kappa, seg_len, width_mean, porosity,
            block_frac, plug_radius)
        solid = ~pore

        px, py = percolates(solid)
        if not (px and py):
            continue
        por = float(pore.mean())
        if not (cfg["por_accept_min"] <= por <= cfg["por_accept_max"]):
            continue

        fname = (f"sample_{sample_id:06d}_str={name}_kap={kappa:05.2f}"
                 f"_psi={np.degrees(psi):05.1f}_len={seg_len:05.1f}"
                 f"_blk={block_frac:.2f}_por={por:.4f}")
        write_gif((1 - np.dstack([solid, solid, solid]).astype(np.uint8)) * 255,
                  os.path.join(cfg["structures_dir"], fname + ".gif"))
        return {
            "sample_id": sample_id,
            "filename": fname + ".gif",
            "stratum": name,
            "kappa": kappa,
            "psi_deg": float(np.degrees(psi)),
            "seg_len": seg_len,
            "width_mean": width_mean,
            "n_seg": n_seg,
            "block_frac": block_frac,
            "plug_radius": plug_radius,
            "n_junctions": n_junc,
            "n_blocked": n_blocked,
            "blocked_frac_real": (n_blocked / n_junc) if n_junc else 0.0,
            "porosity": por,
            "porosity_target": porosity,
            "attempts": attempt + 1,
        }

    logging.warning("sample %d (%s): no accepted structure in %d attempts",
                    sample_id, name, cfg["max_attempts"])
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_lengths(pore, axis):
    """Mean length of consecutive pore runs along `axis`, wrapping."""
    a = pore if axis == 1 else pore.T
    a = np.concatenate([a, a], axis=1)              # wrap so runs are not cut
    starts = a & ~np.roll(a, 1, axis=1)
    n_runs = starts.sum()
    return (a.sum() / n_runs) if n_runs else np.inf


def _wrap_agreement(solid):
    """
    Column agreement across the seam vs inside, the NOTES 2 diagnostic.

    A non-periodic structure solved periodically has a random splice at the
    wrap: the K0 generator measured 0.979 inside and 0.693 across, against a
    chance level of 0.704 for its pore fraction.
    """
    inside = float((solid[:, :-1] == solid[:, 1:]).mean())
    across = float((solid[:, -1] == solid[:, 0]).mean())
    p = float(solid.mean())
    chance = p * p + (1 - p) * (1 - p)
    return inside, across, chance


def self_test(size=192):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    rng = np.random.default_rng(0)

    # --- frame: a single segment must lie along theta, in solver axes --------
    for deg, want in ((0.0, "x"), (90.0, "y")):
        m = _draw_segments(size, [(size // 2, size // 2)], [np.radians(deg)], [60.0])
        ys, xs = np.nonzero(m)
        spread_x, spread_y = xs.ptp(), ys.ptp()
        check(f"a segment at {deg:g} deg runs along {want}",
              (spread_x > spread_y) if want == "x" else (spread_y > spread_x),
              f"|dx|={spread_x} |dy|={spread_y}")

    m = _draw_segments(size, [(size // 2, size // 2)], [np.radians(45.0)], [60.0])
    ys, xs = np.nonzero(m)
    check("a segment at 45 deg advances equally in x and y",
          abs(xs.ptp() - ys.ptp()) <= 1, f"|dx|={xs.ptp()} |dy|={ys.ptp()}")

    # --- periodicity --------------------------------------------------------
    # a segment centred on the corner must appear on BOTH sides of the seam
    m = _draw_segments(size, [(0.0, 0.0)], [np.radians(30.0)], [60.0])
    check("a segment through the origin wraps to both edges",
          m[:, :5].any() and m[:, -5:].any() and m[:5, :].any() and m[-5:, :].any())

    # dilation must not thin the structure at the seam
    line = np.zeros((size, size), bool)
    line[:, 0] = True                                # solid column at the seam
    d = _dilate_periodic(line, 2.0)
    check("periodic dilation wraps symmetrically",
          d[:, 1].all() and d[:, -1].all() and d[:, 2].all() and d[:, -2].all())

    # and the ensemble itself must be continuous across the wrap
    pore, *_ = build_network(rng, size, np.radians(30.0), 6.0, 0.30 * size, 3.0, 0.45)
    inside, across, chance = _wrap_agreement(~pore)
    check("structure is continuous across the wrap (NOTES 2 test)",
          across > inside - 0.03 and across > chance + 0.15,
          f"inside {inside:.3f}  across {across:.3f}  chance {chance:.3f}")

    # --- anisotropy: psi must set the conducting direction ------------------
    def runs(psi_deg, kappa=8.0):
        p, *_ = build_network(np.random.default_rng(7), size, np.radians(psi_deg),
                             kappa, 0.30 * size, 3.0, 0.45)
        return _run_lengths(p, 1), _run_lengths(p, 0)

    rx0, ry0 = runs(0.0)
    check("psi=0 makes pore runs longer along x", rx0 > ry0 * 1.3,
          f"run_x {rx0:.1f}  run_y {ry0:.1f}")
    rx9, ry9 = runs(90.0)
    check("psi=90 makes pore runs longer along y", ry9 > rx9 * 1.3,
          f"run_x {rx9:.1f}  run_y {ry9:.1f}")
    rx4, ry4 = runs(45.0)
    check("psi=45 is balanced between x and y", 0.7 < rx4 / ry4 < 1.4,
          f"run_x {rx4:.1f}  run_y {ry4:.1f}")

    # isotropic stratum must NOT have a preferred axis
    rxi, ryi = runs(0.0, kappa=0.0)
    check("kappa=0 has no preferred axis", 0.7 < rxi / ryi < 1.4,
          f"run_x {rxi:.1f}  run_y {ryi:.1f}")

    # --- the point of the family: dead ends and graded connectivity ---------
    # DATA/aniso has dead_frac ~ .0005 because a thresholded field leaves almost
    # no stranded pore.  A stick network must do better or it is not worth the
    # LBM time.
    try:
        from connect2d import cluster_features
        dead = []
        for s in range(6):
            p, *_ = build_network(np.random.default_rng(100 + s), size,
                                 np.radians(30.0), 3.0, 0.25 * size, 3.0, 0.42)
            if all(percolates(~p)):
                dead.append(cluster_features(p)["dead_frac"])
        check("stranded pore volume actually exists", len(dead) and max(dead) > 0.01,
              f"dead_frac max {max(dead):.4f} over {len(dead)} structures "
              f"(DATA/aniso mean is .0005)")
    except ImportError:
        print("[skip] dead_frac check (connect2d not importable)")

    # --- porosity control and reproducibility -------------------------------
    got = []
    for tgt in (0.3, 0.5):
        p, *_ = build_network(np.random.default_rng(3), size, 0.0, 1.0,
                             0.3 * size, 3.0, tgt)
        got.append(p.mean())
    check("porosity tracks its target", got[0] < got[1], 
          f"target .30 -> {got[0]:.3f}   target .50 -> {got[1]:.3f}")

    a, *_ = build_network(np.random.default_rng(11), size, 0.4, 2.0, 50.0, 3.0, 0.4)
    b, *_ = build_network(np.random.default_rng(11), size, 0.4, 2.0, 50.0, 3.0, 0.4)
    check("generation is reproducible from the seed", np.array_equal(a, b))

    # --- junction blocking --------------------------------------------------
    args = (size, np.radians(30.0), 2.0, 0.30 * size, 3.0, 0.50)

    plain, _, nj0, nb0 = build_network(np.random.default_rng(21), *args,
                                       block_frac=0.0)
    check("block_frac=0 does no blocking", nj0 == 0 and nb0 == 0)

    blocked, _, nj, nb = build_network(np.random.default_rng(21), *args,
                                       block_frac=0.3, plug_radius=2.0)
    check("junctions are found and a fraction is plugged",
          nj > 20 and 0 < nb <= nj, f"{nb} of {nj} junctions plugged")
    check("blocking only ever removes pore", (blocked & ~plain).sum() == 0)

    # THE point of the design: S2 must barely move while connectivity moves a
    # lot.  Both changes are measured RELATIVE to their own scale -- an earlier
    # version compared a percentage change in tortuosity against an absolute
    # change in S2, which flatters the result by construction.
    from baselines2d import two_point_correlation, DIRECTIONS
    from connect2d import directional_tortuosity, cluster_features

    d_por = abs(plain.mean() - blocked.mean())
    s2a = np.concatenate([two_point_correlation(~plain, DIRECTIONS[d], 65)
                          for d in (0, 90)])
    s2b = np.concatenate([two_point_correlation(~blocked, DIRECTIONS[d], 65)
                          for d in (0, 90)])
    rel_s2 = float(np.abs(s2a - s2b).max() / (s2a.max() - s2a.min()))

    ta = directional_tortuosity(plain)["x"][0]
    tb = directional_tortuosity(blocked)["x"][0]
    rel_tort = (tb - ta) / (ta - 1.0) if np.isfinite(ta) and np.isfinite(tb) else np.inf
    d_dead = (cluster_features(blocked)["dead_frac"]
              - cluster_features(plain)["dead_frac"])

    check("blocking barely moves porosity", d_por < 0.03, f"d_porosity {d_por:.4f}")
    # averaged over seeds: which junctions happen to be plugged swings a
    # single structure's tortuosity by a factor of two, so a one-sample
    # threshold here is a coin flip
    deltas = []
    for sd in range(4):
        pl, *_ = build_network(np.random.default_rng(40 + sd), *args, block_frac=0.0)
        bl, *_ = build_network(np.random.default_rng(40 + sd), *args,
                               block_frac=0.4, plug_radius=2.0)
        t0 = directional_tortuosity(pl)["x"][0]
        t1 = directional_tortuosity(bl)["x"][0]
        dd = (cluster_features(bl)["dead_frac"] - cluster_features(pl)["dead_frac"])
        if np.isfinite(t0) and np.isfinite(t1) and t0 > 1.0:
            deltas.append(((t1 - t0) / (t0 - 1.0), dd))
    mt = float(np.mean([d[0] for d in deltas]))
    md = float(np.mean([d[1] for d in deltas]))
    check("blocking moves connectivity, averaged over seeds",
          len(deltas) >= 3 and (mt > 0.10 or md > 0.005),
          f"mean d_tortuosity-above-1 {mt:+.1%}  mean d_dead_frac {md:+.4f}"
          f"  over {len(deltas)} seeds")
    check("blocking never makes the pore space MORE connected",
          all(d[0] >= -1e-9 for d in deltas))

    # NOT a pass/fail claim.  The design intent was that a plug is nearly
    # invisible to S2 while severing a path, but a swept measurement says
    # otherwise: over plug radius 1-2 px and blocking fraction 0.1-0.8 the best
    # ratio of relative tortuosity change to relative S2 change was 2.2, and
    # near the percolation threshold it was worse, not better (NOTES 11.11).
    # At porosity ~0.6 this network is far above threshold, so a plug usually
    # severs nothing and you must place many -- by which point S2 has moved too.
    # Printed so the number is visible; the ensemble-level screen is the real
    # test and this proxy must not be mistaken for it.
    print(f"       [diag] relative dS2 {rel_s2:.1%}  relative d_tortuosity "
          f"{rel_tort:+.1%}  ratio {rel_tort / rel_s2 if rel_s2 else float('inf'):.1f}")

    # a junction sitting on the seam must be plugged once, not four times
    pore = np.zeros((size, size), bool)
    pore[0, :] = True
    pore[:, 0] = True                      # a cross centred on the corner
    junc = np.zeros((size, size), bool)
    junc[0, 0] = junc[0, 1] = junc[1, 0] = True
    junc[0, -1] = junc[-1, 0] = True       # the same junction, seen across the wrap
    _, nj_seam, _ = block_junctions(np.random.default_rng(0), pore, junc, 1.0, 1.0)
    check("a junction on the seam counts once", nj_seam == 1,
          f"{nj_seam} junction(s) found")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output")
    p.add_argument("--n_samples", type=int, default=600)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=12000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--len_min", type=float, default=0.12, help="x size")
    p.add_argument("--len_max", type=float, default=0.55, help="x size")
    p.add_argument("--w_min", type=float, default=2.0)
    p.add_argument("--w_max", type=float, default=5.0)
    p.add_argument("--por_min", type=float, default=0.30)
    p.add_argument("--por_max", type=float, default=0.60)
    p.add_argument("--por_accept_min", type=float, default=0.20)
    p.add_argument("--por_accept_max", type=float, default=0.72)
    p.add_argument("--block_min", type=float, default=0.0,
                   help="fraction of channel junctions plugged with solid; "
                        "0 reproduces the plain Boolean stick network")
    p.add_argument("--block_max", type=float, default=0.35)
    p.add_argument("--plug_min", type=float, default=1.0, help="plug radius, px")
    p.add_argument("--plug_max", type=float, default=2.5)
    p.add_argument("--max_attempts", type=int, default=40)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not a.output:
        p.error("--output is required (or use --self-test)")

    cfg = {k: getattr(a, k) for k in
           ("size", "seed", "len_min", "len_max", "w_min", "w_max",
            "por_min", "por_max", "por_accept_min", "por_accept_max",
            "block_min", "block_max", "plug_min", "plug_max",
            "max_attempts")}
    cfg["structures_dir"] = os.path.join(a.output, "structures")
    os.makedirs(cfg["structures_dir"], exist_ok=True)

    from tqdm import tqdm
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(gen_one, i, cfg) for i in range(a.n_samples)]
        for f in tqdm(as_completed(futs), total=len(futs), desc="channels"):
            r = f.result()
            if r:
                rows.append(r)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    df.to_csv(os.path.join(a.output, "structures.csv"), index=False)
    print(f"\n{len(df)}/{a.n_samples} kept -> {a.output}/structures.csv")
    print(df[["porosity", "kappa", "psi_deg", "seg_len", "n_seg",
              "n_junctions", "n_blocked", "blocked_frac_real", "attempts"]]
          .describe().T[["mean", "std", "min", "50%", "max"]].round(3).to_string())
    print("\nper stratum:")
    print(df.groupby("stratum")[["porosity", "kappa", "n_seg"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
