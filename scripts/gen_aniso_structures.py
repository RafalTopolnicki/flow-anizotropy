"""
Generate 2D periodic porous structures with controlled flow anisotropy.

Why not reuse the k0/k2 generators
----------------------------------
Two reasons, both fatal for a permeability *tensor*:

1. `gen_k0_structures.py` draws continuous wavevector magnitudes, so its fields
   do not close on the domain.  K is only defined on a periodic cell, and the
   solver wraps in both directions, so every wavevector here is an integer
   number of cycles per box.
2. A statistically isotropic ensemble puts the off-diagonal k_xy in the noise.
   The reciprocity residual |k_xy - k_yx| / (k_xx + k_yy) that the solver
   reports scales with the *trace*, so the relative error on k_xy goes as
   trace / |k_xy|: it is ~0.1% when |k_xy| is 15% of the trace and ~9% when it
   is 0.85%.  Since |k_xy| / trace = ½ (k1-k2)/(k1+k2) sin 2ψ, the ensemble
   needs both a real anisotropy ratio and a principal axis rotated away from
   the lattice axes.

Model
-----
Random trigonometric field, thresholded to a target porosity:

    S(r) = sqrt(2/N) Σ_i cos(q_i · r + φ_i),   solid where S > t

The wavevectors are drawn isotropically in a band around k0 and then squashed
by an area-preserving elliptical map with axis ratio A along a direction ψ:

    q = R(ψ) diag(1/A, A) R(ψ)ᵀ q_iso        then rounded to integers

Components along ψ shrink by A, so the wavelength along ψ grows and the solid
features elongate along ψ.  A = 1 reproduces an isotropic field; the resulting
anisotropy is then purely the finite-size effect of Koza09 (σ_α ∝ L⁻¹), which
is still fully determined by the image and is a legitimate — just harder —
prediction target.  Samples are assigned to three strata (iso / moderate /
strong) cyclically by index, so any prefix of the dataset is balanced and an
unshuffled KFold sees every stratum in every fold.

Frame convention
----------------
`grid[i, j] = (x, y) = (j/size, i/size)`, i.e. array axis 1 is the solver's x
and array axis 0 is the solver's y, matching how `read_from_gif` maps image
(row, col) onto `F[x][y]`.  ψ is therefore measured counter-clockwise from +x
in exactly the frame in which `lbm2d-perm` reports `theta_deg`, and the two are
directly comparable.  `--self-test` checks this end to end.

The generator knobs (A, ψ, k0, N) are written to the summary CSV for
stratification and confound analysis.  They are *not* model features — the
whole point is to predict K from the image.

`structures.csv` (and the merged `permeability.csv`) is the source of truth for
every per-sample quantity; the filename repeats them only for readability.  Do
not regex-parse the filename: the fields are `_`-separated and `\w` matches `_`,
so the obvious `str=(\w+)` silently returns `iso_A` rather than `iso`.  The same
class of bug produced all-NaN columns in the k0/deeppore datasets.  Join on
`sample_id` instead.

Usage
-----
    python scripts/gen_aniso_structures.py --output DATA/aniso --n_samples 5000
    python scripts/gen_aniso_structures.py --self-test
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from array2gif import write_gif
from scipy.ndimage import label
from tqdm import tqdm


# Strata: (name, min axis ratio, max axis ratio)
STRATA = [
    ("iso",    1.0, 1.0),
    ("mod",    1.5, 2.5),
    ("strong", 2.5, 4.0),
]

_GRID = None


def _build_grid(size):
    """grid[i, j] = (x, y) with x along array axis 1, y along array axis 0."""
    pts = np.linspace(0, 1, size, endpoint=False)
    xs, ys = np.meshgrid(pts, pts)          # default 'xy': xs[i,j]=pts[j]
    return np.stack([xs, ys], axis=2)


def _init_worker(size):
    global _GRID
    _GRID = _build_grid(size)


# ---------------------------------------------------------------------------
# Field construction
# ---------------------------------------------------------------------------

def sample_wavevectors(rng, n_modes, k0, dk, aniso, psi):
    """Integer wavevectors from an elliptical band; ψ is the elongation axis."""
    theta = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    mag = rng.uniform(max(0.5, k0 - dk), k0 + dk, n_modes)
    q = np.stack([mag * np.cos(theta), mag * np.sin(theta)], axis=1)

    if aniso != 1.0:
        c, s = np.cos(psi), np.sin(psi)
        R = np.array([[c, -s], [s, c]])
        M = R @ np.diag([1.0 / aniso, aniso]) @ R.T
        q = q @ M.T

    # Integer components are what make the cell exactly periodic.
    q = np.rint(q).astype(int)
    q = q[np.any(q != 0, axis=1)]
    return q


def build_field(grid, q_int, rng):
    """sqrt(2/N) Σ cos(q·r + φ) evaluated in one tensordot."""
    n = len(q_int)
    q = q_int * (2.0 * np.pi)
    phase = np.tensordot(grid, q, axes=([2], [1]))      # (size, size, N)
    phase += rng.uniform(0.0, np.pi, n)
    return np.sqrt(2.0 / n) * np.cos(phase, out=phase).sum(axis=2)


def threshold_for_porosity(field, porosity):
    """Exact quantile threshold: solid where field > t, pore fraction = porosity."""
    return float(np.quantile(field, porosity))


# ---------------------------------------------------------------------------
# Percolation
# ---------------------------------------------------------------------------

def percolates(solid):
    """
    (perc_x, perc_y) for the pore phase on the periodic torus.

    Tests whether a pore cluster spans a 3x3 tiling edge to edge, which a
    wrapping cluster always does and a merely large local one cannot.
    4-connectivity, deliberately: a diagonal pinch has zero cross-section and
    conducts essentially nothing under D2Q9 bounce-back, so counting it would
    admit structures whose permeability is numerically zero and which then burn
    the solver's full step budget without converging.
    """
    pore = np.tile(~solid, (3, 3))
    lab, n = label(pore)                     # default structure = 4-connectivity
    if n == 0:
        return False, False
    left, right = set(lab[:, 0]), set(lab[:, -1])
    top, bottom = set(lab[0, :]), set(lab[-1, :])
    span_cols = (left & right) - {0}         # spans along axis 1 = solver x
    span_rows = (top & bottom) - {0}         # spans along axis 0 = solver y
    return bool(span_cols), bool(span_rows)


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------

def gen_one(sample_id, cfg):
    rng = np.random.default_rng(cfg["seed"] + sample_id)
    name, a_lo, a_hi = STRATA[sample_id % len(STRATA)]

    for attempt in range(cfg["max_attempts"]):
        aniso = float(rng.uniform(a_lo, a_hi))
        psi = float(rng.uniform(0.0, np.pi))
        k0 = float(rng.uniform(cfg["k0_min"], cfg["k0_max"]))
        n_modes = int(rng.integers(cfg["modes_min"], cfg["modes_max"] + 1))
        porosity = float(rng.uniform(cfg["por_min"], cfg["por_max"]))

        q = sample_wavevectors(rng, n_modes, k0, cfg["dk"], aniso, psi)
        if len(q) < 5:
            continue

        field = build_field(_GRID, q, rng)
        thresh = threshold_for_porosity(field, porosity)
        solid = field > thresh

        px, py = percolates(solid)
        if not (px and py):
            continue

        por = float(1.0 - solid.mean())
        fname = (f"sample_{sample_id:06d}_str={name}_A={aniso:.2f}"
                 f"_psi={np.degrees(psi):05.1f}_k0={k0:.2f}_por={por:.4f}")

        write_gif((1 - np.dstack([solid, solid, solid])) * 255,
                  os.path.join(cfg["structures_dir"], fname + ".gif"))
        if cfg["save_npy"]:
            np.save(os.path.join(cfg["npy_dir"], fname + ".npy"),
                    solid.astype(np.uint8))

        return {
            "sample_id": sample_id,
            "filename": fname + ".gif",
            "npy": fname + ".npy" if cfg["save_npy"] else "",
            "stratum": name,
            "aniso_A": aniso,
            "psi_deg": float(np.degrees(psi)),
            "k0": k0,
            "n_modes": len(q),
            "porosity": por,
            "porosity_target": porosity,
            "threshold": thresh,
            "attempts": attempt + 1,
        }

    logging.warning("sample %d (%s): no percolating structure in %d attempts",
                    sample_id, name, cfg["max_attempts"])
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test(size):
    """Frame and anisotropy sanity checks that need no solver."""
    grid = _build_grid(size)
    rng = np.random.default_rng(0)
    ok = True

    # A pure q=(k,0) mode must vary along the solver's x = array axis 1.
    f = build_field(grid, np.array([[4, 0]]), np.random.default_rng(1))
    var_along_axis1 = f.var(axis=1).mean()
    var_along_axis0 = f.var(axis=0).mean()
    good = var_along_axis1 > 100 * var_along_axis0
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] q=(4,0) varies along array axis 1 (solver x): "
          f"var {var_along_axis1:.3f} vs {var_along_axis0:.3f}")

    # psi=0 must elongate features along x, i.e. shrink the x wavenumbers.
    for psi_deg in (0.0, 90.0):
        q = sample_wavevectors(rng, 4000, 8.0, 1.0, 3.0, np.radians(psi_deg))
        rms_x, rms_y = np.sqrt((q ** 2).mean(axis=0))
        good = (rms_x < rms_y) if psi_deg == 0.0 else (rms_x > rms_y)
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] psi={psi_deg:5.1f} deg -> "
              f"rms|q_x|={rms_x:5.2f} rms|q_y|={rms_y:5.2f} "
              f"(expect {'q_x < q_y' if psi_deg == 0 else 'q_x > q_y'})")

    # Periodicity: the field must be continuous across the wrap.
    q = sample_wavevectors(rng, 40, 8.0, 1.0, 2.0, 0.7)
    f = build_field(grid, q, np.random.default_rng(2))
    wrap = max(np.abs(f[0] - f[-1]).max(), np.abs(f[:, 0] - f[:, -1]).max())
    step = max(np.abs(np.diff(f, axis=0)).max(), np.abs(np.diff(f, axis=1)).max())
    good = wrap <= 1.5 * step
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] periodic wrap jump {wrap:.4f} "
          f"<= 1.5x largest interior step {step:.4f}")

    # Percolation must reject a structure with a solid wall across x.
    solid = np.zeros((size, size), dtype=bool)
    solid[:, size // 2] = True               # wall spanning all rows, blocks x
    px, py = percolates(solid)
    good = (not px) and py
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] vertical solid wall: perc_x={px} perc_y={py} "
          f"(expect False, True)")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0,
                   help="added to the sample index to seed each structure")
    p.add_argument("--k0_min", type=float, default=4.0)
    p.add_argument("--k0_max", type=float, default=10.0)
    p.add_argument("--dk", type=float, default=1.0)
    p.add_argument("--modes_min", type=int, default=20)
    p.add_argument("--modes_max", type=int, default=80)
    p.add_argument("--porosity_min", type=float, default=0.65)
    p.add_argument("--porosity_max", type=float, default=0.90)
    p.add_argument("--max_attempts", type=int, default=200)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--no_npy", action="store_true",
                   help="skip the .npy copies used by the TDA pipeline")
    p.add_argument("--self-test", action="store_true", dest="selftest")
    args = p.parse_args()

    if args.selftest:
        raise SystemExit(self_test(args.size))
    if not args.output:
        p.error("--output is required (or use --self-test)")

    structures_dir = os.path.join(args.output, "structures")
    npy_dir = os.path.join(args.output, "npy")
    os.makedirs(structures_dir, exist_ok=True)
    if not args.no_npy:
        os.makedirs(npy_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(args.output, "gen.log")),
                  logging.StreamHandler()])

    cfg = {
        "structures_dir": structures_dir,
        "npy_dir": npy_dir,
        "save_npy": not args.no_npy,
        "seed": args.seed,
        "k0_min": args.k0_min, "k0_max": args.k0_max, "dk": args.dk,
        "modes_min": args.modes_min, "modes_max": args.modes_max,
        "por_min": args.porosity_min, "por_max": args.porosity_max,
        "max_attempts": args.max_attempts,
    }

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(args.size,)) as pool:
        futs = [pool.submit(gen_one, i, cfg) for i in range(args.n_samples)]
        for f in tqdm(as_completed(futs), total=len(futs), desc="generating"):
            r = f.result()
            if r is not None:
                rows.append(r)

    df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    df.to_csv(os.path.join(args.output, "structures.csv"), index=False)
    with open(os.path.join(args.output, "filelist.txt"), "w") as fh:
        for n in df.filename:
            fh.write(os.path.join(structures_dir, n) + "\n")

    logging.info("kept %d/%d structures", len(df), args.n_samples)
    print(df.groupby("stratum")[["porosity", "aniso_A", "k0", "attempts"]]
            .describe().T.to_string())


if __name__ == "__main__":
    main()
