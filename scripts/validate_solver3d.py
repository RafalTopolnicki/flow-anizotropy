"""
Check a build of `lbm3d-perm` against an exact analytic solution.

    python scripts/validate_solver3d.py
    python scripts/validate_solver3d.py --solver /path/to/lbm3d-perm

Run this after compiling on a new machine, before committing any real compute.
It takes a couple of minutes and catches a broken build, a bad compiler flag, or
a wrong-endian .raw reader — none of which announce themselves in the output of
a real structure, where you have nothing to compare against.

The test is Poiseuille flow in a slit: a solid slab normal to z, so the flow is
a plane parabola driven along x.  With halfway bounce-back the walls sit half a
lattice spacing outside the last fluid node, which makes the aperture exactly
h = (number of fluid layers) and gives a closed form for the answer:

    u_j = (F / 2 nu) z_j (h - z_j),   z_j = j + 1/2,   j = 0 .. h-1
    k   = nu * sum_j u_j / (F * nz)

Three things are checked:

  1. k_xx and k_yy against that closed form.  Agreement is ~1e-3, NOT machine
     precision, and it must scale as h^-2 — that is the second-order
     wall-position error of halfway bounce-back, and seeing the exponent is a
     stronger check than any single number.
  2. k_zz is exactly zero.  Flow across the slab is impossible, so anything
     else means the forcing or the velocity moment is wrong.  This is what
     caught the 2D code's forcing convention, which reported k_zz = -0.05.
  3. The off-diagonals vanish.  The slit is aligned with the axes, so any
     coupling is numerical.

The z-forced run cannot converge (there is no flow), so the solver returns exit
3 and flags conv_fz = 0.  That is the expected, correct behaviour and the script
accounts for it; --max-steps is kept low so it does not dominate the runtime.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_io3d import save_structure          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOLVER = REPO / "LMB3d" / "lbm3d-perm"
NU = (1.0 / 3.0) * (1.0 - 0.5)          # cs^2 (tau - 0.5), tau = 1


def k_exact(n_fluid, nz):
    h = float(n_fluid)
    z = np.arange(n_fluid) + 0.5
    return float((z * (h - z)).sum()) / (2.0 * nz)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solver", default=str(DEFAULT_SOLVER))
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--apertures", type=int, nargs="+", default=[10, 20])
    p.add_argument("--max-steps", type=int, default=40000)
    p.add_argument("--tol", type=float, default=8e-3,
                   help="max acceptable relative error on k_xx")
    a = p.parse_args()

    if not os.path.exists(a.solver):
        sys.exit(f"solver not found: {a.solver}\n"
                 f"build it with:  cd LMB3d && bash make_perm.sh")

    tmp = Path(tempfile.mkdtemp())
    n = a.size
    ok = True
    errs = []

    print(f"solver {a.solver}")
    print(f"domain {n}^3, apertures {a.apertures}\n")
    print(f"{'h':>4} {'k_xx':>13} {'exact':>13} {'rel err':>9} "
          f"{'k_yy err':>9} {'k_zz':>10} {'max off-diag':>13}")
    print("-" * 78)

    for h in a.apertures:
        ns = n - h
        solid = np.zeros((n, n, n), np.uint8)
        solid[:ns] = 1                       # slab normal to z
        raw = tmp / f"slit{h}.raw"
        save_structure(raw, solid)

        subprocess.run([a.solver, str(raw), str(tmp / f"slit{h}.csv"), "--quiet",
                        "--max-steps", str(a.max_steps)],
                       check=False, capture_output=True)
        try:
            r = pd.read_csv(tmp / f"slit{h}.csv").iloc[-1]
        except Exception as e:
            print(f"{h:4d}   solver produced no row ({e})")
            ok = False
            continue

        ke = k_exact(h, n)
        ex = abs(r.K_xx - ke) / ke
        ey = abs(r.K_yy - ke) / ke
        off = max(abs(r.K_xy), abs(r.K_xz), abs(r.K_yz))
        errs.append((h, ex))
        good = ex < a.tol and ey < a.tol and abs(r.K_zz) < 1e-6 * ke and off < 1e-6 * ke
        ok &= good
        print(f"{h:4d} {r.K_xx:13.8f} {ke:13.8f} {ex:9.2e} {ey:9.2e} "
              f"{r.K_zz:+10.1e} {off:13.1e}  [{'ok' if good else 'FAIL'}]")

    # h^-2 scaling is the real signature of correct bounce-back: the error is a
    # wall-position offset, not a bug, and only a bug would scale differently.
    if len(errs) >= 2:
        (h1, e1), (h2, e2) = errs[0], errs[-1]
        got, want = e1 / e2, (h2 / h1) ** 2
        scaling_ok = abs(got - want) / want < 0.25
        ok &= scaling_ok
        print(f"\nerror ratio h={h1} : h={h2}  measured {got:.2f}, "
              f"expected {want:.2f} (second order)  "
              f"[{'ok' if scaling_ok else 'FAIL'}]")

    print("\nsolver validation", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
