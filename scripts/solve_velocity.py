"""
Solve one structure and keep the velocity fields.

    python scripts/solve_velocity.py DATA/aniso/structures/sample_000000_*.gif -o OUT/s0
    python scripts/solve_velocity.py <structure.gif> -o OUT/s0 --vtk

Runs `lbm2d-perm` on the structure exactly the way the dataset run does — two
solves of the same doubly periodic cell, force along +x then +y, the two columns
of K — and additionally saves the converged velocity field of each run.

Outputs, given `-o OUT/s0`:

    OUT/s0.fx.npy      (ny, nx, 2) float32   velocity under x-forcing
    OUT/s0.fy.npy      (ny, nx, 2) float32   velocity under y-forcing
    OUT/s0.solid.npy   (ny, nx)    uint8     1 = solid, 0 = pore
    OUT/s0.csv                               the usual full tensor row
    OUT/s0.fx.vtk, OUT/s0.fy.vtk             with --vtk

Arrays are indexed ``[solver_y, solver_x]`` with components ``(u_x, u_y)`` — the
same orientation `structure_io.load_structure` returns, so the field and the
mask overlay without transposing either. See `to_vtk.py` for the VTK ordering.

Two consistency checks run automatically and print their numbers, because the
orientation is the one thing here that can be wrong in a way that still looks
like a plausible flow field:

  * **mask agreement** — `macro()` only ever writes U,V on pore nodes and the
    arrays start zeroed, so the velocity field must be *exactly* zero on every
    solid node. A transposed parse breaks this immediately.
  * **flux reproduction** — recomputing q = sum(u)/(L*L0^2) from the field must
    reproduce the `qx_fx`/`qy_fx`/... columns the solver wrote. Agreement is to
    ~1%, not to machine precision, and that is expected: the CSV holds the mean
    over the 20k-step averaging phase while the exported field is the
    instantaneous state at the last step (NOTES 3.5 measured that spread at
    0.2-1.8%). A gross mismatch means a parsing error; a sub-percent one is the
    fluctuation the block average exists to remove.

Note the exported field is the raw LBM moment sum(f*e)/rho, without the +F/(2 rho)
forcing correction, matching the flux convention the whole project uses
(NOTES section 6).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_io import load_structure          # noqa: E402
from to_vtk import write_vtk                     # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOLVER = REPO / "LMB2d" / "lbm2d-perm"
L0 = 4          # lbm.h: characteristic block size, used by volumeavg()


def read_velocity_dat(path, ny, nx):
    """
    Parse one `--velocity` file into (ny, nx, 2).

    `exportvelocity` in lbm.cpp writes, with j (y) as the outer loop and i (x)
    as the inner one:

        for j in range(LY):
            for i in range(LX):
                print(U[i][j], V[i][j])

    so row j*LX + i holds the velocity at (x=i, y=j) and a plain reshape to
    (LY, LX, 2) is already indexed [y][x]. No transpose.
    """
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.shape != (ny * nx, 2):
        raise ValueError(f"{path}: expected {ny*nx} rows of 2, got {raw.shape}")
    return raw.reshape(ny, nx, 2)


def volume_flux(field):
    """q = (sum u_x, sum u_y) / (L * L0^2), reproducing lbm.cpp: volumeavg()."""
    ny, nx = field.shape[:2]
    return (float(field[..., 0].sum()) / (nx * L0 * L0),
            float(field[..., 1].sum()) / (ny * L0 * L0))


def run(args):
    structure = Path(args.structure)
    if not structure.exists():
        sys.exit(f"no such structure: {structure}")
    solver = Path(args.solver)
    if not solver.exists():
        sys.exit(f"solver not found: {solver}\nbuild it with: cd LMB2d && bash make_perm.sh")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_suffix(".csv")
    if csv.exists():
        csv.unlink()        # the solver appends; a stale row would be confusing

    cmd = [str(solver), str(structure), str(csv),
           "--ff", str(args.ff), "--velocity", str(out)]
    for flag, val in (("--eps", args.eps), ("--max-steps", args.max_steps),
                      ("--avg-steps", args.avg_steps)):
        if val is not None:
            cmd += [flag, str(val)]
    if args.quiet:
        cmd.append("--quiet")

    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc == 2:
        sys.exit("solver failed (exit 2)")
    if rc == 3:
        print("\n!! solver exit 3: at least one direction did not converge; "
              "fields are still written but check conv_fx/conv_fy in the CSV\n")

    solid = load_structure(structure)
    ny, nx = solid.shape

    fields = {}
    for tag in ("fx", "fy"):
        dat = Path(f"{out}.{tag}.dat")
        if not dat.exists():
            sys.exit(f"solver did not write {dat}")
        fields[tag] = read_velocity_dat(dat, ny, nx)
        np.save(f"{out}.{tag}.npy", fields[tag].astype(np.float32))
        if not args.keep_dat:
            dat.unlink()
    np.save(f"{out}.solid.npy", solid)

    _report(fields, solid, csv, out)

    if args.vtk:
        for tag, field in fields.items():
            write_vtk(f"{out}.{tag}.vtk", field, name="velocity", solid=solid,
                      ascii_mode=args.ascii,
                      title=f"{structure.name} force=+{tag[-1]}")
            print(f"  wrote {out}.{tag}.vtk")

    return 0


def _report(fields, solid, csv, out):
    import csv as csvmod

    ny, nx = solid.shape
    print(f"\nstructure {nx}x{ny}, porosity {1.0 - solid.mean():.4f}")
    for tag, field in fields.items():
        print(f"  {out}.{tag}.npy  {field.shape} float32")

    print("\ncheck 1 - velocity is exactly zero on solid nodes")
    for tag, field in fields.items():
        n_bad = int((solid[..., None].astype(bool) & (field != 0.0)).any(-1).sum())
        worst = float(np.abs(field[solid.astype(bool)]).max()) if solid.any() else 0.0
        print(f"  {tag}: {n_bad} solid nodes with nonzero velocity "
              f"(max |u| there {worst:.3e})  [{'ok' if n_bad == 0 else 'FAIL'}]")

    row = None
    if csv.exists():
        with open(csv) as fh:
            rows = list(csvmod.DictReader(fh))
        row = rows[-1] if rows else None
    if row is None:
        print("\ncheck 2 - skipped, no CSV row")
        return

    print("\ncheck 2 - flux recomputed from the field vs the solver's block mean")
    print(f"  {'':10s} {'from field':>13s} {'from CSV':>13s} {'rel diff':>10s}")
    for tag in ("fx", "fy"):
        qx, qy = volume_flux(fields[tag])
        for comp, mine in (("qx", qx), ("qy", qy)):
            theirs = float(row[f"{comp}_{tag}"])
            denom = max(abs(theirs), 1e-30)
            print(f"  {comp}_{tag:3s} {mine:13.6e} {theirs:13.6e} "
                  f"{abs(mine - theirs)/denom:10.2%}")

    k = {c: float(row[c]) for c in ("k_xx", "k_yx", "k_xy", "k_yy")}
    print(f"\n  K = [[{k['k_xx']:10.3f} {k['k_xy']:10.3f}]"
          f"\n       [{k['k_yx']:10.3f} {k['k_yy']:10.3f}]]"
          f"   recip_resid {float(row['recip_resid']):.2e}"
          f"   theta {float(row['theta_deg']):.2f} deg")
    print(f"  full row: {csv}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("structure", help="structure .gif (or .ppm/.dat)")
    p.add_argument("-o", "--output", required=True, help="output prefix, e.g. OUT/s0")
    p.add_argument("--solver", default=str(DEFAULT_SOLVER))
    p.add_argument("--ff", type=float, default=1)
    p.add_argument("--eps", type=float)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--avg-steps", type=int)
    p.add_argument("--vtk", action="store_true", help="also write ParaView .vtk files")
    p.add_argument("--ascii", action="store_true", help="ASCII VTK instead of binary")
    p.add_argument("--keep-dat", action="store_true",
                   help="keep the solver's raw text .dat files")
    p.add_argument("--quiet", action="store_true")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
