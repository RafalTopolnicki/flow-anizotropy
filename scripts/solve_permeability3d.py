"""
Run `lbm3d-perm` over a generated 3D structure set and collect the tensors.

The 3D counterpart of `solve_permeability.py`, and the same design for the same
reasons: one CSV per sample under <dataset>/perm/ rather than appending to a
shared file, because concurrent appends from N workers race on the header and
per-sample files make a multi-day run **restartable** — rerun and it solves only
what is missing.  Solver exit 3 (a direction that did not converge) is kept and
flagged, not treated as a failure.

Restartability matters more here than in 2D: this run is days rather than hours.

Usage
-----
    python scripts/solve_permeability3d.py --dataset DATA/aniso3d --workers 14
    python scripts/solve_permeability3d.py --dataset DATA/aniso3d --merge-only

Derived targets
---------------
`k_xy`, `k_xz`, `k_yz` are the **symmetrised** off-diagonals.  K = K^T is exact
for Stokes flow, so averaging the two measurements of each halves the noise on
precisely the components that need it, and the difference that is thrown away is
kept as a per-sample error bar:

    koff_rel_err_xy = |K_xy - K_yx| / |2 k_xy|      etc.

That is the 3D form of the label-noise law in NOTES 3.3, now giving three
independent error bars instead of one — one per off-diagonal target.

**Do not regress eigenvector directions.**  In 2D the fix for theta being
defined mod 180 deg was to predict (cos 2t, sin 2t).  In 3D the principal frame
is worse behaved than that: eigenvectors are sign-ambiguous, their order can
swap when eigenvalues cross, and for a rod or a plate two of them are genuinely
degenerate and not determined by the structure at all (measured in the generator
pilot).  The unambiguous target is the tensor itself — the six components of K,
or equivalently log k_mean plus the five deviatoric components.  The principal
values k1 >= k2 >= k3 and the fractional anisotropy are safe; the axes are not.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_SOLVER = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "LMB3d", "lbm3d-perm")


def solve_one(task):
    solver, structure, out_csv, ff, timeout, extra = task
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return out_csv, 0, "cached"

    cmd = [solver, structure, out_csv, "--quiet", "--ff", str(ff)] + extra
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.path.exists(out_csv):
            os.remove(out_csv)
        return out_csv, -1, "timeout"

    if p.returncode not in (0, 3):
        if os.path.exists(out_csv):
            os.remove(out_csv)
        return out_csv, p.returncode, (p.stderr or "").strip()[-200:]
    return out_csv, p.returncode, "ok"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="directory produced by gen_aniso_structures3d.py")
    ap.add_argument("--solver", default=DEFAULT_SOLVER)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--ff", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="solve only the first N")
    ap.add_argument("--timeout", type=int, default=86400,
                    help="per-structure wall-clock limit in seconds (default 24 h)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--extra", default="",
                    help='extra flags for the solver, e.g. "--eps 5e-4"')
    args = ap.parse_args()

    struct_csv = os.path.join(args.dataset, "structures.csv")
    if not os.path.exists(struct_csv):
        sys.exit(f"no structures.csv in {args.dataset}")
    df = pd.read_csv(struct_csv)
    if args.limit:
        df = df.head(args.limit)

    perm_dir = os.path.join(args.dataset, "perm")
    os.makedirs(perm_dir, exist_ok=True)
    structures_dir = os.path.join(args.dataset, "structures")

    if not args.merge_only:
        if not os.path.exists(args.solver):
            sys.exit(f"solver not found: {args.solver}  (build it with LMB3d/make_perm.sh)")

        extra = args.extra.split()
        tasks = [(args.solver,
                  os.path.join(structures_dir, r.filename),
                  os.path.join(perm_dir, f"{r.sample_id:06d}.csv"),
                  args.ff, args.timeout, extra)
                 for r in df.itertuples()]

        failed, unconverged, cached = [], 0, 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(solve_one, t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs), desc="solving"):
                out, rc, msg = f.result()
                if msg == "cached":
                    cached += 1
                elif rc == 3:
                    unconverged += 1
                elif rc != 0:
                    failed.append((os.path.basename(out), rc, msg))

        print(f"cached {cached}   non-converged (kept, flagged) {unconverged}   "
              f"failed {len(failed)}")
        for name, rc, msg in failed[:10]:
            print(f"  {name}: rc={rc} {msg}")

    # --- merge --------------------------------------------------------------
    parts = []
    for r in df.itertuples():
        pth = os.path.join(perm_dir, f"{r.sample_id:06d}.csv")
        if os.path.exists(pth) and os.path.getsize(pth) > 0:
            try:
                one = pd.read_csv(pth)
            except Exception:
                continue
            one["sample_id"] = r.sample_id
            parts.append(one)

    if not parts:
        sys.exit("nothing to merge")

    perm = pd.concat(parts, ignore_index=True).drop(columns=["filename"])
    out = df.merge(perm, on="sample_id", how="inner", suffixes=("", "_solver"))

    trace = out["K_xx"] + out["K_yy"] + out["K_zz"]
    out["trace"] = trace
    out["log_k_mean"] = np.log(out["k_mean"].where(out["k_mean"] > 0))
    out["k_ratio"] = out["k1"] / out["k3"].where(out["k3"] > 0)
    out["log_k_ratio"] = np.log(out["k_ratio"].where(out["k_ratio"] > 0))

    # Per-off-diagonal label noise, and the quantity it depends on.
    for a, b in (("x", "y"), ("x", "z"), ("y", "z")):
        sym = out[f"k_{a}{b}"]
        asym = (out[f"K_{a}{b}"] - out[f"K_{b}{a}"]).abs()
        out[f"koff_rel_err_{a}{b}"] = asym / (2 * sym).abs()
        out[f"koff_over_trace_{a}{b}"] = sym.abs() / trace.abs()

    dest = os.path.join(args.dataset, "permeability.csv")
    out.to_csv(dest, index=False)
    print(f"merged {len(out)} rows -> {dest}")

    conv = (out.conv_fx == 1) & (out.conv_fy == 1) & (out.conv_fz == 1)
    print(f"all three directions converged: {conv.sum()}/{len(out)}")
    cols = (["recip_resid", "fa", "k_mean", "k_ratio"]
            + [f"koff_over_trace_{a}{b}" for a, b in (("x","y"),("x","z"),("y","z"))]
            + [f"koff_rel_err_{a}{b}" for a, b in (("x","y"),("x","z"),("y","z"))])
    print(out.loc[conv, cols].describe().T.to_string())


if __name__ == "__main__":
    main()
