"""
Run `lbm2d-perm` over a generated structure set and collect the tensors.

Each structure gets its own CSV under <dataset>/perm/ rather than appending to
one shared file: concurrent appends from N workers race on the header, and
per-sample files make the run restartable — rerun the script and it solves only
what is missing.  The merge joins the solver output onto structures.csv so the
generator knobs and the tensor live in one table.

Usage
-----
    python scripts/solve_permeability.py --dataset DATA/aniso --workers 14
    python scripts/solve_permeability.py --dataset DATA/aniso --merge-only
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

DEFAULT_SOLVER = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "LMB2d", "lbm2d-perm")


def solve_one(task):
    solver, structure, out_csv, ff, extra = task
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        return out_csv, 0, "cached"

    cmd = [solver, structure, out_csv, "--quiet", "--ff", str(ff)] + extra
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return out_csv, -1, "timeout"

    # Exit 3 = at least one direction did not converge.  The row is still
    # written and flagged by conv_fx / conv_fy, so keep it and filter later.
    if p.returncode not in (0, 3):
        if os.path.exists(out_csv):
            os.remove(out_csv)
        return out_csv, p.returncode, (p.stderr or "").strip()[-200:]
    return out_csv, p.returncode, "ok"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="directory produced by gen_aniso_structures.py")
    ap.add_argument("--solver", default=DEFAULT_SOLVER)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--ff", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="solve only the first N")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--extra", default="",
                    help="extra flags passed through to the solver, e.g. \"--eps 5e-4\"")
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
            sys.exit(f"solver not found: {args.solver}  (build it with LMB2d/make_perm.sh)")

        extra = args.extra.split()
        tasks = [(args.solver,
                  os.path.join(structures_dir, r.filename),
                  os.path.join(perm_dir, f"{r.sample_id:06d}.csv"),
                  args.ff, extra)
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

        print(f"cached {cached}   non-converged (kept, flagged) {unconverged}   failed {len(failed)}")
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
    out = df.merge(perm, on="sample_id", how="inner",
                   suffixes=("", "_solver"))

    # Derived targets.  theta is defined mod 180 deg, so a model must predict
    # (cos 2t, sin 2t) rather than the angle itself.
    import numpy as np
    t = np.radians(out["theta_deg"])
    out["cos2theta"] = np.cos(2 * t)
    out["sin2theta"] = np.sin(2 * t)
    out["log_k_mean"] = np.log(out["k_mean"].where(out["k_mean"] > 0))
    out["k_ratio"] = out["k1"] / out["k2"].where(out["k2"] > 0)
    out["log_k_ratio"] = np.log(out["k_ratio"].where(out["k_ratio"] > 0))
    # Label-noise proxy for the off-diagonal, from the K = K^T identity.
    out["koff_rel_err"] = ((out["k_xy"] - out["k_yx"]).abs()
                           / (2 * out["k_off"]).abs())
    out["koff_over_trace"] = out["k_off"].abs() / (out["k_xx"] + out["k_yy"]).abs()

    dest = os.path.join(args.dataset, "permeability.csv")
    out.to_csv(dest, index=False)
    print(f"merged {len(out)} rows -> {dest}")

    conv = (out.conv_fx == 1) & (out.conv_fy == 1)
    print(f"both directions converged: {conv.sum()}/{len(out)}")
    print(out.loc[conv, ["recip_resid", "koff_over_trace", "koff_rel_err",
                         "anisotropy", "k_mean"]].describe().T.to_string())


if __name__ == "__main__":
    main()
