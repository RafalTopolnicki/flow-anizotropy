r"""
Assemble per-structure descriptor .npy files into one direction-tagged table.

Reads `<descroot>/<kind>/a<theta>/<name>.npy` for every kind and direction
found, and writes a single CSV keyed by `sample_id`.

Column naming: `{kind}_a{theta}_{i}`, e.g. `ecp_a45_0 … ecp_a45_80`,
`ph_a135_0 … ph_a135_199`.

Deliberately a single pass over everything, rather than the 3D pipeline's
"first direction creates the CSV, later ones `--append` columns" flow. That
flow made the output depend on invocation order and silently produced partial
tables when one direction was missing; here a missing direction is an error you
see immediately, and rerunning is idempotent.

Structures are matched to rows by `sample_id`, parsed from the leading
`sample_%06d` of the filename — never by regex over the descriptive part of the
name, which is the trap recorded in `gen_aniso_structures.py`.

Usage
-----
    python scripts/collect_descriptors.py --dataset DATA/aniso --descroot DESC \
        --output DESC/descriptors.csv
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

SAMPLE_RE = re.compile(r"^sample_(\d{6})")


def sample_id_of(fname: str) -> int:
    m = SAMPLE_RE.match(os.path.basename(fname))
    if not m:
        raise ValueError(f"cannot parse sample_id from {fname!r}")
    return int(m.group(1))


def load_block(dirpath: str, kind: str, theta: str) -> pd.DataFrame:
    """All .npy under dirpath -> DataFrame indexed by sample_id."""
    files = sorted(f for f in os.listdir(dirpath) if f.endswith(".npy"))
    if not files:
        raise SystemExit(f"no .npy files in {dirpath}")

    ids = [sample_id_of(f) for f in files]
    mat = np.stack([np.load(os.path.join(dirpath, f)).ravel() for f in files])
    cols = [f"{kind}_a{theta}_{i}" for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "sample_id", ids)
    return df.set_index("sample_id")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--descroot", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kinds", nargs="+", default=["ecp", "ph"])
    a = p.parse_args()

    base = pd.read_csv(os.path.join(a.dataset, "structures.csv")).set_index("sample_id")
    out = base[["filename", "stratum", "porosity"]].copy()

    total = 0
    for kind in a.kinds:
        kind_dir = os.path.join(a.descroot, kind)
        if not os.path.isdir(kind_dir):
            print(f"  ! no {kind_dir}, skipping")
            continue
        for tag in sorted(os.listdir(kind_dir),
                          key=lambda t: float(t.lstrip("a")) if t.startswith("a") else 1e9):
            if not tag.startswith("a"):
                continue
            theta = tag[1:]
            blk = load_block(os.path.join(kind_dir, tag), kind, theta)
            missing = out.index.difference(blk.index)
            if len(missing):
                sys.exit(f"{kind}/{tag}: {len(missing)} structures have no descriptor "
                         f"(first: {list(missing[:3])}) — rerun that stage")
            out = out.join(blk, how="left")
            total += blk.shape[1]
            print(f"  {kind}/{tag}: {blk.shape[0]} rows x {blk.shape[1]} features")

    if total == 0:
        sys.exit("no descriptors found")

    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    out.reset_index().to_csv(a.output, index=False)

    n_nan = int(out.isna().sum().sum())
    print(f"\n  {len(out)} rows x {total} descriptor features -> {a.output}")
    print(f"  NaNs: {n_nan}")
    if n_nan:
        sys.exit("NaNs present — investigate before training")


if __name__ == "__main__":
    main()
