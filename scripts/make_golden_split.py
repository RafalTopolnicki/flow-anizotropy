"""
Freeze a golden evaluation set: a fixed list of sample_ids, held out forever.

    python scripts/make_golden_split.py --dataset DATA/aniso --output splits/aniso_golden

Writes two files, and they belong in git even though DATA/ does not — the split
is 60 kB and is the one artefact that must never change.  If it drifted, every
learning-curve number computed before and after would be silently incomparable.

    splits/aniso_golden.csv    sample_id, stratum, split   for every usable row
    splits/aniso_golden.json   provenance + sha256 of the golden ids

Ids, not copied data.  `permeability.csv` stays the single source of truth
(NOTES 4.2); a golden set that copied structures or labels would be a second
copy free to drift, which is exactly what the one-structure-format rule exists
to prevent.  Joining on sample_id also means the same split serves CatBoost
(which reads descriptor CSVs) and the CNN (which reads GIFs) without either
knowing about the other.

The CSV lists the *pool* rows too, not just the golden ones, so it records which
rows were considered.  Otherwise a later change to the convergence filter would
quietly enlarge the training pool with nothing to show for it.

The hash is the point of the JSON: every consumer asserts the split it loaded
hashes to this value, which turns "did the golden set change?" from something
you have to remember into something that fails loudly.
"""

import argparse
import hashlib
import json
import os
from datetime import date

import numpy as np
import pandas as pd

# Recorded verbatim in the split manifest, so it is derived from the columns
# actually present rather than hard-coded: 2D has two force directions, 3D
# three, and a manifest claiming the wrong filter is worse than none.
def filter_desc(cols):
    return " and ".join(f"{c} == 1" for c in cols) or "none"


def golden_hash(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in sorted(ids)).encode()).hexdigest()


def load_split(csv_path, json_path=None):
    """
    Read a split back, verifying it is the one the manifest describes.

    Returns (golden_ids, pool_ids) as int arrays.
    """
    df = pd.read_csv(csv_path)
    golden = df.loc[df.split == "golden", "sample_id"].to_numpy()
    pool = df.loc[df.split == "pool", "sample_id"].to_numpy()

    if json_path is None:
        json_path = os.path.splitext(csv_path)[0] + ".json"
    if os.path.exists(json_path):
        meta = json.load(open(json_path))
        got = golden_hash(golden)
        if got != meta["sha256_golden_ids"]:
            raise SystemExit(
                f"golden set in {csv_path} does not match {json_path}\n"
                f"  manifest {meta['sha256_golden_ids']}\n"
                f"  actual   {got}\n"
                "Refusing to run: results would not be comparable with earlier ones.")
    return golden, pool


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="DATA/aniso")
    p.add_argument("--labels", default=None)
    p.add_argument("--output", default="splits/aniso_golden",
                   help="path prefix; .csv and .json are appended")
    p.add_argument("--n-golden", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing split (think hard before using this)")
    a = p.parse_args()

    labels = a.labels or os.path.join(a.dataset, "permeability.csv")
    csv_out, json_out = a.output + ".csv", a.output + ".json"
    if os.path.exists(csv_out) and not a.force:
        raise SystemExit(
            f"{csv_out} already exists.\nA golden set is meant to be frozen; "
            "overwriting it invalidates every number computed against it. "
            "Pass --force only if you really mean to.")

    df = pd.read_csv(labels)
    n0 = len(df)
    conv = [c for c in ("conv_fx", "conv_fy", "conv_fz") if c in df.columns]
    if conv:
        df = df[(df[conv] == 1).all(axis=1)]
    df = df[["sample_id", "stratum"]].sort_values("sample_id").reset_index(drop=True)
    print(f"{n0} rows -> {len(df)} usable ({filter_desc(conv)})")

    # Stratified: |k_off|/trace differs ~7x between strata, so an unbalanced
    # golden set would move R^2 for reasons that have nothing to do with the model.
    rng = np.random.default_rng(a.seed)
    chosen = []
    for s, g in df.groupby("stratum"):
        k = int(round(a.n_golden * len(g) / len(df)))
        chosen.append(rng.choice(g.sample_id.to_numpy(), size=k, replace=False))
    golden = np.sort(np.concatenate(chosen))

    df["split"] = np.where(df.sample_id.isin(golden), "golden", "pool")
    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    df.to_csv(csv_out, index=False)

    counts = {k: v.stratum.value_counts().to_dict() for k, v in df.groupby("split")}
    meta = {
        "dataset": a.dataset, "labels": labels,
        "created": date.today().isoformat(),
        "seed": a.seed, "filter": filter_desc(conv),
        "n_total": int(len(df)),
        "n_golden": int((df.split == "golden").sum()),
        "n_pool": int((df.split == "pool").sum()),
        "stratified_by": "stratum",
        "counts": counts,
        "sha256_golden_ids": golden_hash(golden),
    }
    with open(json_out, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\ngolden {meta['n_golden']}   pool {meta['n_pool']}")
    print(pd.DataFrame(counts).fillna(0).astype(int).to_string())
    print(f"\nsha256(golden ids) {meta['sha256_golden_ids'][:16]}...")
    print(f"-> {csv_out}\n-> {json_out}")
    print("\nCommit both.  They are small, and they are what makes every "
          "learning-curve number reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
