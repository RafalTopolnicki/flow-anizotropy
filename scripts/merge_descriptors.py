r"""
Join descriptor part-tables built from disjoint structure ranges into one CSV.

Why this exists
---------------
A descriptor table is normally built in one pass over `[0, --limit)`, but the
per-structure `.npy` tree it is folded from is an *intermediate* and gets
deleted once `collect_descriptors.py` has written the CSV (it is the thing that
eats the ceph inode quota).  So when the labelled set grows -- DATA/aniso3d
went 2816 -> 5000 when the LBM solve finished -- the head rows cannot be
re-collected without recomputing PH for structures that are already in the CSV.
Instead the tail is computed on its own (`collect_descriptors.py --start`) and
the two parts are joined here.

**A head/tail seam is a silent-corruption shape.**  If any filtration or
imaging parameter differed between the two builds, the result is a table with a
discontinuity at the seam row, every column still numeric, no NaN, and no R^2
that would ever reveal it.  Persistence images are only concatenable because
their imaging ranges are fixed CLI constants rather than derived from the batch
-- so a structure's features do not depend on which run computed them.  That
holds by construction, and this script checks that it also held in fact.

Checks, all fatal unless noted
------------------------------
1. every part has the same columns in the same order
2. no duplicate sample_id inside a part, and none shared between parts
3. the union of sample_ids is exactly [0, --expect), no gaps, no strays
4. filename/stratum/porosity agree row-for-row with the dataset's
   structures.csv -- catches a row-order or identity mix-up, which is the
   failure a pure column check cannot see
5. no NaN and no inf in any feature column
6. **the seam statistic**: per feature, the standardized mean difference
   between consecutive parts, scaled to a z by the part sizes.  The parts are
   disjoint index ranges of one generator whose strata are assigned cyclically
   by sample index, so both are samples of the same distribution and z should
   be O(1).  A changed bandwidth, range, RCAP or phase moves every pixel of the
   image at once and shows up here as a wall of large z.  Aborts above
   `--seam-z` (default 8); `--allow-seam-shift` downgrades it to a warning for
   the case where you *intended* the parts to differ.

Writes `<output>.parts.json` beside the CSV recording which rows came from
which part and what the seam measured, so the provenance of a joined table is
readable later without rerunning anything.

Usage
-----
    python scripts/merge_descriptors.py \
        --parts DESC/aniso3d_throat/descriptors_part_0_2816.csv \
                DESC/aniso3d_throat/descriptors_part_2816_5000.csv \
        --output DESC/aniso3d_throat/descriptors.csv \
        --dataset DATA/aniso3d --expect 5000

    python scripts/merge_descriptors.py --self-test
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

KEY = "sample_id"
META = ["filename", "stratum", "porosity"]


def die(msg: str):
    sys.exit(f"ABORT: {msg}")


def read_part(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        die(f"part does not exist: {path}")
    df = pd.read_csv(path)
    if KEY not in df.columns:
        die(f"{path}: no {KEY} column")
    if df.empty:
        die(f"{path}: no rows")
    return df


def feature_columns(cols) -> list:
    return [c for c in cols if c != KEY and c not in META]


def check_columns(parts, paths):
    """Same columns, same order, everywhere."""
    ref, refp = list(parts[0].columns), paths[0]
    for df, path in zip(parts[1:], paths[1:]):
        cols = list(df.columns)
        if cols == ref:
            continue
        only_ref = [c for c in ref if c not in set(cols)]
        only_new = [c for c in cols if c not in set(ref)]
        if only_ref or only_new:
            die(f"column sets differ between {refp} and {path}: "
                f"{len(only_ref)} only in the first (e.g. {only_ref[:3]}), "
                f"{len(only_new)} only in the second (e.g. {only_new[:3]}). "
                f"The parts were built with different descriptors.")
        first = next(i for i, (x, y) in enumerate(zip(ref, cols)) if x != y)
        die(f"column ORDER differs between {refp} and {path}, first at "
            f"index {first}: {ref[first]!r} vs {cols[first]!r}")


def check_ids(parts, paths, expect):
    """No duplicates, no overlap, and exactly [0, expect) when expect is given."""
    seen = {}
    for df, path in zip(parts, paths):
        ids = df[KEY].to_numpy()
        dup = pd.Index(ids)[pd.Index(ids).duplicated()].unique()
        if len(dup):
            die(f"{path}: {len(dup)} duplicated {KEY} (e.g. {list(dup[:3])})")
        for i in ids:
            if i in seen:
                die(f"{KEY} {i} appears in both {seen[i]} and {path} — the "
                    f"parts are not disjoint, so one would silently win")
            seen[i] = path

    ids = np.array(sorted(seen))
    if expect:
        want = set(range(expect))
        missing, extra = sorted(want - set(ids.tolist())), sorted(set(ids.tolist()) - want)
        if missing:
            die(f"{len(missing)} structures are in no part (first: {missing[:5]}); "
                f"expected a complete [0, {expect})")
        if extra:
            die(f"{len(extra)} sample_ids outside [0, {expect}) (first: {extra[:5]})")
    elif not np.array_equal(ids, np.arange(len(ids))):
        gaps = sorted(set(range(int(ids.max()) + 1)) - set(ids.tolist()))
        die(f"sample_ids are not contiguous from 0: {len(gaps)} gaps "
            f"(first: {gaps[:5]}). Pass --expect if a prefix is intended.")
    return ids


def check_metadata(merged, dataset):
    """filename/stratum/porosity must match structures.csv row for row."""
    path = os.path.join(dataset, "structures.csv")
    if not os.path.isfile(path):
        die(f"no {path} — pass the dataset that the parts were built from")
    base = pd.read_csv(path).set_index(KEY)
    have = [c for c in META if c in merged.columns]
    for col in have:
        if col not in base.columns:
            continue
        ref = base.loc[merged[KEY].to_numpy(), col].to_numpy()
        got = merged[col].to_numpy()
        bad = (~np.isclose(ref.astype(float), got.astype(float), rtol=0, atol=1e-9)
               if col == "porosity" else ref != got)
        if bad.any():
            n, first = int(bad.sum()), int(merged[KEY].to_numpy()[bad][0])
            die(f"{col} disagrees with structures.csv for {n} rows "
                f"(first sample_id {first}: {base.at[first, col]!r} in the "
                f"dataset, {merged.loc[merged[KEY] == first, col].iloc[0]!r} in "
                f"the parts). A part was built against different structures.")
    return have


def check_finite(merged, feats):
    vals = merged[feats].to_numpy(dtype=float, copy=False)
    n_nan, n_inf = int(np.isnan(vals).sum()), int(np.isinf(vals).sum())
    if n_nan or n_inf:
        col = np.array(feats)[(np.isnan(vals) | np.isinf(vals)).any(axis=0)][:3]
        die(f"{n_nan} NaN and {n_inf} inf in the feature columns "
            f"(e.g. {list(col)}) — investigate before training")


def seam_stats(a: pd.DataFrame, b: pd.DataFrame, feats: list) -> dict:
    """
    Standardized mean difference per feature, as a z.

    d = (mean_b - mean_a) / pooled_sd, z = d / sqrt(1/na + 1/nb).  Features with
    no variance in either part contribute nothing (a constant column is equal in
    both or caught by the max-abs-difference term instead).
    """
    A = a[feats].to_numpy(dtype=float, copy=False)
    B = b[feats].to_numpy(dtype=float, copy=False)
    na, nb = len(A), len(B)
    ma, mb = A.mean(axis=0), B.mean(axis=0)
    va, vb = A.var(axis=0, ddof=1), B.var(axis=0, ddof=1)
    pooled = np.sqrt(np.maximum((va + vb) / 2.0, 0.0))

    live = pooled > 0
    z = np.zeros(len(feats))
    z[live] = ((mb - ma)[live] / pooled[live]) / np.sqrt(1.0 / na + 1.0 / nb)

    # A column that is constant in both parts but at DIFFERENT values has zero
    # pooled sd and infinite z; report it separately rather than as a nan.
    frozen = (~live) & (np.abs(mb - ma) > 0)
    order = np.argsort(-np.abs(z))[:5]
    return {
        "n_a": na, "n_b": nb, "n_features": len(feats),
        "max_abs_z": float(np.abs(z).max()) if len(z) else 0.0,
        "n_z_over_5": int((np.abs(z) > 5).sum()),
        "n_frozen_columns_differing": int(frozen.sum()),
        "worst": [{"column": feats[i], "z": float(z[i])} for i in order],
    }


def merge(paths, output, dataset, expect, seam_z, allow_shift, quiet=False):
    parts = [read_part(p) for p in paths]
    check_columns(parts, paths)

    feats = feature_columns(parts[0].columns)
    if not feats:
        die("the parts carry no feature columns")

    ids = check_ids(parts, paths, expect)

    merged = pd.concat(parts, ignore_index=True).sort_values(KEY).reset_index(drop=True)

    meta_checked = check_metadata(merged, dataset) if dataset else []
    check_finite(merged, feats)

    seams = []
    ordered = sorted(zip(paths, parts), key=lambda kv: kv[1][KEY].min())
    for (pa, da), (pb, db) in zip(ordered, ordered[1:]):
        s = seam_stats(da, db, feats)
        s["a"], s["b"] = pa, pb
        s["boundary"] = int(db[KEY].min())
        seams.append(s)

    if not quiet:
        print(f"  parts: {len(parts)}   rows: {len(merged)}   features: {len(feats)}")
        for p, df in ordered:
            print(f"    {os.path.basename(p):<44} "
                  f"{KEY} {int(df[KEY].min())}..{int(df[KEY].max())}  ({len(df)} rows)")
        print(f"  metadata checked against structures.csv: "
              f"{', '.join(meta_checked) if meta_checked else 'SKIPPED (no --dataset)'}")
        for s in seams:
            print(f"  seam at {KEY} {s['boundary']}: max|z| {s['max_abs_z']:.2f} "
                  f"over {s['n_features']} features, {s['n_z_over_5']} above 5")
            for w in s["worst"][:3]:
                print(f"      {w['column']:<28} z = {w['z']:+.2f}")

    hot = [s for s in seams
           if s["max_abs_z"] > seam_z or s["n_frozen_columns_differing"]]
    if hot and not allow_shift:
        s = hot[0]
        die(f"the seam at {KEY} {s['boundary']} does not look like one "
            f"distribution: max|z| = {s['max_abs_z']:.1f} against a threshold of "
            f"{seam_z} ({s['n_z_over_5']} features above 5). The parts are "
            f"disjoint index ranges of one generator, so this says a filtration "
            f"or imaging parameter differed between the builds. Compare the "
            f"ph_params_*.log and config.txt of each part before trusting it; "
            f"pass --allow-seam-shift if the difference is intended.")
    if hot:
        print(f"  WARNING: seam shift accepted by --allow-seam-shift "
              f"(max|z| {hot[0]['max_abs_z']:.1f})")

    tmp = f"{output}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    merged.to_csv(tmp, index=False)
    os.replace(tmp, output)

    side = {
        "output": output, "rows": len(merged), "features": len(feats),
        "expect": expect, "dataset": dataset,
        "parts": [{"path": p, "rows": len(d),
                   "first": int(d[KEY].min()), "last": int(d[KEY].max())}
                  for p, d in ordered],
        "seams": seams, "seam_z_threshold": seam_z,
        "allow_seam_shift": bool(allow_shift),
    }
    with open(f"{output}.parts.json", "w") as fh:
        json.dump(side, fh, indent=2)

    if not quiet:
        print(f"\n  {len(merged)} rows x {len(feats)} features -> {output}")
        print(f"  provenance -> {output}.parts.json")
    return merged


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _fake(tmp, n=40, nfeat=6, seed=0):
    """A dataset directory plus two disjoint part tables over [0, n)."""
    rng = np.random.default_rng(seed)
    ds = os.path.join(tmp, "DATA")
    os.makedirs(ds, exist_ok=True)
    base = pd.DataFrame({
        KEY: np.arange(n),
        "filename": [f"sample_{i:06d}.raw" for i in range(n)],
        "stratum": ["iso", "mod", "strong"] * (n // 3) + ["iso"] * (n % 3),
        "porosity": rng.uniform(0.65, 0.9, n),
    })
    base.to_csv(os.path.join(ds, "structures.csv"), index=False)

    feats = rng.normal(size=(n, nfeat))
    cols = [f"thr_a1_0_0_{i}" for i in range(nfeat)]
    full = pd.concat([base, pd.DataFrame(feats, columns=cols)], axis=1)
    cut = 25
    head, tail = full.iloc[:cut].copy(), full.iloc[cut:].copy()
    ph = os.path.join(tmp, "head.csv"); head.to_csv(ph, index=False)
    pt = os.path.join(tmp, "tail.csv"); tail.to_csv(pt, index=False)
    return ds, ph, pt, cols, full


def _expect_die(fn) -> bool:
    try:
        fn()
    except SystemExit as e:
        return str(e).startswith("ABORT")
    return False


def self_test() -> int:
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL  {name}")

    with tempfile.TemporaryDirectory() as tmp:
        ds, ph, pt, cols, full = _fake(tmp)
        out = os.path.join(tmp, "merged.csv")

        m = merge([ph, pt], out, ds, 40, 8.0, False, quiet=True)
        check("happy path merges", len(m) == 40)
        check("sorted by sample_id", list(m[KEY]) == list(range(40)))
        check("values preserved",
              np.allclose(m[cols].to_numpy(), full[cols].to_numpy()))
        check("parts.json written", os.path.isfile(out + ".parts.json"))
        side = json.load(open(out + ".parts.json"))
        check("parts.json records both ranges",
              [p["first"] for p in side["parts"]] == [0, 25])
        check("order of --parts does not matter",
              len(merge([pt, ph], out, ds, 40, 8.0, False, quiet=True)) == 40)

        # 1. column mismatch
        bad = pd.read_csv(pt).rename(columns={cols[0]: "thr_a9_9_9_0"})
        p = os.path.join(tmp, "badcol.csv"); bad.to_csv(p, index=False)
        check("column set mismatch aborts",
              _expect_die(lambda: merge([ph, p], out, ds, 40, 8.0, False, quiet=True)))

        # 2. column ORDER mismatch, same set
        sw = pd.read_csv(pt)
        sw = sw[[KEY] + META + [cols[1], cols[0]] + cols[2:]]
        p = os.path.join(tmp, "badorder.csv"); sw.to_csv(p, index=False)
        check("column order mismatch aborts",
              _expect_die(lambda: merge([ph, p], out, ds, 40, 8.0, False, quiet=True)))

        # 3. overlapping ids
        ov = pd.read_csv(pt); ov[KEY] = ov[KEY] - 5
        p = os.path.join(tmp, "overlap.csv"); ov.to_csv(p, index=False)
        check("overlapping parts abort",
              _expect_die(lambda: merge([ph, p], out, ds, 40, 8.0, False, quiet=True)))

        # 4. a gap in coverage
        gap = pd.read_csv(pt).iloc[1:]
        p = os.path.join(tmp, "gap.csv"); gap.to_csv(p, index=False)
        check("missing structure aborts",
              _expect_die(lambda: merge([ph, p], out, ds, 40, 8.0, False, quiet=True)))
        check("missing structure aborts without --expect too",
              _expect_die(lambda: merge([ph, p], out, ds, 0, 8.0, False, quiet=True)))

        # 5. duplicated id inside one part
        dup = pd.concat([pd.read_csv(pt), pd.read_csv(pt).iloc[:1]])
        p = os.path.join(tmp, "dup.csv"); dup.to_csv(p, index=False)
        check("duplicate sample_id aborts",
              _expect_die(lambda: merge([ph, p], out, ds, 40, 8.0, False, quiet=True)))

        # 6. NaN
        nan = pd.read_csv(pt); nan.loc[nan.index[0], cols[0]] = np.nan
        p = os.path.join(tmp, "nan.csv"); nan.to_csv(p, index=False)
        check("NaN aborts",
              _expect_die(lambda: merge([ph, p], out, ds, 40, 8.0, False, quiet=True)))

        # 7. metadata mix-up: right ids, wrong filenames
        mx = pd.read_csv(pt); mx["filename"] = "sample_999999.raw"
        p = os.path.join(tmp, "mixup.csv"); mx.to_csv(p, index=False)
        check("metadata mismatch aborts",
              _expect_die(lambda: merge([ph, p], out, ds, 40, 8.0, False, quiet=True)))

        # 8. THE SEAM: tail computed with a different parameter
        sh = pd.read_csv(pt)
        sh[cols] = sh[cols] + 3.0
        p = os.path.join(tmp, "shift.csv"); sh.to_csv(p, index=False)
        check("shifted tail aborts on the seam statistic",
              _expect_die(lambda: merge([ph, p], out, ds, 40, 8.0, False, quiet=True)))
        check("--allow-seam-shift overrides it",
              len(merge([ph, p], out, ds, 40, 8.0, True, quiet=True)) == 40)
        check("an unshifted seam stays well under the threshold",
              json.load(open(out + ".parts.json"))["seams"][0]["max_abs_z"] < 8.0
              if merge([ph, pt], out, ds, 40, 8.0, False, quiet=True) is not None
              else False)

        # 9. a constant column that differs between parts has no pooled sd
        fa, fb = pd.read_csv(ph), pd.read_csv(pt)
        fa[cols[0]] = 1.0
        fb[cols[0]] = 2.0
        pa = os.path.join(tmp, "froz_a.csv"); fa.to_csv(pa, index=False)
        pb = os.path.join(tmp, "froz_b.csv"); fb.to_csv(pb, index=False)
        check("constant-but-different column aborts",
              _expect_die(lambda: merge([pa, pb], out, ds, 40, 8.0, False, quiet=True)))

        # 10. a single part is a legitimate no-op merge (no seam to check)
        one = merge([ph], out, ds, 25, 8.0, False, quiet=True)
        check("one part with --expect matching merges", len(one) == 25)
        check("one part records no seam",
              json.load(open(out + ".parts.json"))["seams"] == [])

    print(f"  self-test: {ok}/{ok + fail} passed")
    return 0 if fail == 0 else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", nargs="+", help="part CSVs over disjoint sample_id ranges")
    p.add_argument("--output", help="merged CSV (written atomically)")
    p.add_argument("--dataset", default=None,
                   help="dataset root; its structures.csv is used to verify that "
                        "filename/stratum/porosity match row for row")
    p.add_argument("--expect", type=int, default=0,
                   help="require the union of sample_ids to be exactly [0, N)")
    p.add_argument("--seam-z", type=float, default=8.0,
                   help="abort if any feature's standardized mean difference "
                        "across a seam exceeds this many sigma (default 8)")
    p.add_argument("--allow-seam-shift", action="store_true",
                   help="downgrade the seam check to a warning")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()

    if a.self_test:
        return self_test()
    if not (a.parts and a.output):
        p.error("--parts and --output are required (or use --self-test)")
    merge(a.parts, a.output, a.dataset, a.expect, a.seam_z, a.allow_seam_shift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
