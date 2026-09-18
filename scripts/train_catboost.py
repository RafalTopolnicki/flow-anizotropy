r"""
CatBoost on descriptor groups, predicting the permeability tensor.

Trains one model per (feature group, target) and reports R^2/MAE under k-fold
CV, so the comparison of interest — direction-aware TDA against the
non-topological directional baselines — is a single table.

Feature groups (matched by column prefix)
-----------------------------------------
    ecp, ph            direction-aware TDA (the wedge filtration)
    tda                ecp + ph
    thr                throat-scale PH (the distance-transform filtration)
    por, tpc, fab      directional baselines
    baselines          por + tpc + fab
    porosity           the scalar alone (the null model)
    all                everything

Targets
-------
    diagonal    k_xx, k_yy
    offdiag     k_off                      <- the headline; the 2D analogue of
                                              the elastic normal-shear terms
    tensor      k_xx, k_yy, k_off
    invariant   log_k_mean, log_k_ratio, cos2theta, sin2theta
    all         tensor + invariant

`theta` is never a target: it is defined mod 180 degrees, so it is predicted as
(cos 2t, sin 2t) via the `invariant` group.

Two things this reports that a plain R^2 table would hide
---------------------------------------------------------
* **Rows are filtered to `conv_fx == conv_fy == 1`.** Non-converged solves carry
  a meaningless tensor; they are flagged, not dropped, by the solver.
* **A label-noise ceiling for k_off.** The solver's reciprocity residual gives a
  per-sample relative error on k_off, so R^2 against a target that noisy is
  bounded above by roughly 1 - var(noise)/var(signal). Without it, a mediocre
  R^2 on k_off looks like a modelling failure when it may be the label.

Usage
-----
    python scripts/train_catboost.py --dataset DATA/aniso \
        --descriptors DESC/aniso/descriptors.csv DESC/aniso/baselines.csv \
        --groups tda baselines fab porosity --targets tensor invariant \
        --output RESULTS/tda_vs_baselines
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import fold_metrics, summarize, fmt, write_metrics_json  # noqa: E402

PREFIXES = {
    "ecp": ("ecp_",),
    "ph": ("ph_",),
    "tda": ("ecp_", "ph_"),
    "por": ("por_",),
    "tpc": ("tpc_",),
    "fab": ("fab_",),
    # 3D only: the directional fabric profile (baselines3d.py). It is a separate
    # prefix on purpose -- "fabdir_" does not match "fab_", so the 10-feature
    # global fabric tensor stays comparable with the 2D `fab` group instead of
    # being diluted by 720 profile features. It must be listed explicitly in
    # `baselines` and `all` or it is silently never used. Matches nothing in the
    # 2D CSVs, so the 2D runs are unaffected.
    "fabdir": ("fabdir_",),
    # Non-additive pore-connectivity features (connect2d.py): spanning-cluster
    # fractions, dead pore volume, and geodesic tortuosity in four directions.
    # Kept OUT of `baselines` and `all` on purpose -- those two names index
    # runs already recorded in RESULTS/summary.csv, and silently widening them
    # would make old and new numbers incomparable under the same label. The
    # combinations get their own names instead.
    "conn": ("conn_",),
    # Throat-scale PH from the distance transform (filtration_throat2d.py).
    # A separate prefix for the same reason as `conn_`: `ph`, `tda` and `all`
    # name runs already in RESULTS/summary.csv and are built on the WEDGE
    # filtration, so the throat descriptors get their own group names rather
    # than silently redefining those.
    "thr": ("thr_",),
    "baselines": ("por_", "tpc_", "fab_", "fabdir_"),
    "all": ("ecp_", "ph_", "por_", "tpc_", "fab_", "fabdir_"),
    "baselines+conn": ("por_", "tpc_", "fab_", "fabdir_", "conn_"),
    "tda+conn": ("ecp_", "ph_", "conn_"),
    "all+conn": ("ecp_", "ph_", "por_", "tpc_", "fab_", "fabdir_", "conn_"),
    "tpc+thr": ("tpc_", "thr_"),
    "baselines+thr": ("por_", "tpc_", "fab_", "fabdir_", "thr_"),
    "tda+thr": ("ecp_", "ph_", "thr_"),
    "all+thr": ("ecp_", "ph_", "por_", "tpc_", "fab_", "fabdir_", "thr_"),
}

TARGET_GROUPS = {
    "diagonal": ["k_xx", "k_yy"],
    "offdiag": ["k_off"],
    "tensor": ["k_xx", "k_yy", "k_off"],
    "invariant": ["log_k_mean", "log_k_ratio", "cos2theta", "sin2theta"],
}
TARGET_GROUPS["all"] = TARGET_GROUPS["tensor"] + TARGET_GROUPS["invariant"]

# Aliases in tensor notation. K12 is the *symmetrised* off-diagonal k_off
# rather than k_xy or k_yx: K = K^T holds exactly for Stokes flow, so averaging
# the two measurements halves the label noise on the component that needs it
# most (see NOTES.md 3.3).
TARGET_GROUPS["k11"] = ["k_xx"]
TARGET_GROUPS["k22"] = ["k_yy"]
TARGET_GROUPS["k12"] = ["k_off"]
TARGET_GROUPS["full_tensor"] = ["k_xx", "k_yy", "k_off"]


def select(df, group):
    if group == "porosity":
        return ["porosity"]
    pref = PREFIXES[group]
    return [c for c in df.columns if c.startswith(pref)]


def resolve_targets(names, cols):
    out = []
    for n in names:
        out.extend(TARGET_GROUPS.get(n, [n]))
    missing = [t for t in out if t not in cols]
    if missing:
        raise SystemExit(f"targets not in the table: {missing}")
    return list(dict.fromkeys(out))


def koff_noise_ceiling(df):
    """
    Upper bound on achievable R^2 for k_off, from the K = K^T residual.

    |k_xy - k_yx| is a direct estimate of the per-sample error on k_off, so its
    variance sets the noise floor. Halved because k_off is the mean of the two,
    which averages the error down by sqrt(2) in variance.
    """
    if not {"k_xy", "k_yx", "k_off"} <= set(df.columns):
        return None
    err = (df["k_xy"] - df["k_yx"]).abs() / 2.0
    var_noise = float((err ** 2).mean())
    var_sig = float(df["k_off"].var())
    return max(0.0, 1.0 - var_noise / var_sig) if var_sig > 0 else None


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--descriptors", nargs="+", required=True)
    p.add_argument("--labels", default=None,
                   help="default: <dataset>/permeability.csv")
    p.add_argument("--groups", nargs="+",
                   default=["tda", "baselines", "fab", "porosity"])
    p.add_argument("--targets", nargs="+", default=["tensor", "invariant"])
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--multioutput", action="store_true",
                   help="one joint model over all targets (CatBoost MultiRMSE) "
                        "instead of one model per target")
    p.add_argument("--strata", nargs="+", default=None,
                   help="restrict to these strata, e.g. mod strong")
    a = p.parse_args()

    import catboost as cb

    labels = a.labels or os.path.join(a.dataset, "permeability.csv")
    if not os.path.exists(labels):
        raise SystemExit(f"no labels at {labels} — has the LBM run finished?")

    df = pd.read_csv(labels)
    for path in a.descriptors:
        extra = pd.read_csv(path)
        drop = [c for c in extra.columns if c != "sample_id" and c in df.columns]
        df = df.merge(extra.drop(columns=drop), on="sample_id", how="inner")

    n0 = len(df)
    conv = [c for c in ("conv_fx", "conv_fy", "conv_fz") if c in df.columns]
    if conv:
        df = df[(df[conv] == 1).all(axis=1)]
    if a.strata:
        df = df[df.stratum.isin(a.strata)]
    if a.limit:
        df = df.head(a.limit)
    df = df.reset_index(drop=True)
    print(f"{n0} rows -> {len(df)} after convergence/stratum filtering")

    targets = resolve_targets(a.targets, df.columns)
    ceiling = koff_noise_ceiling(df)
    if ceiling is not None:
        print(f"k_off label-noise ceiling (from K=K^T): R^2 <= {ceiling:.4f}")

    os.makedirs(a.output, exist_ok=True)
    cv = KFold(a.folds, shuffle=True, random_state=a.seed)
    rows = []
    per_fold = []

    for group in a.groups:
        cols = select(df, group)
        if not cols:
            print(f"  ! group {group}: no columns, skipping")
            continue
        X = df[cols].to_numpy(np.float32)

        if a.multioutput:
            Y = df[targets].to_numpy(np.float64)
            ok = np.isfinite(Y).all(1)
            per = {t: [] for t in targets}
            t0 = time.time()
            for f_i, (tr, te) in enumerate(cv.split(X[ok])):
                m = cb.CatBoostRegressor(iterations=a.iterations, depth=a.depth,
                                         learning_rate=a.lr, random_seed=a.seed,
                                         loss_function="MultiRMSE", verbose=0,
                                         allow_writing_files=False)
                m.fit(X[ok][tr], Y[ok][tr])
                pred = np.atleast_2d(m.predict(X[ok][te]))
                for i, t in enumerate(targets):
                    per[t].append(fold_metrics(Y[ok][te][:, i], pred[:, i], fold=f_i))
            secs = round(time.time() - t0, 1)
            for t in targets:
                summ = summarize(per[t])
                rows.append({"group": group, "n_features": len(cols), "target": t,
                             "mode": "joint", **summ,
                             "n": int(ok.sum()), "seconds": secs})
                per_fold.append({"group": group, "target": t, "mode": "joint",
                                 "n_features": len(cols), "n": int(ok.sum()),
                                 "seconds": secs, "folds": per[t], **summ})
                print(f"  {group:10s} {len(cols):5d}f  {t:14s} [joint] {fmt(summ)}")
            continue

        for t in targets:
            y = df[t].to_numpy(np.float64)
            ok = np.isfinite(y)
            folds = []
            t0 = time.time()
            for f_i, (tr, te) in enumerate(cv.split(X[ok])):
                m = cb.CatBoostRegressor(iterations=a.iterations, depth=a.depth,
                                         learning_rate=a.lr, random_seed=a.seed,
                                         verbose=0, allow_writing_files=False)
                m.fit(X[ok][tr], y[ok][tr])
                folds.append(fold_metrics(y[ok][te], m.predict(X[ok][te]), fold=f_i))
            secs = round(time.time() - t0, 1)
            summ = summarize(folds)
            rows.append({"group": group, "n_features": len(cols), "target": t,
                         "mode": "single", **summ,
                         "n": int(ok.sum()), "seconds": secs})
            per_fold.append({"group": group, "target": t, "mode": "single",
                             "n_features": len(cols), "n": int(ok.sum()),
                             "seconds": secs, "folds": folds, **summ})
            print(f"  {group:10s} {len(cols):5d}f  {t:14s} {fmt(summ)}  ({secs}s)")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.output, "results.csv"), index=False)
    write_metrics_json(a.output,
                       {"kind": "catboost", "folds": a.folds, "seed": a.seed,
                        "iterations": a.iterations, "depth": a.depth, "lr": a.lr,
                        "n_rows": len(df), "koff_ceiling": ceiling},
                       per_fold)
    with open(os.path.join(a.output, "arguments.json"), "w") as fh:
        json.dump({**vars(a), "n_rows": len(df), "koff_ceiling": ceiling}, fh, indent=2)

    print("\n=== R^2 by group and target ===")
    print(res.pivot(index="group", columns="target", values="r2_mean")
             .round(4).to_string())
    print(f"\n-> {a.output}/results.csv")


if __name__ == "__main__":
    main()
