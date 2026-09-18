"""
Data efficiency: how much training data does each descriptor family need?

    python scripts/learning_curve.py --groups tda baselines fab --targets k_off
    python scripts/learning_curve.py --cnn densenet121 --targets k_off
    python scripts/learning_curve.py --no-golden --groups tda fab --targets k_off

The hypothesis is that direction-aware TDA reaches a given accuracy from fewer
structures than the classical baselines.  Everything here exists to make that
comparison fair.

Design
------
**One frozen golden set** (`splits/aniso_golden.csv`), never trained on, scored
identically at every training size and by every feature group.  Holding the
evaluation set fixed is what makes the curves *paired*: the evaluation noise
(~0.016 on R^2 at n=1000) is common to all arms and cancels in the differences
you actually care about.  `--no-golden` falls back to K-fold CV on each subset,
which sets nothing aside but gives every point its own independent evaluation
noise — the wrong place to spend variance here.

**Nested subsets.**  Within a seed, the size-n subset is a prefix of the size-2n
subset, so the curve cannot wiggle from subset luck.  The order is a round-robin
over strata, so every prefix is also stratum-balanced — |k_off|/trace differs
about 7x between strata, and an unbalanced subset would move R^2 for reasons
unrelated to the model.

**Several seeds per size**, because at n=125 a single draw is mostly noise.  The
spread reported per point is across seeds, not folds.

The confound this is really about
---------------------------------
The feature groups differ in dimensionality by 160x — TDA 1124, baselines 779,
tpc 260, fab 7, porosity 1.  At small n a fixed 1500-iteration CatBoost overfits
1124 features and cannot overfit 7, so a naive curve is biased *against* TDA in
exactly the small-n regime the hypothesis is about, and would measure capacity
rather than information.

So early stopping is on by default: an inner validation split is carved from each
training subset (never from the golden set) and the iteration count is chosen per
run.  `--no-early-stop` gives the fixed-budget variant.  **Run both.**  If the
conclusion flips between them it is a statement about capacity, not about data
efficiency, and that is something to find out now rather than in review.

If the result is close, the further control is `--pca K`, which reduces every
group to the same K components: if TDA still wins at matched dimensionality, the
advantage is in the information rather than the feature count.

Output
------
    <output>/runs.csv       one row per (target, group, size, seed)
    <output>/metrics.json   the same with full detail and provenance
    <output>/summary.csv    mean and std across seeds, per (target, group, size)

Resumable: a run already present in runs.csv is skipped, so this is safe to
interrupt.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import fold_metrics, summarize                      # noqa: E402
from make_golden_split import load_split                         # noqa: E402
from train_catboost import PREFIXES, TARGET_GROUPS, select       # noqa: E402

DEFAULT_SIZES = [125, 250, 500, 1000, 2000, 3990]


# ---------------------------------------------------------------------------

def nested_order(df, pool_ids, seed):
    """
    A single ordering of the pool such that every prefix is stratum-balanced.

    Round-robin over the strata, each shuffled independently.  Prefixes are then
    automatically nested (n=125 is the head of n=250) *and* balanced, which a
    proportional-allocation scheme only gets approximately.
    """
    rng = np.random.default_rng(seed)
    pool = df[df.sample_id.isin(pool_ids)]
    queues = [list(rng.permutation(g.sample_id.to_numpy()))
              for _, g in sorted(pool.groupby("stratum"))]
    order = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop())
    return np.array(order)


def fit_catboost(Xtr, ytr, Xte, yte, a, multi=False):
    import catboost as cb

    params = dict(iterations=a.iterations, depth=a.depth, learning_rate=a.lr,
                  random_seed=a.seed, verbose=0, allow_writing_files=False)
    if multi:
        params["loss_function"] = "MultiRMSE"

    if a.early_stop and len(Xtr) >= 40:
        # Inner validation split, carved from the TRAINING subset only.  Never
        # the golden set: choosing the iteration count on the evaluation data
        # would leak, and the leak grows as the subset shrinks — precisely where
        # the curves are being compared.
        rng = np.random.default_rng(a.seed)
        idx = rng.permutation(len(Xtr))
        nval = max(10, int(round(a.val_frac * len(Xtr))))
        vi, ti = idx[:nval], idx[nval:]
        m = cb.CatBoostRegressor(**params, od_type="Iter", od_wait=50)
        m.fit(Xtr[ti], ytr[ti], eval_set=(Xtr[vi], ytr[vi]), use_best_model=True)
        best = int(m.get_best_iteration() or a.iterations)
    else:
        m = cb.CatBoostRegressor(**params)
        m.fit(Xtr, ytr)
        best = a.iterations
    return m.predict(Xte), best


def fit_cnn(df_all, train_ids, test_ids, targets, a):
    """Reuse train_cnn.run_fold, which already takes explicit train/test indices."""
    from argparse import Namespace
    import train_cnn as T

    sub = df_all[df_all.sample_id.isin(np.concatenate([train_ids, test_ids]))]
    sub = sub.reset_index(drop=True)
    pos = {int(s): i for i, s in enumerate(sub.sample_id)}
    tr = np.array([pos[int(s)] for s in train_ids])
    te = np.array([pos[int(s)] for s in test_ids])

    X = T.load_images(a.dataset, sub)
    Y = sub[targets].to_numpy(np.float64)
    args = Namespace(backbone=a.cnn, device=a.device, seed=a.seed, epochs=a.epochs,
                     batch=a.batch, augment=a.augment, lr=a.cnn_lr, wd=a.wd,
                     pretrained=False)
    return T.run_fold(X, Y, tr, te, args, targets), Y[te]


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="DATA/aniso")
    p.add_argument("--labels", default=None)
    p.add_argument("--descriptors", nargs="+",
                   default=["DESC/aniso/descriptors.csv", "DESC/aniso/baselines.csv"])
    p.add_argument("--split", default="splits/aniso_golden.csv")
    p.add_argument("--no-golden", action="store_true",
                   help="K-fold CV on each subset instead of the frozen hold-out")
    p.add_argument("--folds", type=int, default=5, help="only used with --no-golden")
    p.add_argument("--groups", nargs="+",
                   default=["tda", "baselines", "tpc", "fab", "porosity"])
    p.add_argument("--targets", nargs="+", default=["k_off", "k_xx"])
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--output", default="RESULTS/learning_curve")
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    p.add_argument("--pca", type=int, default=0,
                   help="reduce every group to K components first (dimensionality control)")
    p.add_argument("--multioutput", action="store_true")
    # CNN arm
    p.add_argument("--cnn", default=None, help="backbone name; switches to the CNN")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--augment", default="roll")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cnn-lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    a = p.parse_args()
    a.seed = 0

    labels = a.labels or os.path.join(a.dataset, "permeability.csv")
    df = pd.read_csv(labels)
    conv = [c for c in ("conv_fx", "conv_fy", "conv_fz") if c in df.columns]
    if conv:
        df = df[(df[conv] == 1).all(axis=1)]
    df = df.reset_index(drop=True)

    targets = []
    for t in a.targets:
        targets.extend(TARGET_GROUPS.get(t, [t]))
    targets = list(dict.fromkeys(targets))
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise SystemExit(f"targets not in the table: {missing}")

    if a.no_golden:
        golden, pool = np.array([]), df.sample_id.to_numpy()
        print(f"no golden set: {a.folds}-fold CV on each subset, pool {len(pool)}")
    else:
        golden, pool = load_split(a.split)
        print(f"golden {len(golden)} (frozen, verified)   pool {len(pool)}")

    if a.cnn:
        feat = None
        print(f"model: CNN {a.cnn}  augment={a.augment}  epochs={a.epochs}")
    else:
        d = df[["sample_id"]].copy()
        for f in a.descriptors:
            d = d.merge(pd.read_csv(f), on="sample_id", how="left",
                        suffixes=("", "_dup"))
        feat = d
        print(f"model: CatBoost  early_stop={a.early_stop}  "
              f"iterations={a.iterations}  pca={a.pca or 'off'}")

    os.makedirs(a.output, exist_ok=True)
    runs_csv = os.path.join(a.output, "runs.csv")
    done = set()
    rows = []
    if os.path.exists(runs_csv):
        prev = pd.read_csv(runs_csv)
        rows = prev.to_dict("records")
        done = {(r["target"], r["group"], int(r["size"]), int(r["seed"])) for r in rows}
        print(f"resuming: {len(done)} runs already present")

    sizes = [s for s in a.sizes if s <= len(pool)]
    groups = [a.cnn and f"cnn_{a.cnn}" or g for g in (["_"] if a.cnn else a.groups)]
    total = len(targets) * len(groups) * len(sizes) * len(a.seeds)
    print(f"\n{total} runs: {len(targets)} targets x {len(groups)} groups "
          f"x {len(sizes)} sizes x {len(a.seeds)} seeds\n")

    n_run = 0
    for seed in a.seeds:
        order = nested_order(df, pool, seed)
        for size in sizes:
            train_ids = order[:size]
            for gi, group in enumerate(groups):
                if a.cnn:
                    pass
                else:
                    cols = select(feat, a.groups[gi])
                    if not cols:
                        continue
                for target in targets:
                    key = (target, group, size, seed)
                    if key in done:
                        continue
                    t0 = time.time()

                    if a.cnn:
                        pred, ytrue = fit_cnn(df, train_ids, golden, [target], a)
                        pred = np.asarray(pred).reshape(len(golden), -1)[:, 0]
                        ytrue = ytrue[:, 0]
                    else:
                        sub = feat.set_index("sample_id")
                        Xtr = sub.loc[train_ids, cols].to_numpy(np.float32)
                        Xte = sub.loc[golden, cols].to_numpy(np.float32)
                        yi = df.set_index("sample_id")[target]
                        ytr = yi.loc[train_ids].to_numpy(float)
                        ytrue = yi.loc[golden].to_numpy(float)
                        if a.pca:
                            from sklearn.decomposition import PCA
                            k = min(a.pca, Xtr.shape[1], len(Xtr) - 1)
                            pc = PCA(n_components=k, random_state=0).fit(Xtr)
                            Xtr, Xte = pc.transform(Xtr), pc.transform(Xte)
                        a.seed = seed
                        pred, best = fit_catboost(Xtr, ytr, Xte, ytrue, a)

                    m = fold_metrics(ytrue, pred)
                    rec = {"target": target, "group": group, "size": int(size),
                           "seed": int(seed), "n_eval": int(len(ytrue)),
                           "n_features": (256 * 256 if a.cnn else len(cols)),
                           "seconds": round(time.time() - t0, 1),
                           **{k2: v for k2, v in m.items() if k2 != "n_test"}}
                    if not a.cnn:
                        rec["best_iteration"] = best
                    rows.append(rec)
                    n_run += 1
                    print(f"  {target:12s} {group:10s} n={size:5d} seed={seed} "
                          f"R2 {m['r2']:7.4f}  MAE {m['mae']:.4g}  "
                          f"({rec['seconds']}s)")
                    pd.DataFrame(rows).to_csv(runs_csv, index=False)

    res = pd.DataFrame(rows)
    if res.empty:
        print("nothing to do")
        return 0

    agg = (res.groupby(["target", "group", "size"])
              .agg(r2_mean=("r2", "mean"), r2_std=("r2", lambda s: s.std(ddof=1)),
                   mae_mean=("mae", "mean"), mae_std=("mae", lambda s: s.std(ddof=1)),
                   mse_mean=("mse", "mean"), mse_std=("mse", lambda s: s.std(ddof=1)),
                   n_seeds=("seed", "nunique"))
              .reset_index())
    agg.to_csv(os.path.join(a.output, "summary.csv"), index=False)

    with open(os.path.join(a.output, "metrics.json"), "w") as fh:
        json.dump({"split": a.split if not a.no_golden else None,
                   "n_golden": int(len(golden)), "n_pool": int(len(pool)),
                   "sizes": sizes, "seeds": a.seeds, "targets": targets,
                   "groups": groups, "early_stop": bool(a.early_stop),
                   "iterations": a.iterations, "pca": a.pca,
                   "model": f"cnn_{a.cnn}" if a.cnn else "catboost",
                   "runs": rows}, fh, indent=2)

    for t in targets:
        sub = agg[agg.target == t]
        if sub.empty:
            continue
        print(f"\n=== R^2 on the golden set — target {t} ===")
        print(sub.pivot(index="group", columns="size", values="r2_mean")
                 .round(4).to_string())
    print(f"\n-> {runs_csv}\n-> {a.output}/summary.csv\n-> {a.output}/metrics.json")
    print(f"{n_run} new runs this invocation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
