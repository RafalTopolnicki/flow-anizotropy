r"""
CNN baseline: predict the permeability tensor straight from the binary image.

The end-to-end counterpart of `train_catboost.py`. It uses the same targets,
the same 5-fold split (KFold, shuffle, seed 0) and the same results schema, so
its rows can be concatenated with the CatBoost table and compared directly.

Backbones come from `timm`, so any of its models works; the three families used
in the parent projects are the intended ones:

    convnext_tiny  convnext_small      resnet50  resnet101      densenet121  densenet169

Input is the 1-channel binary structure (timm adapts the stem via `in_chans=1`).

Augmentation — the part worth reading
-------------------------------------
The cells are **periodic**, which makes a random circular shift an *exact*
symmetry: the structure is genuinely unchanged as a torus, and every component
of K is untouched. That is the default (`--augment roll`), and it is free
regularisation with no label bookkeeping.

The D4 flips and quarter-turns are also exact for the image, but they are **not**
label-preserving — this is the trap the option exists to handle. Under a
quarter-turn K -> R K R^T gives

    k_xx <-> k_yy ,  k_off -> -k_off ,  cos2t -> -cos2t ,  sin2t -> -sin2t

and under a reflection

    k_xx, k_yy fixed ,  k_off -> -k_off ,  cos2t fixed ,  sin2t -> -sin2t

Naively rotating images while holding the labels fixed would teach the network
that k_off is orientation-independent, i.e. destroy exactly the signal this
study is about. `--augment roll+d4` applies the transforms above to the labels
along with the image; it refuses to run on a target whose transformation rule
is not known rather than guessing.

Scalars invariant under all of D4 — `log_k_mean`, `log_k_ratio`, `k1`, `k2`,
`k_mean`, `anisotropy`, `porosity` — pass through untouched.

Usage
-----
    python scripts/train_cnn.py --self-test
    python scripts/train_cnn.py --dataset DATA/aniso --backbone convnext_tiny \
        --targets tensor invariant --output RESULTS/cnn_convnext
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io import load_structure                      # noqa: E402
from train_catboost import TARGET_GROUPS, resolve_targets, koff_noise_ceiling  # noqa: E402


# --- how each target behaves under the dihedral group ----------------------
# "swap" pairs are exchanged by a quarter-turn.
ROT90_SIGN = {"k_off": -1.0, "cos2theta": -1.0, "sin2theta": -1.0}
FLIP_SIGN = {"k_off": -1.0, "sin2theta": -1.0}
ROT90_SWAP = ("k_xx", "k_yy")
D4_INVARIANT = {"log_k_mean", "log_k_ratio", "k1", "k2", "k_mean",
                "anisotropy", "porosity", "k_xx", "k_yy",
                "k_off", "cos2theta", "sin2theta"}


def d4_transform_labels(y, targets, n_rot, flip):
    """Apply the label side of `n_rot` quarter-turns then an optional flip."""
    y = y.copy()
    idx = {t: i for i, t in enumerate(targets)}

    for _ in range(n_rot % 4):
        if ROT90_SWAP[0] in idx and ROT90_SWAP[1] in idx:
            a, b = idx[ROT90_SWAP[0]], idx[ROT90_SWAP[1]]
            y[:, [a, b]] = y[:, [b, a]]
        for t, s in ROT90_SIGN.items():
            if t in idx:
                y[:, idx[t]] *= s
    if flip:
        for t, s in FLIP_SIGN.items():
            if t in idx:
                y[:, idx[t]] *= s
    return y


def check_d4_supported(targets):
    unknown = [t for t in targets if t not in D4_INVARIANT]
    if unknown:
        raise SystemExit(
            "--augment roll+d4 does not know how these targets transform: "
            f"{unknown}. Add a rule or use --augment roll.")


# ---------------------------------------------------------------------------

def load_images(dataset, df, size=256):
    X = np.zeros((len(df), size, size), dtype=np.uint8)
    for n, r in enumerate(df.itertuples()):
        X[n] = load_structure(os.path.join(dataset, "structures", r.filename))
    return X


def run_fold(X, Y, tr, te, args, targets):
    import torch
    import torch.nn as nn
    import timm

    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    mu, sd = Y[tr].mean(0), Y[tr].std(0)
    sd[sd == 0] = 1.0

    try:
        model = timm.create_model(args.backbone, pretrained=args.pretrained,
                                  in_chans=1, num_classes=len(targets))
    except Exception as e:                       # offline / no weights cached
        print(f"    pretrained weights unavailable ({type(e).__name__}), "
              f"using random init")
        model = timm.create_model(args.backbone, pretrained=False,
                                  in_chans=1, num_classes=len(targets))
    model = model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tr) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    lossf = nn.MSELoss()

    Yn = (Y - mu) / sd
    model.train()
    step = 0
    for ep in range(args.epochs):
        order = rng.permutation(tr)
        for k in range(0, len(order) - args.batch + 1, args.batch):
            idx = order[k:k + args.batch]
            xb = X[idx].astype(np.float32)
            yb = Yn[idx].copy()

            if args.augment != "none":
                # circular shift: exact on a periodic cell, labels untouched
                for n in range(len(xb)):
                    xb[n] = np.roll(xb[n], (int(rng.integers(256)),
                                            int(rng.integers(256))), axis=(0, 1))
            if args.augment == "roll+d4":
                nrot = int(rng.integers(4))
                flp = bool(rng.integers(2))
                if nrot:
                    xb = np.rot90(xb, nrot, axes=(1, 2)).copy()
                if flp:
                    xb = xb[:, :, ::-1].copy()
                # labels must follow the image, in raw units then re-standardise
                yr = d4_transform_labels(yb * sd + mu, targets, nrot, flp)
                yb = (yr - mu) / sd

            xt = torch.from_numpy(xb).unsqueeze(1).to(dev)
            yt = torch.from_numpy(yb.astype(np.float32)).to(dev)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                loss = lossf(model(xt), yt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step < steps - 1:
                sched.step()
            step += 1

    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(te), args.batch):
            xb = torch.from_numpy(X[te[k:k + args.batch]].astype(np.float32)) \
                      .unsqueeze(1).to(dev)
            with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
                preds.append(model(xb).float().cpu().numpy())
    return np.concatenate(preds) * sd + mu


def self_test():
    """The label algebra, checked against K -> R K R^T directly."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    targets = ["k_xx", "k_yy", "k_off", "cos2theta", "sin2theta"]
    rng = np.random.default_rng(0)
    kxx, kyy, koff = 3.0, 1.0, 0.7
    d1, d2 = (kxx - kyy) / 2, koff
    t = 0.5 * np.arctan2(d2, d1)
    y = np.array([[kxx, kyy, koff, np.cos(2 * t), np.sin(2 * t)]])

    def tensor_after(Q):
        K = np.array([[kxx, koff], [koff, kyy]])
        Kp = Q @ K @ Q.T
        a, b, c = Kp[0, 0], Kp[0, 1], Kp[1, 1]
        tt = 0.5 * np.arctan2(b, (a - c) / 2)
        return np.array([a, c, b, np.cos(2 * tt), np.sin(2 * tt)])

    R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    got = d4_transform_labels(y, targets, 1, False)[0]
    check("quarter-turn matches R K Rᵀ", np.allclose(got, tensor_after(R90), atol=1e-9),
          f"{np.round(got, 4)} vs {np.round(tensor_after(R90), 4)}")

    F = np.diag([-1.0, 1.0])
    got = d4_transform_labels(y, targets, 0, True)[0]
    check("reflection matches F K Fᵀ", np.allclose(got, tensor_after(F), atol=1e-9),
          f"{np.round(got, 4)} vs {np.round(tensor_after(F), 4)}")

    got = d4_transform_labels(y, targets, 2, False)[0]
    check("half-turn is the identity on K", np.allclose(got, y[0], atol=1e-12))

    got = d4_transform_labels(d4_transform_labels(y, targets, 1, False), targets, 3, False)
    check("four quarter-turns return to the start", np.allclose(got, y, atol=1e-12))

    check("invariant targets are refused only when unknown",
          check_d4_supported(["k_off", "log_k_mean"]) is None)
    try:
        check_d4_supported(["mystery_target"])
        check("unknown target is rejected", False)
    except SystemExit:
        check("unknown target is rejected", True)

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--labels", default=None)
    p.add_argument("--output")
    p.add_argument("--backbone", default="convnext_tiny")
    p.add_argument("--targets", nargs="+", default=["tensor", "invariant"])
    p.add_argument("--augment", choices=["none", "roll", "roll+d4"], default="roll")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=-1,
                   help="run a single fold (quick check)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--strata", nargs="+", default=None)
    p.add_argument("--self-test", action="store_true", dest="selftest")
    a = p.parse_args()

    if a.selftest:
        raise SystemExit(self_test())
    if not (a.dataset and a.output):
        p.error("--dataset and --output are required (or use --self-test)")

    from sklearn.model_selection import KFold

    labels = a.labels or os.path.join(a.dataset, "permeability.csv")
    if not os.path.exists(labels):
        raise SystemExit(f"no labels at {labels} — has the LBM run finished?")
    df = pd.read_csv(labels)
    if {"conv_fx", "conv_fy"} <= set(df.columns):
        df = df[(df.conv_fx == 1) & (df.conv_fy == 1)]
    if a.strata:
        df = df[df.stratum.isin(a.strata)]
    if a.limit:
        df = df.head(a.limit)
    df = df.reset_index(drop=True)

    targets = resolve_targets(a.targets, df.columns)
    if a.augment == "roll+d4":
        check_d4_supported(targets)

    Y = df[targets].to_numpy(np.float64)
    keep = np.isfinite(Y).all(1)
    df, Y = df[keep].reset_index(drop=True), Y[keep]
    print(f"{len(df)} structures, {len(targets)} targets, backbone {a.backbone}, "
          f"augment {a.augment}")

    ceiling = koff_noise_ceiling(df)
    if ceiling is not None:
        print(f"k_off label-noise ceiling: R^2 <= {ceiling:.4f}")

    t0 = time.time()
    X = load_images(a.dataset, df)
    print(f"images loaded: {X.nbytes / 1024**2:.0f} MB in {time.time() - t0:.0f} s")

    cv = KFold(a.folds, shuffle=True, random_state=a.seed)
    oof = np.full_like(Y, np.nan)
    for f, (tr, te) in enumerate(cv.split(X)):
        if a.only_fold >= 0 and f != a.only_fold:
            continue
        t0 = time.time()
        oof[te] = run_fold(X, Y, tr, te, a, targets)
        print(f"  fold {f}: {time.time() - t0:.0f} s")

    done = np.isfinite(oof).all(1)
    rows = []
    for i, t in enumerate(targets):
        yt, yp = Y[done, i], oof[done, i]
        ss = ((yt - yt.mean()) ** 2).sum()
        rows.append({"group": f"cnn_{a.backbone}", "n_features": 256 * 256,
                     "target": t,
                     "mode": "joint" if len(targets) > 1 else "single",
                     "r2_mean": float(1 - ((yt - yp) ** 2).sum() / ss) if ss > 0 else np.nan,
                     "r2_std": np.nan,
                     "mae_mean": float(np.abs(yt - yp).mean()),
                     "n": int(done.sum())})
        print(f"  {t:14s} R2 {rows[-1]['r2_mean']:7.4f}   MAE {rows[-1]['mae_mean']:.4g}")

    os.makedirs(a.output, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.output, "results.csv"), index=False)
    np.save(os.path.join(a.output, "oof_predictions.npy"), oof)
    with open(os.path.join(a.output, "arguments.json"), "w") as fh:
        json.dump({**vars(a), "targets": targets, "n_rows": len(df),
                   "koff_ceiling": ceiling}, fh, indent=2)
    print(f"\n-> {a.output}/results.csv")


if __name__ == "__main__":
    main()
