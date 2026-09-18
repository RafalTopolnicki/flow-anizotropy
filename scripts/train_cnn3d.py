r"""
3D CNN baseline: predict the permeability tensor straight from the 80^3 volume.

The 3D counterpart of `train_cnn.py`, and the end-to-end counterpart of
`train_catboost.py` on the 3D descriptors.  Same k-fold split (KFold, shuffle,
seed 0), same `scripts/metrics.py` bookkeeping and the same results schema, so
its rows concatenate with the CatBoost rows and are compared directly.

    python scripts/train_cnn3d.py --self-test
    python scripts/train_cnn3d.py --dataset DATA/aniso3d --backbone densenet121 \
        --targets k_xy k_xz k_yz --augment roll+oct --output RESULTS/cnn3d/densenet121_offdiag

Why the backbone is implemented here instead of imported
--------------------------------------------------------
`train_cnn.py` gets its backbones from timm, which is 2D only.  The usual 3D
source is MONAI, but that is one more package to install on whichever machine
has the GPU, and there are no pretrained 3D weights for binary pore space
anyway — the 2D runs that used ImageNet initialisation are not reproducible in
3D whatever library provides the graph.  So DenseNet is written out below, from
Huang et al. (2016) with Conv3d throughout: ~90 lines, no new dependency, and it
runs under `--self-test` on a CPU.

Augmentation — the part worth reading
-------------------------------------
The cells are **triply periodic**, so a random circular shift is an *exact*
symmetry: the structure is unchanged as a torus and every component of K is
untouched.  That is the default (`--augment roll`) and it is free.

`--augment roll+oct` adds the full **octahedral group**: the 6 axis permutations
times the 8 sign patterns, 48 elements, every one of which relabels the grid
exactly — a transpose and some flips, not one voxel interpolated.  They are
*not* label-preserving.  Under the signed permutation P,

    K -> P K P^T ,   i.e.   K'_ab = s_a s_b K_{sigma(a) sigma(b)}

which is the same identity `rotation_covariance3d.py` verified the solver obeys
at machine precision (NOTES 8.6a), and this module reuses that module's axis
convention rather than restating it.  Holding the labels fixed while permuting
the volume would teach the network that the off-diagonals are
orientation-independent — destroying precisely the signal the study is about.

Because s_a s_b is +1 whenever a == b, the diagonal components and the
off-diagonal components transform **independently**: diagonals permute among
themselves, off-diagonals permute among themselves and pick up a sign.  So the
headline target set `k_xy k_xz k_yz` can use the whole 48-element group on its
own — a 48x augmentation that teaches the sign structure directly.  Target sets
that are an *incomplete* family are refused rather than guessed at: a lone
`k_xy` has nowhere to come from, since a permutation can send it to +/-k_xz.

Invariant under the whole group, and so passed through untouched: `log_k_mean`,
`log_k_ratio`, `k_ratio`, `k1`, `k2`, `k3`, `k_mean`, `fa`, `trace`, `porosity`.

Memory
------
Volumes are held as uint8 and cast per batch: 5000 x 80^3 is 2.6 GB as uint8 and
would be 10 GB as float32.  Activations, not weights, set the batch size — the
default of 8 is a fifth of the 2D default for that reason.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structure_io3d import load_structure                        # noqa: E402
from train_catboost import TARGET_GROUPS, resolve_targets        # noqa: E402
from metrics import (fold_metrics, summarize, fmt, write_metrics_json,  # noqa: E402
                     koff_ceilings3d)

AXES = "xyz"

# ---------------------------------------------------------------------------
# The octahedral group, on the grid and on the labels
# ---------------------------------------------------------------------------
# sigma is in solver-axis order and reads exactly as in rotation_covariance3d:
# sigma[a] = the OLD solver axis that becomes NEW axis a.  signs[a] = -1 flips
# new axis a.  Together they are the signed permutation matrix
#     P[a, i] = signs[a] * delta(i, sigma[a])
# and the 6 x 8 = 48 combinations are every orthogonal map that takes the voxel
# grid to itself.
SIGMAS = list(itertools.permutations(range(3)))
SIGNS = list(itertools.product((1, -1), repeat=3))
GROUP = [(s, f) for s in SIGMAS for f in SIGNS]


def group_matrix(sigma, signs):
    P = np.zeros((3, 3))
    for a in range(3):
        P[a, sigma[a]] = signs[a]
    return P


def transform_grid(vol, sigma, signs):
    """
    Apply a group element to a volume batch, exactly.

    `vol` is (..., nz, ny, nx) as `structure_io3d` returns it, so the last three
    axes are transposed into solver order [x, y, z] first and back afterwards —
    the same two no-op transposes `rotation_covariance3d.permute_structure`
    uses, kept so the permutation is written the way it is reasoned about.
    """
    lead = vol.ndim - 3
    pre = tuple(range(lead))
    a = vol.transpose(*pre, lead + 2, lead + 1, lead + 0)      # -> [x, y, z]
    a = a.transpose(*pre, *(lead + s for s in sigma))          # new axis <- old
    for ax, s in enumerate(signs):
        if s < 0:
            a = np.flip(a, axis=lead + ax)
    return np.ascontiguousarray(
        a.transpose(*pre, lead + 2, lead + 1, lead + 0))       # -> [z, y, x]


# --- which targets this knows how to transform -----------------------------
DIAG = ("K_xx", "K_yy", "K_zz")
OFFDIAG = ("k_xy", "k_xz", "k_yz")
DEV_DIAG = ("dev_xx", "dev_yy")          # dev_zz is implied: -(dev_xx + dev_yy)
DEV_OFF = ("dev_xy", "dev_xz", "dev_yz")
OCT_INVARIANT = {"log_k_mean", "log_k_ratio", "k_ratio", "k1", "k2", "k3",
                 "k_mean", "fa", "trace", "porosity"}
FAMILIES = (DIAG, OFFDIAG, DEV_DIAG, DEV_OFF)
# (a, b) index pair in the symmetric tensor, for each off-diagonal name
OFF_PAIRS = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def check_oct_supported(targets):
    """Refuse a target set whose transformation rule is not determined."""
    known = set(OCT_INVARIANT) | {t for fam in FAMILIES for t in fam}
    unknown = [t for t in targets if t not in known]
    if unknown:
        raise SystemExit(
            "--augment roll+oct does not know how these targets transform: "
            f"{unknown}. Add a rule or use --augment roll.")
    for fam in FAMILIES:
        present = [t for t in fam if t in targets]
        if present and len(present) != len(fam):
            raise SystemExit(
                f"--augment roll+oct needs the whole family {list(fam)} to "
                f"transform any of it; got {present}. A permutation can send "
                f"{present[0]} to another member of its family, so a partial "
                "set has nowhere to come from. Use --augment roll.")


def transform_labels(y, targets, sigma, signs):
    """
    The label side of one group element: K -> P K P^T, per family.

    Diagonal and off-diagonal families transform independently, because the
    sign factor s_a s_b is +1 whenever a == b.
    """
    y = y.copy()
    idx = {t: i for i, t in enumerate(targets)}
    P = group_matrix(sigma, signs)

    def apply(diag_names, off_names, diag_implicit=False):
        """Rebuild the symmetric 3x3, map it, write the components back."""
        if not any(n in idx for n in tuple(diag_names) + tuple(off_names)):
            return
        M = np.zeros((len(y), 3, 3))
        if diag_names:
            for a, n in enumerate(diag_names):
                M[:, a, a] = y[:, idx[n]]
            if diag_implicit:           # the third diagonal is minus the others
                M[:, 2, 2] = -(M[:, 0, 0] + M[:, 1, 1])
        for n in off_names:
            a, b = OFF_PAIRS[n.rsplit("_", 1)[1]]
            M[:, a, b] = M[:, b, a] = y[:, idx[n]]
        M = np.einsum("ai,nij,bj->nab", P, M, P)
        if diag_names:
            for a, n in enumerate(diag_names):
                y[:, idx[n]] = M[:, a, a]
        for n in off_names:
            a, b = OFF_PAIRS[n.rsplit("_", 1)[1]]
            y[:, idx[n]] = M[:, a, b]

    # Two independent tensors: the raw K components, and the deviator. Each is
    # mapped as a whole even when only one of its two families is a target,
    # which is why the diagonal names are passed only when present.
    apply([n for n in DIAG if n in idx], [n for n in OFFDIAG if n in idx])
    apply([n for n in DEV_DIAG if n in idx], [n for n in DEV_OFF if n in idx],
          diag_implicit=True)
    return y


# ---------------------------------------------------------------------------
# DenseNet, 3D
# ---------------------------------------------------------------------------
# Huang et al. (2016), Conv3d throughout. The configuration table is the only
# difference between the depths.
DENSENET_CFG = {
    "densenet121": (64, 32, (6, 12, 24, 16)),
    "densenet169": (64, 32, (6, 12, 32, 32)),
    "densenet201": (64, 32, (6, 12, 48, 32)),
    # Not a published depth: a quarter-size net for smoke runs and for checking
    # a pipeline change without waiting for the real thing. Named so it can
    # never be mistaken for one in a results table.
    "densenet_small": (32, 16, (3, 6, 12, 8)),
}


def build_model(name, n_out):
    import torch
    import torch.nn as nn

    if name not in DENSENET_CFG:
        raise SystemExit(f"unknown backbone {name}; have {sorted(DENSENET_CFG)}")
    n_init, growth, blocks = DENSENET_CFG[name]
    bn_size = 4

    class DenseLayer(nn.Module):
        def __init__(self, nin):
            super().__init__()
            mid = bn_size * growth
            self.norm1, self.conv1 = nn.BatchNorm3d(nin), nn.Conv3d(nin, mid, 1, bias=False)
            self.norm2 = nn.BatchNorm3d(mid)
            self.conv2 = nn.Conv3d(mid, growth, 3, padding=1, bias=False)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            h = self.conv1(self.relu(self.norm1(x)))
            h = self.conv2(self.relu(self.norm2(h)))
            return torch.cat([x, h], 1)

    class Transition(nn.Sequential):
        def __init__(self, nin, nout):
            super().__init__(nn.BatchNorm3d(nin), nn.ReLU(inplace=True),
                             nn.Conv3d(nin, nout, 1, bias=False),
                             nn.AvgPool3d(2, stride=2))

    layers = [nn.Conv3d(1, n_init, 7, stride=2, padding=3, bias=False),
              nn.BatchNorm3d(n_init), nn.ReLU(inplace=True),
              nn.MaxPool3d(3, stride=2, padding=1)]
    nf = n_init
    for i, n_layers in enumerate(blocks):
        for _ in range(n_layers):
            layers.append(DenseLayer(nf))
            nf += growth
        if i != len(blocks) - 1:
            layers.append(Transition(nf, nf // 2))
            nf //= 2
    layers += [nn.BatchNorm3d(nf), nn.ReLU(inplace=True),
               nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(nf, n_out)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------

def load_volumes(dataset, df):
    """Every structure as one uint8 array (N, nz, ny, nx)."""
    first = load_structure(os.path.join(dataset, "structures", df.filename.iloc[0]))
    X = np.zeros((len(df),) + first.shape, dtype=np.uint8)
    X[0] = first
    for n, r in enumerate(df.itertuples()):
        if n:
            X[n] = load_structure(os.path.join(dataset, "structures", r.filename))
    return X


def run_fold(X, Y, tr, te, args, targets):
    import torch
    import torch.nn as nn

    dev = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    shape = X.shape[1:]

    mu, sd = Y[tr].mean(0), Y[tr].std(0)
    sd[sd == 0] = 1.0
    Yn = (Y - mu) / sd

    model = build_model(args.backbone, len(targets)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = max(1, len(tr) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == "cuda"))
    lossf = nn.MSELoss()

    model.train()
    step = 0
    for _ in range(args.epochs):
        order = rng.permutation(tr)
        for k in range(0, len(order) - args.batch + 1, args.batch):
            idx = order[k:k + args.batch]
            xb = X[idx].astype(np.float32)
            yb = Yn[idx].copy()

            if args.augment != "none":
                # circular shift: exact on a periodic cell, labels untouched
                for n in range(len(xb)):
                    xb[n] = np.roll(xb[n], tuple(int(rng.integers(s)) for s in shape),
                                    axis=(0, 1, 2))
            if args.augment == "roll+oct":
                sigma, signs = GROUP[int(rng.integers(len(GROUP)))]
                xb = transform_grid(xb, sigma, signs)
                # labels follow the volume, in raw units, then re-standardise
                yb = (transform_labels(yb * sd + mu, targets, sigma, signs) - mu) / sd

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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    """
    The group, the grid map, the label map, and that the two agree.

    The last is the one that matters and the one a 2D port gets wrong: the
    volume is indexed [z, y, x] while the tensor is indexed (x, y, z), so an
    axis convention that is self-consistent but mismatched between the two would
    train the network on silently wrong labels. It is checked without the solver
    by using the **voxel-coordinate covariance**, which is a rank-2 tensor in
    the same frame as K and must therefore transform the same way.
    """
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    # --- the group ---------------------------------------------------------
    mats = [group_matrix(s, f) for s, f in GROUP]
    check("48 distinct elements", len({m.tobytes() for m in mats}) == 48)
    check("all orthogonal", all(np.allclose(m @ m.T, np.eye(3)) for m in mats))
    check("closed under composition",
          all(any(np.array_equal(mats[i] @ mats[j], m) for m in mats)
              for i in range(0, 48, 7) for j in range(0, 48, 11)))

    # --- the grid map, on a non-cubic array so equal-axis bugs cannot hide --
    rng = np.random.default_rng(0)
    vol = (rng.random((5, 7, 11)) < 0.4).astype(np.uint8)
    shapes, round_trips, porosities = set(), True, set()
    for sigma, signs in GROUP:
        p = transform_grid(vol, sigma, signs)
        shapes.add(p.shape)
        porosities.add(round(float(p.mean()), 12))
        inv_sigma = tuple(np.argsort(sigma))
        # undo: unflip in the new frame, then permute back
        back = transform_grid(transform_grid(p, tuple(range(3)), signs),
                              inv_sigma, (1, 1, 1))
        round_trips &= np.array_equal(back, vol)
    check("round-trips to the original", round_trips)
    check("solid fraction untouched", len(porosities) == 1,
          f"{porosities.pop():.6f}")
    check("6 distinct shapes from 6 permutations", len(shapes) == 6, str(sorted(shapes)))

    # --- grid map and label map agree, via the coordinate covariance --------
    def covariance(v):
        """Covariance of solid-voxel coordinates, in solver order (x, y, z)."""
        iz, iy, ix = np.nonzero(v)
        c = np.stack([ix, iy, iz]).astype(float)
        return np.cov(c - c.mean(1, keepdims=True))

    C = covariance(vol)
    worst = 0.0
    for sigma, signs in GROUP:
        got = covariance(transform_grid(vol, sigma, signs))
        P = group_matrix(sigma, signs)
        worst = max(worst, float(np.abs(got - P @ C @ P.T).max()))
    check("grid map matches P C P^T on the coordinate covariance",
          worst < 1e-9, f"max |diff| {worst:.2e} over all 48")

    # --- the label map is P K P^T, for each supported target set -----------
    K = np.array([[3.0, 0.7, -0.4], [0.7, 2.0, 0.25], [-0.4, 0.25, 1.2]])
    for names, build, read in (
        (list(DIAG) + list(OFFDIAG),
         lambda M: [M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]],
         None),
        (list(OFFDIAG), lambda M: [M[0, 1], M[0, 2], M[1, 2]], None),
    ):
        y0 = np.array([build(K)])
        worst = 0.0
        for sigma, signs in GROUP:
            P = group_matrix(sigma, signs)
            want = np.array(build(P @ K @ P.T))
            got = transform_labels(y0, names, sigma, signs)[0]
            worst = max(worst, float(np.abs(got - want).max()))
        check(f"labels match P K P^T for {names[0]}...({len(names)} targets)",
              worst < 1e-12, f"max |diff| {worst:.2e} over all 48")

    # the deviator, whose third diagonal component is implied not stored
    D = K - np.eye(3) * np.trace(K) / 3
    names = list(DEV_DIAG) + list(DEV_OFF)
    y0 = np.array([[D[0, 0], D[1, 1], D[0, 1], D[0, 2], D[1, 2]]])
    worst = 0.0
    for sigma, signs in GROUP:
        P = group_matrix(sigma, signs)
        Dp = P @ D @ P.T
        want = np.array([Dp[0, 0], Dp[1, 1], Dp[0, 1], Dp[0, 2], Dp[1, 2]])
        worst = max(worst, float(np.abs(transform_labels(y0, names, sigma, signs)[0]
                                        - want).max()))
    check("deviator transforms with dev_zz implied", worst < 1e-12,
          f"max |diff| {worst:.2e} over all 48")

    # identity, and that invariants are never touched
    names = list(OFFDIAG) + ["log_k_mean"]
    y0 = np.array([[0.7, -0.4, 0.25, 1.234]])
    got = transform_labels(y0, names, (0, 1, 2), (1, 1, 1))
    check("identity element is a no-op", np.array_equal(got, y0))
    changed = {tuple(transform_labels(y0, names, s, f)[0])[3] for s, f in GROUP}
    check("invariant target never moves", changed == {1.234})

    # --- what it refuses ---------------------------------------------------
    check("complete family accepted", check_oct_supported(list(OFFDIAG)) is None)
    for bad, why in ((["k_xy"], "partial family"),
                     (["mystery"], "unknown target"),
                     (["K_xx", "K_yy"], "partial diagonal")):
        try:
            check_oct_supported(bad)
            check(f"{why} rejected", False)
        except SystemExit:
            check(f"{why} rejected", True)

    # --- the model builds and produces one number per target ----------------
    try:
        import torch
        m = build_model("densenet_small", 3)
        with torch.no_grad():
            out = m(torch.zeros(2, 1, 32, 32, 32))
        check("densenet_small forward pass", tuple(out.shape) == (2, 3),
              f"shape {tuple(out.shape)}, "
              f"{sum(p.numel() for p in m.parameters()) / 1e6:.2f} M params")
    except ImportError:
        print("[skip] forward pass: torch not installed")

    print("\nself-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset")
    p.add_argument("--labels", default=None)
    p.add_argument("--output")
    p.add_argument("--backbone", default="densenet121",
                   choices=sorted(DENSENET_CFG))
    p.add_argument("--targets", nargs="+", default=["k_xy", "k_xz", "k_yz"])
    p.add_argument("--augment", choices=["none", "roll", "roll+oct"], default="roll")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=-1,
                   help="run a single fold (quick check)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
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
        raise SystemExit(
            f"no labels at {labels} — run "
            f"`python scripts/solve_permeability3d.py --dataset {a.dataset} --merge-only`")
    df = pd.read_csv(labels)
    n0 = len(df)
    conv = [c for c in ("conv_fx", "conv_fy", "conv_fz") if c in df.columns]
    if conv:
        df = df[(df[conv] == 1).all(axis=1)]
    if a.strata:
        df = df[df.stratum.isin(a.strata)]
    if a.limit:
        df = df.head(a.limit)
    df = df.reset_index(drop=True)

    targets = resolve_targets(a.targets, df.columns)
    if a.augment == "roll+oct":
        check_oct_supported(targets)

    Y = df[targets].to_numpy(np.float64)
    keep = np.isfinite(Y).all(1)
    df, Y = df[keep].reset_index(drop=True), Y[keep]
    print(f"{n0} rows -> {len(df)} after filtering; {len(targets)} targets, "
          f"backbone {a.backbone}, augment {a.augment}")

    # The labels are still arriving: say which snapshot this run used, because a
    # results directory is only comparable with another from the same one.
    man = os.path.splitext(labels)[0] + "_manifest.json"
    snapshot = json.load(open(man)) if os.path.exists(man) else None
    if snapshot:
        print(f"  label snapshot: {snapshot['n_rows']}/{snapshot['n_structures']} "
              f"merged {snapshot['merged']}, ids {snapshot['sample_id_min']}.."
              f"{snapshot['sample_id_max']}")

    ceilings = koff_ceilings3d(df)
    for t, c in (ceilings or {}).items():
        if t in targets:
            print(f"  {t} label-noise ceiling: R^2 <= {c:.6f}")

    t0 = time.time()
    X = load_volumes(a.dataset, df)
    print(f"volumes loaded: {X.shape[1:]} x {len(X)}, "
          f"{X.nbytes / 1024**3:.2f} GB in {time.time() - t0:.0f} s")

    cv = KFold(a.folds, shuffle=True, random_state=a.seed)
    oof = np.full_like(Y, np.nan)
    test_idx = []
    for f, (tr, te) in enumerate(cv.split(X)):
        if a.only_fold >= 0 and f != a.only_fold:
            continue
        t0 = time.time()
        oof[te] = run_fold(X, Y, tr, te, a, targets)
        test_idx.append(te)
        print(f"  fold {f}: {time.time() - t0:.0f} s")

    # Scored per fold and averaged, through scripts/metrics.py, exactly as
    # train_catboost.py and train_cnn.py do — see the note in train_cnn.py on
    # why pooling the out-of-fold vector instead would be a different statistic.
    n_vox = int(np.prod(X.shape[1:]))
    rows, per_fold = [], []
    for i, t_name in enumerate(targets):
        folds = [fold_metrics(Y[te, i], oof[te, i], fold=f_i)
                 for f_i, te in enumerate(test_idx)]
        summ = summarize(folds)
        n_done = int(np.isfinite(oof[:, i]).sum())
        common = {"group": f"cnn3d_{a.backbone}", "n_features": n_vox,
                  "target": t_name,
                  "mode": "joint" if len(targets) > 1 else "single", "n": n_done}
        rows.append({**common, **summ})
        per_fold.append({**common, "folds": folds, **summ})
        print(f"  {t_name:14s} {fmt(summ)}")

    os.makedirs(a.output, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.output, "results.csv"), index=False)
    np.save(os.path.join(a.output, "oof_predictions.npy"), oof)
    write_metrics_json(a.output,
                       {"kind": "cnn3d", "backbone": a.backbone, "folds": a.folds,
                        "seed": a.seed, "epochs": a.epochs, "batch": a.batch,
                        "augment": a.augment, "n_rows": len(df),
                        "koff_ceilings": ceilings, "label_snapshot": snapshot},
                       per_fold)
    with open(os.path.join(a.output, "arguments.json"), "w") as fh:
        json.dump({**vars(a), "targets": targets, "n_rows": len(df),
                   "koff_ceilings": ceilings, "label_snapshot": snapshot}, fh, indent=2)
    print(f"\n-> {a.output}/results.csv")


if __name__ == "__main__":
    main()
