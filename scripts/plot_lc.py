#!/usr/bin/env python
"""Learning-curve plots for the 3D sample-size study: R^2 and MAE vs n.

    python scripts/plot_lc.py                     # k_xy, k_xz (fixed arm)
    python scripts/plot_lc.py --targets K_xx k_xy k_xz --arm es

Reads RESULTS/lc3d_<tag>/<arm>/runs.csv (written by learning_curve.py, every
row scored on the frozen 999-structure hold-out) and writes one PNG per
(target, metric) to RESULTS/lc3d_plots/lc3d_<tag>_<arm>_<metric>.png.
A target whose runs.csv does not exist is skipped with a warning.

Lines are the mean over seeds, bands +-1 sd across seeds. The five headline
groups are coloured; por/tpc/fab/fabdir are drawn thin and grey because the
`baselines` union hides that tpc does most of its work (standing instruction).
por is not drawn (R^2 ~ 0 would flatten every other curve); its range is
printed in the footer.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order (reference palette slots 1-5); colour follows the group.
MAIN = [
    ("baselines",     "baselines",     "#2a78d6", "o"),
    ("tda",           "tda (cone)",    "#eb6834", "s"),
    ("thr",           "thr (throat)",  "#1baf7a", "D"),
    ("all",           "baselines+tda", "#eda100", "^"),
    ("baselines+thr", "baselines+thr", "#e87ba4", "v"),
]
SEPARATE = ["tpc", "fabdir", "fab"]  # por omitted: R2~0 flattens the rest; range in footer
GREY, INK, MUTED, GRID, SURFACE = "#898781", "#1f1f1e", "#6b6a64", "#e1e0d9", "#fcfcfb"
METRICS = {"r2": ("R² on hold-out", "higher is better"),
           "mae": ("MAE on hold-out", "lower is better")}


def spread_labels(ys, min_gap, top):
    """Nudge label y positions apart (keeps order) so direct labels don't collide,
    pushing up first, then back down so none ends above `top`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + min_gap)
    out[order[-1]] = min(out[order[-1]], top)
    for a, b in zip(order[::-1], order[-2::-1]):
        out[b] = min(out[b], out[a] - min_gap)
    return out


def plot(df, target, arm, metric, path):
    g = df.groupby(["group", "size"])[metric].agg(["mean", "std"]).reset_index()
    sizes = sorted(g["size"].unique())
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    labels = []  # (y at largest n, text, colour)
    for grp in SEPARATE:
        s = g[g.group == grp]
        if s.empty:
            continue
        ax.plot(s["size"], s["mean"], color=GREY, lw=1.0, ls="--", zorder=2)
        labels.append((s["mean"].iloc[-1], grp, MUTED))
    for grp, name, col, mk in MAIN:
        s = g[g.group == grp]
        if s.empty:
            continue
        ax.fill_between(s["size"], s["mean"] - s["std"], s["mean"] + s["std"],
                        color=col, alpha=0.15, lw=0, zorder=3)
        ax.plot(s["size"], s["mean"], color=col, lw=2, marker=mk, ms=6,
                mec=SURFACE, mew=1.2, label=name, zorder=4)
        labels.append((s["mean"].iloc[-1], name, INK))

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.minorticks_off()
    ylo, yhi = ax.get_ylim()
    ys = spread_labels([y for y, _, _ in labels], 0.035 * (yhi - ylo), yhi)
    for (y0, text, col), y in zip(labels, ys):
        ax.annotate(text, xy=(sizes[-1], y0), xytext=(sizes[-1] * 1.12, y),
                    va="center", fontsize=8.5, color=col, annotation_clip=False)
    ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.08)

    ylabel, sense = METRICS[metric]
    ax.set_xlabel("training structures n (log scale)", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(f"{target}: {ylabel} vs training size  ({arm} arm, {sense})",
                 loc="left", fontsize=11, color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="lower right" if metric == "r2" else "upper right",
              frameon=False, fontsize=8.5)
    por = g[g.group == "por"]["mean"]
    note = (f"; por (porosity alone) not drawn: {metric} {por.min():.2f} to {por.max():.2f}"
            if len(por) else "")
    fig.text(0.01, 0.01, "mean over 5 seeds, band = ±1 sd; hold-out = 999 frozen "
             "structures; grey dashed = baseline components" + note,
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.86, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", default=["k_xy", "k_xz"])
    ap.add_argument("--arm", default="fixed", choices=["fixed", "es"])
    ap.add_argument("--metrics", nargs="+", default=["r2", "mae"])
    ap.add_argument("--out", default="RESULTS/lc3d_plots")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for t in a.targets:
        tag = t.lower().replace("_", "")
        runs = f"RESULTS/lc3d_{tag}/{a.arm}/runs.csv"
        if not os.path.exists(runs):
            print(f"skip {t}: {runs} does not exist (run TARGET={t} "
                  f"OUT=RESULTS/lc3d_{tag} bash run_lc3d.sh)", file=sys.stderr)
            continue
        df = pd.read_csv(runs)
        df = df[df.target == t]
        for m in a.metrics:
            path = f"{a.out}/lc3d_{tag}_{a.arm}_{m}.png"
            plot(df, t, a.arm, m, path)
            print("->", path)


if __name__ == "__main__":
    main()
