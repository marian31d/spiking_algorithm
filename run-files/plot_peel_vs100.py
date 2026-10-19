#!/usr/bin/env python3
"""
Plot peel-sweep metrics (vs the 100-peel reference) from all_metrics.csv.

Three stacked panels, one y-axis each, shared categorical x (peel count):
  1. coverage        = matched spikes / reference spikes
  2. min-norm, all clusters
  3. min-norm, good-only clusters
Min-norm = O/min(|GUI|,|FLOW|), GUI->FLOW, spike-weighted mean.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/data/results/peel_compare_vs100")
PEELS = [1, 2, 4, 7, 10, 20, 50, 70, 90]

# categorical slots 1-3 of the reference palette (all-pairs validated, light mode)
SERIES = [
    ("ms51", "#2a78d6", "o"),
    ("mp79", "#eb6834", "s"),
    ("mds2", "#1baf7a", "^"),
]

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

PANELS = [
    ("coverage_pct",                  "Coverage (%)",            "Spike coverage vs 100-peel reference",   (30, 102)),
    ("minnorm_spikewt_all_gui2flow",  "Min-norm (%)",            "Cluster agreement — all clusters",   (93.0, 100.5)),
    ("minnorm_spikewt_good_gui2flow", "Min-norm (%)",            "Cluster agreement — good only",      (93.0, 100.5)),
]


RUN_TITLE = {"full": "whole probe", "pershank": "per shank"}


def is_identity(data, name, peel):
    """True when the peel run reproduced the reference exactly (metrics trivially 100%)."""
    r = data[(name, peel)]
    return r["ref_spikes"] == r["peel_spikes"] == r["matched"]


def load(runs):
    data = {}
    with open(OUT / "all_metrics.csv") as fh:
        for r in csv.DictReader(fh):
            if r["runs"] == runs:
                data[(r["dataset"], int(r["peels"]))] = r
    if not data:
        raise SystemExit(f"no rows with runs={runs!r} in all_metrics.csv")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="pershank", choices=["full", "pershank"])
    args = ap.parse_args()

    data = load(args.runs)
    x = range(len(PEELS))

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    svals = [float(data[(n, p)][c]) for n, _, _ in SERIES for p in PEELS
             for c in ("minnorm_spikewt_all_gui2flow", "minnorm_spikewt_good_gui2flow")]
    slo = min(svals) - 0.4
    clo = min(float(data[(n, p)]["coverage_pct"]) for n, _, _ in SERIES for p in PEELS) - 5

    for ax, (col, ylab, title, ylim) in zip(axes, PANELS):
        ylim = (slo, 100.5) if col.startswith("minnorm") else (clo, 102)
        ax.set_facecolor(SURFACE)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, linestyle="-")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
            ax.spines[side].set_linewidth(0.8)

        for name, color, marker in SERIES:
            y = [float(data[(name, p)][col]) for p in PEELS]
            ax.plot(x, y, color=color, linewidth=2, label=name,
                    clip_on=False, zorder=3)
            for xi, yi, p in zip(x, y, PEELS):
                ident = is_identity(data, name, p)
                ax.plot([xi], [yi], marker=marker, markersize=5.5,
                        color=color,
                        markerfacecolor=SURFACE if ident else color,
                        markeredgecolor=color if ident else SURFACE,
                        markeredgewidth=1.4, clip_on=False, zorder=4)

        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab, fontsize=10, color=INK_2)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)

    # selective direct label: only a genuine interior dip, not the tail of a trend
    col = "minnorm_spikewt_good_gui2flow"
    for name, _, _ in SERIES:
        ys = [float(data[(name, p)][col]) for p in PEELS]
        for i in range(1, len(PEELS) - 1):
            drop = min(ys[i - 1], ys[i + 1]) - ys[i]
            if drop < 0.8:
                continue
            # place the label toward the interior so it cannot overflow the axes
            side = 1 if i < len(PEELS) / 2 else -1
            axes[2].annotate(f"{name}, {PEELS[i]} peels: {ys[i]:.1f}",
                             xy=(i, ys[i]), xytext=(14 * side, 26),
                             textcoords="offset points", fontsize=8.5, color=INK_2,
                             ha="left" if side > 0 else "right",
                             arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1.2))

    axes[0].legend(frameon=False, fontsize=9, loc="lower right",
                   labelcolor=INK_2, handlelength=1.8)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels([str(p) for p in PEELS])
    axes[-1].set_xlabel("Number of matching-pursuit peels", fontsize=10, color=INK_2)

    fig.suptitle(f"Matching pursuit: peel sweep vs 100-peel run ({RUN_TITLE[args.runs]})",
                 fontsize=13, color=INK, x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.952,
             "Min-norm = O/min(|ref|,|peel|), GUI→FLOW, spike-weighted mean. "
             "tol = 0 samples, no time window.",
             fontsize=8.5, color=INK_2, ha="left")
    if any(is_identity(data, n, p) for n, _, _ in SERIES for p in PEELS):
        fig.text(0.055, 0.932,
                 "Hollow markers: peel output identical to the reference, so 100% is "
                 "an identity, not a measurement.",
                 fontsize=8.5, color=INK_2, ha="left")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"peel_sweep_vs100_{args.runs}.{ext}", dpi=300,
                    facecolor=SURFACE, bbox_inches="tight")
    print("wrote", OUT / f"peel_sweep_vs100_{args.runs}.png")


if __name__ == "__main__":
    main()
