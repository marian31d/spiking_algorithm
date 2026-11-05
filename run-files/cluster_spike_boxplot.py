#!/usr/bin/env python3
"""
Per-cluster box plot showing WHEN each cluster is active.

For every cluster its spikes are ordered by time; the box spans the spike 25%
of the way through to the spike 75% of the way through (so the box covers the
middle half of that cluster's spikes), with the median spike marked inside.
Whiskers show the full extent by default -- first spike to last spike -- since
for "when does this unit exist" the extremes are the point, not outliers.

A cluster whose box covers the whole recording fires throughout. A narrow box
near one end is a unit that is only present for part of the session: exactly
the units a short `freeze_sec` learning window would miss, or learn and then
never see again.

Other metrics (--metric score|amplitude) give the same box plot for the
per-spike value distribution instead of time.

Example:
  python3 cluster_spike_boxplot.py --phy /path/to/phy --sort median \
      --vline 300,900,1800,3600 --out spans.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_FILE = {"time": "spike_times.npy", "score": "template_scores.npy",
               "amplitude": "amplitudes.npy"}
GOOD, MUA, INK, INK_2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load_labels(phy: Path):
    for fname, col in (("cluster_KSLabel.tsv", "KSLabel"), ("cluster_group.tsv", "group")):
        f = phy / fname
        if f.exists():
            df = pd.read_csv(f, sep="\t")
            if "cluster_id" in df.columns and col in df.columns:
                return {int(k): str(v).lower() for k, v in zip(df["cluster_id"], df[col])}
    return {}


def load_fs(phy: Path, override=None):
    if override:
        return float(override)
    try:
        ops = np.load(phy / "ops.npy", allow_pickle=True).item()
        return float(ops.get("fs") or ops["settings"]["fs"])
    except Exception:
        pass
    try:
        for line in (phy / "params.py").read_text().splitlines():
            if line.strip().startswith("sample_rate"):
                return float(line.split("=")[1])
    except Exception:
        pass
    print("  (could not read fs; assuming 20000 Hz -- pass --fs to override)")
    return 20000.0


def per_cluster_stats(clu, val, whis, method="inverted_cdf"):
    """Middle-50% box per cluster, plus whiskers (full range, or 1.5*IQR).

    `method` is passed to np.percentile. The default 'inverted_cdf' returns an
    actual data point -- the spike 25%/50%/75% of the way through the cluster's
    spikes -- rather than numpy's default 'linear', which interpolates between
    the two spikes bracketing that position and so returns a time at which no
    spike occurred. The two agree to milliseconds for clusters with many
    spikes, but diverge wildly for small ones: a 4-spike cluster puts the
    linear Q1 three quarters of the way between its 1st and 2nd spike, which
    here was over 6500 s from either.
    """
    order = np.argsort(clu, kind="stable")
    c, v = clu[order], val[order]
    ids, starts = np.unique(c, return_index=True)
    bounds = np.append(starts, len(c))

    rows = []
    for i, cid in enumerate(ids):
        x = v[bounds[i]:bounds[i + 1]]
        if x.size == 0:
            continue
        q1, med, q3 = np.percentile(x, [25, 50, 75], method=method)
        if whis == "range":
            lo, hi = float(x.min()), float(x.max())
        else:
            iqr = q3 - q1
            inside = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
            lo, hi = (float(inside.min()), float(inside.max())) if inside.size else (float(x.min()), float(x.max()))
        rows.append(dict(cluster_id=int(cid), n=int(x.size), q1=float(q1),
                         med=float(med), q3=float(q3), whislo=lo, whishi=hi))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phy", required=True, help="phy/Kilosort results directory")
    ap.add_argument("--metric", default="time", choices=list(METRIC_FILE))
    ap.add_argument("--fs", default=None, help="Sample rate; read from ops.npy if omitted")
    ap.add_argument("--units", default="s", choices=["s", "min", "h"], help="Time units (--metric time)")
    ap.add_argument("--quantile_method", default="inverted_cdf",
                    choices=["inverted_cdf", "linear", "lower", "nearest"],
                    help="How Q1/median/Q3 are taken. Default returns a real spike; "
                         "'linear' is numpy's interpolating default.")
    ap.add_argument("--whis", default=None, choices=["range", "iqr"],
                    help="Whiskers: full first-to-last extent (default for time) or 1.5*IQR")
    ap.add_argument("--sqrt", action="store_true", help="Plot sqrt(metric) (score/amplitude)")
    ap.add_argument("--sort", default="median", choices=["id", "median", "count", "start", "span"])
    ap.add_argument("--good_only", action="store_true")
    ap.add_argument("--min_spikes", type=int, default=1)
    ap.add_argument("--vline", default=None,
                    help="Comma-separated times to mark (e.g. freeze_sec candidates), in --units")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    phy = Path(args.phy).resolve()
    is_time = args.metric == "time"
    whis = args.whis or ("range" if is_time else "iqr")

    clu = np.load(phy / "spike_clusters.npy").astype(np.int64).ravel()
    val = np.load(phy / METRIC_FILE[args.metric]).astype(np.float64).ravel()
    if clu.size != val.size:
        raise SystemExit(f"length mismatch: spike_clusters {clu.size} vs {METRIC_FILE[args.metric]} {val.size}")

    if is_time:
        fs = load_fs(phy, args.fs)
        val = val / fs / {"s": 1.0, "min": 60.0, "h": 3600.0}[args.units]
        unit = f"time ({args.units})"
    else:
        if args.sqrt:
            val = np.sqrt(np.maximum(val, 0))
        unit = {"score": "template score", "amplitude": "amplitude"}[args.metric]
        if args.sqrt:
            unit = f"sqrt({unit})"

    labels = load_labels(phy)
    df = per_cluster_stats(clu, val, whis, method=args.quantile_method)
    df["label"] = [labels.get(c, "") for c in df["cluster_id"]]
    df["span"] = df["whishi"] - df["whislo"]
    df["iqr"] = df["q3"] - df["q1"]

    n_all = len(df)
    df = df[df["n"] >= args.min_spikes]
    # counts of the whole population, taken before --good_only drops anything, so
    # the legend keeps both entries and still reports what was filtered out
    pop_good = int((df["label"] == "good").sum())
    pop_mua = len(df) - pop_good
    if args.good_only:
        df = df[df["label"] == "good"]
    if df.empty:
        raise SystemExit("no clusters left after filtering")

    key = {"id": "cluster_id", "median": "med", "count": "n",
           "start": "whislo", "span": "span"}[args.sort]
    df = df.sort_values(key).reset_index(drop=True)

    # time reads naturally left-to-right, so lay those out horizontally
    horiz = is_time
    n = len(df)
    if horiz:
        fig, ax = plt.subplots(figsize=(11, max(5.0, min(40.0, 0.085 * n + 2))))
    else:
        fig, ax = plt.subplots(figsize=(max(8.0, min(40.0, 0.09 * n + 3)), 6.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x" if horiz else "y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)

    recs = df.to_dict("records")
    bxp_stats = [{"label": "", "whislo": r["whislo"], "q1": r["q1"], "med": r["med"],
                  "q3": r["q3"], "whishi": r["whishi"], "fliers": []} for r in recs]
    bp = ax.bxp(bxp_stats, positions=np.arange(n), widths=0.7, showfliers=False,
                patch_artist=True, vert=not horiz)
    for i, r in enumerate(recs):
        col = GOOD if r["label"] == "good" else MUA
        bp["boxes"][i].set(facecolor=col, edgecolor=col, linewidth=0.6, alpha=0.45)
        bp["medians"][i].set(color=col, linewidth=1.4)
        for k in ("whiskers", "caps"):
            for art in bp[k][2 * i:2 * i + 2]:
                art.set(color=col, linewidth=0.8)

    if args.vline:
        for v in [float(t) for t in args.vline.split(",")]:
            (ax.axvline if horiz else ax.axhline)(v, color=INK, linewidth=1.3,
                                                  linestyle="--", alpha=0.85, zorder=1)
            if horiz:
                # dark, bold, on a background patch so it stays readable under the title
                ax.annotate(f"{v:g}", xy=(v, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(2, 3), textcoords="offset points",
                            fontsize=9, fontweight="bold", color=INK,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE,
                                      edgecolor="none"))

    n_good = int((df["label"] == "good").sum())
    ax.plot([], [], color=GOOD, linewidth=6, alpha=0.45, label=f"good ({pop_good})")
    ax.plot([], [], color=MUA, linewidth=6, alpha=0.45,
            label=f"mua/other ({pop_mua}{', not shown' if args.good_only else ''})")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")

    step = max(1, n // (60 if horiz else 40))
    ticks = np.arange(0, n, step)
    tlabels = df["cluster_id"].values[::step]
    if horiz:
        ax.set_ylim(-1, n)
        ax.set_yticks(ticks); ax.set_yticklabels(tlabels, fontsize=6)
        ax.set_ylabel(f"cluster id (sorted by {args.sort})", fontsize=10, color=INK_2)
        ax.set_xlabel(unit, fontsize=10, color=INK_2)
    else:
        ax.set_xlim(-1, n)
        ax.set_xticks(ticks); ax.set_xticklabels(tlabels, rotation=90, fontsize=7)
        ax.set_xlabel(f"cluster id (sorted by {args.sort})", fontsize=10, color=INK_2)
        ax.set_ylabel(unit, fontsize=10, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    whis_txt = "whiskers: first to last spike" if whis == "range" else "whiskers: 1.5·IQR"
    ax.set_title(args.title or f"Per-cluster {unit} — box = middle 50% of spikes, {whis_txt} — {phy.name}",
                 fontsize=12, color=INK, loc="center",
                 pad=22 if (args.vline and horiz) else 10)   # room for the --vline labels

    out = Path(args.out) if args.out else phy / f"cluster_{args.metric}_box.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    df.to_csv(out.with_suffix(".csv"), index=False)

    print(f"{n} clusters plotted (of {n_all}); {n_good} good")
    if is_time:
        total = float(val.max())
        print(f"  recording spans 0 - {total:.0f} {args.units}")
        print(f"  median cluster: middle 50% of its spikes within {df['iqr'].median():.0f} {args.units}"
              f" ({100*df['iqr'].median()/total:.0f}% of recording)")
        for frac in (0.5, 0.25):
            k = int((df["span"] < frac * total).sum())
            print(f"  {k}/{n} clusters span < {frac:.0%} of the recording")
    print(f"wrote {out}\nwrote {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
