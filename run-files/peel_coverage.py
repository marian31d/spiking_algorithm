#!/usr/bin/env python3
"""
Per-peel coverage of a Kilosort run by one or more flows outputs.

For every Kilosort spike, spike_peels.npy says which matching-pursuit peel found it
(0-based in the file, shown as peel 1..N). Each flows output is matched 1:1 to the
Kilosort spikes by time (match_spike_times_tol, +-tol samples), and for every peel:
  covered       = Kilosort spikes of that peel that have a flows spike
  same template = covered spikes whose flows template id == the Kilosort spike_templates id
Both are reported as % of that peel's Kilosort spikes.

Only spikes with t >= t_start are used, on BOTH sides (like --maskflows 1).
t_start defaults to the Kilosort run's freeze_sec (0 if it has none / inf).

Inputs:
  --ks_dir   Kilosort output with spike_times.npy, spike_templates.npy, spike_peels.npy, ops.npy
  --flows    one or more flows output dirs with spike_times.npy and
             spike_templates.npy (or spike_clusters.npy holding the template ids)

Outputs (in --out, default: the parent of the first --flows dir):
  peel_coverage.csv      flows, tol, peel, ks_spikes, covered, covered_pct,
                         covered_same_template, covered_same_template_pct
  peel_coverage_pct.png  coverage histogram, black tick = same-template coverage

Run:
python peel_coverage.py \
  --ks_dir /data/results/mS51_res/IIR/updated_clu/time/peel_specified/3600 \
  --flows  /data/results/mS51_res/flows_c/no_th /data/results/mS51_res/flows_c/with_Th \
  --tol 0
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_templates_timewindow import match_spike_times_tol

# fixed categorical order, one colour per flows output
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#1f1f1d", "#5f5e5a", "#e4e3df", "#fcfcfb"


def load_template_ids(flows_dir: Path) -> np.ndarray:
    for name in ("spike_templates.npy", "spike_clusters.npy"):
        p = flows_dir / name
        if p.exists():
            return np.load(p).astype(np.int64, copy=False).reshape(-1)
    raise FileNotFoundError(f"no spike_templates.npy or spike_clusters.npy in {flows_dir}")


def peel_coverage(ks_times, ks_templates, ks_peels, flows_times, flows_templates, n_peels, tol):
    """Return per-peel counts: covered, covered with the same template."""
    i_ks, i_fl = match_spike_times_tol(ks_times, flows_times, tol=tol)
    covered = np.bincount(ks_peels[i_ks], minlength=n_peels)
    same = ks_templates[i_ks] == flows_templates[i_fl]
    covered_same = np.bincount(ks_peels[i_ks][same], minlength=n_peels)
    return covered, covered_same


def plot_coverage(rows, labels, title, path: Path):
    peels = np.array(sorted({r["peel"] for r in rows}))
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(peels)), 5), facecolor=SURFACE)

    w = 0.8 / len(labels)
    for i, lbl in enumerate(labels):
        rr = sorted((r for r in rows if r["flows"] == lbl), key=lambda r: r["peel"])
        off = (i - (len(labels) - 1) / 2) * w
        ax.bar(peels + off, [r["covered_pct"] for r in rr], w, color=COLORS[i % len(COLORS)],
               label=f"{lbl}: covered", edgecolor=SURFACE, linewidth=1)
        ax.scatter(peels + off, [r["covered_same_template_pct"] for r in rr], marker="_",
                   s=max(40, 450 * w), color=INK, linewidths=2, zorder=3,
                   label="covered with the same template" if i == len(labels) - 1 else None)

    ax.set_ylim(0, 100)
    ax.set_xticks(peels)
    ax.set_xlabel("Peel (matching-pursuit iteration that found the Kilosort spike)", color=INK)
    ax.set_ylabel("Covered (% of that peel's spikes)", color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK2)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks_dir", required=True, type=str, help="Kilosort output dir (needs spike_peels.npy)")
    ap.add_argument("--flows", required=True, nargs="+", type=str, help="one or more flows output dirs")
    ap.add_argument("--labels", nargs="+", default=None, help="names for the flows dirs (default: dir names)")
    ap.add_argument("--tol", default=0, type=int, help="matching tolerance in samples (default 0)")
    ap.add_argument("--t_start", default=None, type=float,
                    help="only use spikes with t >= t_start seconds, on both sides "
                         "(default: the Kilosort run's freeze_sec, or 0)")
    ap.add_argument("--out", default=None, type=str, help="output dir (default: parent of the first --flows dir)")
    ap.add_argument("--title", default=None, type=str, help="plot title")
    args = ap.parse_args()

    ks_dir = Path(args.ks_dir).resolve()
    flows_dirs = [Path(f).resolve() for f in args.flows]
    labels = args.labels or [f.name for f in flows_dirs]
    if len(labels) != len(flows_dirs):
        raise ValueError("--labels must have one name per --flows dir")
    out_dir = Path(args.out).resolve() if args.out else flows_dirs[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = np.load(ks_dir / "ops.npy", allow_pickle=True).item()
    fs = float(ops.get("fs", 20000))
    t_start = args.t_start
    if t_start is None:
        freeze = ops.get("settings", {}).get("freeze_sec", None)
        t_start = float(freeze) if freeze is not None and np.isfinite(freeze) else 0.0
    t0 = int(round(t_start * fs))
    print(f"Kilosort: {ks_dir}")
    print(f"Window: t >= {t_start:g} s ({t0} samples, fs={fs:g}), tol={args.tol}")

    ks_times = np.load(ks_dir / "spike_times.npy").astype(np.int64, copy=False).reshape(-1)
    ks_templates = np.load(ks_dir / "spike_templates.npy").astype(np.int64, copy=False).reshape(-1)
    ks_peels = np.load(ks_dir / "spike_peels.npy").astype(np.int64, copy=False).reshape(-1)
    keep = ks_times >= t0
    ks_times, ks_templates, ks_peels = ks_times[keep], ks_templates[keep], ks_peels[keep]
    n_peels = int(ks_peels.max()) + 1
    n_ks = np.bincount(ks_peels, minlength=n_peels)
    print(f"Kilosort spikes in window: {ks_times.size} over {n_peels} peels")

    rows = []
    for lbl, fdir in zip(labels, flows_dirs):
        f_times = np.load(fdir / "spike_times.npy").astype(np.int64, copy=False).reshape(-1)
        f_templates = load_template_ids(fdir)
        keep = f_times >= t0
        f_times, f_templates = f_times[keep], f_templates[keep]

        covered, covered_same = peel_coverage(ks_times, ks_templates, ks_peels,
                                              f_times, f_templates, n_peels, args.tol)
        print(f"{lbl}: {f_times.size} flows spikes, covered {100 * covered.sum() / ks_times.size:.1f}% "
              f"of Kilosort spikes ({100 * covered_same.sum() / ks_times.size:.1f}% with the same template)")
        for p in range(n_peels):
            n = int(n_ks[p])
            rows.append({
                "flows": lbl, "tol": args.tol, "peel": p + 1, "ks_spikes": n,
                "covered": int(covered[p]),
                "covered_pct": round(100.0 * covered[p] / n, 2) if n else float("nan"),
                "covered_same_template": int(covered_same[p]),
                "covered_same_template_pct": round(100.0 * covered_same[p] / n, 2) if n else float("nan"),
            })

    csv_path = out_dir / "peel_coverage.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    title = args.title or (f"Share of each peel's Kilosort spikes matched by flows "
                           f"({ks_dir.parent.name}/{ks_dir.name}, t >= {t_start:g} s, tol {args.tol})")
    png_path = out_dir / "peel_coverage_pct.png"
    plot_coverage(rows, labels, title, png_path)

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
