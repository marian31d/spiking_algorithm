#!/usr/bin/env python3
"""
Collect peel-sweep metrics from compare_templates_timewindow summaries.

Scans /data/results/peel_compare_vs100/<animal>/<runs>/<peel>/summary.txt and writes:
  all_metrics.csv                        - one row per (dataset, runs, peel)
  matching_pursuit_filled.csv            - three-table layout, runs="full"
  matching_pursuit_pershank_filled.csv   - three-table layout, runs="pershank"
"""

import csv
import re
from pathlib import Path

OUT = Path("/data/results/peel_compare_vs100")
PEELS = [1, 2, 4, 7, 10, 20, 50, 70, 90]
ANIMALS = [("mS51", "ms51"), ("mP79", "mp79"), ("mds2", "mds2")]
RUNS = [("full", "matching_pursuit_filled.csv"),
        ("pershank", "matching_pursuit_pershank_filled.csv")]


def parse(txt):
    """Pull the coverage + min-norm figures out of one summary.txt."""
    blk = txt.split("Min-norm (O/min(|GUI|,|FLOW|)) best-match stats")[1]
    allblk = blk.split("ALL clusters:")[1].split("GOOD-only:")[0]
    goodblk = blk.split("GOOD-only:")[1]
    g = lambda pat, s=txt: re.search(pat, s).group(1)
    return dict(
        ref_spikes=int(g(r"GUI total spikes:\s*(\d+)")),
        peel_spikes=int(g(r"FLOW total spikes:\s*(\d+)")),
        matched=int(g(r"Matched spikes:\s*(\d+)")),
        coverage_pct=round(float(g(r"Matched spikes/GUI window spikes:\s*([\d.]+)%")), 2),
        minnorm_spikewt_good_gui2flow=round(float(g(r"GUI.FLOW:.*?spike-wt=([\d.]+)%", goodblk)), 2),
        minnorm_spikewt_all_gui2flow=round(float(g(r"GUI.FLOW:.*?spike-wt=([\d.]+)%", allblk)), 2),
        minnorm_spikewt_all_flow2gui=round(float(g(r"FLOW.GUI:.*?spike-wt=([\d.]+)%", allblk)), 2),
    )


def table(rows, runs, col):
    out = []
    for _, lab in ANIMALS:
        vals = []
        for p in PEELS:
            r = rows.get((lab, runs, p))
            vals.append(f"{r[col]:.1f}" if r else "")
        out.append([lab] + vals + [""])
    return out


def main():
    rows, missing = {}, []
    for adir, lab in ANIMALS:
        for runs, _ in RUNS:
            for p in PEELS:
                f = OUT / adir / runs / str(p) / "summary.txt"
                if not f.exists():
                    missing.append(str(f))
                    continue
                rows[(lab, runs, p)] = dict(dataset=lab, runs=runs, peels=p, **parse(f.read_text()))

    cols = ["dataset", "runs", "peels", "ref_spikes", "peel_spikes", "matched", "coverage_pct",
            "minnorm_spikewt_good_gui2flow", "minnorm_spikewt_all_gui2flow",
            "minnorm_spikewt_all_flow2gui"]
    with open(OUT / "all_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for runs, _ in RUNS:
            for _, lab in ANIMALS:
                for p in PEELS:
                    if (lab, runs, p) in rows:
                        w.writerow(rows[(lab, runs, p)])

    hdr = ["peel number"] + [str(p) for p in PEELS]
    blocks = [("coverage", "coverage_pct"),
              ("score (all clusters)", "minnorm_spikewt_all_gui2flow"),
              ("score (good only)", "minnorm_spikewt_good_gui2flow")]
    for runs, fname in RUNS:
        if not any(k[1] == runs for k in rows):
            continue
        with open(OUT / fname, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            for i, (title, col) in enumerate(blocks):
                if i:
                    w.writerow([""] * 11)
                w.writerow(hdr + [title])
                w.writerows(table(rows, runs, col))
        print("wrote", OUT / fname)

    if missing:
        print(f"\nMISSING {len(missing)} summaries:")
        for m in missing[:10]:
            print("  ", m)


if __name__ == "__main__":
    main()
