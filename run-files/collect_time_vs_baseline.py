#!/usr/bin/env python3
"""
Collect freeze-time sweep metrics (each time/<freeze_sec> run vs baseline_det/1).

Scans /data/results/time_compare_vs_baseline/<animal>/<freeze_sec>/summary.txt and writes:
  all_metrics.csv        - one row per (dataset, freeze_sec), plus the baseline2 noise floor
  time_freeze_filled.csv - three-table layout (coverage / all-cluster / good-only min-norm)

Datasets have different freeze grids (each ends near its own recording length), so the
table columns are the union and a dataset is blank where it has no run.
"""

import argparse
import csv
import re
from pathlib import Path

DEFAULT_ROOT = "/data/results/time_compare_vs_baseline"
GRID = {"mS51": [300, 900, 1800, 2400, 3600, 5400, 10800, 14400, 20400, 20754],
        "mP79": [300, 900, 1800, 2400, 3600, 5400, 10800, 12600, 12824],
        "mds2": [300, 900, 1800, 2400, 3600, 5400, 10800, 13800, 14051]}

# each dataset's all-data freeze run: freeze_sec >= its recording length, so every
# model is fit on the whole recording while still taking the finite-freeze path
ALL_DATA = {"mS51": 20754, "mP79": 12824, "mds2": 14051}
LABEL = {"mS51": "ms51", "mP79": "mp79", "mds2": "mds2"}
COLS = sorted({t for g in GRID.values() for t in g})


def parse(txt):
    blk = txt.split("Min-norm (O/min(|GUI|,|FLOW|)) best-match stats")[1]
    allblk = blk.split("ALL clusters:")[1].split("GOOD-only:")[0]
    goodblk = blk.split("GOOD-only:")[1]
    g = lambda pat, s=txt: re.search(pat, s).group(1)
    return dict(
        ref_spikes=int(g(r"GUI total spikes:\s*(\d+)")),
        ref_in_window=int(g(r"GUI total spikes:\s*\d+\s*\(in window: (\d+)\)")),
        run_spikes=int(g(r"FLOW total spikes:\s*(\d+)")),
        run_in_window=int(g(r"FLOW total spikes:\s*\d+\s*\(in window: (\d+)\)")),
        matched=int(g(r"Matched spikes:\s*(\d+)")),
        coverage_pct=round(float(g(r"Matched spikes/GUI window spikes:\s*([\d.]+)%")), 2),
        minnorm_spikewt_good_gui2flow=round(float(g(r"GUI.FLOW:.*?spike-wt=([\d.]+)%", goodblk)), 2),
        minnorm_spikewt_all_gui2flow=round(float(g(r"GUI.FLOW:.*?spike-wt=([\d.]+)%", allblk)), 2),
        minnorm_spikewt_all_flow2gui=round(float(g(r"FLOW.GUI:.*?spike-wt=([\d.]+)%", allblk)), 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="directory holding <animal>/<freeze_sec>/summary.txt")
    ap.add_argument("--csv_name", default="time_freeze_filled.csv")
    ap.add_argument("--ref_is_max", action="store_true",
                    help="Reference is each dataset's longest freeze run, so that run has no "
                         "row of its own (it would be a 100%% self-comparison); leave it blank.")
    args = ap.parse_args()
    global OUT
    OUT = Path(args.root)

    rows, missing = {}, []
    skip = {a: (max(GRID[a]) if args.ref_is_max else None) for a in GRID}
    for adir in GRID:
        for key in ["baseline2"] + [str(t) for t in GRID[adir]]:
            if skip[adir] is not None and key == str(skip[adir]):
                continue
            f = OUT / adir / key / "summary.txt"
            if not f.exists():
                missing.append(str(f))
                continue
            rows[(LABEL[adir], key)] = dict(dataset=LABEL[adir], freeze_sec=key,
                                            **parse(f.read_text()))

    cols = ["dataset", "freeze_sec", "ref_spikes", "ref_in_window", "run_spikes", "run_in_window",
            "matched", "coverage_pct",
            "minnorm_spikewt_good_gui2flow", "minnorm_spikewt_all_gui2flow",
            "minnorm_spikewt_all_flow2gui"]
    with open(OUT / "all_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for adir in GRID:
            for key in ["baseline2"] + [str(t) for t in GRID[adir]]:
                if (LABEL[adir], key) in rows:
                    w.writerow(rows[(LABEL[adir], key)])

    hdr = ["freeze sec"] + [str(t) for t in COLS]
    blocks = [("coverage", "coverage_pct"),
              ("score (all clusters)", "minnorm_spikewt_all_gui2flow"),
              ("score (good only)", "minnorm_spikewt_good_gui2flow")]
    with open(OUT / args.csv_name, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        for i, (title, col) in enumerate(blocks):
            if i:
                w.writerow([""] * (len(COLS) + 2))
            w.writerow(hdr + [title])
            for adir in GRID:
                lab = LABEL[adir]
                vals = []
                for t in COLS:
                    r = rows.get((lab, str(t)))
                    vals.append(f"{r[col]:.1f}" if r else "")
                w.writerow([lab] + vals + [""])
    print("wrote", OUT / args.csv_name)

    if any((LABEL[a], "baseline2") in rows for a in GRID):
        print("\nrun-to-run noise floor (baseline_det/2 vs /1):")
    for adir in GRID:
        r = rows.get((LABEL[adir], "baseline2"))
        if r:
            print(f"  {LABEL[adir]}: coverage={r['coverage_pct']}%  "
                  f"all={r['minnorm_spikewt_all_gui2flow']}%  "
                  f"good={r['minnorm_spikewt_good_gui2flow']}%")
    missing = [m for m in missing if "baseline2" not in m]
    if missing:
        print(f"\nMISSING {len(missing)}:")
        for m in missing[:12]:
            print("  ", m)


if __name__ == "__main__":
    main()
