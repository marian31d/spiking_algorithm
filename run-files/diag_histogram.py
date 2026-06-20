#!/usr/bin/env python3
"""
Diagnostic histograms for the spike detector.

Two questions, two plots:

  PLOT 1 - ISI histogram of the FLOW (C) detector
      Inter-spike intervals, pooled and per-template.
      Per-template intervals shorter than the refractory period are
      physically impossible for a single neuron -> they are DUPLICATES
      (the re-triggering we suspect). This plot quantifies them.

  PLOT 2 - cost function (cf) at Kilosort spike times
      Reads a cf trace exported from the C code (one win_cf per sample)
      and, at each Kilosort spike, takes the max cf in a small window.
      If most true spikes have cf BELOW threshold  -> threshold too high
                                                       (sensitivity problem).
      If most true spikes have cf ABOVE threshold  -> they ARE detectable,
                                                       the detector logic is
                                                       dropping them.
      This is the measurement that decides why recall is ~24%.

Run PLOT 1 immediately (needs only the C output). PLOT 2 needs the cf
trace - see the comment block at the bottom for the ~6 lines to add to the C.

Example:
  python diag_histogram.py \
      --flow /data/results/mP79_res/flows_c/debug \
      --ks   /data/results/mP79_res/mp79-det-3pc-frozen-temp-2peels-7.6 \
      --cf_trace /data/results/mP79_res/flows_c/debug/cf_trace.npy \
      --start_sample 72000000 \
      --threshold 64 --refractory 20 \
      --out ./diag_out
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 20000  # Hz, only used to label axes in ms


# ----------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------
def load_times_and_labels(phy_dir: Path):
    """spike_times.npy is required; labels come from spike_clusters or
    spike_templates, whichever exists (the C writes spike_templates)."""
    phy_dir = Path(phy_dir)
    times = np.load(phy_dir / "spike_times.npy").astype(np.int64).reshape(-1)
    labels = None
    for name in ("spike_clusters.npy", "spike_templates.npy"):
        p = phy_dir / name
        if p.exists():
            labels = np.load(p).astype(np.int64).reshape(-1)
            break
    if labels is not None and labels.shape[0] != times.shape[0]:
        print(f"  [warn] times ({times.shape[0]}) and labels "
              f"({labels.shape[0]}) length mismatch -> ignoring labels")
        labels = None
    return times, labels


# ----------------------------------------------------------------------
# ISI
# ----------------------------------------------------------------------
def pooled_isi(times):
    return np.diff(np.sort(times))


def per_template_isi(times, labels):
    """intervals between consecutive spikes of the SAME template."""
    out = []
    for c in np.unique(labels):
        tc = np.sort(times[labels == c])
        if tc.size > 1:
            out.append(np.diff(tc))
    return np.concatenate(out) if out else np.array([], dtype=np.int64)


def top_duplicating_templates(times, labels, refractory, k=10):
    rows = []
    for c in np.unique(labels):
        tc = np.sort(times[labels == c])
        if tc.size < 2:
            continue
        d = np.diff(tc)
        viol = int((d < refractory).sum())
        rows.append((int(c), tc.size, viol, 100.0 * viol / d.size))
    rows.sort(key=lambda r: -r[2])
    return rows[:k]


def plot_isi(times, labels, refractory, out_path):
    pooled = pooled_isi(times)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (0,0) pooled, zoomed to 0-100 samples, 1-sample bins
    ax = axes[0, 0]
    ax.hist(pooled, bins=np.arange(0, 101, 1))
    ax.axvline(refractory, color="r", ls="--",
               label=f"refractory = {refractory} samp ({refractory/FS*1000:.1f} ms)")
    ax.set_title("Pooled ISI (all templates) - 0-100 samples")
    ax.set_xlabel("interval (samples)")
    ax.set_ylabel("count")
    ax.legend()

    # (0,1) pooled, full range, log y
    ax = axes[0, 1]
    hi = int(np.percentile(pooled, 99.5)) if pooled.size else 1
    ax.hist(pooled, bins=np.linspace(0, max(hi, 2), 200), log=True)
    ax.set_title("Pooled ISI - full range (log y)")
    ax.set_xlabel("interval (samples)")
    ax.set_ylabel("count (log)")

    pooled_viol_frac = float((pooled < refractory).mean()) if pooled.size else 0.0

    if labels is not None:
        per = per_template_isi(times, labels)
        # (1,0) per-template, zoomed
        ax = axes[1, 0]
        ax.hist(per, bins=np.arange(0, 101, 1), color="C1")
        ax.axvline(refractory, color="r", ls="--")
        ax.set_title("Per-template ISI (SAME template) - 0-100 samples")
        ax.set_xlabel("interval (samples)")
        ax.set_ylabel("count")

        per_viol = int((per < refractory).sum())
        per_frac = (per_viol / per.size) if per.size else 0.0

        # (1,1) text summary
        ax = axes[1, 1]
        ax.axis("off")
        txt = (
            "DUPLICATE SUMMARY\n"
            "-----------------\n"
            f"total FLOW spikes:            {times.size:,}\n\n"
            f"per-template intervals:       {per.size:,}\n"
            f"  < refractory ({refractory} samp):     {per_viol:,}\n"
            f"  fraction (= duplicate rate): {100*per_frac:.1f}%\n\n"
            f"pooled intervals < {refractory} samp:    {100*pooled_viol_frac:.1f}%\n"
            "(pooled mixes real coincidences\n"
            " from different neurons with dups;\n"
            " per-template is the clean signal)"
        )
        ax.text(0.02, 0.98, txt, va="top", ha="left",
                fontsize=12, family="monospace")
    else:
        for ax in (axes[1, 0], axes[1, 1]):
            ax.axis("off")
            ax.text(0.5, 0.5, "no template labels found\n(per-template ISI skipped)",
                    ha="center", va="center")
        per = np.array([])
        per_viol, per_frac = 0, 0.0

    fig.suptitle("FLOW detector inter-spike-interval diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # console echo (so you can paste numbers back without the image)
    print("\n=== ISI / duplicate diagnostics ===")
    print(f"total FLOW spikes              : {times.size:,}")
    print(f"pooled intervals < {refractory:>2} samples : {100*pooled_viol_frac:5.1f}%")
    if labels is not None:
        print(f"per-template intervals        : {per.size:,}")
        print(f"per-template < {refractory:>2} samples     : {per_viol:,} "
              f"({100*per_frac:.1f}%)   <-- duplicate rate")
        print("\ntop duplicating templates (template_id, n_spikes, n_violations, %):")
        for cid, n, v, pc in top_duplicating_templates(times, labels, refractory):
            print(f"   id={cid:<5} n={n:<8} viol={v:<8} {pc:5.1f}%")
    print(f"\nsaved -> {out_path}")


# ----------------------------------------------------------------------
# cf at Kilosort spike times
# ----------------------------------------------------------------------
def plot_cf(cf_trace_path, ks_dir, start_sample, threshold, win, out_path):
    cf = np.load(cf_trace_path).astype(np.float64).reshape(-1)  # win_cf per sample, index 0 == start_sample
    ks_times = np.load(Path(ks_dir) / "spike_times.npy").astype(np.int64).reshape(-1)

    # align KS spike times into the cf-trace index space
    rel = ks_times - int(start_sample)
    in_range = (rel >= 0) & (rel < cf.size)
    rel = rel[in_range]
    if rel.size == 0:
        print("[cf] no Kilosort spikes fall inside the cf trace window - "
              "check --start_sample and the dump length.")
        return

    # alignment-agnostic: take the MAX cf in +/- win samples around each KS time.
    # this dodges the SPIKE_TIME_OFFSET / time-convention mismatch entirely.
    try:
        from scipy.ndimage import maximum_filter1d
        cf_rollmax = maximum_filter1d(cf, size=2 * win + 1, mode="nearest")
        cf_at_ks = cf_rollmax[rel]
    except Exception as e:
        print(f"[cf] scipy unavailable ({e}); falling back to slow loop")
        cf_at_ks = np.empty(rel.size)
        for k, r in enumerate(rel):
            a = max(0, r - win); b = min(cf.size, r + win + 1)
            cf_at_ks[k] = cf[a:b].max()

    # background distribution: cf at all samples (subsampled for the plot)
    sub = cf[:: max(1, cf.size // 200000)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    xhi = float(np.percentile(cf_at_ks, 99))
    xhi = max(xhi, threshold * 2)

    ax = axes[0]
    ax.hist(cf_at_ks, bins=np.linspace(0, xhi, 120), color="C0",
            label="cf at Kilosort spikes")
    ax.axvline(threshold, color="r", ls="--", label=f"threshold = {threshold:g}")
    ax.set_title("cf at TRUE (Kilosort) spike times")
    ax.set_xlabel("max cost function near spike")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[1]
    ax.hist(cf_at_ks, bins=np.linspace(0, xhi, 120), density=True, alpha=0.6,
            label="at Kilosort spikes")
    ax.hist(sub, bins=np.linspace(0, xhi, 120), density=True, alpha=0.6,
            label="everywhere (background)")
    ax.axvline(threshold, color="r", ls="--")
    ax.set_yscale("log")
    ax.set_title("true spikes vs background (log y, density)")
    ax.set_xlabel("cost function")
    ax.set_ylabel("density (log)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # the decisive numbers
    n = cf_at_ks.size
    def frac_below(thr):
        return 100.0 * float((cf_at_ks < thr).mean())
    print("\n=== cf at Kilosort spike times ===")
    print(f"Kilosort spikes inside cf window : {n:,} "
          f"(of {ks_times.size:,} total)")
    print(f"median cf at true spikes         : {np.median(cf_at_ks):.1f}")
    print(f"fraction below threshold {threshold:g}      : {frac_below(threshold):5.1f}%"
          "   <-- hard ceiling on recall at this threshold")
    for thr in (threshold, threshold * 0.5625, threshold * 0.25):  # 64 -> 36 -> 16
        print(f"   would-be-missed if TH={thr:6.1f} : {frac_below(thr):5.1f}%")
    print(f"\nsaved -> {out_path}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True,
                    help="dir with the C detector output (the 23.5M-spike one)")
    ap.add_argument("--ks", default=None,
                    help="dir with Kilosort output (for the cf plot)")
    ap.add_argument("--cf_trace", default=None,
                    help="path to win_cf-per-sample .npy exported from the C code")
    ap.add_argument("--start_sample", type=int, default=72000000,
                    help="abs sample index the cf trace starts at (== C start_sample)")
    ap.add_argument("--threshold", type=float, default=64.0,
                    help="TH_LEARNED_SQ from the C code")
    ap.add_argument("--refractory", type=int, default=20,
                    help="REFRACTORY_SAMPLES from the C code")
    ap.add_argument("--win", type=int, default=50,
                    help="+/- samples to search for the cf peak around each KS spike")
    ap.add_argument("--out", default="./diag_out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading FLOW detector output from {args.flow} ...")
    times, labels = load_times_and_labels(args.flow)
    print(f"  {times.size:,} spikes, "
          f"{'with' if labels is not None else 'no'} template labels")
    plot_isi(times, labels, args.refractory, out / "isi_histogram.png")

    if args.cf_trace and Path(args.cf_trace).exists() and args.ks:
        plot_cf(args.cf_trace, args.ks, args.start_sample,
                args.threshold, args.win, out / "cf_at_kilosort.png")
    else:
        print("\n[cf plot skipped] provide --cf_trace and --ks to run it.")
        print("    (see the C-dump instructions at the bottom of this file)")


if __name__ == "__main__":
    main()

