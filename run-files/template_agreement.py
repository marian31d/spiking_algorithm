#!/usr/bin/env python3
"""
Time-match C (FLOW) spikes to Kilosort (GUI) spikes, then analyze template assignment:
  - of time-matched pairs, how often do the templates AGREE?
  - when they DISAGREE, are the two templates similar (twins)?  uses similar_templates.npy

This separates "missed/extra spikes" (a detection problem) from
"same spike, different template label" (a twin-template problem).

Usage:
    python template_agreement.py \
        --c-times   flows_c/2peels/spike_times.npy \
        --c-temps   flows_c/2peels/spike_templates.npy \
        --ks-times  GUI/spike_times.npy \
        --ks-temps  GUI/spike_templates.npy \
        --sim       GUI/similar_templates.npy \
        --tol 2
"""
import argparse
import numpy as np


def match_by_time(c_t, ks_t, tol):
    """Greedy nearest-time match: for each C spike, find closest KS spike within tol.
    Returns arrays of matched indices (ic, iks)."""
    order = np.argsort(ks_t)
    ks_sorted = ks_t[order]
    ic, iks = [], []
    used = np.zeros(len(ks_t), bool)
    for i, t in enumerate(c_t):
        # binary search nearest
        j = np.searchsorted(ks_sorted, t)
        best = -1; bestd = tol + 1
        for jj in (j-1, j, j+1):
            if 0 <= jj < len(ks_sorted):
                d = abs(int(ks_sorted[jj]) - int(t))
                if d <= tol and d < bestd and not used[order[jj]]:
                    bestd = d; best = order[jj]
        if best >= 0:
            used[best] = True
            ic.append(i); iks.append(best)
    return np.array(ic, dtype=np.int64), np.array(iks, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c-times", required=True)
    ap.add_argument("--c-temps", required=True)
    ap.add_argument("--ks-times", required=True)
    ap.add_argument("--ks-temps", required=True)
    ap.add_argument("--sim", default=None, help="similar_templates.npy (n_templates x n_templates)")
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--offset", type=int, default=0,
                    help="manual constant added to KS times (default 0; both absolute)")
    ap.add_argument("--sim-thresh", type=float, default=0.8,
                    help="cosine above this = 'twin' templates")
    args = ap.parse_args()

    c_raw = np.load(args.c_times)
    k_raw = np.load(args.ks_times)
    print(f"C times dtype:  {c_raw.dtype}   KS times dtype: {k_raw.dtype}")
    if c_raw.dtype == np.float32 or k_raw.dtype == np.float32:
        print("  *** WARNING: float32 cannot represent ~72,000,000 exactly "
              "(only ~7 sig digits). Times may be corrupted. Re-save as int64. ***")

    c_t  = c_raw.astype(np.int64).ravel()
    c_tp = np.load(args.c_temps).astype(np.int64).ravel()
    k_t  = k_raw.astype(np.int64).ravel()
    k_tp = np.load(args.ks_temps).astype(np.int64).ravel()

    print(f"C spikes: {len(c_t)},  KS spikes: {len(k_t)}")
    print(f"C  time range:  [{c_t.min()}, {c_t.max()}]")
    print(f"KS time range:  [{k_t.min()}, {k_t.max()}]")

    # auto-detect offset DISABLED: comparing first-N medians is wrong when the two
    # runs cover different windows (KS from 0, C from 72M) — their first spikes are
    # different events. If both are absolute (same fs, same file), use offset=0 and
    # just compare on the overlapping window. Override with --offset if truly needed.
    offset = args.offset
    if offset != 0:
        print(f"  applying manual offset: shifting KS times by +{offset}")
        k_t = k_t + offset
    # restrict BOTH to the overlapping window so recall denominators are fair
    lo = max(c_t.min(), k_t.min()); hi = min(c_t.max(), k_t.max())
    print(f"overlapping time window: [{lo}, {hi}]  "
          f"({'OK' if hi>lo else 'NO OVERLAP'})")
    cmask = (c_t >= lo) & (c_t <= hi)
    kmask = (k_t >= lo) & (k_t <= hi)
    c_t, c_tp = c_t[cmask], c_tp[cmask]
    k_t, k_tp = k_t[kmask], k_tp[kmask]
    print(f"in-window: C={len(c_t)}, KS={len(k_t)}")

    ic, iks = match_by_time(c_t, k_t, args.tol)
    print(f"time-matched pairs (tol={args.tol}): {len(ic)}")
    print(f"  C spikes matched:  {len(ic)}/{len(c_t)}  ({100*len(ic)/len(c_t):.1f}%)")
    print(f"  KS spikes matched: {len(np.unique(iks))}/{len(k_t)}  ({100*len(np.unique(iks))/len(k_t):.1f}%)")

    # of the time-matched pairs, do templates agree?
    c_match_tp = c_tp[ic]
    k_match_tp = k_tp[iks]
    agree = (c_match_tp == k_match_tp)
    print(f"\nof {len(ic)} time-matched pairs:")
    print(f"  template AGREES:    {agree.sum()}  ({100*agree.mean():.1f}%)")
    print(f"  template DISAGREES: {(~agree).sum()}  ({100*(~agree).mean():.1f}%)")

    # for disagreements, are the two templates twins (high similarity)?
    if args.sim is not None:
        sim = np.load(args.sim)
        print(f"\n  similar_templates matrix: {sim.shape}")
        dis = ~agree
        cc = c_match_tp[dis]; kk = k_match_tp[dis]
        valid = (cc < sim.shape[0]) & (kk < sim.shape[1])
        sims = sim[cc[valid], kk[valid]]
        twin = sims >= args.sim_thresh
        print(f"  of {dis.sum()} disagreements:")
        print(f"    TWIN (cos>={args.sim_thresh}): {twin.sum()}  ({100*twin.mean():.1f}%)"
              f"  <- same spike, just different similar-template label")
        print(f"    genuinely different: {(~twin).sum()}  ({100*(~twin).mean():.1f}%)")
        print(f"    median similarity of disagreeing pairs: {np.median(sims):.3f}")
        # effective agreement: agree OR twin
        eff = agree.sum() + twin.sum()
        print(f"\n  EFFECTIVE agreement (same template OR twin): "
              f"{eff}/{len(ic)}  ({100*eff/len(ic):.1f}%)")

    # time offset of matched pairs (the filter group delay)
    dt = c_t[ic] - k_t[iks]
    print(f"\ntime offset (C - KS) of matched pairs:")
    print(f"  median={np.median(dt):.1f}, mean={dt.mean():.2f} samples")
    vals, counts = np.unique(dt, return_counts=True)
    for v, n in sorted(zip(vals, counts), key=lambda x: -x[1])[:5]:
        print(f"    offset {v:+d}: {n} pairs ({100*n/len(dt):.1f}%)")


if __name__ == "__main__":
    main()
