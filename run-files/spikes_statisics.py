import os
import csv
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from compare_templates_timewindow import match_spike_times_tol, build_overlap


def time_window(times, clu, t_start=0, t_end=-1, fs=20000, label=""):
    """
    Keep only the spikes with t_start <= time < t_end, as in
    compare_templates_timewindow.py (--t_start / --t_end).

    t_start, t_end: seconds; t_end <= 0 means no end. With t_start 0 and no end,
                    nothing is removed.
    fs:             sampling rate, to turn seconds into samples
    label:          if given, prints how many spikes were kept
    Returns (times, clu) of the kept spikes.
    """
    lo = int(t_start * fs)
    hi = int(t_end * fs)
    if lo < 0:
        raise ValueError("t_start must be >= 0 (seconds).")
    if hi > 0 and hi <= lo:
        raise ValueError("t_end must be > t_start (seconds).")
    if lo == 0 and hi <= 0:
        return times, clu
    keep = times >= lo
    if hi > 0:
        keep &= times < hi
    if label:
        print(f"  {label} spikes kept: {int(keep.sum())}/{times.size}")
    return times[keep], clu[keep]


def spikes_by_shank(ops, Wall, shank, clu, times):
    """
    Keep only the spikes whose cluster sits on one shank.

    Each template is assigned to the shank of its peak channel (the channel with
    the most energy in `Wall`, summed over PCs), using ops['probe']['kcoords'].
    Spikes are kept when their cluster's template peaks on `shank`. This needs
    uncurated Kilosort output, where cluster id == template row of `Wall`
    (asserted).

    ops:    Kilosort ops dict (needs 'probe' with 'kcoords', and 'xc'/'yc')
    Wall:   (n_templates, n_channels, n_pcs) template array (Wall.npy)
    shank:  shank id as in kcoords
    clu, times: spike_clusters and spike_times of the same sorting
    Returns (filtered_times, filtered_clu), or (None, None) if ops has no probe.
    """
    probe = ops.get('probe', None)
    if probe is None:
        print("Warning: probe not found in ops.")
        return None, None
    K, NCHAN, NPCS = Wall.shape
    xc = np.asarray(ops['xc'], dtype=np.float64)
    yc = np.asarray(ops['yc'], dtype=np.float64)
    assert xc.shape[0] == NCHAN, f"xc has {xc.shape[0]} chans, Wall has {NCHAN}"

    # guard: spike_clusters must map to template rows (uncurated KS output)
    assert clu.max() < K, (
        f"clu.max()={clu.max()} >= K={K}; spike_clusters don't map to Wall rows "
        f"(curated/merged sort?). isin filter would be wrong.")

    # peak channel per template (energy argmax; use ptp to match phy exactly)
    energy = (Wall**2).sum(axis=2)        # (K, NCHAN)
    peak_ch = energy.argmax(axis=1)       # (K,)
    # peak_ch = (Wall.max(axis=2) - Wall.min(axis=2)).argmax(axis=1)  # phy-style ptp

    # assign each template to the shank of its PEAK channel (single-valued)
    kcoords = np.asarray(probe['kcoords'])
    peak_shank = kcoords[peak_ch]                          # (K,) shank per template
    filtered_templates = np.where(peak_shank == shank)[0]

    keep = np.isin(clu, filtered_templates)               # boolean mask over spikes
    filtered_times = times[keep]
    filtered_clu   = clu[keep]
    print(f"Shank {shank}: {len(filtered_templates)} templates, "
          f"{keep.sum()} spikes after filtering.\n")
    return filtered_times, filtered_clu

def compare_spikes_by_shank(ops, gui_times, gui_clu, shank_times, shank_clu, shank, out_dir, tol=0):
    """
    Compare the GUI sorting with one other sorting (e.g. a single-shank run) over
    the whole recording:
      - coverage: % of GUI spikes that have a partner in the other sorting
        (1:1 matching within +-tol samples, see match_spike_times_tol)
      - min_norm: overlap / min(|GUI|,|other|), best per GUI cluster,
        spike-weighted; cluster sizes count matched spikes only
    When the other sorting covers one shank only, restrict the GUI spikes to that
    shank first (spikes_by_shank); otherwise spikes on other shanks count as missed.

    `ops` and `out_dir` are not used; `shank` only labels the printout.
    Returns (coverage [%], min_norm [0-1], matched, gui_n).
    """
    gui_n = gui_times.size
    i_gui, i_shank = match_spike_times_tol(gui_times, shank_times, tol=max(tol, 0))
    matched = i_gui.size
    coverage = 100.0 * matched / gui_times.size

    M, _, _ = build_overlap(gui_clu[i_gui], shank_clu[i_shank])
    row = M.sum(axis=1, keepdims=True).astype(np.float32)  # |g|
    col = M.sum(axis=0, keepdims=True).astype(np.float32)  # |o|
    mn = M / np.maximum(np.minimum(row, col), 1.0)         # min_norm matrix
    best = mn.max(axis=1)                                  # best per GUI cluster
    w = M.sum(axis=1)                                      # spikes per GUI cluster
    min_norm = float((best * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    print(f"Shank {shank}: {matched} matched spikes, coverage={coverage:.2f}%, min_norm={min_norm:.4f}\n")
    return coverage, min_norm, matched, gui_n


def compare_shank_runs(ops, gui_times, gui_clu, gui_Wall, flow_dir, out_dir, tol=0, name="",
                       t_start=0, t_end=-1, maskflows=0):
    """
    Compare every single-shank run in `flow_dir` (its shank_<N> folders) with the
    whole-probe GUI run restricted to the same shank.

    For each shank_<N>, the GUI spikes are filtered to shank N with spikes_by_shank
    (each template assigned to the shank of its peak channel) and compared with that
    shank run by compare_spikes_by_shank (coverage and spike-weighted min_norm,
    matching within +-tol samples). N is the probe's kcoords value, the number
    Kilosort uses when it names the shank_<N> folder.

    A last row 'all' pools the shanks: total matched / total GUI spikes, and min_norm
    weighted by each shank's matched spikes (the same as one spike-weighted average
    over all GUI clusters, since every cluster belongs to one shank).
    Time window, as in compare_templates_timewindow: the GUI spikes are always cut
    to [t_start, t_end) seconds; each shank run is cut too only when maskflows=1.
    Matching and scores use the spikes in the window only. The total spike count of
    the GUI run and of all flow shank runs is printed before and after the cut.

    Saves out_dir / f"shank_runs_vs_whole_{name}.csv" and returns the rows. The
    *_spikes columns are the spikes used (in the window), *_total the whole run.
    """
    fs = ops['fs']
    windowed = t_start > 0 or t_end > 0
    shank_dirs = sorted(Path(flow_dir).glob("shank_*"), key=lambda p: int(p.name.split("_")[1]))
    rows = []
    for shank_dir in shank_dirs:
        shank = int(shank_dir.name.split("_")[1])
        gt_all, gc_all = spikes_by_shank(ops, gui_Wall, shank, gui_clu, gui_times)
        if gt_all is None:
            break
        gt, gc = time_window(gt_all, gc_all, t_start, t_end, fs,
                             label=f"GUI shank {shank}" if windowed else "")
        st_all, sc_all, _ = load_sorting(shank_dir)
        st, sc = st_all, sc_all
        if maskflows:
            st, sc = time_window(st_all, sc_all, t_start, t_end, fs,
                                 label=f"FLOW shank {shank}" if windowed else "")
        coverage, min_norm, matched, gui_n = compare_spikes_by_shank(
            ops, gt, gc, st, sc, shank, out_dir, tol=tol)
        rows.append({"dataset": name, "shank": shank,
                     "whole_run_total": int(gt_all.size), "whole_run_spikes": gui_n,
                     "shank_run_total": int(st_all.size), "shank_run_spikes": int(st.size),
                     "matched": matched,
                     "coverage_pct": round(coverage, 2), "min_norm": round(min_norm, 4)})

    if not rows:
        print(f"No shank_<N> folders compared in {flow_dir}")
        return rows

    # pooled over shanks
    n_gui = sum(r["whole_run_spikes"] for r in rows)
    n_match = sum(r["matched"] for r in rows)
    mn_all = sum(r["min_norm"] * r["matched"] for r in rows if np.isfinite(r["min_norm"])) / max(n_match, 1)
    rows.append({"dataset": name, "shank": "all",
                 "whole_run_total": sum(r["whole_run_total"] for r in rows), "whole_run_spikes": n_gui,
                 "shank_run_total": sum(r["shank_run_total"] for r in rows),
                 "shank_run_spikes": sum(r["shank_run_spikes"] for r in rows), "matched": n_match,
                 "coverage_pct": round(100.0 * n_match / max(n_gui, 1), 2), "min_norm": round(mn_all, 4)})

    csv_path = Path(out_dir) / f"shank_runs_vs_whole_{name}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # totals over the whole run and in the time window (as in compare_templates_timewindow);
    # the GUI sums over the compared shanks equal the whole run when every shank has a folder
    gui_in = time_window(gui_times, gui_clu, t_start, t_end, fs)[0].size
    gui_note = ("" if rows[-1]["whole_run_total"] == gui_times.size
                else f"; {rows[-1]['whole_run_total']} (in window: {n_gui}) on the compared shanks")
    flow_note = "" if maskflows or not windowed else ", not cut: maskflows=0"
    print(f"{name}: GUI total spikes:  {gui_times.size}  (in window: {gui_in}){gui_note}")
    print(f"{name}: FLOW total spikes (all shanks): {rows[-1]['shank_run_total']}  "
          f"(in window: {rows[-1]['shank_run_spikes']}{flow_note})")
    print(f"{name}: all shanks pooled: coverage={rows[-1]['coverage_pct']:.2f}%, "
          f"min_norm={rows[-1]['min_norm']:.4f}\nSaved: {csv_path}\n")
    return rows


def interval_scores(time, ops, gui_times, gui_clu, other_times, other_clu, tol=0):
    """
    Partition the recording into equal intervals of `time` seconds and, per interval,
    compare the GUI sorting against ONE other sorting:
      - coverage: % of GUI spikes in the interval that matched an 'other' spike
      - min_norm: overlap / min(|GUI|,|other|), spike-weighted best per GUI cluster
    Both sortings are cut to each interval before matching, so a match never
    crosses an interval boundary. The recording length is taken from
    ops['Nbatches'] * ops['batch_size'] and ops['fs'].

    time:  interval length in seconds
    tol:   matching tolerance in samples
    Returns (coverage, min_norm, matched, gui_n, edges, centers), one value per
    interval: coverage in % (NaN if either sorting has no spikes there), min_norm
    in 0-1 (NaN if nothing matched), matched / gui_n spike counts, and the
    interval edges and centers in seconds.
    """
    interval = int(round(time * ops['fs']))              # interval length in samples
    total_time = (ops['Nbatches'] * ops['batch_size'])   # total length in samples
    n_partition = int(np.ceil(total_time / interval))    # number of intervals

    coverage = np.full(n_partition, np.nan)
    min_norm = np.full(n_partition, np.nan)
    matched  = np.zeros(n_partition, dtype=np.int64)
    gui_n    = np.zeros(n_partition, dtype=np.int64)

    for i in range(n_partition):
        lo = interval * i
        hi = interval * (i + 1)

        # mask BOTH sortings to this interval so matches can't cross a boundary
        g_in = (gui_times >= lo) & (gui_times < hi)
        o_in = (other_times >= lo) & (other_times < hi)
        gt, gc = gui_times[g_in], gui_clu[g_in]
        ot, oc = other_times[o_in], other_clu[o_in]

        gui_n[i] = gt.size
        if gt.size == 0 or ot.size == 0:
            continue

        i_gui, i_oth = match_spike_times_tol(gt, ot, tol=max(tol, 0))
        matched[i] = i_gui.size
        coverage[i] = 100.0 * matched[i] / gt.size

        if i_gui.size > 0:
            M, _, _ = build_overlap(gc[i_gui], oc[i_oth])
            row = M.sum(axis=1, keepdims=True).astype(np.float32)  # |g|
            col = M.sum(axis=0, keepdims=True).astype(np.float32)  # |o|
            mn = M / np.maximum(np.minimum(row, col), 1.0)         # min_norm matrix
            best = mn.max(axis=1)                                  # best per GUI cluster
            w = M.sum(axis=1)                                      # spikes per GUI cluster
            min_norm[i] = float((best * w).sum() / w.sum()) if w.sum() > 0 else np.nan

    edges   = np.arange(n_partition + 1) * time
    centers = edges[:-1] + time / 2
    return coverage, min_norm, matched, gui_n, edges, centers


def compare_spikes_by_interval(time, ops, gui_times, gui_clu,
                               base_times, base_clu, flow_times, flow_clu,
                               out_dir, tol=0, name=""):
    """
    Compute per-interval coverage and min_norm for GUI-vs-baseline and GUI-vs-flow,
    then plot each metric as two grouped bars per interval.

    Runs interval_scores twice with the same GUI sorting: once against the
    baseline (base_times/base_clu) and once against the flow (flow_times/flow_clu).
    The figure has coverage (top, 0-100 %) and min_norm (bottom, 0-1); the bar
    labels are 'Baseline vs Max freeze duration' and f'Baseline vs {time}', so
    the flow sorting should be the run whose freeze_sec equals `time`.
    Saved as out_dir / f"interval_coverage_minnorm_{name}.png" (`name` is also
    the figure title).

    Returns a dict with the per-interval arrays: coverage_baseline, coverage_flow,
    min_norm_baseline, min_norm_flow, matched_baseline, matched_flow,
    gui_spikes, edges.
    """
    cov_b, mn_b, matched_b, gui_n, edges, centers = interval_scores(
        time, ops, gui_times, gui_clu, base_times, base_clu, tol=tol)
    cov_f, mn_f, matched_f, _, _, _ = interval_scores(
        time, ops, gui_times, gui_clu, flow_times, flow_clu, tol=tol)

    # grouped bars: two per interval, offset by half a bar width each
    bw = time * 0.35                 # bar width
    off = bw / 2                     # offset from interval center

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.bar(centers - off, cov_b, width=bw, align='center',
            color='C2', label='Baseline vs Max freeze duration')
    ax1.bar(centers + off, cov_f, width=bw, align='center',
            color='C0', label=f'Baseline vs {time}')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_ylim(0, 100)
    ax1.set_title('Spike coverage per interval (matched / GUI spikes)')
    ax1.legend(fontsize=8)

    ax2.bar(centers - off, mn_b, width=bw, align='center',
            color='C2', label='Baseline vs Max freeze duration')
    ax2.bar(centers + off, mn_f, width=bw, align='center',
            color='C0', label=f'Baseline vs {time}')
    ax2.set_ylabel('min_norm')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (s)')
    ax2.set_title('min_norm per interval')
    ax2.legend(fontsize=8)

    ax2.set_xticks(edges)
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.suptitle(name, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / f"interval_coverage_minnorm_{name}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "coverage_baseline": cov_b, "coverage_flow": cov_f,
        "min_norm_baseline": mn_b, "min_norm_flow": mn_f,
        "matched_baseline": matched_b, "matched_flow": matched_f,
        "gui_spikes": gui_n, "edges": edges,
    }

def load_sorting(d):
    """
    Load one Kilosort output folder.
    Returns (times, clu, Wall): spike_times and spike_clusters as flat int64
    arrays, and Wall.npy as float64 (n_templates, n_channels, n_pcs).
    """
    d = Path(d)
    times = np.load(d / "spike_times.npy").astype(np.int64, copy=False).reshape(-1)
    clu   = np.load(d / "spike_clusters.npy").astype(np.int64, copy=False).reshape(-1)
    Wall  = np.load(d / "Wall.npy").astype(np.float64, copy=False)
    return times, clu, Wall


def main():
    """
    Run one analysis for every dataset in `datasets`, chosen by the flags below
    (checked in this order, the first one set to True wins):
      - shank_runs: compare_shank_runs, every flow/shank_<N> run vs the GUI
        (whole-probe) run filtered to shank N, one CSV per dataset in `out_dir`
      - intervals: compare_spikes_by_interval, GUI vs baseline and GUI vs flow
        in intervals of `time` seconds, figures saved in `out_dir`
      - shanks:    spikes_by_shank, print how many GUI templates/spikes sit on
                   each shank
      - compare:   compare_spikes_by_shank, GUI vs the flow's shank_<N> folder
    Each dataset entry gives the GUI, baseline and flow folders and the shank;
    ops (fs, batch size, probe) are read from the GUI folder. `tol` is the
    matching tolerance in samples.

    Time window (as --t_start / --t_end / --maskflows in compare_templates_timewindow):
    only spikes with t_start <= time < t_end (seconds) are used. The GUI is always
    cut; the baseline and flow runs are cut too only when maskflows = 1.
    t_start = 0 and t_end = -1 use the whole recording.
    """
    # Parameters
    time = 5400
    tol = 0
    t_start = 5400      # seconds; drop spikes before this (e.g. 5400 for the module/5400 runs)
    t_end = -1       # seconds; > 0 drops spikes from here on, -1 = no end
    maskflows = 1    # 1: cut the baseline and flow runs to the same window as the GUI
    shank_runs = True   # each flow/shank_<N> run vs the GUI run filtered to shank N
    intervals = False
    shanks = False
    compare = False
    out_dir = Path("/data/results/interval_plots")   # where the 3 PNGs go
    out_dir.mkdir(parents=True, exist_ok=True)

    # one entry per dataset: gui / base / flow directories
    datasets = {
        "mS51_49": {
            "base":  Path("/data/results/mS51_res/IIR/updated_clu/time/20754"),
            "gui": Path("/data/results/mS51_res/IIR/updated_clu/baseline_det/1"),
            "flow": Path("/data/results/mS51_res/IIR/updated_clu/time/module/5400"),
            "shank": 8,#5,  # shank to analyze for this dataset
        },
        "mP79_17": {
            "base":  Path("/data/results/mP79_res/IIR/updated_clu/time/12824"),
            "gui": Path("/data/results/mP79_res/IIR/updated_clu/baseline_det/1"),
            "flow": Path("/data/results/mP79_res/IIR/updated_clu/time/module/5400"),
            "shank": 12,#1,  # shank to analyze for this dataset
        },
        "mDS2_02": {
            "base":  Path("/data/results/mds2_res/IIR/updated_clu/time/14051"),
            "gui": Path("/data/results/mds2_res/IIR/updated_clu/baseline_det/1"),
            "flow": Path("/data/results/mds2_res/IIR/updated_clu/time/module/5400"),
            "shank": 4,#3,  # shank to analyze for this dataset
        },
    }

    peels = [1, 2, 4, 7, 10, 20, 50, 70, 90]

    windowed = t_start > 0 or t_end > 0
    for name, dirs in datasets.items():
        if windowed:
            print(f"{name}: time window t_start={t_start} s, "
                  f"t_end={'end' if t_end <= 0 else f'{t_end} s'}, maskflows={maskflows}")
        if shank_runs:
            ops = np.load(os.path.join(dirs["gui"], "ops.npy"), allow_pickle=True).item()
            gui_times, gui_clu, gui_Wall = load_sorting(dirs["gui"])
            # the GUI is cut to the window inside compare_shank_runs, so it can count both
            compare_shank_runs(ops, gui_times, gui_clu, gui_Wall, dirs["flow"],
                               out_dir, tol=tol, name=name,
                               t_start=t_start, t_end=t_end, maskflows=maskflows)
            continue

        if intervals:
            X = 1
        else:
            X = dirs['shank']
        for shank in range(X):
            gui_dir = dirs["gui"]
            shank_dir = dirs["flow"] / f"shank_{dirs['shank']}"
            # ops per dataset (fs / Nbatches / batch_size), taken from that dataset's GUI dir
            ops = np.load(os.path.join(gui_dir, "ops.npy"), allow_pickle=True).item()
            
            gui_times,  gui_clu,  gui_Wall  = load_sorting(dirs["gui"])
            base_times, base_clu, base_Wall = load_sorting(dirs["base"])
            flow_times, flow_clu, flow_Wall = load_sorting(shank_dir) #dirs["flow"]
            fs = ops['fs']
            gui_times, gui_clu = time_window(gui_times, gui_clu, t_start, t_end, fs,
                                             label="GUI" if windowed else "")
            if maskflows:
                base_times, base_clu = time_window(base_times, base_clu, t_start, t_end, fs,
                                                   label="BASE" if windowed else "")
                flow_times, flow_clu = time_window(flow_times, flow_clu, t_start, t_end, fs,
                                                   label="FLOW" if windowed else "")
            print(f"\n=== {name} === shank={shank} === spikes={len(flow_times)} ===\n")

            if intervals:
                compare_spikes_by_interval(
                    time, ops, gui_times, gui_clu,
                    base_times, base_clu, flow_times, flow_clu,
                    out_dir=out_dir, tol=tol, name=name,
                )
            elif shanks:
                #print(ops['probe']['kcoords'])
                spikes_by_shank(ops, gui_Wall, shank, gui_clu, gui_times)
            elif compare:
                compare_spikes_by_shank(ops, gui_times, gui_clu, flow_times, flow_clu, dirs['shank'], out_dir, tol=0)
            else:
                print("No comparison mode selected. Set intervals=True or shanks=True or compare=True.")

if __name__ == "__main__":
    main()