import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path
from compare_templates_timewindow import match_spike_times_tol, build_overlap


def spikes_by_shank(ops, Wall, shank, clu):
    probe = ops.get('probe', None)
    if probe is None:
        print("Warning: probe not found in ops.")
        return None, None
    chan_shank = (np.array(probe['kcoords']) == shank)   # channels in this shank
    K, NCHAN, NPCS = Wall.shape
    xc = np.asarray(ops['xc'], dtype=np.float64)
    yc = np.asarray(ops['yc'], dtype=np.float64)
    assert xc.shape[0] == NCHAN, f"xc has {xc.shape[0]} chans, Wall has {NCHAN}"
    nC = int(ops['settings']['nearest_chans'])           # KS default ~10

    energy = (Wall**2).sum(axis=2)        # (K, NCHAN)
    peak_ch = energy.argmax(axis=1)       # (K,)
    d2 = (xc[:, None] - xc[None, :])**2 + (yc[:, None] - yc[None, :])**2  # (NCHAN, NCHAN)

    Wall_masked = np.zeros_like(Wall)
    for k in range(K):
        near = np.argsort(d2[peak_ch[k]])[:nC]
        Wall_masked[k, near, :] = Wall[k, near, :]

    chan_mask = np.any(Wall_masked != 0, axis=2)         # (K, NCHAN) kept channels per template
    filtered_templates = np.where(np.any(chan_mask[:, chan_shank], axis=1))[0]
    filtered_spikes = np.isin(clu, filtered_templates)
    print(f"Shank {shank}: {len(filtered_templates)} templates, "
          f"{filtered_spikes.sum()} spikes after filtering.")
    return filtered_spikes, filtered_templates


def interval_scores(time, ops, gui_times, gui_clu, other_times, other_clu, tol=0):
    """
    Partition the recording into equal intervals of `time` seconds and, per interval,
    compare the GUI sorting against ONE other sorting:
      - coverage: % of GUI spikes in the interval that matched an 'other' spike
      - min_norm: overlap / min(|GUI|,|other|), spike-weighted best per GUI cluster
    Returns (coverage, min_norm, matched, gui_n, edges, centers).
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
        coverage[i] = 100.0 * i_gui.size / gt.size

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
            color='C2', label='GUI vs baseline')
    ax1.bar(centers + off, cov_f, width=bw, align='center',
            color='C0', label='GUI vs flow')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_ylim(0, 100)
    ax1.set_title('Spike coverage per interval (matched / GUI spikes)')
    ax1.legend(fontsize=8)

    ax2.bar(centers - off, mn_b, width=bw, align='center',
            color='C2', label='GUI vs baseline')
    ax2.bar(centers + off, mn_f, width=bw, align='center',
            color='C0', label='GUI vs flow')
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
    d = Path(d)
    times = np.load(d / "spike_times.npy").astype(np.int64, copy=False).reshape(-1)
    clu   = np.load(d / "spike_clusters.npy").astype(np.int64, copy=False).reshape(-1)
    return times, clu


def main():
    # Parameters
    time = 3600
    tol = 0
    out_dir = Path("/data/results/interval_plots")   # where the 3 PNGs go
    out_dir.mkdir(parents=True, exist_ok=True)

    # one entry per dataset: gui / base / flow directories
    datasets = {
        "mS51": {
            "gui":  Path("/data/results/mS51_res/IIR/matching_pursuit/2peels/2ndOrd-filt"),
            "base": Path("/data/results/mS51_res/IIR/time/20400/2ndOrd-filt"),
            "flow": Path("/data/results/mS51_res/IIR/time/3600/2ndOrd-filt"),
        },
        "mP79": {
            "gui":  Path("/data/results/mP79_res/IIR/matching_pursuit/2peels/2ndOrd-filt"),
            "base": Path("/data/results/mP79_res/IIR/time/12600/2ndOrd-filt"),
            "flow": Path("/data/results/mP79_res/IIR/time/3600/2ndOrd-filt"),
        },
        "mDS2": {
            "gui":  Path("/data/results/mDS2_res/IIR/matching_pursuit/2peels/2ndOrd-filt"),
            "base": Path("/data/results/mDS2_res/IIR/time/13800/2ndOrd-filt"),
            "flow": Path("/data/results/mDS2_res/IIR/baseline/2ndOrd-filt"),
        },
    }

    for name, dirs in datasets.items():
        print(f"\n=== {name} ===")
        # ops per dataset (fs / Nbatches / batch_size), taken from that dataset's GUI dir
        ops = np.load(os.path.join(dirs["gui"], "ops.npy"), allow_pickle=True).item()

        gui_times,  gui_clu  = load_sorting(dirs["gui"])
        base_times, base_clu = load_sorting(dirs["base"])
        flow_times, flow_clu = load_sorting(dirs["flow"])

        compare_spikes_by_interval(
            time, ops, gui_times, gui_clu,
            base_times, base_clu, flow_times, flow_clu,
            out_dir=out_dir, tol=tol, name=name,
        )


if __name__ == "__main__":
    main()