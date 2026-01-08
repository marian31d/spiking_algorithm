#!/usr/bin/env python3
"""
Compute peak-to-trough spike amplitude (µV) per cluster from Kilosort outputs.

Usage:
    python cluster_peak_to_trough_uV.py /path/to/kilosort_output
"""

import sys
from pathlib import Path
import numpy as np

# -------- HARDWARE CONSTANT --------
UV_PER_BIT = 0.195   # µV per ADC count
# ----------------------------------


def load_first_existing(base, names):
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python cluster_peak_to_trough_uV.py <kilosort_results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path.cwd() / "cluster_peak_to_trough_uV.tsv"

    templates = np.load(results_dir / "templates.npy")   # (K, T, C)
    spike_clusters = np.load(results_dir / "spike_clusters.npy")

    # Whitening inverse
    W_inv = None
    p_inv = load_first_existing(results_dir, ["whitening_mat_inv.npy", "whitening_matrix_inv.npy"])
    p_w = load_first_existing(results_dir, ["whitening_mat.npy", "whitening_matrix.npy"])

    if p_inv is not None:
        W_inv = np.load(p_inv)
    elif p_w is not None:
        W = np.load(p_w)
        W_inv = np.linalg.inv(W)

    K, T, C = templates.shape

    clusters = np.unique(spike_clusters)

    with open(out_path, "w") as f:
        f.write(
            "cluster_id\t"
            "template_id\t"
            "best_channel\t"
            "trough_uV\t"
            "peak_uV\t"
            "peak_to_trough_uV\n"
        )

        for k in clusters:
            if k < 0 or k >= K:
                continue

            tpl = templates[k].astype(np.float32)

            # unwhiten
            if W_inv is not None:
                tpl = tpl @ W_inv.T

            # convert to µV
            tpl_uV = tpl * UV_PER_BIT

            # best channel = max peak-to-trough
            p2p_per_ch = np.ptp(tpl_uV, axis=0)
            ch = int(np.argmax(p2p_per_ch))

            waveform = tpl_uV[:, ch]
            trough = float(np.min(waveform))
            peak = float(np.max(waveform))
            p2t = peak - trough

            f.write(
                f"{k}\t"
                f"{k}\t"
                f"{ch}\t"
                f"{trough:.3f}\t"
                f"{peak:.3f}\t"
                f"{p2t:.3f}\n"
            )

    print(f"Saved peak-to-trough amplitudes to:\n{out_path.resolve()}")


if __name__ == "__main__":
    main()
