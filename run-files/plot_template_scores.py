"""Plot each template's matched-filter score over time, and report summary stats.

Reads `spike_times.npy`, `spike_clusters.npy`, and `template_scores.npy` from a
Kilosort4 results directory (`template_scores.npy` is the raw score compared
against `Th**2` in `template_matching.run_matching`'s cnd1/cnd2 -- see
`io.save_to_phy`). For each template/cluster, saves a column plot (one vertical
line per spike, from 0 to its score) of score vs. spike time, and reports the
median and 5th-percentile score.

Usage
-----
    python plot_template_scores.py <results_dir> [--out OUT_DIR]
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from kilosort.io import load_ops


def load_data(results_dir):
    results_dir = Path(results_dir)
    spike_times = np.load(results_dir / 'spike_times.npy')
    spike_clusters = np.load(results_dir / 'spike_clusters.npy')
    scores_path = results_dir / 'template_scores.npy'
    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. This file is written by the patched "
            "`io.save_to_phy`, so results sorted before that change won't "
            "have it -- re-run kilosort on this dataset first."
            )
    scores = np.load(scores_path)

    ops = load_ops(results_dir / 'ops.npy', device=torch.device('cpu'))
    fs = ops['fs']

    return spike_times / fs, spike_clusters, scores


def summarize(spike_times, spike_clusters, scores):
    rows = []
    for cl in np.unique(spike_clusters):
        mask = spike_clusters == cl
        cs = scores[mask]
        rows.append({
            'template': int(cl),
            'n_spikes': int(mask.sum()),
            'median_score': float(np.median(cs)),
            'p5_score': float(np.percentile(cs, 5)),
            'min_score': float(cs.min()),
        })
    return rows


def plot_template(cl, t, s, median, p5, minimum, out_dir):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.vlines(t, 0, s, linewidth=0.5, alpha=0.4)
    ax.axhline(median, color='C1', linestyle='--', linewidth=1,
               label=f'median = {median:.3g}')
    ax.axhline(p5, color='C3', linestyle=':', linewidth=1,
               label=f'5th pct = {p5:.3g}')
    ax.axhline(minimum, color='C4', linestyle='-.', linewidth=1,
               label=f'min = {minimum:.3g}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Score')
    ax.set_ylim(bottom=0)
    ax.set_title(f'Template {cl} (n={len(s)} spikes)')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_dir / f'template_{cl:04d}_score.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', help='Path to kilosort4 results directory')
    parser.add_argument('--out', default=None,
                        help='Output directory for plots/summary '
                             '(default: <results_dir>/score_plots)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out) if args.out else results_dir / 'score_plots'
    out_dir.mkdir(exist_ok=True, parents=True)

    t, spike_clusters, scores = load_data(results_dir)
    summary = summarize(t, spike_clusters, scores)

    for row in summary:
        mask = spike_clusters == row['template']
        plot_template(
            row['template'], t[mask], scores[mask],
            row['median_score'], row['p5_score'], row['min_score'], out_dir
            )

    csv_path = out_dir / 'score_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['template', 'n_spikes', 'median_score', 'p5_score', 'min_score'])
        writer.writeheader()
        writer.writerows(summary)

    print(f'{len(summary)} templates')
    print(f'{"template":>8}  {"n_spikes":>8}  {"median":>10}  {"p5":>10}  {"min":>10}')
    for row in summary:
        print(f'{row["template"]:>8}  {row["n_spikes"]:>8}  '
              f'{row["median_score"]:>10.3g}  {row["p5_score"]:>10.3g}  '
              f'{row["min_score"]:>10.3g}')
    print(f'\nPlots + {csv_path.name} saved to {out_dir}')


if __name__ == '__main__':
    main()
