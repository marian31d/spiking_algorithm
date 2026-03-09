#!/usr/bin/env python3
"""
Direct template-to-template comparison: FLOWS offline (Wall.npy) vs Kilosort GUI (templates.npy).

This bypasses spike matching entirely and asks:
  "For each FLOWS template, which GUI template is most similar, and how similar is it?"

Shapes expected:
  Wall.npy      : (K_flow, n_pcs, n_chan)   — FLOWS offline PC-projected templates
  templates.npy : (K_gui,  n_time, n_chan)  — GUI Phy-style waveform templates

Because the two are in different spaces (PC-projected vs raw waveform), we compare them
in TWO ways:

  1. CHANNEL-POWER comparison (recommended):
     Collapse time/PC dimension → per-channel RMS power vector → cosine similarity.
     Works across spaces. Tells you: "do the same channels carry signal for this template?"

  2. RAW COSINE on flattened vectors (sanity check only):
     Only meaningful if Wall and templates are in the same space — they usually are NOT,
     so treat this as a supplementary check.

Run:
  python compare_templates_direct.py \
    --wall    /path/to/flows/offline/Wall.npy \
    --gui_tpl /path/to/gui/phy/templates.npy \
    --out     /path/to/output_dir \
    [--top_k 5]                    # how many best matches to report per template

Outputs:
  template_match_summary.txt   — human-readable report
  template_match_scores.csv    — full (K_flow × K_gui) best-match table
  heatmap_channel_power_sim.png
  heatmap_raw_cosine_sim.png   (if shapes allow)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Cosine similarity matrix
# ─────────────────────────────────────────────
def cosine_sim(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """A: (M, D), B: (N, D) → S: (M, N)"""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), eps)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), eps)
    return A @ B.T


# ─────────────────────────────────────────────
# Channel-power vectors
# ─────────────────────────────────────────────
def channel_power(T: np.ndarray) -> np.ndarray:
    """
    T: (K, D1, n_chan)  →  P: (K, n_chan)
    RMS power per channel, collapsed over D1 (time or n_pcs).
    """
    return np.sqrt((T.astype(np.float32) ** 2).mean(axis=1))  # (K, n_chan)


# ─────────────────────────────────────────────
# Heatmap helper
# ─────────────────────────────────────────────
def save_heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str, path: Path):
    fig, ax = plt.subplots(figsize=(max(8, mat.shape[1] // 4), max(6, mat.shape[0] // 4)))
    im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall",    required=True, type=str,
                    help="Path to FLOWS offline Wall.npy  (K_flow, n_pcs, n_chan)")
    ap.add_argument("--gui_tpl", required=True, type=str,
                    help="Path to GUI templates.npy        (K_gui, n_time, n_chan)")
    ap.add_argument("--out",     default=".", type=str)
    ap.add_argument("--top_k",   default=5,   type=int,
                    help="Report top-K GUI matches per FLOWS template")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────
    print("Loading Wall.npy ...")
    Wall = np.load(args.wall)       # (K_flow, n_pcs, n_chan)
    print(f"  Wall shape:     {Wall.shape}  — expected (K_flow, n_pcs, n_chan)")

    print("Loading templates.npy ...")
    GUI_T = np.load(args.gui_tpl)   # (K_gui, n_time, n_chan)
    print(f"  GUI tpl shape:  {GUI_T.shape}  — expected (K_gui, n_time, n_chan)")

    # Validate last dim matches (n_chan must agree)
    if Wall.ndim != 3 or GUI_T.ndim != 3:
        raise ValueError(f"Both arrays must be 3-D. Got Wall:{Wall.ndim}D, GUI:{GUI_T.ndim}D")

    n_chan_wall = Wall.shape[2]
    n_chan_gui  = GUI_T.shape[2]
    if n_chan_wall != n_chan_gui:
        print(f"\n⚠️  Channel count mismatch: Wall has {n_chan_wall} channels, GUI has {n_chan_gui}.")
        print("   This can happen if Kilosort masked bad channels before saving templates.")
        print("   Truncating to the smaller channel count for comparison.")
        n_chan = min(n_chan_wall, n_chan_gui)
        Wall  = Wall[:, :, :n_chan]
        GUI_T = GUI_T[:, :, :n_chan]
    else:
        n_chan = n_chan_wall

    K_flow = Wall.shape[0]
    K_gui  = GUI_T.shape[0]
    print(f"\n  K_flow={K_flow},  K_gui={K_gui},  n_chan={n_chan}")

    # ── Method 1: Channel-power cosine similarity ──
    print("\nComputing channel-power similarity (recommended) ...")
    P_flow = channel_power(Wall)    # (K_flow, n_chan)
    P_gui  = channel_power(GUI_T)   # (K_gui,  n_chan)
    sim_ch = cosine_sim(P_flow, P_gui)  # (K_flow, K_gui)
    print(f"  Similarity matrix shape: {sim_ch.shape}")
    print(f"  Mean max similarity (per FLOW template): {sim_ch.max(axis=1).mean():.4f}")

    # ── Method 2: Raw flattened cosine (same-space sanity check) ──
    flat_flow = Wall.reshape(K_flow, -1)    # (K_flow, n_pcs * n_chan)
    flat_gui  = GUI_T.reshape(K_gui,  -1)   # (K_gui,  n_time * n_chan)

    sim_raw = None
    if flat_flow.shape[1] == flat_gui.shape[1]:
        print("\nComputing raw flattened cosine similarity (same-space check) ...")
        sim_raw = cosine_sim(flat_flow, flat_gui)
        print(f"  Mean max similarity (per FLOW template): {sim_raw.max(axis=1).mean():.4f}")
    else:
        print(f"\nSkipping raw cosine (shapes differ: {flat_flow.shape[1]} vs {flat_gui.shape[1]} features)")
        print("  This is expected — Wall (PC-space) and templates (waveform-space) are different dims.")

    # ── Build match table ──────────────────────
    print("\nBuilding best-match table ...")
    rows = []
    for i in range(K_flow):
        ch_scores = sim_ch[i]
        top_idx   = np.argsort(ch_scores)[::-1][:args.top_k]

        row = {"FLOW_template": i}
        for rank, j in enumerate(top_idx):
            row[f"GUI_match_{rank+1}"]      = int(j)
            row[f"chan_power_sim_{rank+1}"]  = float(ch_scores[j])
            if sim_raw is not None:
                row[f"raw_cosine_sim_{rank+1}"] = float(sim_raw[i, j])

        row["best_GUI_template"]       = int(top_idx[0])
        row["best_chan_power_sim"]     = float(ch_scores[top_idx[0]])
        row["best_chan_power_sim_pct"] = float(100 * 0.5 * (ch_scores[top_idx[0]] + 1))  # [-1,1]→[0,100]

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "template_match_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Heatmaps ──────────────────────────────
    print("\nGenerating heatmaps ...")
    save_heatmap(
        sim_ch,
        title="Channel-power cosine similarity: FLOWS Wall vs GUI templates",
        xlabel=f"GUI templates (K_gui={K_gui})",
        ylabel=f"FLOWS templates (K_flow={K_flow})",
        path=out_dir / "heatmap_channel_power_sim.png"
    )
    if sim_raw is not None:
        save_heatmap(
            sim_raw,
            title="Raw flattened cosine similarity: FLOWS Wall vs GUI templates",
            xlabel=f"GUI templates (K_gui={K_gui})",
            ylabel=f"FLOWS templates (K_flow={K_flow})",
            path=out_dir / "heatmap_raw_cosine_sim.png"
        )

    # ── Summary report ─────────────────────────
    best_sims = sim_ch.max(axis=1)
    thresholds = [0.9, 0.8, 0.7, 0.5]

    lines = []
    lines.append("=" * 60)
    lines.append("FLOWS vs GUI — Direct Template Comparison Summary")
    lines.append("=" * 60)
    lines.append(f"Wall.npy:      {args.wall}  → shape {Wall.shape}")
    lines.append(f"templates.npy: {args.gui_tpl}  → shape {GUI_T.shape}")
    lines.append(f"n_channels compared: {n_chan}")
    lines.append(f"K_flow={K_flow},  K_gui={K_gui}")
    lines.append("")
    lines.append("Channel-power cosine similarity (FLOW → best GUI match):")
    lines.append(f"  Mean:    {best_sims.mean():.4f}")
    lines.append(f"  Median:  {np.median(best_sims):.4f}")
    lines.append(f"  Min:     {best_sims.min():.4f}")
    lines.append(f"  Max:     {best_sims.max():.4f}")
    lines.append("")
    lines.append("Fraction of FLOW templates with best match ≥ threshold:")
    for thr in thresholds:
        frac = (best_sims >= thr).mean()
        lines.append(f"  ≥ {thr:.2f} : {100*frac:.1f}%  ({int((best_sims>=thr).sum())}/{K_flow})")
    lines.append("")

    if sim_raw is not None:
        best_raw = sim_raw.max(axis=1)
        lines.append("Raw flattened cosine similarity (FLOW → best GUI match):")
        lines.append(f"  Mean:    {best_raw.mean():.4f}")
        lines.append(f"  Median:  {np.median(best_raw):.4f}")
        lines.append("")

    lines.append(f"Outputs written to: {out_dir}")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    (out_dir / "template_match_summary.txt").write_text(summary_text)
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
