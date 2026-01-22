#!/usr/bin/env python3
"""
GUI vs FLOW comparison using spike overlap-derived scores.

Pairwise overlap matrix (within matched spikes):
  M[g,f] = #matched spikes assigned to GUI cluster g AND FLOW cluster f

Cluster sizes (within matched spikes):
  |g| = sum_f M[g,f]
  |f| = sum_g M[g,f]

Scores per pair (g,f):
  precision(g,f) = overlap / |f|
  recall(g,f)    = overlap / |g|
  f1(g,f)        = 2*overlap / (|g| + |f|)
  min_norm(g,f)  = overlap / min(|g|, |f|)      <-- what you asked for

Features:
- Spike-time matching (±tol samples)
- Overlap matrix between GUI clusters and FLOW clusters
- Score matrices for precision/recall/f1/min_norm
- Hungarian 1:1 mapping (maximizes raw overlap counts)
- Per-GUI best match by EACH score (no 1:1 constraint)
- Summary reports NON-1:1 best-match stats for:
    * all clusters
    * good-only clusters (based on TSV labels)
  for BOTH directions: GUI→FLOW and FLOW→GUI
  as percentages: mean, spike-weighted mean, sqrt(spike)-weighted mean
- Heatmaps: row-normalized overlap + precision + recall + f1 + min_norm

Run:
python compare_gui_flow_scores.py \
  --gui_phy  /path/to/gui/phy \
  --flow_phy /path/to/flow/phy \
  --tol 0 \
  --out /path/to/outdir
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Optional labels (good/mua/etc.)
# -----------------------------
def load_labels(phy_dir: str):
    phy_dir = os.path.abspath(phy_dir)
    candidates = [
        ("cluster_KSLabel.tsv", "KSLabel"),
        ("cluster_group.tsv", "group"),
    ]
    for fname, col in candidates:
        path = os.path.join(phy_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, sep="\t")
            if "cluster_id" in df.columns and col in df.columns:
                labels = dict(
                    zip(df["cluster_id"].astype(int).values,
                        df[col].astype(str).str.lower().values)
                )
                return labels, fname
        except Exception:
            pass
    return {}, None


# -----------------------------
# Spike-time matching
# -----------------------------
def match_spike_times_exact(t1: np.ndarray, t2: np.ndarray):
    s1 = np.argsort(t1)
    s2 = np.argsort(t2)
    t1s, t2s = t1[s1], t2[s2]
    _, i1s, i2s = np.intersect1d(t1s, t2s, return_indices=True)
    return s1[i1s], s2[i2s]


def match_spike_times_tol(t1: np.ndarray, t2: np.ndarray, tol: int):
    s1 = np.argsort(t1)
    s2 = np.argsort(t2)
    a = t1[s1]
    b = t2[s2]
    i = j = 0
    m1, m2 = [], []
    while i < len(a) and j < len(b):
        dt = a[i] - b[j]
        if abs(dt) <= tol:
            m1.append(s1[i])
            m2.append(s2[j])
            i += 1
            j += 1
        elif dt < -tol:
            i += 1
        else:
            j += 1
    return np.array(m1, dtype=np.int64), np.array(m2, dtype=np.int64)


# -----------------------------
# Overlap matrix
# -----------------------------
def build_overlap(gui_clu, flow_clu):
    gui_ids = np.unique(gui_clu)
    flow_ids = np.unique(flow_clu)
    gi = {int(cid): idx for idx, cid in enumerate(gui_ids)}
    fi = {int(cid): idx for idx, cid in enumerate(flow_ids)}
    M = np.zeros((len(gui_ids), len(flow_ids)), dtype=np.int64)
    for g, f in zip(gui_clu, flow_clu):
        M[gi[int(g)], fi[int(f)]] += 1
    return M, gui_ids.astype(int), flow_ids.astype(int)


# -----------------------------
# Template similarity (cluster-template cosine similarity)
# -----------------------------
def _load_templates_and_spike_templates(phy_dir: Path):
    """
    Loads:
      - templates.npy: (n_templates, n_time, n_channels) or similar
      - spike_templates.npy: (n_spikes,) template index per spike
    Returns (templates, spike_templates) or (None, None) if missing.
    """
    t_path = phy_dir / "templates.npy"
    st_path = phy_dir / "spike_templates.npy"
    if (not t_path.exists()) or (not st_path.exists()):
        return None, None
    try:
        templates = np.load(t_path)  # usually float32
        spike_templates = np.load(st_path).astype(np.int64, copy=False).reshape(-1)
        return templates, spike_templates
    except Exception:
        return None, None


def build_cluster_templates_from_matched_spikes(
    templates: np.ndarray,
    spike_templates: np.ndarray,
    matched_spike_idx: np.ndarray,
    matched_cluster_ids: np.ndarray,
    cluster_ids: np.ndarray,
):
    """
    Build one representative template vector per cluster_id, using only matched spikes.

    templates:        (n_templates, ...) waveform tensor
    spike_templates:  (n_spikes,) template index per spike
    matched_spike_idx indices into spike_templates for the matched spikes
    matched_cluster_ids cluster assignment for those matched spikes (same length as matched_spike_idx)
    cluster_ids:      list of unique clusters to return in this order

    Returns:
      C: (n_clusters, n_features) float32
    """
    # Flatten each KS template to a vector
    T = templates.reshape(templates.shape[0], -1).astype(np.float32, copy=False)

    # template index for each matched spike
    mt = spike_templates[matched_spike_idx]  # (n_matched,)
    mc = matched_cluster_ids.astype(np.int64, copy=False)  # (n_matched,)

    # map cluster_id -> row index
    cid_to_row = {int(cid): i for i, cid in enumerate(cluster_ids)}
    C = np.zeros((len(cluster_ids), T.shape[1]), dtype=np.float32)

    # accumulate weighted sums per cluster by counting templates used
    # (fast approach: for each cluster, histogram of template IDs among matched spikes)
    for cid in cluster_ids:
        cid_int = int(cid)
        row = cid_to_row[cid_int]
        mask = (mc == cid_int)
        if not np.any(mask):
            continue
        t_ids = mt[mask]
        # counts per template id
        uniq, cnt = np.unique(t_ids, return_counts=True)
        # weighted sum of templates
        # (ensure indices are in range)
        valid = (uniq >= 0) & (uniq < T.shape[0])
        uniq = uniq[valid]
        cnt = cnt[valid].astype(np.float32, copy=False)
        if uniq.size == 0:
            continue
        C[row, :] = (T[uniq, :] * cnt[:, None]).sum(axis=0) / float(cnt.sum())

    return C


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-12):
    """
    A: (G, D), B: (F, D)  ->  S: (G, F)
    """
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    An = np.linalg.norm(A, axis=1, keepdims=True)
    Bn = np.linalg.norm(B, axis=1, keepdims=True)
    A_unit = A / np.maximum(An, eps)
    B_unit = B / np.maximum(Bn, eps)
    return A_unit @ B_unit.T



# -----------------------------
# Hungarian mapping (global 1:1) maximizing overlap counts
# -----------------------------
def hungarian_maximize_overlap(M: np.ndarray):
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-M)
        pairs = [(int(rr), int(cc), int(M[rr, cc])) for rr, cc in zip(r, c)]
        pairs = [p for p in pairs if p[2] > 0]
        return pairs, "hungarian"
    except Exception:
        # Greedy fallback
        M2 = M.copy()
        pairs = []
        used_r, used_c = set(), set()
        while True:
            rr, cc = np.unravel_index(np.argmax(M2), M2.shape)
            best = int(M2[rr, cc])
            if best <= 0:
                break
            if rr in used_r or cc in used_c:
                M2[rr, cc] = -1
                continue
            pairs.append((int(rr), int(cc), best))
            used_r.add(rr)
            used_c.add(cc)
            M2[rr, :] = -1
            M2[:, cc] = -1
        return pairs, "greedy"


# -----------------------------
# Best-match stats (optionally filtered)
# -----------------------------
def best_match_stats(score: np.ndarray, M: np.ndarray, ids: np.ndarray, keep_set: set | None, axis: int):
    """
    score: (G x F) matrix
    M:     (G x F) overlap counts
    ids:   GUI ids if axis==1, FLOW ids if axis==0
    keep_set: if not None, only clusters whose id in keep_set are included
    axis:
      1 => GUI→FLOW (best per GUI row)
      0 => FLOW→GUI (best per FLOW col)
    """
    if keep_set is None:
        keep = np.ones(len(ids), dtype=bool)
    else:
        keep = np.array([int(cid) in keep_set for cid in ids], dtype=bool)

    if keep.sum() == 0:
        return {
            "n_considered": 0,
            "mean_best": 0.0,
            "spike_weighted_mean_best": 0.0,
            "sqrt_spike_weighted_mean_best": 0.0,
        }

    if axis == 1:
        # GUI → FLOW
        score_sub = score[keep, :]
        M_sub = M[keep, :]
        best = score_sub.max(axis=1).astype(np.float64, copy=False)
        spikes = M_sub.sum(axis=1).astype(np.float64, copy=False)
    elif axis == 0:
        # FLOW → GUI
        score_sub = score[:, keep]
        M_sub = M[:, keep]
        best = score_sub.max(axis=0).astype(np.float64, copy=False)
        spikes = M_sub.sum(axis=0).astype(np.float64, copy=False)
    else:
        raise ValueError("axis must be 0 or 1")

    mean = float(best.mean()) if best.size else 0.0
    sw_sum = float(spikes.sum())
    spike_weighted = float((best * spikes).sum() / sw_sum) if sw_sum > 0 else 0.0

    sqrt_w = np.sqrt(spikes)
    sqrt_sum = float(sqrt_w.sum())
    sqrt_weighted = float((best * sqrt_w).sum() / sqrt_sum) if sqrt_sum > 0 else 0.0

    return {
        "n_considered": int(best.size),
        "mean_best": mean,
        "spike_weighted_mean_best": spike_weighted,
        "sqrt_spike_weighted_mean_best": sqrt_weighted,
    }


# -----------------------------
# Heatmap helper
# -----------------------------
def save_heatmap(mat: np.ndarray, title: str, path: Path, cbar_label: str):
    plt.figure(figsize=(12, 10))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.xlabel("FLOW clusters (columns)")
    plt.ylabel("GUI clusters (rows)")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui_phy", required=True, type=str)
    ap.add_argument("--flow_phy", required=True, type=str)
    ap.add_argument("--tol", default=0, type=int)
    ap.add_argument("--out", default=".", type=str)
    args = ap.parse_args()

    gui_phy = Path(args.gui_phy).resolve()
    flow_phy = Path(args.flow_phy).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reading spike times/clusters...")
    gui_times = np.load(gui_phy / "spike_times.npy").astype(np.int64, copy=False).reshape(-1)
    gui_clu   = np.load(gui_phy / "spike_clusters.npy").astype(np.int64, copy=False).reshape(-1)
    flow_times = np.load(flow_phy / "spike_times.npy").astype(np.int64, copy=False).reshape(-1)
    flow_clu   = np.load(flow_phy / "spike_clusters.npy").astype(np.int64, copy=False).reshape(-1)

    print("Reading templates for template similarity (if available)...")
    gui_templates, gui_spike_templates = _load_templates_and_spike_templates(gui_phy)
    flow_templates, flow_spike_templates = _load_templates_and_spike_templates(flow_phy)

    have_templates = (gui_templates is not None) and (flow_templates is not None)
    if not have_templates:
        print("  (templates.npy / spike_templates.npy missing in GUI or FLOW; template similarity will be skipped)")



    if gui_times.shape[0] != gui_clu.shape[0]:
        raise RuntimeError("GUI spike_times/spike_clusters length mismatch.")
    if flow_times.shape[0] != flow_clu.shape[0]:
        raise RuntimeError("FLOW spike_times/spike_clusters length mismatch.")

    print(f"Matching spikes by time (tol={args.tol} samples)...")
    if args.tol <= 0:
        i_gui, i_flow = match_spike_times_exact(gui_times, flow_times)
    else:
        i_gui, i_flow = match_spike_times_tol(gui_times, flow_times, tol=args.tol)

    n_match = int(len(i_gui))
    if n_match == 0:
        raise RuntimeError("No matched spikes found. Try higher --tol or verify folders.")

    gui_m = gui_clu[i_gui]
    flow_m = flow_clu[i_flow]

    gui_labels, gui_label_file = load_labels(str(gui_phy))
    flow_labels, flow_label_file = load_labels(str(flow_phy))

    gui_good_set = {cid for cid, lab in gui_labels.items() if lab == "good"}
    flow_good_set = {cid for cid, lab in flow_labels.items() if lab == "good"}

    # Overlap
    M, gui_ids, flow_ids = build_overlap(gui_m, flow_m)

    template_sim = None
    if have_templates:
        print("Computing cluster templates + template similarity matrix (cosine)...")

        # Build per-cluster template vectors using matched spikes only
        gui_C = build_cluster_templates_from_matched_spikes(
            templates=gui_templates,
            spike_templates=gui_spike_templates,
            matched_spike_idx=i_gui,
            matched_cluster_ids=gui_m,
            cluster_ids=gui_ids,
        )
        flow_C = build_cluster_templates_from_matched_spikes(
            templates=flow_templates,
            spike_templates=flow_spike_templates,
            matched_spike_idx=i_flow,
            matched_cluster_ids=flow_m,
            cluster_ids=flow_ids,
        )

        template_sim = cosine_similarity_matrix(gui_C, flow_C)  # (G, F)



    # sizes
    row = M.sum(axis=1, keepdims=True).astype(np.float32)  # |g|
    col = M.sum(axis=0, keepdims=True).astype(np.float32)  # |f|

    # Scores
    precision = M / np.maximum(col, 1.0)
    recall    = M / np.maximum(row, 1.0)
    f1        = (2.0 * M) / np.maximum(row + col, 1.0)
    min_norm  = M / np.maximum(np.minimum(row, col), 1.0)  # overlap / min(|g|,|f|)

    # 1:1 mapping (overlap-max)
    pairs, method = hungarian_maximize_overlap(M)

    # Per-GUI best match by EACH score
    best_rows = []
    for i, g in enumerate(gui_ids):
        j_prec = int(np.argmax(precision[i, :]))
        j_rec  = int(np.argmax(recall[i, :]))
        j_f1   = int(np.argmax(f1[i, :]))
        j_min  = int(np.argmax(min_norm[i, :]))

        if template_sim is not None:
            j_tmp = int(np.argmax(template_sim[i, :]))
            best_tmp_id = int(flow_ids[j_tmp])
            best_tmp_val = float(template_sim[i, j_tmp])
            best_tmp_pct = 100.0 * (0.5 * (best_tmp_val + 1.0))  # maps [-1,1] -> [0,1]

        else:
            best_tmp_id = -1
            best_tmp_val = 0.0
            best_tmp_pct = 0.0


        best_rows.append({
            "GUI_cluster": int(g),
            "GUI_label": gui_labels.get(int(g), ""),

            "best_FLOW_by_precision": int(flow_ids[j_prec]),
            "precision": float(precision[i, j_prec]),
            "precision_percent": float(100.0 * precision[i, j_prec]),

            "best_FLOW_by_recall": int(flow_ids[j_rec]),
            "recall": float(recall[i, j_rec]),
            "recall_percent": float(100.0 * recall[i, j_rec]),

            "best_FLOW_by_f1": int(flow_ids[j_f1]),
            "f1": float(f1[i, j_f1]),
            "f1_percent": float(100.0 * f1[i, j_f1]),

            "best_FLOW_by_min_norm": int(flow_ids[j_min]),
            "min_norm": float(min_norm[i, j_min]),
            "min_norm_percent": float(100.0 * min_norm[i, j_min]),

            "GUI_spikes_matched": int(M[i, :].sum()),

            "best_FLOW_by_template_sim": best_tmp_id,
            "template_sim": best_tmp_val,   
            "template_sim_percent": best_tmp_pct,

        })

    df_best = pd.DataFrame(best_rows)
    df_best.to_csv(out_dir / "per_gui_best_match.csv", index=False)

    # Hungarian mapping CSV with ALL scores
    map_rows = []
    for r, c, ov in sorted(pairs, key=lambda x: -x[2]):
        g = int(gui_ids[r])
        f = int(flow_ids[c])
        map_rows.append({
            "GUI_cluster": g,
            "GUI_label": gui_labels.get(g, ""),
            "FLOW_cluster": f,
            "FLOW_label": flow_labels.get(f, ""),

            "overlap": int(ov),

            "precision": float(precision[r, c]),
            "precision_percent": float(100.0 * precision[r, c]),

            "recall": float(recall[r, c]),
            "recall_percent": float(100.0 * recall[r, c]),

            "f1": float(f1[r, c]),
            "f1_percent": float(100.0 * f1[r, c]),

            "min_norm": float(min_norm[r, c]),
            "min_norm_percent": float(100.0 * min_norm[r, c]),

            "GUI_spikes_matched": int(M[r, :].sum()),
            "FLOW_spikes_matched": int(M[:, c].sum()),

            "template_sim": float(template_sim[r, c]) if template_sim is not None else 0.0,
            "template_sim_percent": float(100.0 * template_sim[r, c]) if template_sim is not None else 0.0,

        })
    df_map = pd.DataFrame(map_rows)
    df_map.to_csv(out_dir / "hungarian_mapping.csv", index=False)

    # Heatmaps
    print("Generating heatmaps...")
    M_row = M.astype(np.float32)
    rs = M_row.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    M_row_norm = M_row / rs

    save_heatmap(
        M_row_norm,
        title=f"Row-normalized overlap (matched spikes={n_match})",
        path=out_dir / "heatmap_overlap_row_norm.png",
        cbar_label="fraction of GUI cluster spikes",
    )
    
    save_heatmap(
        f1,
        title=f"Pairwise F1: 2O/(|GUI|+|FLOW|) (tol={args.tol})",
        path=out_dir / "heatmap_f1.png",
        cbar_label="F1",
    )
    save_heatmap(
        min_norm,
        title=f"Min-normalized overlap: O/min(|GUI|,|FLOW|) (tol={args.tol})",
        path=out_dir / "heatmap_min_norm.png",
        cbar_label="min_norm",
    )

    if template_sim is not None:
        save_heatmap(
            template_sim,
            title=f"Template similarity (cosine) (tol={args.tol})",
            path=out_dir / "heatmap_template_similarity.png",
            cbar_label="cosine similarity",
        )


    # Stats: ALL clusters + GOOD-only, BOTH directions, ALL scores
    def pct(x: float) -> float:
        return 100.0 * float(x)


    # helper to format a stats block
    def stats_block(name: str, score_mat: np.ndarray):
        all_gui = best_match_stats(score_mat, M, gui_ids, None, axis=1)
        all_flow = best_match_stats(score_mat, M, flow_ids, None, axis=0)

        good_gui = best_match_stats(score_mat, M, gui_ids, gui_good_set, axis=1)
        good_flow = best_match_stats(score_mat, M, flow_ids, flow_good_set, axis=0)

        lines = []
        lines.append(f"{name} best-match stats (percent):")

        lines.append("  ALL clusters:")
        lines.append(f"    GUI→FLOW: mean={pct(all_gui['mean_best']):.2f}%, "
                     f"spike-wt={pct(all_gui['spike_weighted_mean_best']):.2f}%, "
                     f"sqrt-wt={pct(all_gui['sqrt_spike_weighted_mean_best']):.2f}%  (n={all_gui['n_considered']})")
        lines.append(f"    FLOW→GUI: mean={pct(all_flow['mean_best']):.2f}%, "
                     f"spike-wt={pct(all_flow['spike_weighted_mean_best']):.2f}%, "
                     f"sqrt-wt={pct(all_flow['sqrt_spike_weighted_mean_best']):.2f}%  (n={all_flow['n_considered']})")

        lines.append("  GOOD-only:")
        lines.append(f"    GUI→FLOW: mean={pct(good_gui['mean_best']):.2f}%, "
                     f"spike-wt={pct(good_gui['spike_weighted_mean_best']):.2f}%, "
                     f"sqrt-wt={pct(good_gui['sqrt_spike_weighted_mean_best']):.2f}%  (n={good_gui['n_considered']})")
        lines.append(f"    FLOW→GUI: mean={pct(good_flow['mean_best']):.2f}%, "
                     f"spike-wt={pct(good_flow['spike_weighted_mean_best']):.2f}%, "
                     f"sqrt-wt={pct(good_flow['sqrt_spike_weighted_mean_best']):.2f}%  (n={good_flow['n_considered']})")

        return lines

    # Build summary
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("GUI vs FLOW SUMMARY (ALL scores, ALL clusters + GOOD-only)")
    summary_lines.append("=" * 70)
    summary_lines.append(f"GUI phy:  {gui_phy}")
    summary_lines.append(f"FLOW phy: {flow_phy}")
    summary_lines.append(f"tol (samples): {args.tol}")
    summary_lines.append("")
    summary_lines.append(f"Matched spikes: {n_match}")
    summary_lines.append(f"GUI unique clusters (matched spikes):  {len(gui_ids)}")
    summary_lines.append(f"FLOW unique clusters (matched spikes): {len(flow_ids)}")
    summary_lines.append("")
    summary_lines.append(f"Label file (GUI):  {gui_label_file}")
    summary_lines.append(f"Label file (FLOW): {flow_label_file}")
    summary_lines.append(f"GUI good clusters (from TSV):  {len(gui_good_set)}")
    summary_lines.append(f"FLOW good clusters (from TSV): {len(flow_good_set)}")
    summary_lines.append("")
    summary_lines.append(f"Hungarian 1:1 mapping method: {method} (max overlap counts)")
    summary_lines.append(f"Mapped pairs (nonzero overlap): {len(pairs)}")
    summary_lines.append("")

    if template_sim is not None:
        summary_lines += stats_block("Template similarity (cosine)", template_sim)
        summary_lines.append("")


    summary_lines += stats_block("Precision (O/|FLOW|)", precision)
    summary_lines.append("")
    summary_lines += stats_block("Recall (O/|GUI|)", recall)
    summary_lines.append("")
    summary_lines += stats_block("F1 (2O/(|GUI|+|FLOW|))", f1)
    summary_lines.append("")
    summary_lines += stats_block("Min-norm (O/min(|GUI|,|FLOW|))", min_norm)
    summary_lines.append("")
    summary_lines.append("Outputs:")
    summary_lines.append(f"  {out_dir / 'hungarian_mapping.csv'}")
    summary_lines.append(f"  {out_dir / 'per_gui_best_match.csv'}")
    summary_lines.append(f"  {out_dir / 'heatmap_overlap_row_norm.png'}")
    summary_lines.append(f"  {out_dir / 'heatmap_precision.png'}")
    summary_lines.append(f"  {out_dir / 'heatmap_recall.png'}")
    summary_lines.append(f"  {out_dir / 'heatmap_f1.png'}")
    summary_lines.append(f"  {out_dir / 'heatmap_min_norm.png'}")
    summary_lines.append("")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))

    # Terminal quick view: top 10 GUI good by min_norm and by f1
    df_best_good = df_best[df_best["GUI_label"] == "good"].copy()
    if not df_best_good.empty:
        print("\nTOP 10 GUI 'good' by best min_norm_percent:")
        print(df_best_good.sort_values("min_norm_percent", ascending=False).head(10).to_string(index=False))
        print("\nTOP 10 GUI 'good' by best f1_percent:")
        print(df_best_good.sort_values("f1_percent", ascending=False).head(10).to_string(index=False))

    if "template_sim_percent" in df_best_good.columns:
        print("\nTOP 10 GUI 'good' by best template_sim_percent:")
        print(df_best_good.sort_values("template_sim_percent", ascending=False).head(10).to_string(index=False))

    else:
        print("\n(no GUI 'good' units found in labels)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
