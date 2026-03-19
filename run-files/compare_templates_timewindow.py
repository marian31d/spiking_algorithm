#!/usr/bin/env python3
"""
GUI vs. FLOW Evaluator (Time-Windowed & Cluster-Aware)

This script evaluates the accuracy of the FLOW spike sorting phase.
It includes a Cluster-Aware sweeping algorithm to prevent high-density 
probe collisions from destroying the F1 score, making it suitable for 
validating SoC hardware constraints.
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
# Template loading & processing
# -----------------------------
def _load_templates_and_spike_templates(phy_dir: Path, spike_mask: np.ndarray | None = None):
    t_path = phy_dir / "templates.npy"
    st_path = phy_dir / "spike_templates.npy"
    if (not t_path.exists()) or (not st_path.exists()):
        return None, None
    try:
        templates = np.load(t_path)
        spike_templates = np.load(st_path).astype(np.int64, copy=False).reshape(-1)
        if spike_mask is not None:
            try:
                if spike_mask.shape[0] == spike_templates.shape[0]:
                    spike_templates = spike_templates[spike_mask]
            except Exception:
                pass
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
    T = templates.reshape(templates.shape[0], -1).astype(np.float32, copy=False)
    mt = spike_templates[matched_spike_idx]
    mc = matched_cluster_ids.astype(np.int64, copy=False)

    cid_to_row = {int(cid): i for i, cid in enumerate(cluster_ids)}
    C = np.zeros((len(cluster_ids), T.shape[1]), dtype=np.float32)

    for cid in cluster_ids:
        cid_int = int(cid)
        row = cid_to_row[cid_int]
        mask = (mc == cid_int)
        if not np.any(mask):
            continue
        t_ids = mt[mask]
        uniq, cnt = np.unique(t_ids, return_counts=True)
        valid = (uniq >= 0) & (uniq < T.shape[0])
        uniq = uniq[valid]
        cnt = cnt[valid].astype(np.float32, copy=False)
        if uniq.size == 0:
            continue
        C[row, :] = (T[uniq, :] * cnt[:, None]).sum(axis=0) / float(cnt.sum())

    return C

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-12):
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    An = np.linalg.norm(A, axis=1, keepdims=True)
    Bn = np.linalg.norm(B, axis=1, keepdims=True)
    A_unit = A / np.maximum(An, eps)
    B_unit = B / np.maximum(Bn, eps)
    return A_unit @ B_unit.T

# -----------------------------
# Hungarian mapping
# -----------------------------
def hungarian_maximize_overlap(M: np.ndarray):
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-M)
        pairs = [(int(rr), int(cc), int(M[rr, cc])) for rr, cc in zip(r, c)]
        pairs = [p for p in pairs if p[2] > 0]
        return pairs, "hungarian"
    except Exception:
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
# Best-match stats
# -----------------------------
def best_match_stats(score: np.ndarray, M: np.ndarray, ids: np.ndarray, keep_set: set | None, axis: int):
    if keep_set is None:
        keep = np.ones(len(ids), dtype=bool)
    else:
        keep = np.array([int(cid) in keep_set for cid in ids], dtype=bool)

    if keep.sum() == 0:
        return {
            "n_considered": 0, "mean_best": 0.0,
            "spike_weighted_mean_best": 0.0, "sqrt_spike_weighted_mean_best": 0.0,
        }

    if axis == 1:
        score_sub = score[keep, :]
        M_sub = M[keep, :]
        best = score_sub.max(axis=1).astype(np.float64, copy=False)
        spikes = M_sub.sum(axis=1).astype(np.float64, copy=False)
    elif axis == 0:
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

def save_heatmap(mat: np.ndarray, title: str, path: Path, cbar_label: str):
    plt.figure(figsize=(12, 10))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label=cbar_label)
    plt.title(title)
    plt.xlabel("FLOW clusters (columns)")
    plt.ylabel("GUI clusters (rows)")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------
# Main Execution
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui_phy", required=True, type=str, help="Path to GUI results")
    ap.add_argument("--flow_phy", required=True, type=str, help="Path to FLOW results")
    ap.add_argument("--t_start", required=True, type=float, help="Start time in SECONDS")
    ap.add_argument("--t_end", default=-1.0, type=float, help="End time in SECONDS (-1 for end of file)")
    ap.add_argument("--t_duration", default=-1.0, type=float, help="Duration in SECONDS (Overrides t_end if > 0)")
    ap.add_argument("--tol", default=12, type=int, help="Spike jitter tolerance (samples)")
    ap.add_argument("--out", default=".", type=str)
    args = ap.parse_args()

    fs = 20000 
    gui_phy = Path(args.gui_phy)
    flow_phy = Path(args.flow_phy)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Spike Data...")
    gui_times = np.load(gui_phy / "spike_times.npy").reshape(-1)
    gui_clu   = np.load(gui_phy / "spike_clusters.npy").reshape(-1)
    
    flow_times = np.load(flow_phy / "spike_times.npy").reshape(-1)
    flow_clu   = np.load(flow_phy / "spike_clusters.npy").reshape(-1)

    # --- STRICT TIME WINDOW ALIGNMENT ---
    start_sample = int(args.t_start * fs)
    if args.t_duration > 0:
        end_sample = start_sample + int(args.t_duration * fs)
    elif args.t_end > 0:
        end_sample = int(args.t_end * fs)
    else:
        # Prevent 11% Recall bugs by mapping to exactly what FLOW processed
        end_sample = flow_times.max() + 1 if len(flow_times) > 0 else np.inf

    print(f"Comparing window: {args.t_start}s to {end_sample/fs if end_sample != np.inf else 'End of File'}s")
    
    gui_mask = (gui_times >= start_sample) & (gui_times < end_sample)
    flow_mask = (flow_times >= start_sample) & (flow_times < end_sample)

    gui_t_win, gui_c_win = gui_times[gui_mask], gui_clu[gui_mask]
    flow_t_win, flow_c_win = flow_times[flow_mask], flow_clu[flow_mask]

    print(f"GUI spikes in window: {len(gui_t_win)}")
    print(f"FLOW spikes in window: {len(flow_t_win)}")

    # --- CLUSTER-AWARE OVERLAP MATRIX ---
    print("Computing Cluster-Aware Overlap Matrix...")
    
    sg = np.argsort(gui_t_win)
    gui_t_s = gui_t_win[sg]
    gui_c_s = gui_c_win[sg]
    
    sf = np.argsort(flow_t_win)
    flow_t_s = flow_t_win[sf]
    flow_c_s = flow_c_win[sf]
    
    gui_ids = np.unique(gui_c_s)
    flow_ids = np.unique(flow_c_s)
    
    gi = {int(cid): idx for idx, cid in enumerate(gui_ids)}
    fi = {int(cid): idx for idx, cid in enumerate(flow_ids)}
    
    M = np.zeros((len(gui_ids), len(flow_ids)), dtype=np.int64)
    
    left_idx = np.searchsorted(flow_t_s, gui_t_s - args.tol, side='left')
    right_idx = np.searchsorted(flow_t_s, gui_t_s + args.tol, side='right')
    
    matched_gui_local = []
    matched_flow_local = []

    for i in range(len(gui_t_s)):
        l = left_idx[i]
        r = right_idx[i]
        if l < r:
            g_idx = gi[int(gui_c_s[i])]
            matched_f_clusters = flow_c_s[l:r]
            
            # Prevent density collisions by crediting all unique matches in the window
            for fc in np.unique(matched_f_clusters):
                f_idx = fi[int(fc)]
                M[g_idx, f_idx] += 1
                
            closest_f = l + np.argmin(np.abs(flow_t_s[l:r] - gui_t_s[i]))
            matched_gui_local.append(sg[i])
            matched_flow_local.append(sf[closest_f])

    gui_abs_indices = np.where((gui_times >= start_sample) & (gui_times < end_sample))[0]
    flow_abs_indices = np.where((flow_times >= start_sample) & (flow_times < end_sample))[0]

    i_gui = gui_abs_indices[matched_gui_local]
    i_flow = flow_abs_indices[matched_flow_local]
    
    gui_m = gui_clu[i_gui]
    flow_m = flow_clu[i_flow]

    n_match = len(i_gui)
    print(f"Matched Pairs found: {n_match}")

    if n_match == 0:
        raise RuntimeError("No matched spikes found in this window. Verify t_start.")

    gui_labels, gui_label_file = load_labels(str(gui_phy))
    flow_labels, flow_label_file = load_labels(str(flow_phy))

    gui_good_set = {cid for cid, lab in gui_labels.items() if lab == "good"}
    flow_good_set = {cid for cid, lab in flow_labels.items() if lab == "good"}

    # --- TEMPLATE SIMILARITY ---
    print("Loading Waveforms for Similarity check...")
    gui_templates, gui_spike_templates = _load_templates_and_spike_templates(gui_phy)
    flow_templates, flow_spike_templates = _load_templates_and_spike_templates(flow_phy)

    have_templates = (gui_templates is not None and flow_templates is not None)
    template_sim = None

    if have_templates:
        print("Computing cluster templates + template similarity matrix (cosine)...")
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
        template_sim = cosine_similarity_matrix(gui_C, flow_C)

    # --- TRUE CLUSTER SIZES FOR MATH ---
    # Using the raw window counts ensures Precision/Recall don't break >100%
    row = np.array([np.sum(gui_c_win == g) for g in gui_ids]).astype(np.float32).reshape(-1, 1)
    col = np.array([np.sum(flow_c_win == f) for f in flow_ids]).astype(np.float32).reshape(1, -1)

    precision = M / np.maximum(col, 1.0)
    recall    = M / np.maximum(row, 1.0)
    f1        = (2.0 * M) / np.maximum(row + col, 1.0)
    min_norm  = M / np.maximum(np.minimum(row, col), 1.0)

    # 1:1 mapping
    pairs, method = hungarian_maximize_overlap(M)

    # Per-GUI best match
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
            best_tmp_pct = 100.0 * (0.5 * (best_tmp_val + 1.0))
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

    # --- HEATMAPS ---
    print("Generating heatmaps...")
    M_row = M.astype(np.float32)
    rs = M_row.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    M_row_norm = M_row / rs

    save_heatmap(M_row_norm, f"Row-normalized overlap", out_dir / "heatmap_overlap_row_norm.png", "fraction of GUI spikes")
    save_heatmap(f1, f"Pairwise F1", out_dir / "heatmap_f1.png", "F1")
    save_heatmap(min_norm, f"Min-normalized overlap", out_dir / "heatmap_min_norm.png", "min_norm")

    if template_sim is not None:
        save_heatmap(template_sim, f"Template similarity (cosine)", out_dir / "heatmap_template_similarity.png", "cosine similarity")

    def pct(x: float) -> float:
        return 100.0 * float(x)

    def stats_block(name: str, score_mat: np.ndarray):
        all_gui = best_match_stats(score_mat, M, gui_ids, None, axis=1)
        all_flow = best_match_stats(score_mat, M, flow_ids, None, axis=0)
        good_gui = best_match_stats(score_mat, M, gui_ids, gui_good_set, axis=1)
        good_flow = best_match_stats(score_mat, M, flow_ids, flow_good_set, axis=0)

        lines = [f"{name} best-match stats (percent):", "  ALL clusters:",
                 f"    GUI->FLOW: mean={pct(all_gui['mean_best']):.2f}%, spike-wt={pct(all_gui['spike_weighted_mean_best']):.2f}%",
                 f"    FLOW->GUI: mean={pct(all_flow['mean_best']):.2f}%, spike-wt={pct(all_flow['spike_weighted_mean_best']):.2f}%",
                 "  GOOD-only:",
                 f"    GUI->FLOW: mean={pct(good_gui['mean_best']):.2f}%, spike-wt={pct(good_gui['spike_weighted_mean_best']):.2f}%",
                 f"    FLOW->GUI: mean={pct(good_flow['mean_best']):.2f}%, spike-wt={pct(good_flow['spike_weighted_mean_best']):.2f}%"]
        return lines

    summary_lines = ["=" * 70, "GUI vs FLOW SUMMARY (Cluster-Aware)", "=" * 70,
                     f"GUI phy:  {gui_phy}", f"FLOW phy: {flow_phy}",
                     f"Matched spikes: {n_match}", ""]

    if template_sim is not None:
        summary_lines += stats_block("Template similarity (cosine)", template_sim) + [""]

    summary_lines += stats_block("Precision (O/|FLOW|)", precision) + [""]
    summary_lines += stats_block("Recall (O/|GUI|)", recall) + [""]
    summary_lines += stats_block("F1 (2O/(|GUI|+|FLOW|))", f1) + [""]
    summary_lines += stats_block("Min-norm (O/min(|GUI|,|FLOW|))", min_norm) + [""]

    (out_dir / "summary.txt").write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))
    print("\nDONE.")

if __name__ == "__main__":
    main()
