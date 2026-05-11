#!/usr/bin/env python3
"""
Compute QC criteria per cluster using ONLY Kilosort outputs.
Includes an interactive popularity (spike count) filter and outputs a QC TSV.
"""

import argparse
from pathlib import Path
import numpy as np

# Try SciPy; if missing, fall back to scipy.special
try:
    from scipy.stats import chi2
    _HAS_SCIPY_STATS = True
except Exception:
    _HAS_SCIPY_STATS = False
    from scipy.special import gammainc


# ==========================================
# --- MATH & STATS HELPERS ---
# ==========================================

def chi2_cdf(x, df):
    """Chi-square CDF. Uses scipy.stats if available; otherwise scipy.special."""
    x = np.asarray(x, dtype=np.float64)
    df = float(df)
    if _HAS_SCIPY_STATS:
        return chi2.cdf(x, df=df)
    return gammainc(df / 2.0, x / 2.0)

def stable_inv_cov(X, reg_eps=1e-6, svd_floor=1e-8):
    """Return mean and inverse covariance with SVD stabilization."""
    X = np.asarray(X, dtype=np.float64)
    mu = np.mean(X, axis=0)
    Xc = X - mu

    cov = np.cov(Xc, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = cov + np.eye(cov.shape[0], dtype=np.float64) * reg_eps

    U, s, Vt = np.linalg.svd(cov, full_matrices=False)
    s[s < svd_floor] = svd_floor
    inv_cov = (U / s) @ Vt
    return mu, inv_cov

def mahalanobis_d2(X, mu, inv_cov):
    """Mahalanobis distance squared for each row in X."""
    X = np.asarray(X, dtype=np.float64) - mu
    return np.einsum("ij,jk,ik->i", X, inv_cov, X)


# ==========================================
# --- METRIC CALCULATORS ---
# ==========================================

def compute_isi_index_paper(spike_times_samples, fs, isi1_ms=2.0, isi2_ms=20.0):
    """
    ISI-index = N(0-2ms) / ( N(0-20ms) * (2/20) )
    """
    st = np.asarray(spike_times_samples, dtype=np.int64)
    if st.size < 3:
        return np.nan

    st = np.sort(st)
    isi_ms = np.diff(st) * 1000.0 / float(fs)

    n_0_20 = np.sum((isi_ms > 0) & (isi_ms <= isi2_ms))
    n_0_2  = np.sum((isi_ms > 0) & (isi_ms <= isi1_ms))

    expected_0_2 = n_0_20 * (isi1_ms / isi2_ms)
    if expected_0_2 <= 0:
        return np.nan
    return float(n_0_2 / expected_0_2)

def compute_l_ratio(cluster_id, clu, X_all, p, max_in, max_other):
    """Classic L-ratio using PCA coefficients."""
    cid = int(cluster_id)
    in_idx = np.flatnonzero(clu == cid)
    n_in_total = int(in_idx.size)
    
    if n_in_total < 20:
        return np.nan

    in_use = np.random.choice(in_idx, size=max_in, replace=False) if n_in_total > max_in else in_idx
    X_in = X_all[in_use]

    if X_in.shape[0] < 50:
        return np.nan

    mu, inv_cov = stable_inv_cov(X_in)

    out_idx = np.flatnonzero(clu != cid)
    if out_idx.size == 0:
        return 0.0
    
    if out_idx.size > max_other:
        out_idx = np.random.choice(out_idx, size=max_other, replace=False)

    d2 = mahalanobis_d2(X_all[out_idx], mu, inv_cov)
    tail = 1.0 - chi2_cdf(d2, df=p)

    return float(np.sum(tail) / max(n_in_total, 1))

def compute_template_p2t(cluster_id, templates):
    """Template peak-to-trough on best channel in template units."""
    if templates is None:
        return np.nan
    k = int(cluster_id)
    if k < 0 or k >= templates.shape[0]:
        return np.nan
    
    tpl = templates[k].astype(np.float32)
    p2p = np.ptp(tpl, axis=0)
    return float(np.max(p2p))


# ==========================================
# --- FILTERING ---
# ==========================================

def filter_by_popularity(clu, cluster_ids):
    """Prints spike counts per cluster (sorted by count) and filters based on user input threshold."""
    # Count spikes for each cluster
    counts = {cid: np.sum(clu == cid) for cid in cluster_ids}
    
    # Sort the dictionary by spike count in descending order
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    
    print("\n--- Cluster Popularity (Spike Counts) ---")
    for cid, count in sorted_counts.items():
        print(f"Cluster {cid:4d}: {count:6d} spikes")
    
    try:
        threshold_str = input("\nEnter minimum spike count threshold to keep a cluster (or press Enter for 0): ")
        threshold = int(threshold_str) if threshold_str.strip() else 0
    except ValueError:
        print("Invalid input. Defaulting to 0.")
        threshold = 0
        
    # Keep clusters that meet the threshold
    kept_clusters = [cid for cid, count in sorted_counts.items() if count >= threshold]
    eliminated = len(cluster_ids) - len(kept_clusters)
    
    return kept_clusters, eliminated

# ==========================================
# --- MAIN EXECUTION ---
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks_dir", required=True, help="Kilosort output folder")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--max_other", type=int, default=30000, help="Max non-cluster spikes for L-ratio")
    ap.add_argument("--max_in", type=int, default=5000, help="Max in-cluster spikes for covariance")
    ap.add_argument("--isi_thresh", type=float, default=0.2, help="ISI-index threshold (pass if <)")
    ap.add_argument("--lratio_thresh", type=float, default=0.05, help="L-ratio threshold (pass if <)")
    ap.add_argument("--p2t_thresh", type=float, default=50.0, help="Template p2t threshold (0 disables)")
    args = ap.parse_args()

    ks_dir = Path(args.ks_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Data ---
    print("Loading data...")
    st = np.load(ks_dir / "spike_times.npy").astype(np.int64).reshape(-1)
    clu = np.load(ks_dir / "spike_clusters.npy").astype(np.int64).reshape(-1)

    ops_path = ks_dir / "ops.npy"
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
        fs = float(ops["fs"])
    else:
        raise FileNotFoundError("ops.npy not found. Needed to get fs for ISI/firing rate.")

    tpl_path = ks_dir / "templates.npy"
    templates = np.load(tpl_path) if tpl_path.exists() else None
    if templates is None:
        print("[WARN] templates.npy not found: template p2t will be NaN.")

    pc_feat_path = ks_dir / "pc_features.npy"
    if not pc_feat_path.exists():
        raise FileNotFoundError("pc_features.npy not found. Cannot compute L-ratio.")

    pc_features = np.load(pc_feat_path)
    X_all = pc_features.reshape(pc_features.shape[0], -1).astype(np.float32)
    p = int(X_all.shape[1])

    duration_s = float(st.max()) / fs if st.size else np.nan
    all_cluster_ids = np.unique(clu)
    initial_cluster_count = len(all_cluster_ids)

    # --- 2. Interactive Popularity Filtering ---
    print(f"\nInitial total clusters found: {initial_cluster_count}")
    filtered_cluster_ids, eliminated_by_popularity = filter_by_popularity(clu, all_cluster_ids)

    # --- 3. Compute Metrics & Write Output ---
    print("\nComputing QC metrics...")
    header = [
        "cluster_id", "n_spikes", "duration_s", "firing_rate_hz",
        "isi_index", "isi_pass", "l_ratio", "l_pass",
        "template_p2t_units", "p2t_pass",
    ]
    lines = ["\t".join(header)]

    # Trackers for the stats funnel
    funnel_stats = {
        "passed_popularity": len(filtered_cluster_ids),
        "failed_isi": 0,
        "failed_lratio": 0,
        "failed_p2t": 0,
        "passed_all_qc": 0
    }

    for cid in filtered_cluster_ids:
        idx = np.flatnonzero(clu == cid)
        n_spikes = int(idx.size)
        fr = (n_spikes / duration_s) if (np.isfinite(duration_s) and duration_s > 0) else np.nan

        isi = compute_isi_index_paper(st[idx], fs=fs)
        isi_pass = int(np.isfinite(isi) and (isi < args.isi_thresh))

        lr = compute_l_ratio(cid, clu, X_all, p, args.max_in, args.max_other)
        l_pass = int(np.isfinite(lr) and (lr < args.lratio_thresh))

        p2t = compute_template_p2t(cid, templates)
        p2t_pass = int(np.isfinite(p2t) and (p2t >= args.p2t_thresh)) if args.p2t_thresh > 0 else -1

        # Funnel logic tracking
        passed_all = True
        if not isi_pass:
            funnel_stats["failed_isi"] += 1
            passed_all = False
        if not l_pass:
            funnel_stats["failed_lratio"] += 1
            passed_all = False
        if p2t_pass == 0:
            funnel_stats["failed_p2t"] += 1
            passed_all = False
            
        if passed_all:
            funnel_stats["passed_all_qc"] += 1

        lines.append(
            f"{int(cid)}\t{n_spikes}\t{duration_s:.6f}\t{fr:.6f}\t"
            f"{isi if np.isfinite(isi) else 'nan'}\t{isi_pass}\t"
            f"{lr if np.isfinite(lr) else 'nan'}\t{l_pass}\t"
            f"{p2t if np.isfinite(p2t) else 'nan'}\t{p2t_pass}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    # --- 4. Print Summary Stats ---
    print("\n=========================================")
    print("        FILTERING & QC STATISTICS        ")
    print("=========================================")
    print(f"Total Initial Clusters:       {initial_cluster_count}")
    print(f"1. Popularity Filter:       - {eliminated_by_popularity} eliminated")
    print(f"   Remaining:                 {funnel_stats['passed_popularity']}")
    print(f"2. ISI-Index (<{args.isi_thresh}):      - {funnel_stats['failed_isi']} failed")
    print(f"3. L-Ratio (<{args.lratio_thresh}):        - {funnel_stats['failed_lratio']} failed")
    if args.p2t_thresh > 0:
        print(f"4. Template P2T (>={args.p2t_thresh}):   - {funnel_stats['failed_p2t']} failed")
    else:
        print(f"4. Template P2T:              Disabled")
    print("-----------------------------------------")
    print(f"Clusters passing ALL steps:   {funnel_stats['passed_all_qc']}")
    print("=========================================\n")
    print(f"Saved TSV: {out_path}")


if __name__ == "__main__":
    main()