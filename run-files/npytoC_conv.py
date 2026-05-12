#!/usr/bin/env python3
"""
Compute QC criteria per cluster using ONLY Kilosort outputs.
Includes an interactive popularity (spike count) filter and outputs a QC TSV.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

def compute_l_ratio(template_id, temp, X_all, p, max_in, max_other):
    """Classic L-ratio using PCA coefficients."""
    tid = int(template_id)
    in_idx = np.flatnonzero(temp == tid)
    n_in_total = int(in_idx.size)
    
    if n_in_total < 20:
        return np.nan

    in_use = np.random.choice(in_idx, size=max_in, replace=False) if n_in_total > max_in else in_idx
    X_in = X_all[in_use]

    if X_in.shape[0] < 50:
        return np.nan

    mu, inv_cov = stable_inv_cov(X_in)

    out_idx = np.flatnonzero(temp != tid)
    if out_idx.size == 0:
        return 0.0
    
    if out_idx.size > max_other:
        out_idx = np.random.choice(out_idx, size=max_other, replace=False)

    d2 = mahalanobis_d2(X_all[out_idx], mu, inv_cov)
    tail = 1.0 - chi2_cdf(d2, df=p)

    return float(np.sum(tail) / max(n_in_total, 1))

def compute_template_p2t(cluster_id, templates, tpl_assign, amplitudes, winv):
    """
    Calculates the true physical Peak-to-Trough by unwhitening the template
    and scaling it by the median amplitude of its assigned spikes.
    """
    if templates is None:
        return np.nan
        
    k = int(cluster_id)
    if k < 0 or k >= templates.shape[0]:
        return np.nan
    
    # 1. Get the base template shape: (n_samples, n_channels)
    tpl = templates[k].astype(np.float32)
    
    # 2. Unwhiten the template to get back to physical channel space
    if winv is not None:
        tpl = np.dot(tpl, winv)
        
    # 3. Calculate the Peak-to-Trough on the best channel
    base_p2p = np.max(np.ptp(tpl, axis=0))
    
    # 4. Find the median amplitude scaling factor for this specific template
    spike_indices = np.flatnonzero(tpl_assign == k)
    if spike_indices.size == 0:
        return 0.0 # No spikes, amplitude is effectively 0
        
    median_scale = np.median(amplitudes[spike_indices])
    
    # 5. True Physical Amplitude
    true_p2t = float(base_p2p * median_scale)
    
    return true_p2t


# ==========================================
# --- FILTERING ---
# ==========================================

def filter_by_popularity(tpl_assign, template_ids, duration_s):
    """Calculates spikes per hour, draws CDF, prints stats, and asks for threshold."""
    
    # --- Convert recording duration to hours ---
    duration_h = duration_s / 3600.0
    if duration_h <= 0:
        print("[WARN] Invalid duration. Skipping popularity filter.")
        return template_ids, 0
    
    # Calculate frequency in Spikes per Hour for each template
    counts = {tid: np.sum(tpl_assign == tid) for tid in template_ids}
    freqs = {tid: count / duration_h for tid, count in counts.items()}
    
    # Extract just the frequency values for math
    freq_array = np.array(list(freqs.values()))
    
    if len(freq_array) == 0:
        return [], 0

    # Calculate requested statistics
    min_f = np.min(freq_array)
    max_f = np.max(freq_array)
    p1 = np.percentile(freq_array, 1)
    p99 = np.percentile(freq_array, 99)
    
    print("\n--- Popularity Statistics (Spikes per Hour) ---")
    print(f"Minimum:         {min_f:.2f} spikes/hr")
    print(f"1st Percentile:  {p1:.2f} spikes/hr")
    print(f"99th Percentile: {p99:.2f} spikes/hr")
    print(f"Maximum:         {max_f:.2f} spikes/hr")
    
    # --- Plot the CDF ---
    sorted_freqs = np.sort(freq_array)
    # y-axis is the cumulative probability (from 0 to 1)
    cdf = np.arange(1, len(sorted_freqs) + 1) / len(sorted_freqs)
    
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_freqs, cdf, lw=2, label='CDF')
    plt.axvline(p1, color='red', linestyle='--', label=f'1% ({p1:.1f}/hr)')
    plt.axvline(p99, color='green', linestyle='--', label=f'99% ({p99:.1f}/hr)')
    
    plt.xlabel("Spikes per Hour")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Template Appearances per Hour")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show plot without blocking the terminal prompt
    plt.show(block=False)
    plt.pause(0.1) 
    
    # --- Get User Input ---
    try:
        threshold_str = input("\nEnter minimum Spikes per Hour threshold (or press Enter for 0): ")
        threshold = float(threshold_str) if threshold_str.strip() else 0.0
    except ValueError:
        print("Invalid input. Defaulting to 0.0 spikes/hr.")
        threshold = 0.0
        
    plt.close() # Close the graph window after user presses enter
        
    # Apply filter based on spikes per hour
    kept_templates = [tid for tid, f in freqs.items() if f >= threshold]
    eliminated = len(template_ids) - len(kept_templates)
    
    return kept_templates, eliminated

# ==========================================
# --- MAIN EXECUTION ---
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks_dir", required=True, help="Kilosort output folder")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--fs", type=float, default=20000.0, help="Sampling rate (fallback if ops.npy is missing)") # <--- NEW
    ap.add_argument("--max_other", type=int, default=30000, help="Max non-template spikes for L-ratio")
    ap.add_argument("--max_in", type=int, default=5000, help="Max in-template spikes for covariance")
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
   
    tpl_assign = np.load(ks_dir / "spike_templates.npy").astype(np.int64).reshape(-1)

    #Load amplitudes and whitening matrix
    amplitudes = np.load(ks_dir / "amplitudes.npy").astype(np.float64).reshape(-1)
    
    winv_path = ks_dir / "whitening_mat_inv.npy"
    if winv_path.exists():
        winv = np.load(winv_path)
    else:
        print("[WARN] whitening_mat_inv.npy not found. P2T will be in whitened space.")
        winv = None

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
    
    # Get unique template IDs
    duration_s = float(st.max()) / fs if st.size else np.nan
    all_template_ids = np.unique(tpl_assign)
    initial_template_count = len(all_template_ids)

    # --- 2. Interactive Popularity Filtering ---
    print(f"\nInitial total templates found: {initial_template_count}")
    filtered_template_ids, eliminated_by_popularity = filter_by_popularity(tpl_assign, all_template_ids, duration_s)
    # --- 3. Compute Metrics & Write Output ---
    print("\nComputing QC metrics...")
    
    # Change header from template_id to template_id
    header = [
        "template_id", "n_spikes", "duration_s", "firing_rate_hz",
        "isi_index", "isi_pass", "l_ratio", "l_pass",
        "template_p2t_units", "p2t_pass",
    ]
    lines = ["\t".join(header)]

    funnel_stats = {
        "passed_popularity": len(filtered_template_ids),
        "failed_isi": 0,
        "failed_lratio": 0,
        "failed_p2t": 0,
        "passed_all_qc": 0
    }

    for tid in filtered_template_ids:
        # Evaluate based on template assignment
        idx = np.flatnonzero(tpl_assign == tid)
        n_spikes = int(idx.size)
        fr = (n_spikes / duration_s) if (np.isfinite(duration_s) and duration_s > 0) else np.nan

        isi = compute_isi_index_paper(st[idx], fs=fs)
        isi_pass = int(np.isfinite(isi) and (isi < args.isi_thresh))

        lr = compute_l_ratio(tid, tpl_assign, X_all, p, args.max_in, args.max_other)
        l_pass = int(np.isfinite(lr) and (lr < args.lratio_thresh))

        p2t = compute_template_p2t(tid, templates, tpl_assign, amplitudes, winv)
        p2t_pass = int(np.isfinite(p2t) and (p2t >= args.p2t_thresh)) if args.p2t_thresh > 0 else -1

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
            f"{int(tid)}\t{n_spikes}\t{duration_s:.6f}\t{fr:.6f}\t"
            f"{isi if np.isfinite(isi) else 'nan'}\t{isi_pass}\t"
            f"{lr if np.isfinite(lr) else 'nan'}\t{l_pass}\t"
            f"{p2t if np.isfinite(p2t) else 'nan'}\t{p2t_pass}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    # --- 4. Print Summary Stats ---
    print("\n=========================================")
    print("        FILTERING & QC STATISTICS        ")
    print("=========================================")
    print(f"Total Initial Templates:      {initial_template_count}")
    print(f"1. Popularity Filter:       - {eliminated_by_popularity} eliminated")
    print(f"   Remaining:                 {funnel_stats['passed_popularity']}")
    print(f"2. ISI-Index (<{args.isi_thresh}):      - {funnel_stats['failed_isi']} failed")
    print(f"3. L-Ratio (<{args.lratio_thresh}):        - {funnel_stats['failed_lratio']} failed")
    if args.p2t_thresh > 0:
        print(f"4. Template P2T (>={args.p2t_thresh}):   - {funnel_stats['failed_p2t']} failed")
    else:
        print(f"4. Template P2T:              Disabled")
    print("-----------------------------------------")
    print(f"Templates passing ALL steps:  {funnel_stats['passed_all_qc']}")
    print("=========================================\n")
    print(f"Saved TSV: {out_path}")

if __name__ == "__main__":
    main()
