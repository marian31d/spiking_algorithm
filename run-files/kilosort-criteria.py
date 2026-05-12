# ==========================================
# --- MAIN EXECUTION ---
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks_dir", required=True, help="Kilosort output folder")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--fs", type=float, default=20000.0, help="Sampling rate (fallback if ops.npy is missing)")
    ap.add_argument("--max_other", type=int, default=30000, help="Max non-cluster spikes for L-ratio")
    ap.add_argument("--max_in", type=int, default=5000, help="Max in-cluster spikes for covariance")
    ap.add_argument("--isi_thresh", type=float, default=0.2, help="ISI-index threshold (pass if <)")
    ap.add_argument("--lratio_thresh", type=float, default=0.05, help="L-ratio threshold (pass if <)")
    ap.add_argument("--p2t_thresh", type=float, default=50.0, help="Template p2t threshold (0 disables)")
    args = ap.parse_args()

    ks_dir = Path(args.ks_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if out_path.is_dir():
        out_path = out_path / "qc_metrics.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Data ---
    print("Loading data...")
    st = np.load(ks_dir / "spike_times.npy").astype(np.int64).reshape(-1)
    
    # WE LOAD BOTH CLUSTERS AND TEMPLATES NOW
    clu = np.load(ks_dir / "spike_clusters.npy").astype(np.int64).reshape(-1)
    tpl_assign = np.load(ks_dir / "spike_templates.npy").astype(np.int64).reshape(-1)

    amplitudes = np.load(ks_dir / "amplitudes.npy").astype(np.float64).reshape(-1)
    
    winv_path = ks_dir / "whitening_mat_inv.npy"
    winv = np.load(winv_path) if winv_path.exists() else None

    ops_path = ks_dir / "ops.npy"
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
        fs = float(ops["fs"])
    else:
        print(f"[WARN] ops.npy not found. Defaulting to provided --fs: {args.fs} Hz")
        fs = args.fs

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

    # --- 2. Interactive Popularity Filtering (ON CLUSTERS) ---
    print(f"\nInitial total clusters found: {initial_cluster_count}")
    # Note: We pass 'clu' here instead of tpl_assign
    filtered_cluster_ids, eliminated_by_popularity = filter_by_popularity(clu, all_cluster_ids, duration_s)

    # --- 3. Compute Metrics & Write Output ---
    print("\nComputing QC metrics...")
    
    # New Header: We now output the Cluster ID AND its dominant Template ID
    header = [
        "cluster_id", "dominant_template_id", "n_spikes", "duration_s", "firing_rate_hz",
        "isi_index", "isi_pass", "l_ratio", "l_pass",
        "template_p2t_units", "p2t_pass",
    ]
    lines = ["\t".join(header)]

    funnel_stats = {
        "passed_popularity": len(filtered_cluster_ids),
        "failed_isi": 0,
        "failed_lratio": 0,
        "failed_p2t": 0,
        "passed_all_qc": 0
    }
    
    # LIST TO COLLECT GOOD TEMPLATES FOR C++ EXPORT
    good_templates_for_online = []

    for cid in filtered_cluster_ids:
        # EVALUATE EVERYTHING BASED ON CLUSTER ID (cid)
        idx = np.flatnonzero(clu == cid)
        n_spikes = int(idx.size)
        fr = (n_spikes / duration_s) if (np.isfinite(duration_s) and duration_s > 0) else np.nan

        isi = compute_isi_index_paper(st[idx], fs=fs)
        isi_pass = int(np.isfinite(isi) and (isi < args.isi_thresh))

        lr = compute_l_ratio(cid, clu, X_all, p, args.max_in, args.max_other)
        l_pass = int(np.isfinite(lr) and (lr < args.lratio_thresh))

        # --- FIND THE DOMINANT TEMPLATE FOR THIS CLUSTER ---
        cluster_templates = tpl_assign[idx]
        if cluster_templates.size > 0:
            dominant_template = np.bincount(cluster_templates).argmax()
        else:
            dominant_template = -1

        # Calculate P2T based on the dominant template!
        p2t = compute_template_p2t(dominant_template, templates, tpl_assign, amplitudes, winv)
        p2t_pass = int(np.isfinite(p2t) and (p2t >= args.p2t_thresh)) if args.p2t_thresh > 0 else -1

        # Evaluate if the cluster passed everything
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
            # Add the dominant template to our export list
            if dominant_template != -1:
                good_templates_for_online.append(dominant_template)

        lines.append(
            f"{int(cid)}\t{int(dominant_template)}\t{n_spikes}\t{duration_s:.6f}\t{fr:.6f}\t"
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
    
    # --- 5. Export the filtered Wall matrix ---
    # Convert list to unique IDs in case multiple clusters map to the same template
    unique_good_templates = list(set(good_templates_for_online))
    filter_and_save_wall(ks_dir, out_path.parent, unique_good_templates)

if __name__ == "__main__":
    main()
