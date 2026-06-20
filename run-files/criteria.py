#!/usr/bin/env python3
"""
Compute QC criteria per cluster using ONLY Kilosort outputs (no SpikeInterface).

Outputs TSV with:
- firing_rate_hz
- ISI-index (paper-style) + pass (<0.2)
- L-ratio (from pc_features.npy) + pass (<0.05)
- peak-to-trough amplitude in µV (from unwhitened template if whitening available) + pass (>= 50 µV)

REQUIREMENTS (Kilosort outputs):
  - ops.npy
  - spike_times.npy
  - spike_clusters.npy
  - templates.npy
  - pc_features.npy   (needed for L-ratio)

OPTIONAL (recommended for correct mapping cluster->template):
  - spike_templates.npy

OPTIONAL (for µV conversion from templates):
  - whitening_mat_inv.npy  (best)
    OR whitening_mat.npy   (will invert)

Usage:
  python criteria_kilosort_only.py --ks_dir /path/to/ks --out /path/to/criteria.tsv

Optional knobs:
  --uv_per_bit 0.195
  --p2t_thresh_uv 50
  --isi_thresh 0.2
  --lratio_thresh 0.05
  --max_other 30000
  --max_in 5000
"""

import argparse
from pathlib import Path
import numpy as np

try:
    from scipy.stats import chi2
    _HAS_SCIPY_STATS = True
except Exception:
    _HAS_SCIPY_STATS = False
    from scipy.special import gammainc  # still SciPy, but lighter than stats


def chi2_cdf(x, df):
    x = np.asarray(x, dtype=np.float64)
    df = float(df)
    if _HAS_SCIPY_STATS:
        return chi2.cdf(x, df=df)
    # CDF = gammainc(k/2, x/2)
    return gammainc(df / 2.0, x / 2.0)


def compute_isi_index_paper(spike_times_samples, fs, isi1_ms=2.0, isi2_ms=20.0):
    st = np.asarray(spike_times_samples, dtype=np.int64)
    if st.size < 3:
        return np.nan

    st = np.sort(st)
    isi_ms = np.diff(st) * 1000.0 / float(fs)

    n_0_20 = np.sum((isi_ms > 0) & (isi_ms <= isi2_ms))
    n_0_2  = np.sum((isi_ms > 0) & (isi_ms <= isi1_ms))

    expected_0_2 = n_0_20 * (isi1_ms / isi2_ms)  # 2/20
    if expected_0_2 <= 0:
        return np.nan
    return float(n_0_2 / expected_0_2)


def stable_inv_cov(X, reg_eps=1e-6, svd_floor=1e-8):
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
    X = np.asarray(X, dtype=np.float64) - mu
    return np.einsum("ij,jk,ik->i", X, inv_cov, X)


def load_first_existing(base: Path, names):
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


def load_whitening_inverse(ks_dir: Path):
    """
    Try to load whitening inverse from common filenames.
    Returns W_inv (C,C) or None if unavailable.
    """
    p_inv = load_first_existing(ks_dir, ["whitening_mat_inv.npy", "whitening_matrix_inv.npy"])
    p_w   = load_first_existing(ks_dir, ["whitening_mat.npy", "whitening_matrix.npy"])

    if p_inv is not None:
        W_inv = np.load(p_inv)
        return W_inv

    if p_w is not None:
        W = np.load(p_w)
        try:
            return np.linalg.inv(W)
        except np.linalg.LinAlgError:
            # fall back to pseudo-inverse
            return np.linalg.pinv(W)

    return None


def dominant_template_per_cluster(clu, spike_templates, cluster_ids, n_templates):
    """
    Map each cluster_id -> dominant template id based on spike_templates votes.
    Returns dict cluster_id -> template_id.
    """
    mapping = {}
    for cid in cluster_ids:
        idx = np.flatnonzero(clu == cid)
        if idx.size == 0:
            continue
        tpls = spike_templates[idx]
        tpls = tpls[(tpls >= 0) & (tpls < n_templates)]
        if tpls.size == 0:
            continue
        counts = np.bincount(tpls, minlength=n_templates)
        mapping[int(cid)] = int(np.argmax(counts))
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks_dir", required=True, help="Kilosort output folder")
    ap.add_argument("--out", required=True, help="Output TSV path")

    ap.add_argument("--uv_per_bit", type=float, default=0.195, help="µV per ADC count")
    ap.add_argument("--p2t_thresh_uv", type=float, default=50.0, help="P2T threshold in µV (pass if >=)")

    ap.add_argument("--isi_thresh", type=float, default=0.2, help="ISI-index threshold (pass if <)")
    ap.add_argument("--lratio_thresh", type=float, default=0.05, help="L-ratio threshold (pass if <)")

    ap.add_argument("--max_other", type=int, default=30000, help="Max non-cluster spikes for L-ratio")
    ap.add_argument("--max_in", type=int, default=5000, help="Max in-cluster spikes for covariance")

    args = ap.parse_args()

    ks_dir = Path(args.ks_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- required Kilosort outputs ---
    st = np.load(ks_dir / "spike_times.npy").astype(np.int64).reshape(-1)
    clu = np.load(ks_dir / "spike_clusters.npy").astype(np.int64).reshape(-1)

    ops = np.load(ks_dir / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])

    templates = np.load(ks_dir / "templates.npy")  # (K, T, C)
    K, T, C = templates.shape

    # PCA features for L-ratio
    pc_feat_path = ks_dir / "pc_features.npy"
    if not pc_feat_path.exists():
        raise FileNotFoundError(
            "pc_features.npy not found in ks_dir. Can't compute L-ratio without it."
        )
    pc_features = np.load(pc_feat_path)  # often (N, nPC, nCh) or similar
    X_all = pc_features.reshape(pc_features.shape[0], -1).astype(np.float32)
    p = int(X_all.shape[1])

    # optional: spike_templates.npy to map clusters->templates correctly
    spike_templates_path = ks_dir / "spike_templates.npy"
    spike_templates = None
    if spike_templates_path.exists():
        spike_templates = np.load(spike_templates_path).astype(np.int64).reshape(-1)

    # whitening inverse (for converting templates back towards ADC counts)
    W_inv = load_whitening_inverse(ks_dir)
    if W_inv is None:
        print("[WARN] Whitening inverse not found. p2t_uV from templates will be NaN.")
    else:
        W_inv = np.asarray(W_inv)
        if W_inv.shape != (C, C):
            print(f"[WARN] Whitening inverse shape {W_inv.shape} doesn't match templates C={C}. Disabling unwhitening.")
            W_inv = None

    # duration for firing rate
    duration_s = float(st.max()) / fs if st.size else np.nan

    cluster_ids = np.unique(clu)

    # cluster -> template mapping
    if spike_templates is not None and spike_templates.size == clu.size:
        cl_to_tpl = dominant_template_per_cluster(clu, spike_templates, cluster_ids, K)
    else:
        cl_to_tpl = {}  # fallback later

    def l_ratio(cluster_id):
        cid = int(cluster_id)

        in_idx = np.flatnonzero(clu == cid)
        n_in_total = int(in_idx.size)
        if n_in_total < 20:
            return np.nan

        if n_in_total > args.max_in:
            in_use = np.random.choice(in_idx, size=args.max_in, replace=False)
        else:
            in_use = in_idx

        X_in = X_all[in_use]
        if X_in.shape[0] < 50:
            return np.nan

        mu, inv_cov = stable_inv_cov(X_in)

        out_idx = np.flatnonzero(clu != cid)
        if out_idx.size == 0:
            return 0.0
        if out_idx.size > args.max_other:
            out_idx = np.random.choice(out_idx, size=args.max_other, replace=False)

        d2 = mahalanobis_d2(X_all[out_idx], mu, inv_cov)
        tail = 1.0 - chi2_cdf(d2, df=p)

        return float(np.sum(tail) / max(n_in_total, 1))

    def p2t_uV_from_template_for_cluster(cluster_id):
        """
        Compute peak-to-trough (µV) from the dominant template of this cluster:
          template -> (optional) unwhiten -> *uv_per_bit -> best-channel p2t.
        Returns (p2t_uV, template_id, best_channel_index)
        """
        cid = int(cluster_id)

        # choose template id
        if cid in cl_to_tpl:
            tid = cl_to_tpl[cid]
        else:
            # fallback: assume cluster id == template id (not always true)
            tid = cid

        if tid < 0 or tid >= K:
            return (np.nan, tid, -1)

        tpl = templates[tid].astype(np.float32)  # (T, C)

        # unwhiten if possible
        if W_inv is None:
            return (np.nan, tid, -1)

        tpl_unw = tpl @ W_inv.T  # (T, C)

        # convert to µV
        tpl_uV = tpl_unw * float(args.uv_per_bit)

        # best channel = max p2t
        p2p_per_ch = np.ptp(tpl_uV, axis=0)  # (C,)
        best_ch = int(np.argmax(p2p_per_ch))
        p2t_uV = float(p2p_per_ch[best_ch])

        return (p2t_uV, tid, best_ch)

    header = [
        "cluster_id",
        "n_spikes",
        "duration_s",
        "firing_rate_hz",
        "isi_index",
        "isi_pass",
        "l_ratio",
        "l_pass",
        "template_id_used",
        "best_channel",
        "p2t_uV",
        "p2t_pass",
    ]

    lines = ["\t".join(header)]

    for cid in cluster_ids:
        idx = np.flatnonzero(clu == cid)
        n_spikes = int(idx.size)

        fr = (n_spikes / duration_s) if (np.isfinite(duration_s) and duration_s > 0) else np.nan

        isi = compute_isi_index_paper(st[idx], fs=fs)
        isi_pass = int(np.isfinite(isi) and (isi < args.isi_thresh))

        lr = l_ratio(cid)
        l_pass = int(np.isfinite(lr) and (lr < args.lratio_thresh))

        p2t_uV, tid, best_ch = p2t_uV_from_template_for_cluster(cid)
        p2t_pass = int(np.isfinite(p2t_uV) and (p2t_uV >= args.p2t_thresh_uv))

        lines.append(
            f"{int(cid)}\t{n_spikes}\t"
            f"{duration_s:.6f}\t{fr:.6f}\t"
            f"{isi if np.isfinite(isi) else 'nan'}\t{isi_pass}\t"
            f"{lr if np.isfinite(lr) else 'nan'}\t{l_pass}\t"
            f"{int(tid)}\t{int(best_ch)}\t"
            f"{p2t_uV if np.isfinite(p2t_uV) else 'nan'}\t{p2t_pass}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved TSV: {out_path}")
    if W_inv is None:
        print("[NOTE] p2t_uV is NaN because whitening inverse was not available/matching.")


if __name__ == "__main__":
    main()
