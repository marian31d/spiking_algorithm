#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import json

from kilosort.io import load_probe, BinaryFiltered, save_ops, load_ops
from kilosort.run_kilosort import (
    initialize_ops,
    compute_preprocessing,
    compute_drift_correction,
    detect_spikes,
    cluster_spikes,
)
from kilosort import preprocessing
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.gui.logger import setup_logger
from kilosort import template_matching
from kilosort.io import save_to_phy

"""
Two-stage Kilosort4 flow:
  1) OFFLINE (first hour): learn templates (Wall) + export ops
  2) ONLINE/ON-CHIP (rest): reuse exported ops + run template matching (extract) only

This version:
- DOES NOT filter Wall
- Writes offline statistics (good/mua counts, total clusters, etc.)
- Exports Phy on CPU in ONLINE to avoid CUDA OOM inside save_to_phy()
- Uses correct Wall layout for extract vs save_to_phy:
    * extract() uses (K, PCs, Chan)
    * save_to_phy() expects (K, Chan, PCs)
"""

logger = setup_logger(__name__)

HOUR_SEC = 2400.0  # edit as you like


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_for_online(ops: dict, Wall: torch.Tensor, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    save_ops(ops, results_dir=out_dir)
    np.save(out_dir / "Wall.npy", Wall.detach().cpu().numpy())
    logger.info(f"[EXPORT] Saved ops.npy + Wall.npy to: {out_dir}")


def write_offline_stats(phy_dir: Path, out_dir: Path, Wall: torch.Tensor, st: np.ndarray):
    """
    Writes offline statistics into:
      - out_dir/offline_stats.txt
      - out_dir/offline_stats.json

    Uses (if present):
      - cluster_KSLabel.tsv or cluster_group.tsv
      - spike_clusters.npy
      - spike_templates.npy
    """
    phy_dir = Path(phy_dir)
    out_dir = Path(out_dir)

    # -------- read labels (cluster_id -> label) --------
    labels = {}
    labels_file_used = None
    candidates = [("cluster_KSLabel.tsv", "KSLabel"), ("cluster_group.tsv", "group")]

    for fname, labcol in candidates:
        f = phy_dir / fname
        if not f.exists():
            continue
        lines = f.read_text().strip().splitlines()
        if len(lines) < 2:
            continue
        header = lines[0].split("\t")
        if "cluster_id" not in header or labcol not in header:
            continue
        i_id = header.index("cluster_id")
        i_lab = header.index(labcol)

        for ln in lines[1:]:
            parts = ln.split("\t")
            if len(parts) <= max(i_id, i_lab):
                continue
            try:
                cid = int(parts[i_id])
            except Exception:
                continue
            lab = str(parts[i_lab]).strip().lower()
            labels[cid] = lab

        labels_file_used = fname
        break

    label_counts = {}
    good_cluster_ids = []
    mua_cluster_ids = []

    if labels_file_used is not None and len(labels) > 0:
        for cid, lab in labels.items():
            label_counts[lab] = label_counts.get(lab, 0) + 1
            if lab == "good":
                good_cluster_ids.append(cid)
            elif lab == "mua":
                mua_cluster_ids.append(cid)
        good_cluster_ids = sorted(set(good_cluster_ids))
        mua_cluster_ids = sorted(set(mua_cluster_ids))

    # -------- spike/template arrays (optional) --------
    spike_clusters_path = phy_dir / "spike_clusters.npy"
    spike_templates_path = phy_dir / "spike_templates.npy"

    spike_clusters = None
    spike_templates = None

    if spike_clusters_path.exists():
        spike_clusters = np.load(spike_clusters_path).astype(np.int64, copy=False)
    if spike_templates_path.exists():
        spike_templates = np.load(spike_templates_path).astype(np.int64, copy=False)

    # -------- compute useful stats --------
    K = int(Wall.shape[0])
    n_spikes = int(st.shape[0])

    st_clusters = st[:, 1].astype(np.int64, copy=False) if st.ndim == 2 and st.shape[1] >= 2 else None
    unique_st_clusters = int(np.unique(st_clusters).size) if st_clusters is not None else None

    spc_stats = {}
    if st_clusters is not None:
        u, counts = np.unique(st_clusters, return_counts=True)
        spc_stats = {
            "clusters_with_spikes": int(u.size),
            "spikes_per_cluster_min": int(counts.min()) if counts.size else None,
            "spikes_per_cluster_median": float(np.median(counts)) if counts.size else None,
            "spikes_per_cluster_mean": float(counts.mean()) if counts.size else None,
            "spikes_per_cluster_p90": float(np.quantile(counts, 0.90)) if counts.size else None,
            "spikes_per_cluster_max": int(counts.max()) if counts.size else None,
        }

    label_spike_counts = {}
    if spike_clusters is not None and labels_file_used is not None and len(labels) > 0:
        labs = np.array([labels.get(int(c), "unlabeled") for c in spike_clusters], dtype=object)
        u, cts = np.unique(labs, return_counts=True)
        label_spike_counts = {str(k): int(v) for k, v in zip(u, cts)}

    summary = {
        "phy_dir": str(phy_dir),
        "labels_file_used": labels_file_used,
        "Wall_num_templates_K": K,
        "offline_num_spikes_st": n_spikes,
        "offline_unique_clusters_in_st": unique_st_clusters,
        "clusters_in_tsv_total": int(len(labels)) if labels_file_used is not None else None,
        "cluster_label_counts_from_tsv": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))) if label_counts else None,
        "good_clusters_in_tsv": int(len(good_cluster_ids)) if labels_file_used is not None else None,
        "mua_clusters_in_tsv": int(len(mua_cluster_ids)) if labels_file_used is not None else None,
        "label_spike_counts_from_spike_clusters": label_spike_counts if label_spike_counts else None,
        "spikes_per_cluster_stats_from_st": spc_stats if spc_stats else None,
        "has_spike_clusters_npy": bool(spike_clusters_path.exists()),
        "has_spike_templates_npy": bool(spike_templates_path.exists()),
        "note": "Wall is NOT filtered. These stats are for reporting only.",
    }

    (out_dir / "offline_stats.json").write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append("OFFLINE FLOW STATISTICS")
    lines.append(f"phy_dir: {phy_dir}")
    lines.append(f"labels_file_used: {labels_file_used}")
    lines.append("")
    lines.append(f"Wall templates (K): {K}")
    lines.append(f"Spikes in st: {n_spikes}")
    lines.append(f"Unique st clusters: {unique_st_clusters}")
    lines.append("")
    if labels_file_used is not None:
        lines.append(f"Clusters in TSV total: {len(labels)}")
        lines.append("Label counts (TSV):")
        for k, v in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  {k}: {v}")
        lines.append(f"Good clusters: {len(good_cluster_ids)}")
        lines.append(f"MUA clusters: {len(mua_cluster_ids)}")
    else:
        lines.append("No TSV labels found (cluster_KSLabel.tsv / cluster_group.tsv).")

    if label_spike_counts:
        lines.append("")
        lines.append("Spike counts by label (from spike_clusters.npy):")
        for k, v in sorted(label_spike_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"  {k}: {v}")

    if spc_stats:
        lines.append("")
        lines.append("Spikes-per-cluster (from st[:,1]) stats:")
        for k, v in spc_stats.items():
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append(f"Wrote: {out_dir / 'offline_stats.txt'}")
    lines.append(f"Wrote: {out_dir / 'offline_stats.json'}")
    (out_dir / "offline_stats.txt").write_text("\n".join(lines))

    logger.info(f"[OFFLINE-STATS] Wrote offline_stats.txt/json into {out_dir}")


def build_bfile_from_ops(bin_path, probe, ops, device, tmin=0.0, tmax=np.inf):
    s = ops["settings"]
    nchan = int(s["n_chan_bin"])
    fs = float(s["fs"])
    NT = int(ops.get("NT", s.get("batch_size", 60000)))
    nt = int(ops.get("nt", 61))
    nt0min = int(ops.get("nt0min", 20))

    hp_filter = preprocessing.get_highpass_filter(
        fs=fs,
        cutoff=float(s.get("highpass_cutoff", 300.0)),
        device=device,
    )

    Wrot = ops.get("Wrot", None)
    if Wrot is None:
        raise ValueError("ops['Wrot'] missing.")

    dshift = ops.get("dshift", None)
    if dshift is not None:
        if isinstance(dshift, torch.Tensor):
            dshift = dshift.detach().cpu().numpy()
        else:
            dshift = np.asarray(dshift)

    bfile = BinaryFiltered(
        filename=str(bin_path),
        n_chan_bin=nchan,
        fs=int(fs),
        NT=NT,
        nt=nt,
        nt0min=nt0min,
        chan_map=probe["chanMap"],
        hp_filter=hp_filter,
        whiten_mat=Wrot,
        dshift=dshift,
        device=device,
        do_CAR=bool(s.get("do_CAR", True)),
        artifact_threshold=float(s.get("artifact_threshold", np.inf)),
        invert_sign=bool(s.get("invert_sign", False)),
        dtype=str(s.get("data_dtype", "int16")),
        tmin=float(tmin),
        tmax=float(tmax),
    )

    if bfile.dshift is not None:
        nb = int(bfile.n_batches)
        ds = np.asarray(bfile.dshift).reshape(-1)
        if len(ds) < nb:
            pad = np.full(nb - len(ds), ds[-1], dtype=ds.dtype)
            bfile.dshift = np.concatenate([ds, pad], axis=0)

    return bfile


def offline_flow(bin_path: Path, probe_path: Path, out_dir: Path,
                 fs: float, nchan: int, dtype: str):
    out_dir = ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = load_probe(probe_path)

    user_settings = {
        "fs": float(fs),
        "n_chan_bin": int(nchan),
        "tmin": 0.0,
        "tmax": float(HOUR_SEC),
        "n_pcs": 3,
        "nearest_chans": 10,
        "position_limit": 100.0,
        "nt": 61,
        "nt0min": 20,
        "batch_size": 60000,
        "highpass_cutoff": 300.0,
    }

    settings = {**DEFAULT_SETTINGS, **user_settings}
    settings["filename"] = str(bin_path)
    settings["data_dir"] = str(bin_path.parent)
    settings["results_dir"] = str(out_dir)

    ops, settings = initialize_ops(settings, probe, dtype, True, False, device, False, False)
    ops["settings"] = settings
    ops["filename"] = settings["filename"]
    ops["data_dir"] = settings["data_dir"]

    ops = compute_preprocessing(ops, device=device)
    ops, bfile, st0 = compute_drift_correction(ops, device=device)
    st, tF, Wall0, clu0 = detect_spikes(ops, device=device, bfile=bfile)

    clu, Wall, st, tF = cluster_spikes(st, tF, ops, device, bfile)

    phy_dir = out_dir / "phy_offline"
    phy_dir.mkdir(parents=True, exist_ok=True)

    spike_clusters = st[:, 1].astype(np.int32)
    imin = bfile.imin

    ret = save_to_phy(
        st=st,
        clu=spike_clusters,
        tF=tF,
        Wall=Wall,
        probe=probe,
        ops=ops,
        imin=imin,
        results_dir=phy_dir,
        data_dtype=ops["settings"].get("data_dtype", "int16"),
        save_extra_vars=False,
    )
    phy_used = Path(ret[0]) if isinstance(ret, (list, tuple)) else phy_dir

    write_offline_stats(phy_used, out_dir, Wall, st)

    export_for_online(ops, Wall, out_dir)
    logger.info("[OFFLINE] Done.")
    return ops, Wall


def online_flow(bin_path: Path, probe_path: Path,
                exported_offline_dir: Path, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = load_probe(probe_path)

    ops = load_ops(exported_offline_dir / "ops.npy", device=device)
    Wall_np = np.load(exported_offline_dir / "Wall.npy")
    Wall = torch.from_numpy(Wall_np).to(device)

    ops["settings"]["tmin"] = float(HOUR_SEC)
    ops["settings"]["tmax"] = float(np.inf)

    bfile_rest = build_bfile_from_ops(bin_path, probe, ops, device, float(HOUR_SEC), float(np.inf))

    n_pcs = int(ops["settings"]["n_pcs"])

    # Keep Wall in (K, Chan, PCs) for Phy export later
    Wall_phy = Wall

    # Create a separate version for extract (K, PCs, Chan) if needed
    Wall_match = Wall
    if Wall_match.ndim == 3 and Wall_match.shape[-1] == n_pcs:
        Wall_match = Wall_match.permute(0, 2, 1).contiguous()

    logger.info(f"[ONLINE] Running extraction with {len(Wall_match)} templates.")
    st, tF, ops = template_matching.extract(ops, bfile_rest, Wall_match, device=device)

    imin = bfile_rest.imin

    phy_dir = out_dir / "phy"
    phy_dir.mkdir(parents=True, exist_ok=True)

    spike_clusters = st[:, 1].astype(np.int32)

    # ---------------------------
    # Export to Phy on CPU to avoid CUDA OOM AND keep correct Wall layout for save_to_phy
    # save_to_phy expects Wall as (K, Chan, PCs) where PCs = n_pcs
    # ---------------------------

    Wall_export = Wall_phy
    # If somehow Wall_phy is (K, PCs, Chan), convert back
    if Wall_export.ndim == 3 and Wall_export.shape[-1] != n_pcs and Wall_export.shape[1] == n_pcs:
        Wall_export = Wall_export.permute(0, 2, 1).contiguous()
    Wall_export = Wall_export.detach().cpu()

    # tF export on CPU
    tF_export = tF.detach().cpu() if isinstance(tF, torch.Tensor) else tF

    # ops['wPCA'] must be CPU torch tensor for save_to_phy (it calls .contiguous()).
    if "wPCA" in ops:
        if isinstance(ops["wPCA"], np.ndarray):
            ops["wPCA"] = torch.from_numpy(ops["wPCA"]).contiguous()
        elif isinstance(ops["wPCA"], torch.Tensor):
            ops["wPCA"] = ops["wPCA"].detach().cpu().contiguous()

    # Geometry fields must be numpy arrays (postprocessing uses torch.from_numpy on them)
    for k in ["xc", "yc", "kcoords", "chanMap", "xblk", "yblk"]:
        if k in ops and isinstance(ops[k], torch.Tensor):
            ops[k] = ops[k].detach().cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_to_phy(
        st=st,
        clu=spike_clusters,
        tF=tF_export,
        Wall=Wall_export,
        probe=probe,
        ops=ops,
        imin=imin,
        results_dir=phy_dir,
        data_dtype=ops["settings"].get("data_dtype", "int16"),
        save_extra_vars=False,
    )

    np.save(out_dir / "st.npy", st)
    np.save(out_dir / "tF.npy", tF_export.detach().cpu().numpy() if isinstance(tF_export, torch.Tensor) else np.array(tF_export))
    save_ops(ops, results_dir=out_dir)

    logger.info("[ONLINE] Done.")
    return st, tF, ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, type=str)
    ap.add_argument("--probe", required=True, type=str)
    ap.add_argument("--out_offline", required=True, type=str)
    ap.add_argument("--out_online", required=True, type=str)
    ap.add_argument("--fs", required=True, type=float)
    ap.add_argument("--nchan", required=True, type=int)
    ap.add_argument("--dtype", default="int16", type=str)
    args = ap.parse_args()

    offline_flow(Path(args.bin), Path(args.probe), Path(args.out_offline), args.fs, args.nchan, args.dtype)
    online_flow(Path(args.bin), Path(args.probe), Path(args.out_offline), Path(args.out_online))


if __name__ == "__main__":
    main()
