#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch

from kilosort.io import load_probe, BinaryFiltered, save_ops, load_ops
from kilosort.run_kilosort import (
    initialize_ops,
    compute_preprocessing,
    compute_drift_correction,
    detect_spikes,
    cluster_spikes,
)
# New Import for Quality Metrics
from kilosort import quality_metrics
from kilosort.io import BinaryFiltered
from kilosort import preprocessing
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.preprocessing import get_highpass_filter
from kilosort.gui.logger import setup_logger
from kilosort import template_matching
from kilosort.io import save_to_phy, load_probe

logger = setup_logger(__name__)

HOUR_SEC = 1800.0

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def export_for_online(ops: dict, Wall: torch.Tensor, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    save_ops(ops, results_dir=out_dir)
    np.save(out_dir / "Wall.npy", Wall.detach().cpu().numpy())
    logger.info(f"[EXPORT] Saved filtered ops.npy + Wall.npy to: {out_dir}")

def build_bfile_from_ops(bin_path, probe, ops, device, tmin=0.0, tmax=np.inf):
    s = ops["settings"]
    nchan = int(s["n_chan_bin"])
    fs = float(s["fs"])
    shift = s.get("shift", 0.0)
    shift = 0.0 if shift is None else float(shift)
    scale = s.get("scale", 1.0)
    scale = 1.0 if scale is None else float(scale)
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
        ds = np.asarray(bfile.dshift)
        if ds.ndim != 1: ds = ds.reshape(-1)
        if len(ds) < nb:
            pad = np.full(nb - len(ds), ds[-1], dtype=ds.dtype)
            ds_padded = np.concatenate([ds, pad], axis=0)
            bfile.dshift = ds_padded
    return bfile

def offline_flow_first_hour(bin_path: Path, probe_path: Path, out_dir: Path,
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
    settings["filename"]   = str(bin_path)
    settings["data_dir"]   = str(bin_path.parent)
    settings["results_dir"] = str(out_dir)

    ops, settings = initialize_ops(settings, probe, dtype, True, False, device, False, False)
    ops["settings"] = settings
    ops["filename"] = settings["filename"]
    ops["data_dir"] = settings["data_dir"]

    ops = compute_preprocessing(ops, device=device)
    ops, bfile, st0 = compute_drift_correction(ops, device=device)
    st, tF, Wall0, clu0 = detect_spikes(ops, device=device, bfile=bfile)
    
    # (D) Full Clustering
    clu, Wall, st, tF = cluster_spikes(st, tF, ops, device, bfile)

    # --- NEW: QUALITY METRICS & FILTERING ---
    logger.info("[OFFLINE] Calculating quality metrics...")
    # Calculate automated labels (good/mua/noise)
    ops, df = quality_metrics.compute_quality_metrics(ops, bfile, st, Wall, device)
    
    # Identify 'good' clusters
    # KS4 stores labels in ops['KSLabel'] or via the returned dataframe
    good_mask = (ops['KSLabel'] == 'good')
    good_indices = np.where(good_mask)[0]
    
    if len(good_indices) == 0:
        logger.warning("[OFFLINE] No 'good' clusters found! Exporting all clusters to avoid crash.")
    else:
        logger.info(f"[OFFLINE] Filtering: Keeping {len(good_indices)} 'good' clusters out of {len(Wall)}.")
        # Filter Wall: (K, Chan, PCs) -> (K_good, Chan, PCs)
        Wall = Wall[good_indices]
        
        # Update ops to reflect filtered clusters (optional but cleaner)
        if 'iCC' in ops: ops['iCC'] = ops['iCC'][good_indices]
        if 'iU' in ops: ops['iU'] = ops['iU'][good_indices]
    # ----------------------------------------

    np.save(out_dir / "Wall.npy", Wall.detach().cpu().numpy())
    export_for_online(ops, Wall, out_dir)
    logger.info("[OFFLINE] Done.")
    return ops, Wall

def online_flow_rest(bin_path: Path, probe_path: Path,
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
    if Wall.ndim == 3 and Wall.shape[-1] == n_pcs:
        Wall = Wall.permute(0, 2, 1).contiguous()

    logger.info(f"[ONLINE] Running extraction with {len(Wall)} filtered templates.")
    st, tF, ops = template_matching.extract(ops, bfile_rest, Wall, device=device)
    
    imin = bfile_rest.imin
    phy_dir = out_dir / "phy"
    phy_dir.mkdir(parents=True, exist_ok=True)

    def ensure_ops_tensors(ops, device):
        keys = ["iCC", "iCC_mask", "iU", "wPCA", "yblk", "xblk"]
        for k in keys:
            if k in ops:
                ops[k] = torch.as_tensor(ops[k], device=device)
        return ops

    ops = ensure_ops_tensors(ops, device)
    
    save_to_phy(st, st[:, 1].astype(np.int32), tF.to(device), Wall, probe, ops, imin, phy_dir, ops["settings"].get("data_dtype", "int16"), False)

    np.save(out_dir / "st.npy", st)
    save_ops(ops, results_dir=out_dir)
    logger.info("[ONLINE] Done.")

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

    offline_flow_first_hour(Path(args.bin), Path(args.probe), Path(args.out_offline), args.fs, args.nchan, args.dtype)
    online_flow_rest(Path(args.bin), Path(args.probe), Path(args.out_offline), Path(args.out_online))

if __name__ == "__main__":
    main()