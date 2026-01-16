#!/usr/bin/env python3
"""
Two-stage Kilosort4 flow:
  1) OFFLINE (first hour): learn templates (Wall) + export ops
  2) ONLINE/ON-CHIP (rest): reuse exported ops + run template matching (extract) only
  
Note:
this code passses all the clutsers from the offline to the online including mua clusters.

Example:
python flows.py \
  --bin /path/to/data.dat \
  --probe /path/to/probe.json \
  --out_offline /path/to/offline_out \
  --out_online  /path/to/online_out \
  --fs 20000 \
  --nchan 128 \
  --dtype int16
"""

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
from kilosort.io import BinaryFiltered
from kilosort import preprocessing
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.preprocessing import get_highpass_filter
from kilosort.gui.logger import setup_logger
from kilosort import template_matching
from kilosort.io import save_to_phy, load_probe

logger = setup_logger(__name__)

HOUR_SEC = 3600.0


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_for_online(ops: dict, Wall: torch.Tensor, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    save_ops(ops, results_dir=out_dir)
    np.save(out_dir / "Wall.npy", Wall.detach().cpu().numpy())
    logger.info(f"[EXPORT] Saved ops.npy + Wall.npy to: {out_dir}")


def build_bfile_from_ops(bin_path, probe, ops, device, tmin=0.0, tmax=np.inf):
    """
    Build BinaryFiltered from ops/probe and PAD dshift so it covers all batches.

    Drift padding rule:
      - If online run has more batches than offline dshift length,
        repeat the LAST drift value for the remaining batches.
    """

    s = ops["settings"]

    nchan = int(s["n_chan_bin"])
    fs = float(s["fs"])

    # shift/scale can be None in ops/settings -> set safe defaults
    shift = s.get("shift", 0.0)
    shift = 0.0 if shift is None else float(shift)

    scale = s.get("scale", 1.0)
    scale = 1.0 if scale is None else float(scale)

    NT = int(ops.get("NT", s.get("batch_size", 60000)))
    nt = int(ops.get("nt", 61))
    nt0min = int(ops.get("nt0min", 20))

    # highpass filter
    hp_filter = preprocessing.get_highpass_filter(
        fs=fs,
        cutoff=float(s.get("highpass_cutoff", 300.0)),
        device=device,
    )

    # whitening matrix must exist from offline stage
    Wrot = ops.get("Wrot", None)
    if Wrot is None:
        raise ValueError("ops['Wrot'] missing. Offline stage must compute whitening and save ops.")

    # offline drift vector (might be shorter than online batches)
    dshift = ops.get("dshift", None)
    if dshift is not None:
        # ensure it's a CPU numpy array (BinaryFiltered will index it)
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
        dshift=dshift,  # temporary (we will pad below if needed)
        device=device,
        do_CAR=bool(s.get("do_CAR", True)),
        artifact_threshold=float(s.get("artifact_threshold", np.inf)),
        invert_sign=bool(s.get("invert_sign", False)),
        dtype=str(s.get("data_dtype", "int16")),
        tmin=float(tmin),
        tmax=float(tmax),
    )

    # --------- PAD DRIFT TO MATCH ONLINE RUN LENGTH ----------
    if bfile.dshift is not None:
        nb = int(bfile.n_batches)
        ds = np.asarray(bfile.dshift)

        if ds.ndim != 1:
            ds = ds.reshape(-1)

        if len(ds) < nb:
            # repeat last offline drift value for remaining batches
            pad = np.full(nb - len(ds), ds[-1], dtype=ds.dtype)
            ds_padded = np.concatenate([ds, pad], axis=0)
            bfile.dshift = ds_padded  # update bfile so indexing is safe

        # (optional) print to confirm
        # print("dshift len:", len(bfile.dshift), "n_batches:", bfile.n_batches)

    return bfile


def offline_flow_first_hour(bin_path: Path, probe_path: Path, out_dir: Path,
                            fs: float, nchan: int, dtype: str):
    """
    Offline:
      - init ops (with DEFAULT_SETTINGS merged)
      - compute preprocessing (hp + whitening)
      - compute drift correction (datashift.run)
      - detect spikes (includes initial clustering + extract)
      - final clustering + merge (cluster_spikes)
      - export ops + final templates (Wall)
    """
    out_dir = ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = load_probe(probe_path)

    # Minimal overrides; everything else comes from DEFAULT_SETTINGS
    user_settings = {
        "fs": float(fs),
        "n_chan_bin": int(nchan),

        # restrict to first hour for OFFLINE learning
        "tmin": 0.0,
        "tmax": float(HOUR_SEC),

        # keep consistent with your needs
        "n_pcs": 3,
        "nearest_chans": 10,
        "position_limit": 100.0,

        # important to have explicit nt/nt0min if you want stable shapes
        "nt": 61,
        "nt0min": 20,

        # optional
        "batch_size": 60000,
        "highpass_cutoff": 300.0,
    }

    # MERGE DEFAULTS (fixes KeyError like templates_from_data, duplicate_spike_ms, etc.)
    settings = {**DEFAULT_SETTINGS, **user_settings}
    settings["filename"]   = str(bin_path)
    settings["data_dir"]   = str(bin_path.parent)
    settings["results_dir"] = str(out_dir)

    # IMPORTANT: these must NOT be inside settings (initialize_ops will reject them)
    do_CAR = True
    invert_sign = False
    save_preprocessed_copy = False
    data_dtype = dtype

    # initialize_ops returns (ops, settings)
    ops, settings = initialize_ops(
        settings=settings,
        probe=probe,
        data_dtype=data_dtype,
        do_CAR=do_CAR,
        invert_sign=invert_sign,
        device=device,
        save_preprocessed_copy=save_preprocessed_copy,
        gui_mode=False,
    )
    ops["settings"] = settings  # keep the returned normalized settings

    # compute_preprocessing expects these at top-level in ops
    ops["filename"] = settings["filename"]
    ops["data_dir"] = settings["data_dir"]
    # optional but often useful
    ops["results_dir"] = settings.get("results_dir", ops.get("results_dir", None))


    # (A) preprocessing (hp + whitening)
    ops = compute_preprocessing(ops, device=device, file_object=None)

    # (B) drift correction (datashift + drift-corrected bfile)
    ops, bfile, st0 = compute_drift_correction(
        ops, device=device, file_object=None, clear_cache=False, verbose=False
    )

    # (C) detect + initial clustering + extract
    st, tF, Wall0, clu0 = detect_spikes(
        ops, device=device, bfile=bfile, clear_cache=False, verbose=False
    )

    # (D) final clustering + merge -> final templates Wall
    clu, Wall, st, tF = cluster_spikes(
        st=st, tF=tF, ops=ops, device=device, bfile=bfile, clear_cache=False, verbose=False
    )
    np.save(out_dir / "Wall.npy", Wall.detach().cpu().numpy())
    export_for_online(ops, Wall, out_dir)
    logger.info("[OFFLINE] Done.")
    return ops, Wall


def online_flow_rest(bin_path: Path, probe_path: Path,
                     exported_offline_dir: Path, out_dir: Path):
    """
    Online/on-chip-ish:
      - load ops + Wall
      - run template_matching.extract on t=[1h, end]
    """
    out_dir = ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = load_probe(probe_path)

    ops = load_ops(exported_offline_dir / "ops.npy", device=device)
    Wall_np = np.load(exported_offline_dir / "Wall.npy")
    Wall = torch.from_numpy(Wall_np).to(device)

    # process the rest of the file only
    ops["settings"]["tmin"] = float(HOUR_SEC)
    ops["settings"]["tmax"] = float(np.inf)

    bfile_rest = build_bfile_from_ops(
        bin_path=bin_path,
        probe=probe,
        ops=ops,
        device=device,
        tmin=float(HOUR_SEC),
        tmax=float(np.inf),
    )

    Wall_np = np.load(exported_offline_dir / "Wall.npy")
    Wall = torch.from_numpy(Wall_np).to(device)

    # ---- FIX TEMPLATE AXIS ORDER ----
    n_pcs = int(ops["settings"]["n_pcs"])  # should be 3

    # Wall coming from cluster_spikes is usually (K, C, n_pcs).
    # template_matching.prepare_matching expects (K, n_pcs, C) in your KS version.
    if Wall.ndim == 3 and Wall.shape[-1] == n_pcs:
        # (K, C, n_pcs) -> (K, n_pcs, C)
        Wall = Wall.permute(0, 2, 1).contiguous()

    # sanity prints (optional but helpful)
    print("Wall shape used for extract:", tuple(Wall.shape), "n_pcs:", n_pcs)
    # ---------------------------------


    st, tF, ops = template_matching.extract(ops, bfile_rest, Wall, device=device)
    clu = st[:, 1].astype(np.int32)   # cluster == template id

    # ---- load Wall learned offline ----
    Wall_np = np.load(exported_offline_dir / "Wall.npy")
    Wall = torch.from_numpy(Wall_np).to(device)

    # ---- make "clusters" for phy: cluster == template id ----
    # st is (n_spikes, 3): [time, template_id, score]
    spike_clusters = st[:, 1].astype(np.int32)

    # ---- IMPORTANT: imin makes spike_times global in the original binary ----
    imin = bfile_rest.imin  # start sample index of the online segment

    # ---- write a phy-compatible folder ----
    phy_dir = out_dir / "phy"          # e.g. out_online/phy
    phy_dir.mkdir(parents=True, exist_ok=True)

    def ensure_ops_tensors_on_device(ops, device):
        # keys that are used downstream by compute_spike_positions / template logic
        keys = ["iCC", "iCC_mask", "iU", "wPCA"]
        for k in keys:
            if k in ops and not isinstance(ops[k], torch.Tensor):
                ops[k] = torch.as_tensor(ops[k], device=device)
            elif k in ops and isinstance(ops[k], torch.Tensor) and ops[k].device != device:
                ops[k] = ops[k].to(device)

        # sometimes these exist and can participate indirectly
        for k in ["yblk", "xblk"]:
            if k in ops and isinstance(ops[k], torch.Tensor) and ops[k].device != device:
                ops[k] = ops[k].to(device)

        return ops

    device = tF.device
    ops = ensure_ops_tensors_on_device(ops, device)

    # also make sure templates are on same device
    if isinstance(Wall, torch.Tensor) and Wall.device != device:
        Wall = Wall.to(device)



    ret = save_to_phy(
        st=st,
        clu=spike_clusters,       # cluster == template id
        tF=tF.to(device),
        Wall=Wall,
        probe=probe,
        ops=ops,
        imin=imin,
        results_dir=phy_dir,
        data_dtype=ops["settings"].get("data_dtype", "int16"),
        save_extra_vars=False
    )

    results_dir = ret[0]

    print("✅ Phy-compatible output written to:", results_dir)

    np.save(out_dir / "st.npy", st)
    np.save(out_dir / "tF.npy", tF.cpu().numpy() if hasattr(tF, "cpu") else np.array(tF))
    save_ops(ops, results_dir=out_dir)

    logger.info("[ONLINE] Done.")
    logger.info(f"[ONLINE] Saved: {out_dir/'st.npy'} and {out_dir/'tF.npy'}")
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

    bin_path = Path(args.bin).expanduser().resolve()
    probe_path = Path(args.probe).expanduser().resolve()
    out_offline = Path(args.out_offline).expanduser().resolve()
    out_online = Path(args.out_online).expanduser().resolve()

    offline_flow_first_hour(
        bin_path=bin_path,
        probe_path=probe_path,
        out_dir=out_offline,
        fs=args.fs,
        nchan=args.nchan,
        dtype=args.dtype,
    )

    online_flow_rest(
        bin_path=bin_path,
        probe_path=probe_path,
        exported_offline_dir=out_offline,
        out_dir=out_online,
    )


if __name__ == "__main__":
    main()
