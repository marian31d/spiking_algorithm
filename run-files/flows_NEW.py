#!/usr/bin/env python3
#This version of flows is deteministic and uses the exact same functions the gui uses to run
# =========================
# DETERMINISM / REPRODUCIBILITY (must be before torch import)
# =========================
import os
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import numpy as np
# Seed Python & NumPy
random.seed(0)
np.random.seed(0)
# =========================

# Seed Torch
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# cuDNN settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable TF32 (important on Ampere+ GPUs)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

#------------------------------------------------------
import argparse
from pathlib import Path
import json

from kilosort.io import load_probe, BinaryFiltered, save_ops, load_ops
from kilosort.run_kilosort import _sort
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.gui.logger import setup_logger
from kilosort import template_matching
from kilosort.preprocessing import get_drift_matrix
from kilosort.preprocessing import get_whitening_matrix
from scipy.signal import butter, lfilter
"""
Two-stage Kilosort4 flow:
  1) OFFLINE (first hour): learn templates (Wall) + export ops
  2) ONLINE/ON-CHIP (rest): reuse exported ops + run template matching (extract) only

#python flows.py   --bin "/media/mariandawud/Extreme SSD/mS51/dat/mS51_49/mS51_49.dat"   --probe /home/mariandawud/kilosort/ms51/probe_ms51_new.json   --out_offline "/media/mariandawud/Extreme SSD/results/mS51_res/flows_new/offline_20mins-51"   --out_online "/media/mariandawud/Extreme SSD/results/mS51_res/flows_new/online_20mins-51"   --fs 20000   --nchan 135   --dtype int16


This version:
- DOES NOT filter Wall
- Writes RUN statistics for BOTH the offline and online phases via write_run_stats()
"""

logger = setup_logger(__name__)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _list_output_files(results_dir):
    """Scan results_dir (recursively) and return the files this run produced.

    Excludes the *_stats.txt / *_stats.json files themselves so that re-runs
    don't list their own previous output.
    """
    results_dir = Path(results_dir)
    out = []
    if not results_dir.exists():
        return out
    for p in sorted(results_dir.rglob("*")):
        if p.is_file() and not p.name.endswith(("_stats.txt", "_stats.json")):
            try:
                sz = int(p.stat().st_size)
            except OSError:
                sz = -1
            out.append({"path": str(p.relative_to(results_dir)), "bytes": sz})
    return out


def _shape_str(x):
    """Best-effort 'shape' string for tensors / arrays / None."""
    if x is None:
        return "None"
    try:
        return str(tuple(x.shape))
    except Exception:
        return "?"


def offline_bfile_params(ops, probe):
    """Reconstruct the bfile parameters used by the OFFLINE _sort() run.

    The offline BinaryFiltered object lives inside _sort(), so we synthesize
    an equivalent parameter dict from the exported ops/settings (+ globals).
    """
    s = ops.get("settings", {}) or {}
    has_wrot = ops.get("Wrot") is not None
    has_dshift = ops.get("dshift") is not None
    return {
        "class": "BinaryFiltered (kilosort _sort internal)",
        "filter": "kilosort default zero-phase FFT highpass",
        "n_chan_bin": int(s["n_chan_bin"]) if s.get("n_chan_bin") is not None else None,
        "fs": float(s["fs"]) if s.get("fs") is not None else None,
        "batch_size_NT": int(s["batch_size"]) if s.get("batch_size") is not None else None,
        "nt": int(ops.get("nt", 61)),
        "nt0min": int(ops.get("nt0min", 20)),
        "n_chan_map": int(len(probe["chanMap"])),
        "do_CAR": bool(DO_CAR),
        "invert_sign": bool(INVERT_SIGN),
        "dtype": str(DATA_DTYPE),
        "artifact_threshold": float(s.get("artifact_threshold", np.inf)),
        "tmin": float(s.get("tmin", 0.0)),
        "tmax": float(s.get("tmax", np.inf)),
        "whiten_mat": ("learned ops['Wrot'] shape " + _shape_str(ops.get("Wrot"))) if has_wrot else "None",
        "dshift": ("on, ops['dshift'] shape " + _shape_str(ops.get("dshift"))) if has_dshift else "off",
    }


def write_run_stats(phase, results_dir, Wall, st, HOUR_SEC,
                    bin_path=None, probe_path=None,
                    bf_params=None, input_files=None,
                    dshift_state=None, wrot_source=None,
                    extra=None):
    """
    Unified statistics writer for BOTH the offline and online flows.

    Writes:
      - results_dir/<phase>_stats.txt
      - results_dir/<phase>_stats.json   (phase = "offline" | "online")

    Records, for each run:
      - phase (offline / online)
      - input files (binary, probe, and for online the imported ops/Wall/Wrot)
      - output files produced in results_dir
      - bfile parameters (n_chan_bin, fs, batch size, nt, chan_map, CAR, dtype,
        filter, whitening, tmin/tmax, ...)
      - dshift on / off
      - Wrot matrix import path / source
      - number of spikes, number of clusters/templates, and label breakdowns

    Uses (if present):
      - cluster_KSLabel.tsv or cluster_group.tsv
      - spike_clusters.npy
      - spike_templates.npy
    """
    results_dir = Path(results_dir)
    phase = str(phase).lower()

    # -------- output files produced by this run (scan BEFORE writing stats) --------
    output_files = _list_output_files(results_dir)

    # -------- read labels (cluster_id -> label) --------
    labels = {}
    labels_file_used = None
    candidates = [("cluster_KSLabel.tsv", "KSLabel"), ("cluster_group.tsv", "group")]

    for fname, labcol in candidates:
        f = results_dir / fname
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
    spike_clusters_path = results_dir / "spike_clusters.npy"
    spike_templates_path = results_dir / "spike_templates.npy"

    spike_clusters = None
    spike_templates = None

    if spike_clusters_path.exists():
        spike_clusters = np.load(spike_clusters_path).astype(np.int64, copy=False)
    if spike_templates_path.exists():
        spike_templates = np.load(spike_templates_path).astype(np.int64, copy=False)

    # -------- compute useful stats --------
    K = int(Wall.shape[0]) if Wall is not None else None
    n_spikes = int(st.shape[0]) if st is not None else None

    st_clusters = st[:, 1].astype(np.int64, copy=False) if (st is not None and st.ndim == 2 and st.shape[1] >= 2) else None
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

    # -------- assemble machine-readable summary --------
    resolved_inputs = input_files if input_files else {
        "input_binary": str(bin_path) if bin_path is not None else None,
        "probe": str(probe_path) if probe_path is not None else None,
    }

    summary = {
        "phase": phase,
        "results_dir": str(results_dir),
        "HOUR_SEC": HOUR_SEC,
        "io": {
            "input_files": resolved_inputs,
            "output_files": output_files,
        },
        "bfile_params": bf_params,
        "dshift": dshift_state,
        "wrot_source": wrot_source,
        "counts": {
            "Wall_num_templates_K": K,
            "num_spikes": n_spikes,
            "unique_clusters_in_st": unique_st_clusters,
            "clusters_in_tsv_total": int(len(labels)) if labels_file_used is not None else None,
            "good_clusters_in_tsv": int(len(good_cluster_ids)) if labels_file_used is not None else None,
            "mua_clusters_in_tsv": int(len(mua_cluster_ids)) if labels_file_used is not None else None,
        },
        "labels_file_used": labels_file_used,
        "cluster_label_counts_from_tsv": dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))) if label_counts else None,
        "label_spike_counts_from_spike_clusters": label_spike_counts if label_spike_counts else None,
        "spikes_per_cluster_stats_from_st": spc_stats if spc_stats else None,
        "has_spike_clusters_npy": bool(spike_clusters_path.exists()),
        "has_spike_templates_npy": bool(spike_templates_path.exists()),
        "note": "Wall is NOT filtered. These stats are for reporting only.",
    }
    if extra:
        summary["extra"] = extra

    (results_dir / f"{phase}_stats.json").write_text(json.dumps(summary, indent=2, default=str))

    # -------- assemble human-readable txt --------
    L = []
    L.append(f"{phase.upper()} FLOW STATISTICS")
    L.append("=" * 48)
    L.append(f"phase            : {phase}")
    L.append(f"results_dir      : {results_dir}")
    L.append(f"HOUR_SEC         : {HOUR_SEC}")
    L.append("")

    L.append("INPUT FILES")
    for k, v in (resolved_inputs or {}).items():
        L.append(f"  {k}: {v}")
    L.append("")

    L.append("OUTPUT FILES")
    if output_files:
        for it in output_files:
            L.append(f"  {it['path']}  ({it['bytes']} bytes)")
    else:
        L.append("  (none found)")
    L.append("")

    L.append("BFILE PARAMETERS")
    if bf_params:
        for k, v in bf_params.items():
            L.append(f"  {k}: {v}")
    else:
        L.append("  (not provided)")
    L.append("")

    L.append(f"dshift           : {dshift_state}")
    L.append(f"Wrot source      : {wrot_source}")
    L.append("")

    L.append("COUNTS")
    L.append(f"  Wall templates (K)      : {K}")
    L.append(f"  Spikes (st)             : {n_spikes}")
    L.append(f"  Unique clusters in st   : {unique_st_clusters}")
    if labels_file_used is not None:
        L.append(f"  Clusters in TSV total   : {len(labels)}")
        L.append(f"  Good clusters           : {len(good_cluster_ids)}")
        L.append(f"  MUA clusters            : {len(mua_cluster_ids)}")
        L.append("  Label counts (TSV):")
        for k, v in sorted(label_counts.items(), key=lambda x: (-x[1], x[0])):
            L.append(f"    {k}: {v}")
    else:
        L.append("  No TSV labels found (cluster_KSLabel.tsv / cluster_group.tsv).")

    if label_spike_counts:
        L.append("")
        L.append("  Spike counts by label (from spike_clusters.npy):")
        for k, v in sorted(label_spike_counts.items(), key=lambda x: (-x[1], x[0])):
            L.append(f"    {k}: {v}")

    if spc_stats:
        L.append("")
        L.append("  Spikes-per-cluster (from st[:,1]) stats:")
        for k, v in spc_stats.items():
            L.append(f"    {k}: {v}")

    L.append("")
    L.append(f"Wrote: {results_dir / (phase + '_stats.txt')}")
    L.append(f"Wrote: {results_dir / (phase + '_stats.json')}")
    (results_dir / f"{phase}_stats.txt").write_text("\n".join(L))

    logger.info(f"[{phase.upper()}-STATS] Wrote {phase}_stats.txt/json into {results_dir}")

# Use this instead of BinaryFiltered in order to change the filter typ to IIR biquad
b, a = butter(3, 300, fs=20000, btype='high')   # your exact biquad
class BiquadFiltered(BinaryFiltered):
    def filter(self, X, ops=None, ibatch=None, skip_preproc=False):
        if self.chan_map is not None:
            X = X[self.chan_map]
        if self.invert_sign:
            X = X * -1
        X = X - X.mean(1, keepdim=True)
        if self.do_CAR:
            X = X - torch.median(X, 0)[0]          # same median-CAR
        if skip_preproc:
            return X
        # --- biquad replaces zero-phase FFT highpass ---
        Xnp = lfilter(b, a, X.cpu().numpy(), axis=1)   # causal, per channel
        X = torch.from_numpy(Xnp).to(X.device).float()
        # --- whitening + drift, identical to parent ---
        if self.whiten_mat is not None:
            if self.dshift is not None and ops is not None and ibatch is not None:
                M = get_drift_matrix(ops, self.dshift[ibatch], device=self.device)
                X = (M @ self.whiten_mat) @ X
            else:
                X = self.whiten_mat @ X
        return X

def build_bfile_from_ops(bin_path, probe, ops, Wrot, tmin=0.0, tmax=np.inf, dshift=None):
    s = ops["settings"]
    nchan = int(s["n_chan_bin"])
    fs = float(s["fs"])
    NT = int(s["batch_size"])
    nt = int(ops.get("nt", 61))
    nt0min = int(ops.get("nt0min", 20))

    hp_filter = ops["fwav"]
    #Wrot = ops.get("Wrot", None) 
    print(Wrot)
#BinaryFiltered
    bfile = BiquadFiltered(
        filename=str(bin_path),
        n_chan_bin=nchan,
        fs=int(fs),
        NT=NT,
        nt=nt,
        nt0min=nt0min,
        chan_map=probe["chanMap"],
        hp_filter=hp_filter,
        whiten_mat=Wrot, #when limiting whitening pass None 
        dshift=dshift,
        device=DEVICE,
        do_CAR=DO_CAR,
        artifact_threshold=float(s.get("artifact_threshold", np.inf)),
        invert_sign=INVERT_SIGN,
        dtype=DATA_DTYPE,
        tmin=float(tmin), 
        tmax=float(tmax),
    )
    # Recalculate the whitening matrix excluding all batches after 1200
    #wrot = get_whitening_matrix(bfile, probe['xc'], probe['yc'], nskip=ops['settings']['nskip'], nrange=ops['settings']['whitening_range'])
    #bfile.whiten_mat = wrot  
    print("batches number: ", bfile.n_batches)

    # Capture the exact parameters used to build this (online) bfile, for stats.
    bf_params = {
        "class": "BiquadFiltered",
        "filter": "causal IIR butter(order=3, highpass=300Hz, fs=%g) per-channel" % fs,
        "n_chan_bin": nchan,
        "fs": fs,
        "batch_size_NT": NT,
        "nt": nt,
        "nt0min": nt0min,
        "n_chan_map": int(len(probe["chanMap"])),
        "do_CAR": bool(DO_CAR),
        "invert_sign": bool(INVERT_SIGN),
        "dtype": str(DATA_DTYPE),
        "artifact_threshold": float(s.get("artifact_threshold", np.inf)),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "whiten_mat": ("Wrot provided, shape " + _shape_str(Wrot)) if Wrot is not None else "None (whitening disabled)",
        "dshift": ("on, shape " + _shape_str(dshift)) if dshift is not None else "off",
        "n_batches": int(bfile.n_batches),
    }
    return bfile, bf_params


def offline_flow(bin_path: Path, probe_path: Path, out_dir: Path, fs: float, nchan: int, HOUR_SEC: float):
    out_dir = ensure_dir(out_dir)
    probe = load_probe(probe_path)

    user_settings = {
        "fs": float(fs),
        "n_chan_bin": int(nchan),
        "tmin": 0.0,
        "tmax": float(HOUR_SEC),
        "n_pcs": 3,
        #"batch_size": 32768,
        #"nearest_chans": 8,
    }

    results_dir = Path(out_dir)

    settings = {**DEFAULT_SETTINGS, **user_settings}
    settings["filename"] = [Path(bin_path)]
    settings["data_dir"] = settings["filename"][0].parent
    settings["results_dir"] = results_dir

    ops, st, clu, tF, Wall, _, _, _, _ =_sort(settings["filename"],
                   results_dir, probe, settings, DATA_DTYPE, DEVICE, DO_CAR, CLEAR_CACHE, INVERT_SIGN,
                   SAVE_PREPROCESSED_COPY, VERBOSE_LOG, SAVE_EXTRA_VARS, file_object=None, progress_bar=None, gui_sorter=None)

    # ----- unified stats (offline phase) -----
    bf_params = offline_bfile_params(ops, probe)
    write_run_stats(
        "offline", results_dir, Wall, st, HOUR_SEC,
        bin_path=bin_path, probe_path=probe_path,
        bf_params=bf_params,
        input_files={
            "input_binary": str(bin_path),
            "probe": str(probe_path),
        },
        dshift_state=("on" if ops.get("dshift") is not None else "off"),
        wrot_source=("learned this run (ops['Wrot'])" if ops.get("Wrot") is not None else "None"),
    )
    logger.info("Offline stats written")
    return


def online_flow(bin_path: Path, probe_path: Path,
                offline_dir: Path, out_dir: Path, HOUR_SEC: float, debug: bool):
    out_dir = ensure_dir(out_dir)
    probe = load_probe(probe_path)

    # Load resources from offline/GUI run
    ops = load_ops(offline_dir / "ops.npy", device=DEVICE)
    Wall_np = np.load(offline_dir / "Wall.npy")
    Wall = torch.from_numpy(Wall_np).to(DEVICE)

    # ======================================================
    # SIMULATE SoC HARDWARE CONSTRAINTS
    # ======================================================
    # Limit Matching Pursuit Iterations (Peels) for real-time logic
    #ops['settings']['max_peels'] = 2
    #ops['max_peels'] = 2 

    #load Wrot learned from first 1200 batches corresponding to 1hr of data
    wrot_path = offline_dir / "Wrot_1200.npy"
    Wrot = np.load(wrot_path, allow_pickle=True)
    if Wrot is not None:
        Wrot = torch.from_numpy(Wrot).to(DEVICE)
    else:
        print("Wrot not found in directory.")
    #Wrot = None
    # Set window for the 'rest' of the data
    ops['settings']['tmin'] = float(HOUR_SEC)
    ops['tmin'] = float(HOUR_SEC) # Sync root
    ops['settings']['tmax'] = float(np.inf)
    ops['tmax'] = float(np.inf)   # Sync root
   
    #=====================================================
    #FOR DEBUGGING PURPOSES - GUI IDENTICAL SHIFT
    batch_size = int(ops['settings']['batch_size'])   # 60000
    fs = float(ops['settings']['fs'])
    batch_offset = int(round(HOUR_SEC * fs)) // batch_size   # 72000000 // 60000 = 1200
    dshift_online = ops['dshift'][batch_offset:]
    print(len(ops['dshift']), batch_offset, len(dshift_online))
    #=====================================================
    # Build binary file object
    bfile_rest, bf_params = build_bfile_from_ops(bin_path, probe, ops, Wrot, tmin=float(HOUR_SEC), tmax=float(np.inf), dshift=dshift_online)
    # ======================================================
    # PREPARE Wall FOR TEMPLATE MATCHING
    n_pcs = int(ops["settings"]["n_pcs"])
    if Wall.shape[1] == n_pcs:
        Wall_match = Wall
    elif Wall.shape[2] == n_pcs:
        Wall_match = Wall.permute(0, 2, 1).contiguous()
    else:
        raise AssertionError(f"Could not identify n_pcs axis in Wall shape {Wall.shape}")
    #=====================================================
    # DEBUGGING: dump cf
    # from torch.nn.functional import conv1d
    # NT  = ops['batch_size']                 # batch size in samples (check the actual value)
    # nt  = ops['nt']
    # S   = 72005707
    # B_id = S // NT
    # X = bfile_rest.padded_batch_to_torch(B_id, ops)        # the batch matching C sample 108030000
    # W = ops['wPCA'].contiguous()
    # nm = (Wall_match**2).sum(-1).sum(-1)
    # B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=ops['nt']//2)
    # B = torch.einsum('ijk, kjl -> il', Wall_match, B)
    # Cf = torch.relu(B)**2 / nm.unsqueeze(-1)
    # nt = ops['nt']; Cf[:, :nt] = 0; Cf[:, -nt:] = 0
    # kcol = (S - B_id * NT) + nt  
    # print("KS cf at col nt+kcol:", float(Cf[:, kcol+nt].max()), "tid:", int(Cf[:, kcol+nt].argmax()))
    # print("KS frac cf>Th^2:", float((Cf.max(0).values > ops['Th_learned']**2).float().mean()))

    # ----------------------------------------
    print(f"[ONLINE] Running extraction with {len(Wall_match)} templates.")
    # Core SoC simulation: Template Matching
    # We pass the original un-normalized 'Wall_match' here so Kilosort 
    # runs exactly as it always has.
    st, tF, ops = template_matching.extract(ops, bfile_rest, Wall_match, device=DEVICE)

    # Convert to absolute coordinates
    st = np.asarray(st)
    fs = float(ops["settings"]["fs"])
    sample_offset = np.int64(round(HOUR_SEC * fs))
    st_abs = st.copy()
    st_abs[:, 0] += sample_offset

    spike_times = st_abs[:, 0].astype(np.int64, copy=False)
    spike_templates = st_abs[:, 1].astype(np.int64, copy=False)
    spike_clusters = spike_templates.copy()

    # ======================================================
    # FIX: AUTOMATIC LABEL AND METADATA COPYING
    # ======================================================
    import shutil
    for label_file in ["cluster_KSLabel.tsv", "cluster_group.tsv"]:
        src = offline_dir / label_file
        if src.exists():
            shutil.copy(src, out_dir / label_file)
            print(f"[ONLINE] Copied {label_file} to online directory.")

    # Save all results compatible with Phy/Comparison scripts
    np.save(out_dir / "spike_times.npy", spike_times)
    np.save(out_dir / "spike_templates.npy", spike_templates)
    np.save(out_dir / "spike_clusters.npy", spike_clusters)
    np.save(out_dir / "st.npy", st_abs)
    np.save(out_dir / "tF.npy", tF.detach().cpu().numpy() if isinstance(tF, torch.Tensor) else np.array(tF))

    # Save templates pc_feature_indices ops for complete environment reproduction
    if (offline_dir / "templates.npy").exists():
        shutil.copy(offline_dir / "templates.npy", out_dir / "templates.npy")
    if (offline_dir / "templates.npy").exists():
        shutil.copy(offline_dir / "pc_feature_ind.npy", out_dir / "pc_feature_ind.npy")
    save_ops(ops, results_dir=out_dir)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Exports for Soc
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    export_dir = os.path.join(out_dir, "export")
    os.makedirs(export_dir, exist_ok=True)
    print(f"[ONLINE] STORING EXPORTS FOR SOC IN {export_dir}")
    # ======================================================
    # SAVES DRIFT MATRIX & WHITENING
    # ======================================================
    # Calculate the median drift across all batches
    """ stable_drift_state = np.median(ops['dshift'][:batch_offset] , axis=0)
    M = get_drift_matrix(ops, stable_drift_state, device=DEVICE) """
    Wrot = ops['Wrot']
    """ Wsh = (M @ Wrot)
    Wsh_np = Wsh.detach().cpu().numpy() if torch.is_tensor(Wsh) else Wsh
    np.save(os.path.join(export_dir, "Wsh"), Wsh_np) """
    Wrot_np = Wrot.detach().cpu().numpy() if torch.is_tensor(Wrot) else Wrot
    np.save(os.path.join(export_dir, "Wrot"), Wrot_np)              
    #print(f"[ONLINE] Wsh and Wrot matrices saved with shape {Wsh_np.shape}")
    # ----------------------------------------------------------
    #  FOR DEBUGGING PURPOSES: OUTPUT WSH MATRIX FOR ALL BATCHES
    if debug:
        batch_offset = 1200 
        dshift = ops['dshift']
        Wsh_stack = []
        for b0 in range(batch_offset, dshift.shape[0]):
            M = get_drift_matrix(ops, dshift[b0], device=DEVICE)   # (NCHAN, NCHAN)
            Wsh_b = (M @ Wrot).detach().cpu().numpy().astype(np.float32)
            Wsh_stack.append(Wsh_b)

        Wsh_stack = np.stack(Wsh_stack, axis=0)   # (n_online_batches, NCHAN, NCHAN)
        np.save(os.path.join(export_dir, "Wsh_per_batch.npy"), Wsh_stack)
        print(f"[ONLINE] Wsh_per_batch and Wrot matrices saved with shape {Wsh_stack.shape}")
        print(Wsh_stack.shape)
    # ======================================================
    # EXPORT wPCA FOR BARE-METAL C
    # ======================================================
    wPCA_tensor = ops.get('wPCA', None)
    if wPCA_tensor is not None:
        wPCA_np = wPCA_tensor.detach().cpu().numpy() if torch.is_tensor(wPCA_tensor) else wPCA_tensor
        np.save(os.path.join(export_dir, "wPCA.npy"), wPCA_np)
        print(f"[ONLINE] Exported wPCA master shapes for C hardware: {wPCA_np.shape}")
    # ======================================================
    # EXPORT Wall AND NM FOR BARE-METAL C
    # ======================================================
    Wall_np = Wall.detach().cpu().numpy() if torch.is_tensor(Wall) else Wall
    np.save(os.path.join(out_dir, "Wall.npy"), Wall_np)
    np.save(os.path.join(export_dir, "Wall.npy"), Wall_np)
    nm = (Wall ** 2).sum(axis=-1).sum(axis=-1)
    np.save(os.path.join(export_dir, "Wall_nm.npy"), nm.detach().cpu().numpy().astype(np.float32))
    print(f"[ONLINE] Exported Wall templates: {Wall.shape} and nm: {nm.shape}")

    logger.info("EXPORTS WRITTEN FOR SOC HARDWARE SIMULATION.")
    #======================================================

    # ----- unified stats (online phase) -----
    wrot_source = str(wrot_path) if Wrot is not None else "None (whitening disabled)"
    write_run_stats(
        "online", out_dir, Wall, st_abs, HOUR_SEC,
        bin_path=bin_path, probe_path=probe_path,
        bf_params=bf_params,
        input_files={
            "input_binary": str(bin_path),
            "probe": str(probe_path),
            "offline_dir": str(offline_dir),
            "ops_npy": str(offline_dir / "ops.npy"),
            "Wall_npy": str(offline_dir / "Wall.npy"),
            "Wrot_npy": str(wrot_path),
        },
        dshift_state=(f"on (online slice ops['dshift'][{batch_offset}:], {len(dshift_online)} batches)"
                      if dshift_online is not None else "off"),
        wrot_source=wrot_source,
        extra={"sample_offset": int(sample_offset), "batch_offset": int(batch_offset)},
    )
    logger.info("Online stats written")

    print("[ONLINE] Online Phase Complete.")
    return


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--bin", required=True, type=str)
  ap.add_argument("--probe", required=True, type=str)
  ap.add_argument("--out_offline", required=True, type=str)
  ap.add_argument("--out_online", required=True, type=str)
  ap.add_argument("--fs", required=True, type=float)
  ap.add_argument("--nchan", required=True, type=int)
  ap.add_argument("--HOUR_SEC", type=float, required=True)
  args = ap.parse_args()

  global DO_CAR, CLEAR_CACHE, INVERT_SIGN, SAVE_PREPROCESSED_COPY, VERBOSE_LOG, SAVE_EXTRA_VARS, DATA_DTYPE, DEVICE

  DO_CAR = True
  CLEAR_CACHE = False
  INVERT_SIGN = False
  SAVE_PREPROCESSED_COPY = False
  VERBOSE_LOG = False
  SAVE_EXTRA_VARS = True
  DATA_DTYPE = 'int16'
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  #offline_flow(Path(args.bin), Path(args.probe), Path(args.out_offline), args.fs, args.nchan, args.HOUR_SEC)
  online_flow(Path(args.bin), Path(args.probe), Path(args.out_offline), Path(args.out_online), args.HOUR_SEC, debug=True)

if __name__ == "__main__":
    main()