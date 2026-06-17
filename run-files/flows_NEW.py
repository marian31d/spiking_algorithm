
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
from scipy.signal import butter, lfilter
"""
Two-stage Kilosort4 flow:
  1) OFFLINE (first hour): learn templates (Wall) + export ops
  2) ONLINE/ON-CHIP (rest): reuse exported ops + run template matching (extract) only

#python flows.py   --bin "/media/mariandawud/Extreme SSD/mS51/dat/mS51_49/mS51_49.dat"   --probe /home/mariandawud/kilosort/ms51/probe_ms51_new.json   --out_offline "/media/mariandawud/Extreme SSD/results/mS51_res/flows_new/offline_20mins-51"   --out_online "/media/mariandawud/Extreme SSD/results/mS51_res/flows_new/online_20mins-51"   --fs 20000   --nchan 135   --dtype int16


This version:
- DOES NOT filter Wall
- Writes offline statistics (good/mua counts, total clusters, etc.)
"""

logger = setup_logger(__name__)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_offline_stats(results_dir: Path, Wall: torch.Tensor, st: np.ndarray, HOUR_SEC: float):
    """
    Writes offline statistics into:
      - results_dir/offline_stats.txt
      - results_dir/offline_stats.json

    Uses (if present):
      - cluster_KSLabel.tsv or cluster_group.tsv
      - spike_clusters.npy
      - spike_templates.npy
    """
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
        "results_dir": str(results_dir),
        "HOUR_SEC": HOUR_SEC,
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

    (results_dir / "offline_stats.json").write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append("OFFLINE FLOW STATISTICS")
    lines.append(f"results_dir: {results_dir}")
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
    lines.append(f"Wrote: {results_dir / 'offline_stats.txt'}")
    lines.append(f"Wrote: {results_dir / 'offline_stats.json'}")
    (results_dir / "offline_stats.txt").write_text("\n".join(lines))

    logger.info(f"[OFFLINE-STATS] Wrote offline_stats.txt/json into {results_dir}")

# Use this instead of BinaryFiltered in order to change the filter typ to IIR biquad
b, a = butter(2, 300, btype='highpass', fs=20000)   # your exact biquad
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

def build_bfile_from_ops(bin_path, probe, ops, tmin=0.0, tmax=np.inf, dshift=None):
    s = ops["settings"]
    nchan = int(s["n_chan_bin"])
    fs = float(s["fs"])
    NT = int(s["batch_size"])
    nt = int(ops.get("nt", 61))
    nt0min = int(ops.get("nt0min", 20))

    hp_filter = ops["fwav"]

    Wrot = ops.get("Wrot", None)


    bfile = BinaryFiltered(
        filename=str(bin_path),
        n_chan_bin=nchan,
        fs=int(fs),
        NT=NT,
        nt=nt,
        nt0min=nt0min,
        chan_map=probe["chanMap"],
        hp_filter=hp_filter,          # from ops
        whiten_mat=Wrot,              # from ops
        dshift=dshift,
        device=DEVICE,
        do_CAR=DO_CAR,
        artifact_threshold=float(s.get("artifact_threshold", np.inf)),
        invert_sign=INVERT_SIGN,
        dtype=DATA_DTYPE,
        tmin=float(tmin),
        tmax=float(tmax),
    )
    return bfile


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

    write_offline_stats(results_dir, Wall, st, HOUR_SEC)
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
    ops['settings']['max_peels'] = 2
    ops['max_peels'] = 2

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
    bfile_rest = build_bfile_from_ops(bin_path, probe, ops, tmin=float(HOUR_SEC), tmax=float(np.inf), dshift=dshift_online)
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
    from torch.nn.functional import conv1d
    B_id = 600
    X = bfile_rest.padded_batch_to_torch(B_id, ops)        # the batch matching C sample 108030000
    W = ops['wPCA'].contiguous()
    nm = (Wall_match**2).sum(-1).sum(-1)
    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=ops['nt']//2)
    B = torch.einsum('ijk, kjl -> il', Wall_match, B)
    Cf = torch.relu(B)**2 / nm.unsqueeze(-1)
    nt = ops['nt']; Cf[:, :nt] = 0; Cf[:, -nt:] = 0
    kcol = 30000
    print("KS cf at col nt+kcol:", float(Cf[:, nt+kcol].max()), "tid:", int(Cf[:, nt+kcol].argmax()))
    print("KS frac cf>Th^2 (batch 600):", float((Cf.max(0).values > ops['Th_learned']**2).float().mean()))

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
    stable_drift_state = np.median(ops['dshift'][:batch_offset] , axis=0)
    M = get_drift_matrix(ops, stable_drift_state, device=DEVICE)
    Wrot = ops['Wrot']
    Wsh = (M @ Wrot)
    Wsh_np = Wsh.detach().cpu().numpy() if torch.is_tensor(Wsh) else Wsh
    np.save(os.path.join(export_dir, "Wsh"), Wsh_np)
    Wrot_np = Wrot.detach().cpu().numpy() if torch.is_tensor(Wrot) else Wrot
    np.save(os.path.join(export_dir, "Wrot"), Wrot_np)              
    print(f"[ONLINE] Wsh and Wrot matrices saved with shape {Wsh_np.shape}")
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
