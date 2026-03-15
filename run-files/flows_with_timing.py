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
import time
from contextlib import contextmanager
from collections import defaultdict

from kilosort.io import load_probe, BinaryFiltered, save_ops, load_ops
from kilosort.run_kilosort import _sort
from kilosort.parameters import DEFAULT_SETTINGS
from kilosort.gui.logger import setup_logger
from kilosort import template_matching

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


def _fmt_seconds(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - 60 * m - 3600 * h
    if h > 0:
        return f"{h:d}h {m:02d}m {s:05.2f}s"
    if m > 0:
        return f"{m:d}m {s:05.2f}s"
    return f"{s:.2f}s"


class ProgressTracker:
    """Attach human-readable descriptions to internal tqdm bars and time them."""

    def __init__(self, phase_name: str, labels: list[str]):
        self.phase_name = phase_name
        self.labels = list(labels)
        self.records = []
        self._assigned = 0
        self._orig_init = None
        self._orig_close = None

    def __enter__(self):
        import tqdm.std
        import tqdm.auto

        self._orig_init = tqdm.std.tqdm.__init__
        self._orig_close = tqdm.std.tqdm.close
        tracker = self

        def patched_init(bar, *args, **kwargs):
            desc = kwargs.get("desc")
            if not desc and tracker._assigned < len(tracker.labels):
                desc = tracker.labels[tracker._assigned]
                kwargs["desc"] = desc
            tracker._orig_init(bar, *args, **kwargs)
            assigned_desc = getattr(bar, "desc", desc)
            if not assigned_desc and tracker._assigned < len(tracker.labels):
                assigned_desc = tracker.labels[tracker._assigned]
                try:
                    bar.set_description_str(assigned_desc)
                except Exception:
                    pass
            bar._flows_phase_name = tracker.phase_name
            bar._flows_desc = assigned_desc or f"{tracker.phase_name}: progress bar {tracker._assigned + 1}"
            bar._flows_t0 = time.perf_counter()
            bar._flows_recorded = False
            tracker._assigned += 1

        def patched_close(bar, *args, **kwargs):
            if getattr(bar, "_flows_phase_name", None) == tracker.phase_name and not getattr(bar, "_flows_recorded", False):
                tracker.records.append({
                    "desc": getattr(bar, "_flows_desc", f"{tracker.phase_name}: unnamed bar"),
                    "seconds": time.perf_counter() - getattr(bar, "_flows_t0", time.perf_counter()),
                    "total": getattr(bar, "total", None),
                    "n": getattr(bar, "n", None),
                })
                bar._flows_recorded = True
            return tracker._orig_close(bar, *args, **kwargs)

        tqdm.std.tqdm.__init__ = patched_init
        tqdm.std.tqdm.close = patched_close
        tqdm.auto.tqdm.__init__ = patched_init
        tqdm.auto.tqdm.close = patched_close
        return self

    def __exit__(self, exc_type, exc, tb):
        import tqdm.std
        import tqdm.auto

        tqdm.std.tqdm.__init__ = self._orig_init
        tqdm.std.tqdm.close = self._orig_close
        tqdm.auto.tqdm.__init__ = self._orig_init
        tqdm.auto.tqdm.close = self._orig_close
        return False


def write_phase_timing_summary(results_dir: Path, phase_name: str, phase_seconds: float, tracker: ProgressTracker | None = None,
                               extra_lines: list[str] | None = None, filename: str | None = None):
    filename = filename or f"{phase_name.lower()}_timing_summary.txt"
    lines = [
        f"{phase_name.upper()} TIMING SUMMARY",
        f"total_time: {_fmt_seconds(phase_seconds)} ({phase_seconds:.3f} s)",
        "",
    ]

    if tracker is not None and tracker.records:
        lines.append("Sub-phase progress bars:")
        for idx, rec in enumerate(tracker.records, start=1):
            progress = ""
            if rec.get("n") is not None and rec.get("total") is not None:
                progress = f" [{rec['n']}/{rec['total']}]"
            lines.append(f"  {idx}. {rec['desc']}{progress}: {_fmt_seconds(rec['seconds'])} ({rec['seconds']:.3f} s)")
        lines.append("")
    elif tracker is not None:
        lines.append("No internal progress bars were captured for this phase.")
        lines.append("")

    if extra_lines:
        lines.append("Additional summary:")
        lines.extend([f"  {line}" for line in extra_lines])
        lines.append("")

    out = results_dir / filename
    out.write_text("\n".join(lines))
    logger.info(f"[{phase_name}] Wrote timing summary to {out}")


def append_timing_to_offline_stats(results_dir: Path, phase_seconds: float, tracker: ProgressTracker | None = None):
    txt_path = results_dir / "offline_stats.txt"
    json_path = results_dir / "offline_stats.json"

    if txt_path.exists():
        lines = txt_path.read_text().splitlines()
        lines.append("")
        lines.append("TIMING SUMMARY")
        lines.append(f"Offline total time: {_fmt_seconds(phase_seconds)} ({phase_seconds:.3f} s)")
        if tracker is not None and tracker.records:
            for idx, rec in enumerate(tracker.records, start=1):
                lines.append(f"  {idx}. {rec['desc']}: {_fmt_seconds(rec['seconds'])} ({rec['seconds']:.3f} s)")
        txt_path.write_text("\n".join(lines))

    if json_path.exists():
        summary = json.loads(json_path.read_text())
        summary["offline_total_seconds"] = phase_seconds
        summary["offline_total_human"] = _fmt_seconds(phase_seconds)
        summary["offline_progress_bars"] = tracker.records if tracker is not None else []
        json_path.write_text(json.dumps(summary, indent=2))


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

def build_bfile_from_ops(bin_path, probe, ops, tmin=0.0, tmax=np.inf, zero_dshift=False):
    s = ops["settings"]
    nchan = int(s["n_chan_bin"])
    fs = float(s["fs"])
    NT = int(s["batch_size"])
    nt = int(ops.get("nt", 61))
    nt0min = int(ops.get("nt0min", 20))

    hp_filter = ops["fwav"]

    Wrot = ops.get("Wrot", None)

    # Extract dshift from ops, converting to numpy if needed
    dshift = ops.get("dshift", None)
    if dshift is not None:
        if isinstance(dshift, torch.Tensor):
            dshift = dshift.detach().cpu().numpy()
        else:
            dshift = np.asarray(dshift)

    if zero_dshift:
        # Online/SoC phase: zero out drift correction.
        # Pass explicit zeros (not None) so BinaryFiltered applies zero shift
        # per batch without recomputing drift from scratch.
        if dshift is not None:
            dshift = np.zeros_like(dshift)
        else:
            # ops had no dshift — compute n_batches from file size (tmax may be np.inf)
            n_chan_bin = int(ops["settings"]["n_chan_bin"])
            bytes_per_sample = 2  # int16
            total_samples = os.path.getsize(str(bin_path)) // (n_chan_bin * bytes_per_sample)
            online_start_sample = int(round(float(ops["settings"]["tmin"]) * float(ops["settings"]["fs"])))
            n_batches_online = int(np.ceil((total_samples - online_start_sample) / float(ops["settings"]["batch_size"])))
            dshift = np.zeros(n_batches_online, dtype=np.float32)
        logger.info(f"[ONLINE] dshift zeroed ({len(dshift)} batches) — no drift correction applied.")

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

    # pad dshift to n_batches if needed (keeps shapes aligned) — offline only
    if bfile.dshift is not None:
        nb = int(bfile.n_batches)
        ds = np.asarray(bfile.dshift).reshape(-1)
        if len(ds) < nb:
            pad = np.full(nb - len(ds), ds[-1], dtype=ds.dtype)
            bfile.dshift = np.concatenate([ds, pad], axis=0)

    return bfile


def offline_flow(bin_path: Path, probe_path: Path, out_dir: Path,
                 fs: float, nchan: int, HOUR_SEC: float):
    out_dir = ensure_dir(out_dir)
    probe = load_probe(probe_path)
#-----------------------------------------------------------------------------------
    cm = np.asarray(probe["chanMap"])
    assert cm.ndim == 1
    assert int(cm.min()) >= 0, f"chanMap min {cm.min()} < 0"
    assert int(cm.max()) < int(nchan), f"chanMap max {cm.max()} >= nchan {nchan}"
#-----------------------------------------------------------------------------------

    user_settings = {
        "fs": float(fs),
        "n_chan_bin": int(nchan),
        "tmin": 0.0,
        "tmax": float(HOUR_SEC),
        "n_pcs": 6,
        "batch_size": 10000,
    }

    results_dir = Path(out_dir)

    settings = {**DEFAULT_SETTINGS, **user_settings}
    settings["filename"] = [Path(bin_path)]
    settings["data_dir"] = settings["filename"][0].parent
    settings["results_dir"] = results_dir

    logger.info("[OFFLINE] Starting offline sort")
    logger.info(f"[OFFLINE] Window: tmin={settings['tmin']} s, tmax={settings['tmax']} s")
    offline_labels = [
        "OFFLINE 1/3 - drift/spike detect pass 1",
        "OFFLINE 2/3 - drift/spike detect pass 2",
        "OFFLINE 3/3 - clustering / template learning",
    ]
    offline_t0 = time.perf_counter()
    with ProgressTracker("offline", offline_labels) as tracker:
        ops, st, clu, tF, Wall, _, _, _, _ =_sort(settings["filename"],
                       results_dir, probe, settings, DATA_DTYPE, DEVICE, DO_CAR, CLEAR_CACHE, INVERT_SIGN,
                       SAVE_PREPROCESSED_COPY, VERBOSE_LOG, SAVE_EXTRA_VARS, file_object=None, progress_bar=None, gui_sorter=None)
    offline_seconds = time.perf_counter() - offline_t0

    write_offline_stats(results_dir, Wall, st, HOUR_SEC)
    append_timing_to_offline_stats(results_dir, offline_seconds, tracker)
    write_phase_timing_summary(
        results_dir,
        "offline",
        offline_seconds,
        tracker=tracker,
        extra_lines=[
            f"window_seconds={HOUR_SEC}",
            f"n_templates={int(Wall.shape[0])}",
            f"n_spikes={int(st.shape[0])}",
        ],
        filename="offline_timing_summary.txt",
    )
    logger.info(f"[OFFLINE] Completed in {_fmt_seconds(offline_seconds)}")
    logger.info("Offline stats written")
    return

def online_flow(bin_path: Path, probe_path: Path,
                offline_dir: Path, out_dir: Path, HOUR_SEC: float):
    out_dir = ensure_dir(out_dir)
    probe = load_probe(probe_path)

    ops = load_ops(offline_dir / "ops.npy", device=DEVICE)
    Wall_np = np.load(offline_dir / "Wall.npy")
    Wall = torch.from_numpy(Wall_np).to(DEVICE)

    ops["settings"]["tmin"] = float(HOUR_SEC)
    ops["settings"]["tmax"] = float(np.inf)

    bfile_rest = build_bfile_from_ops(bin_path, probe, ops, tmin=float(HOUR_SEC), tmax=float(np.inf),
                                      zero_dshift=True)  # SoC has no drift estimates

    n_pcs = int(ops["settings"]["n_pcs"])

    assert Wall.ndim == 3, f"Expected 3D Wall, got shape {Wall.shape}"

    if Wall.shape[1] == n_pcs:
        # already (K, n_pcs, n_chan)
        Wall_match = Wall
    elif Wall.shape[2] == n_pcs:
        # saved as (K, n_chan, n_pcs) -> convert to (K, n_pcs, n_chan)
        Wall_match = Wall.permute(0, 2, 1).contiguous()
    else:
        raise AssertionError(
            f"Could not identify n_pcs axis in Wall. "
            f"Wall shape={Wall.shape}, n_pcs={n_pcs}"
        )

    logger.info(f"[ONLINE] Wall shape: {tuple(Wall_match.shape)}")
    logger.info(f"[ONLINE] Running extraction with {len(Wall_match)} templates.")
    online_labels = [
        "ONLINE 1/2 - template extraction over remaining data",
        "ONLINE 2/2 - extraction finalize / template updates",
    ]
    online_t0 = time.perf_counter()
    with ProgressTracker("online", online_labels) as tracker:
        st, tF, ops = template_matching.extract(ops, bfile_rest, Wall_match, device=DEVICE)
    online_seconds = time.perf_counter() - online_t0

    print("\n=== CHANNEL DEBUG ===")
    print("HOUR_SEC / tmin:", ops["settings"]["tmin"])
    print("probe_path:", probe_path)
    print("probe n_chan:", probe["n_chan"])
    cm = probe["chanMap"]
    print("chanMap min/max/len:", int(cm.min()), int(cm.max()), len(cm))

    print("ops n_chan_bin:", ops.get("n_chan_bin", None))
    print("ops n_chan:", ops.get("n_chan", None))


    st = np.asarray(st)

    # Shift spike times from relative (online window) to absolute (full file) coordinates.
    # extract() counts samples from batch 0 of bfile_rest, which starts at HOUR_SEC seconds.
    # Adding HOUR_SEC * fs converts to the same absolute sample space as the GUI spike_times.npy.
    fs = float(ops["settings"]["fs"])
    sample_offset = np.int64(round(HOUR_SEC * fs))
    st_abs = st.copy()
    st_abs[:, 0] += sample_offset
    logger.info(f"[ONLINE] Shifted spike times by +{sample_offset} samples ({HOUR_SEC}s * {fs}Hz) -> absolute coordinates.")
    logger.info(f"[ONLINE] spike_times range: {int(st_abs[:,0].min())} - {int(st_abs[:,0].max())} samples")

    # Save st.npy in absolute coordinates too (consistent with spike_times.npy)
    np.save(out_dir / "st.npy", st_abs)
    np.save(out_dir / "tF.npy", tF.detach().cpu().numpy() if isinstance(tF, torch.Tensor) else np.array(tF))
    save_ops(ops, results_dir=out_dir)

    spike_times = st_abs[:, 0].astype(np.int64, copy=False)
    spike_templates = st_abs[:, 1].astype(np.int64, copy=False)

    # For phy-style tools: cluster = template for online (no merge/split step)
    spike_clusters = spike_templates

    np.save(out_dir / "spike_times.npy", spike_times)
    np.save(out_dir / "spike_templates.npy", spike_templates)
    np.save(out_dir / "spike_clusters.npy", spike_clusters)

    templates = np.load(offline_dir / "templates.npy")
    np.save(out_dir / "templates.npy", templates)

    write_phase_timing_summary(
        out_dir,
        "online",
        online_seconds,
        tracker=tracker,
        extra_lines=[
            f"window_start_seconds={HOUR_SEC}",
            f"n_templates={int(Wall_match.shape[0])}",
            f"n_spikes={int(st_abs.shape[0])}",
        ],
        filename="online_timing_summary.txt",
    )
    logger.info(f"[ONLINE] Completed in {_fmt_seconds(online_seconds)}")
    logger.info("[ONLINE] Done.")
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

  run_t0 = time.perf_counter()
  offline_flow(Path(args.bin), Path(args.probe), Path(args.out_offline), args.fs, args.nchan, args.HOUR_SEC)
  online_flow(Path(args.bin), Path(args.probe), Path(args.out_offline), Path(args.out_online), args.HOUR_SEC)
  total_seconds = time.perf_counter() - run_t0
  summary_lines = [
      "FLOW RUN SUMMARY",
      f"total_runtime: {_fmt_seconds(total_seconds)} ({total_seconds:.3f} s)",
      f"offline_dir: {args.out_offline}",
      f"online_dir: {args.out_online}",
      f"HOUR_SEC: {args.HOUR_SEC}",
  ]
  summary_path = Path(args.out_online) / "flow_run_summary.txt"
  summary_path.write_text("\n".join(summary_lines))
  logger.info(f"[FLOW] Completed full run in {_fmt_seconds(total_seconds)}")
  logger.info(f"[FLOW] Wrote overall summary to {summary_path}")


if __name__ == "__main__":
    main()
