import numpy as np
import torch
import json
import os
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt
from kilosort.template_matching import prepare_matching

# --- Helper Function for C Header Export ---

def write_c_float_header(array_data, array_name, filename, export_dir, comment=None):
    """Flattens a numpy array and writes it as a C float header file.
    `comment` (optional) is written as an extra // line under the first one."""
    flat_data = array_data.flatten()
    with open(filename, "w") as f:
        f.write(f"// Header file for {export_dir}\n")
        if comment:
            f.write(f"// {comment}\n")
        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")

        # Declare as const float (single precision)
        f.write(f"const float {array_name}[{len(flat_data)}] = {{\n")
        
        for i, val in enumerate(flat_data):
            # Format with 6 decimal places and the 'f' suffix
            f.write(f"{val:.6f}f, ")
            
            # Newline every 10 values for readability
            if (i + 1) % 10 == 0:
                f.write("\n")      
        f.write("\n};\n\n")
        f.write(f"#endif // {array_name.upper()}_H\n")
        
    print(f"Generated {filename} ({len(flat_data)} elements as floats), size of: {flat_data.nbytes} bytes.")    
    print(array_name, "max: ", np.max(array_data), "min: ", np.min(array_data), "type: ", array_data.dtype, "dimensions: ", array_data.shape)

def get_iir_weights(result_dir):
    # --- Filter Specifications ---
    fs = 20000   # Sampling rate (Hz)
    fc = 300     # High-pass cutoff frequency (Hz)
    order = 3    # 2nd-order Biquad filter
    
    # Generate IIR Butterworth coefficients
    b, a = signal.butter(order, fc, btype='highpass', fs=fs)
    print("IIR Weights", "max: ", np.max(np.concatenate([a, b])), "min: ", np.min(np.concatenate([a, b])))
    # --- Format the C Header File ---
    # Note: We use {{ and }} to escape literal curly braces in Python f-strings
    c_header_content = f"""// Auto-generated IIR Biquad Coefficients
    // Type: High-pass Butterworth
    // Order: {order}
    // Sampling Rate: {fs} Hz
    // Cutoff Frequency: {fc} Hz
    
    #ifndef IIR_WEIGHTS_H
    #define IIR_WEIGHTS_H
    
    // Biquad Coefficient Array
    // Index map: [0]=b0, [1]=b1, [2]=b2, [3]=a1, [4]=a2
    // Note: a0 is mathematically always 1.0, so it is omitted.
    const float iir_weights[7] = {{
        {b[0]:.8f}f,
        {b[1]:.8f}f,
        {b[2]:.8f}f,
        {b[3]:.8f}f,
        {a[1]:.8f}f,
        {a[2]:.8f}f,
        {a[3]:.8f}f
    }};
    
    #endif // IIR_WEIGHTS_H
    """
    
    # --- Write to File ---
    iir_path = os.path.join(result_dir, "iir_weights.h")
    with open(iir_path, "w") as f:
        f.write(c_header_content)
    
    print(f"Successfully generated {iir_path} as an array. Size of: {np.array(a).nbytes + np.array(b).nbytes} bytes.")

def mask_Wall(Wall, ops, nC=None):
    """
    Wall: (K, NCHAN, NPCS)  e.g. (173, 126, 3)
    Keeps each template's nC nearest channels (by probe geometry around the
    template's peak channel), zeros all others. Returns masked Wall + nm.
    """
    K, NCHAN, NPCS = Wall.shape

    xc = np.asarray(ops['xc'], dtype=np.float64)
    yc = np.asarray(ops['yc'], dtype=np.float64)
    assert xc.shape[0] == NCHAN, f"xc has {xc.shape[0]} chans, Wall has {NCHAN}"

    if nC is None:
        nC = int(ops['settings']['nearest_chans'])   # KS default ~10

    # peak channel per template = channel carrying the most energy
    energy = (Wall**2).sum(axis=2)        # (K, NCHAN)
    peak_ch = energy.argmax(axis=1)       # (K,)

    # squared distance between every pair of channels
    d2 = (xc[:, None] - xc[None, :])**2 + (yc[:, None] - yc[None, :])**2  # (NCHAN, NCHAN)

    Wall_masked = np.zeros_like(Wall)
    for k in range(K):
        near = np.argsort(d2[peak_ch[k]])[:nC]    # nC nearest channels to this template's peak
        Wall_masked[k, near, :] = Wall[k, near, :]

    kept = (np.abs(Wall_masked).sum(axis=2) > 0).sum(axis=1)  # channels kept per template
    print(f"[MASK] nearest_chans={nC}; kept per template: "
          f"min={kept.min()} max={kept.max()} (NCHAN={NCHAN})")
    return Wall_masked

def group_templates(indices, probe):
    """
    Group templates by spatial position according to probe shanks.
    Loads probe and check kcoords in it to group templates by location.
    """
    kcoord = np.array(probe['kcoords'])
    nchan = probe['n_chan']
    groups = [0] * len(indices) #indix = the id of the group, values = the list of templates
    for ti in range(len(indices)):
        t = indices[ti]
        for j in range(max(kcoord)+1): #search for group id with minimum similar channels/
            ind = np.where(kcoord==j)[0] #get channels in this group
            common = len(set(ind) & set(t))
            if common > 7:
                groups[ti] = j
                break
    groups_np = np.array(groups)
    return groups_np

def th_per_temp(result_dir):
    """
    Returns an array of the 5 percintile score for each template
    """
    result_dir = Path(result_dir)
    spike_clusters = np.load(result_dir / 'spike_clusters.npy')
    print(f"number of detected spikes: {len(spike_clusters)}")
    scores_path = result_dir / 'template_scores.npy'
    if not scores_path.exists():
        raise FileNotFoundError(
            f"{scores_path} not found. This file is written by the patched "
            "`io.save_to_phy`, so results sorted before that change won't "
            "have it -- re-run kilosort on this dataset first."
            )
    scores = np.load(scores_path)
    n_templates = spike_clusters.max() + 1
    p5_by_template = np.full(n_templates, np.nan)

    for cl in np.unique(spike_clusters):
        mask = spike_clusters == cl
        cs = scores[mask]
        p5_by_template[cl] = np.percentile(cs, 5)

    return p5_by_template


def to_numpy(x):
    """torch tensor or array-like -> numpy array"""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def write_shank_info_header(filename, src, shank_ids, nchan, offset, chanMap):
    """Layout of the combined per-shank arrays: sizes as #defines, the rest as const int arrays."""
    S, C = chanMap.shape

    def arr(name, vals):
        return f"const int {name}[{len(vals)}] = {{{', '.join(str(int(v)) for v in vals)}}};\n"

    with open(filename, "w") as f:
        f.write(f"// Header file for {src}\n")
        f.write("// Per-shank layout of Wall.h, Wall_nm.h, TH.h, Wrot.h, wPCA.h\n")
        f.write("#ifndef SHANK_INFO_H\n#define SHANK_INFO_H\n\n")
        f.write(f"#define N_SHANKS {S}\n#define NCHAN_MAX {C}\n#define NCLU_ALL {int(offset[-1])}\n\n")
        f.write("// shank number N of each shank_<N> folder (the probe's kcoords value)\n")
        f.write(arr("shank_id", shank_ids))
        f.write("// real channels per shank; local channels nchan..NCHAN_MAX-1 are zero padding\n")
        f.write(arr("shank_nchan", nchan))
        f.write("// templates of shank s are rows shank_offset[s] .. shank_offset[s+1]-1\n")
        f.write(arr("shank_offset", offset))
        f.write("// .dat channel of local channel c on shank s: shank_chanMap[s*NCHAN_MAX + c], -1 = padding\n")
        f.write(f"const int shank_chanMap[{S * C}] = {{\n")
        for row in chanMap:
            f.write("    " + ", ".join(str(int(v)) for v in row) + ",\n")
        f.write("};\n\n#endif // SHANK_INFO_H\n")
    print(f"Generated {filename}: {S} shanks, channels per shank {list(map(int, nchan))}, "
          f"{int(offset[-1])} templates")


def export_per_shank(flow_dir, out_dir):
    """
    Combine the single-shank Kilosort runs in `flow_dir` (its shank_<N> folders,
    as written by run_kilosort with shank_idx) into one C header per array.

    Shanks have different channel counts, so each shank's channel axis is
    zero-padded to NCHAN_MAX (the largest shank). Templates are stacked shank by
    shank: shank s owns rows shank_offset[s] .. shank_offset[s+1]-1.

      Wall.h        (NCLU_ALL, NCHAN_MAX, NPCS)       masked templates (mask_Wall per shank)
      Wall_nm.h     (NCLU_ALL,)                       squared norm per template (unmasked, as in single mode)
      TH.h          (NCLU_ALL,)                       5th-percentile score per template
      Wrot.h        (N_SHANKS, NCHAN_MAX, NCHAN_MAX)  whitening matrix per shank
      wPCA.h        (N_SHANKS, NPCS, NT)              PC basis per shank (each run learns its own)
      shank_info.h  N_SHANKS, NCHAN_MAX, NCLU_ALL, shank_id, shank_nchan,
                    shank_offset and shank_chanMap (see write_shank_info_header)
    Padded channels are zero in Wall and Wrot, so they add nothing to the dot products.
    """
    flow_dir, out_dir = Path(flow_dir), Path(out_dir)
    shank_dirs = sorted(flow_dir.glob("shank_*"), key=lambda p: int(p.name.split("_")[1]))
    if not shank_dirs:
        raise FileNotFoundError(f"no shank_<N> folders in {flow_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ops_l = [np.load(d / "ops.npy", allow_pickle=True).item() for d in shank_dirs]
    Wall_l = [np.load(d / "Wall.npy") for d in shank_dirs]
    ids = [int(d.name.split("_")[1]) for d in shank_dirs]
    S = len(shank_dirs)
    nchan = np.array([W.shape[1] for W in Wall_l])
    K = np.array([W.shape[0] for W in Wall_l])
    C, NPCS = int(nchan.max()), Wall_l[0].shape[2]
    offset = np.concatenate([[0], np.cumsum(K)])
    nt_pca = to_numpy(ops_l[0]["wPCA"]).shape

    Wall_all = np.zeros((offset[-1], C, NPCS), dtype=np.float32)
    nm_all = np.zeros(offset[-1], dtype=np.float32)
    TH_all = np.zeros(offset[-1], dtype=np.float32)
    Wrot_all = np.zeros((S, C, C), dtype=np.float32)
    wPCA_all = np.zeros((S,) + nt_pca, dtype=np.float32)
    chanMap = np.full((S, C), -1, dtype=np.int64)

    for s, (d, ops, W) in enumerate(zip(shank_dirs, ops_l, Wall_l)):
        c, a, b = nchan[s], offset[s], offset[s + 1]
        print(f"--- {d.name}: {K[s]} templates, {c} channels")
        Wall_all[a:b, :c] = mask_Wall(W, ops)
        nm_all[a:b] = (W ** 2).sum(axis=(1, 2))
        th = th_per_temp(d)
        assert th.size == K[s], f"{d.name}: TH has {th.size} entries, Wall has {K[s]} templates"
        if np.isnan(th).any():
            print(f"[WARN] {d.name}: {np.isnan(th).sum()} templates without spikes, TH is NaN")
        TH_all[a:b] = th
        Wrot = to_numpy(ops["Wrot"])
        assert Wrot.shape == (c, c), f"{d.name}: Wrot {Wrot.shape} vs {c} channels"
        Wrot_all[s, :c, :c] = Wrot
        wPCA_all[s] = to_numpy(ops["wPCA"])
        chanMap[s, :c] = np.asarray(ops["chanMap"])

    th_learned = sorted({float(o.get("Th_learned", 8.0)) for o in ops_l})
    print(f"Th_learned over shanks = {th_learned}  (compare scores against Th_learned^2 in C)")

    src = str(flow_dir)
    see = "shanks stacked, layout in shank_info.h"
    write_c_float_header(Wall_all, "Wall", out_dir / "Wall.h", src,
                         comment=f"shape (NCLU_ALL, NCHAN_MAX, NPCS) = {Wall_all.shape}, {see}")
    write_c_float_header(nm_all, "Wall_nm", out_dir / "Wall_nm.h", src,
                         comment=f"shape (NCLU_ALL,) = {nm_all.shape}, {see}")
    write_c_float_header(TH_all, "TH", out_dir / "TH.h", src,
                         comment=f"shape (NCLU_ALL,) = {TH_all.shape}, {see}")
    write_c_float_header(Wrot_all, "Wrot", out_dir / "Wrot.h", src,
                         comment=f"shape (N_SHANKS, NCHAN_MAX, NCHAN_MAX) = {Wrot_all.shape}, {see}")
    write_c_float_header(wPCA_all, "wPCA", out_dir / "wPCA.h", src,
                         comment=f"shape (N_SHANKS, NPCS, NT) = {wPCA_all.shape}, {see}")
    write_shank_info_header(out_dir / "shank_info.h", src, ids, nchan, offset, chanMap)


def main():
    # --- 1. Set your directory ---
    per_shank = False   # True: combine the shank_<N> runs in flow_dir into one header per array
    flow_dir = "/data/results/mds2_res/IIR/updated_clu/time/module/5400"   # used when per_shank
    export_dir = "/data/results/mds2_res/IIR/updated_clu/time/3600/peel_specified"
    dataset_dir = "/data/spiking_algorithm/mds2"
    result_dir = os.path.join(dataset_dir, "phase1")   # whole-probe headers, next to phase1/flows.c (or phase2)
    num_shanks = 8  # Set this according to your probe geometry

    if per_shank:
        # per-shank headers go next to pershank/flows.c
        shank_out = os.path.join(dataset_dir, "pershank")
        export_per_shank(flow_dir, shank_out)
        get_iir_weights(shank_out)
        return

    # --- 2. Load Ops ---
    ops_path = os.path.join(export_dir, "ops.npy")
    ops = np.load(ops_path, allow_pickle=True).item()

    # --- 3. Load Wall (Templates) ---
    Wall_np = np.load(os.path.join(export_dir, "Wall.npy"))
    Wall_h = os.path.join(result_dir, "Wall.h")
    Wall_masked_np = mask_Wall(Wall_np, ops)
    write_c_float_header(Wall_masked_np, "Wall", Wall_h, export_dir)

    # --- 4. Export nm[] — one scalar per template ---
    Wall_nm_masked = (Wall_masked_np ** 2).sum(axis=-1).sum(axis=-1).astype(np.float32)  # (K,)
    Wall_nm = (Wall_np ** 2).sum(axis=-1).sum(axis=-1).astype(np.float32)  # (K,)
    nm_h = os.path.join(result_dir, "Wall_nm.h")
    write_c_float_header(Wall_nm, "Wall_nm", nm_h, export_dir)

    # --- 5. Export Th_learned (detection threshold) ---
    Th_learned = float(ops.get("Th_learned", 8.0))
    print(f"Th_learned = {Th_learned}  (compare scores against {Th_learned**2:.2f} in C)")
    
    # --- 6. Load 'wPCA' (PC templates) ---
    wPCA = ops.get('wPCA', None)
    if wPCA is not None:
        wPCA = wPCA.detach().cpu().numpy() if torch.is_tensor(wPCA) else wPCA
    wPCA_h = os.path.join(result_dir, "wPCA.h")
    write_c_float_header(wPCA, "wPCA", wPCA_h, export_dir)

    # --- 7. Extract and Write Wrot (Whitening Matrix) ---
    Wrot = ops.get("Wrot", None)
    if Wrot is not None:
        if isinstance(Wrot, torch.Tensor):
            Wrot = Wrot.detach().cpu().numpy()
        Wrot_h = os.path.join(result_dir, "Wrot.h")
        write_c_float_header(Wrot, "Wrot", Wrot_h, export_dir)
    else:
        print("Wrot not found in ops.")

    """ # --- 8. Extract and Write Mul (Drift Shift * whitening) ---
    Wsh = np.load(os.path.join(export_dir, "Wsh.npy"))
    if Wsh is not None:
        if isinstance(Wsh, torch.Tensor):
            Wsh = Wsh.detach().cpu().numpy()
        Wsh_h = os.path.join(result_dir, "Wsh.h")
        write_c_float_header(Wsh, "Wsh", Wsh_h, export_dir)
    else:
        print("Wsh not found in directory.")

    # --- 8b. Export per-batch Wsh stack for full drift ---
    Wsh_pb = np.load(os.path.join(export_dir, "Wsh_per_batch.npy"))
    # shape (n_online_batches, NCHAN, NCHAN)
    n_batches = Wsh_pb.shape[0]
    print(f"Wsh_per_batch shape: {Wsh_pb.shape} -> {n_batches} batches")
    Wsh_pb_h = os.path.join(result_dir, "Wsh_per_batch.h")
    write_c_float_header(Wsh_pb.astype(np.float32), "Wsh_per_batch", Wsh_pb_h, export_dir)
    """
    # --- 9. Get IIR filter weights 
    get_iir_weights(result_dir)

    # --- 10. Get template groups according to locations
    # Outputs an array such that the index is the template's id and the value is the group number according to location.
    """ indices_path = os.path.join(export_dir, "pc_feature_ind.npy")
    probe_path = os.path.join(result_dir, "probe_mp79.json")  #change probe name
    with open(probe_path, 'r') as f:     #change probe name
        probe = json.load(f)
    indices = np.load(indices_path, allow_pickle=True)
    groups_np = group_templates(indices, probe)
    print(np.unique(groups_np, axis=0).shape[0], "groups;", np.bincount(groups_np).sum(), "assigned")
    template_groups_h = os.path.join(result_dir, "spatial_template_groups.h")
    write_c_float_header(groups_np, "spatial_template_groups", template_groups_h, export_dir)
"""
    # --- 11. Get per template threshold
    thresholds = th_per_temp(export_dir)
    th_h = os.path.join(result_dir, "TH.h")
    write_c_float_header(thresholds, "TH", th_h, export_dir) 



if __name__ == "__main__":
    main()
