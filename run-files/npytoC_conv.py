import numpy as np
import torch
import json
import os
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt
from kilosort.template_matching import prepare_matching

# --- Helper Function for C Header Export ---

def write_c_float_header(array_data, array_name, filename):
    """Flattens a numpy array and writes it as a C float header file."""
    flat_data = array_data.flatten()
    with open(filename, "w") as f:
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
    order = 2    # 2nd-order Biquad filter
    
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
    const float iir_weights[5] = {{
        {b[0]:.8f}f,
        {b[1]:.8f}f,
        {b[2]:.8f}f,
        {a[1]:.8f}f,
        {a[2]:.8f}f
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



def main():
    # --- 1. Set your directory ---
    export_dir = "/data/results/mP79_res/IIR/time/3600/2ndOrd-filt"
    result_dir = "/data/spiking_algorithm/mp79"
    num_shanks = 8  # Set this according to your probe geometry

    # --- 2. Load Ops ---
    ops_path = os.path.join(export_dir, "ops.npy")
    ops = np.load(ops_path, allow_pickle=True).item()

    # --- 3. Load Wall (Templates) ---
    Wall_np = np.load(os.path.join(export_dir, "Wall.npy"))
    Wall_h = os.path.join(result_dir, "Wall.h")
    Wall_masked_np = mask_Wall(Wall_np, ops)
    write_c_float_header(Wall_masked_np, "Wall", Wall_h)

    # --- 4. Export nm[] — one scalar per template ---
    Wall_nm_masked = (Wall_masked_np ** 2).sum(axis=-1).sum(axis=-1).astype(np.float32)  # (K,)
    Wall_nm = (Wall_np ** 2).sum(axis=-1).sum(axis=-1).astype(np.float32)  # (K,)
    nm_h = os.path.join(result_dir, "Wall_nm.h")
    write_c_float_header(Wall_nm, "Wall_nm", nm_h)

    # --- 5. Export Th_learned (detection threshold) ---
    Th_learned = float(ops.get("Th_learned", 8.0))
    print(f"Th_learned = {Th_learned}  (compare scores against {Th_learned**2:.2f} in C)")
    
    # --- 6. Load 'wPCA' (PC templates) ---
    wPCA = ops.get('wPCA', None)
    if wPCA is not None:
        wPCA = wPCA.detach().cpu().numpy() if torch.is_tensor(wPCA) else wPCA
    wPCA_h = os.path.join(result_dir, "wPCA.h")
    write_c_float_header(wPCA, "wPCA", wPCA_h)

    # --- 7. Extract and Write Wrot (Whitening Matrix) ---
    Wrot = ops.get("Wrot", None)
    if Wrot is not None:
        if isinstance(Wrot, torch.Tensor):
            Wrot = Wrot.detach().cpu().numpy()
        Wrot_h = os.path.join(result_dir, "Wrot.h")
        write_c_float_header(Wrot, "Wrot", Wrot_h)
    else:
        print("Wrot not found in ops.")

    """ # --- 8. Extract and Write Mul (Drift Shift * whitening) ---
    Wsh = np.load(os.path.join(export_dir, "Wsh.npy"))
    if Wsh is not None:
        if isinstance(Wsh, torch.Tensor):
            Wsh = Wsh.detach().cpu().numpy()
        Wsh_h = os.path.join(result_dir, "Wsh.h")
        write_c_float_header(Wsh, "Wsh", Wsh_h)
    else:
        print("Wsh not found in directory.")

    # --- 8b. Export per-batch Wsh stack for full drift ---
    Wsh_pb = np.load(os.path.join(export_dir, "Wsh_per_batch.npy"))
    # shape (n_online_batches, NCHAN, NCHAN)
    n_batches = Wsh_pb.shape[0]
    print(f"Wsh_per_batch shape: {Wsh_pb.shape} -> {n_batches} batches")
    Wsh_pb_h = os.path.join(result_dir, "Wsh_per_batch.h")
    write_c_float_header(Wsh_pb.astype(np.float32), "Wsh_per_batch", Wsh_pb_h)
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
    write_c_float_header(groups_np, "spatial_template_groups", template_groups_h) """

    # --- 11. Get per template threshold
    thresholds = th_per_temp(export_dir)
    th_h = os.path.join(result_dir, "TH.h")
    write_c_float_header(thresholds, "TH", th_h)



if __name__ == "__main__":
    main()
