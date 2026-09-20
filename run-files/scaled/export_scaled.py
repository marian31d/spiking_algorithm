from fileinput import filename

import numpy as np
import torch
import json
import os
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt
from kilosort.template_matching import prepare_matching

# scaling factors for fixed point representation in C
Q15 = 2**15
Q30 = 2**30

# --- Helper Function for C Header Export ---
def write_c_int_header(array_data, array_name, filename, export_dir, scale_factor=None, type_c = np.int32):
    """Flattens a numpy array and writes it as a C float header file."""
    if scale_factor == Q15:
        scaled = np.round(array_data * Q15)
        if type_c == np.int16:
            assert scaled.max() <= 32767 and scaled.min() >= -32768, \
                f"Overflow int16: [{scaled.min():.1f}, {scaled.max():.1f}]"
        elif type_c == np.int32:
            assert scaled.max() <= 2**31-1 and scaled.min() >= -(2**31), \
                f"Overflow int32: [{scaled.min():.1f}, {scaled.max():.1f}]"
        array_data = np.round(array_data * Q15).astype(type_c)
    elif scale_factor == Q30:
        scaled = np.round(array_data * Q30)
        assert scaled.max() < (Q30+1) and scaled.min() > -(Q30+1), "Data exceeds Q30 range"
        array_data = np.round(array_data * Q30).astype(type_c)
    else:
        array_data = np.round(array_data).astype(type_c)

    flat_data = array_data.flatten()

    ctype_map = {np.int16: "int16_t", np.int32: "int32_t", np.int64: "int64_t"}
    ctype_str = ctype_map.get(type_c, "int32_t")

    with open(filename, "w") as f:
        f.write(f"// Header file for {export_dir}\n")
        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")
        f.write(f"#include <stdint.h>\n\n")
        f.write(f"const {ctype_str} {array_name}[{len(flat_data)}] = {{\n")
        
        for i, val in enumerate(flat_data):
            f.write(f"{val}, ")
            # Newline every 10 values for readability
            if (i + 1) % 10 == 0:
                f.write("\n")      
        f.write("\n};\n\n")
        f.write(f"#endif // {array_name.upper()}_H\n")  
    print(f"Generated {filename} ({len(flat_data)} elements as {ctype_str}), size of: {flat_data.nbytes} bytes.")
    print(array_name, "max: ", np.max(array_data), "min: ", np.min(array_data), "type: ", array_data.dtype, "dimensions: ", array_data.shape)


def get_iir_weights(result_dir):
    # --- Filter Specifications ---
    fs = 20000   # Sampling rate (Hz)
    fc = 300     # High-pass cutoff frequency (Hz)
    order = 2    # 2nd-order Biquad filter
    
    # Generate IIR Butterworth coefficients
    b, a = signal.butter(order, fc, btype='highpass', fs=fs)
    # IIR: b/a coefficients are already in [-1, 1], Q15 is perfect
    b_q15 = (b * Q15).astype(np.int16)
    a_q15 = (a[1:] * Q15).astype(np.int16)
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
    const int16_t iir_weights[5] = {{
        {b_q15[0]}, 
        {b_q15[1]}, 
        {b_q15[2]}, 
        {a_q15[0]}, 
        {a_q15[1]}
    }};
    
    #endif // IIR_WEIGHTS_H
    """
    
    # --- Write to File ---
    iir_path = os.path.join(result_dir, "iir_weights.h")
    with open(iir_path, "w") as f:
        f.write(c_header_content)
    
    print(f"Successfully generated {iir_path} as an array.")

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
    Wall_h = os.path.join(result_dir, "Wall_SCALED.h")
    Wall_masked_np = mask_Wall(Wall_np, ops)
    write_c_int_header(Wall_masked_np, "Wall_SCALED", Wall_h, Q15, np.int32)

    # --- 4. Export nm[] — one scalar per template ---
    Wall_nm_masked = (Wall_masked_np ** 2).sum(axis=-1).sum(axis=-1).astype(np.float32)  # (K,)
    Wall_nm = (Wall_np ** 2).sum(axis=-1).sum(axis=-1).astype(np.float32)  # (K,)
    nm_h = os.path.join(result_dir, "Wall_nm_SCALED.h")
    write_c_int_header(Wall_nm_masked, "Wall_nm_SCALED", nm_h, export_dir, None, np.int32)

    # --- 5. Export Th_learned (detection threshold) ---
    Th_learned = float(ops.get("Th_learned", 8.0))
    print(f"Th_learned = {Th_learned}  (compare scores against {Th_learned**2:.2f} in C)")
    
    # --- 6. Load 'wPCA' (PC templates) ---
    wPCA = ops.get('wPCA', None)
    if wPCA is not None:
        wPCA = wPCA.detach().cpu().numpy() if torch.is_tensor(wPCA) else wPCA
    WPCA_h = os.path.join(result_dir, "wPCA_SCALED.h")
    write_c_int_header(wPCA, "wPCA_SCALED", WPCA_h, export_dir, Q15, np.int16)

    # --- 7. Extract and Write Wrot (Whitening Matrix) ---
    Wrot = ops.get("Wrot", None)
    if Wrot is not None:
        if isinstance(Wrot, torch.Tensor):
            Wrot = Wrot.detach().cpu().numpy()
        Wrot_h = os.path.join(result_dir, "Wrot_SCALED.h")
        write_c_int_header(Wrot, "Wrot_SCALED", Wrot_h, export_dir, Q15, np.int16)
    else:
        print("Wrot not found in ops.")

    # --- 8. Get IIR filter weights 
    get_iir_weights(result_dir) 

    # --- 9. Get per template threshold
    thresholds = th_per_temp(export_dir)
    th_h = os.path.join(result_dir, "TH_SCALED.h")
    write_c_int_header(thresholds, "TH_SCALED", th_h, export_dir, None, np.int16)

if __name__ == "__main__":
    main()
