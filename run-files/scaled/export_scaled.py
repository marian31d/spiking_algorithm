import numpy as np
import torch
import os
from scipy import signal
from scipy.signal import butter, filtfilt

# scaling factors for fixed point representation in C
Q15 = 2**15
Q30 = 2**30

# --- Helper Function for C Header Export ---
def write_c_int_header(array_data, array_name, filename, scale_factor=None, type_c = np.int32):
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
    print(f"Generated {filename} ({len(flat_data)} elements as {ctype_str}).")

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
    
def main():
    # --- 1. Set your directory ---
    #
    export_dir = "/data/results/mds2_res/flows_final_shift/batch-def/offline_3600/export"
    offline_dir = "/data/results/mds2_res/flows_final_shift/batch-def/offline_3600"
    result_dir = "/data/results/mS51_res/spiking_algorithm/mds2"

    # --- 2. Load Wall (Templates) ---
    Wall_np = np.load(os.path.join(export_dir, "Wall.npy"))
    Wall_h = os.path.join(result_dir, "Wall_SCALED.h")
    write_c_int_header(Wall_np, "Wall_SCALED", Wall_h, Q15, np.int32)

    # --- 3. Export nm[] — one scalar per template ---
    #nm = (Wall_np ** 2).sum(axis=-1).sum(axis=-1).astype(np.float32)  # (K,)
    Wall_nm_np = np.load(os.path.join(export_dir, "Wall_nm.npy"))
    nm_h = os.path.join(result_dir, "Wall_nm.h")
    write_c_int_header(Wall_nm_np, "Wall_nm", nm_h, None, np.int32)

    # --- 4. Load Ops ---
    ops_path = os.path.join(offline_dir, "ops.npy")
    ops = np.load(ops_path, allow_pickle=True).item()

    # --- 5. Export Th_learned (detection threshold) ---
    Th_learned = float(ops.get("Th_learned", 8.0))
    print(f"Th_learned = {Th_learned}  (compare scores against {Th_learned**2:.2f} in C)")
    
    # --- 6. Load 'wPCA' (PC templates) ---
    WPCA_np = np.load(os.path.join(export_dir, "wPCA.npy"))
    WPCA_h = os.path.join(result_dir, "wPCA_SCALED.h")
    write_c_int_header(WPCA_np, "wPCA_SCALED", WPCA_h, Q15, np.int16)

    # --- 7. Extract and Write Wrot (Whitening Matrix) ---
    Wrot = ops.get("Wrot", None)
    if Wrot is not None:
        if isinstance(Wrot, torch.Tensor):
            Wrot = Wrot.detach().cpu().numpy()
        Wrot_h = os.path.join(result_dir, "Wrot_SCALED.h")
        write_c_int_header(Wrot, "Wrot_SCALED", Wrot_h, Q15, np.int16)
    else:
        print("Wrot not found in ops.")

    # --- 8. Extract and Write Mul (Drift Shift * whitening) ---
    Wsh = np.load(os.path.join(export_dir, "Wsh.npy"))
    if Wsh is not None:
        if isinstance(Wsh, torch.Tensor):
            Wsh = Wsh.detach().cpu().numpy()
        Wsh_h = os.path.join(result_dir, "Wsh_SCALED.h")
        write_c_int_header(Wsh, "Wsh_SCALED", Wsh_h, Q15, np.int16)
    else:
        print("Wsh not found in directory.")
    # --- 9. Get template cross correlation (ctc) --- 
    ctc = np.load(os.path.join(export_dir, "ctc.npy"))
    ctc_h = os.path.join(result_dir, "ctc_SCALED.h")
    #write_c_int_header(ctc, "ctc", ctc_h, Q30)

    # --- 10. Get IIR filter weights 
    get_iir_weights(result_dir)

if __name__ == "__main__":
    main()
