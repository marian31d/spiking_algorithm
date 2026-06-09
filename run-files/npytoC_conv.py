import numpy as np
import torch
import os
from scipy import signal
from scipy.signal import butter, filtfilt

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
        
    print(f"Generated {filename} ({len(flat_data)} elements as floats)")

def get_iir_weights(result_dir):
    # --- Filter Specifications ---
    fs = 20000   # Sampling rate (Hz)
    fc = 300     # High-pass cutoff frequency (Hz)
    order = 2    # 2nd-order Biquad filter
    
    # Generate IIR Butterworth coefficients
    b, a = signal.butter(order, fc, btype='highpass', fs=fs)
    
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
    
    print(f"Successfully generated {iir_path} as an array.")
    
def main():
    # --- 1. Set your directory ---
    #
    export_dir = "/data/results/mds2_res/flows_final_shift/updated_dshift/gui-online-shift/export"
    offline_dir = "/data/results/mds2_res/flows_final_shift/updated_dshift/gui-online-shift"
    result_dir = "/data/spiking_algorithm/mds2"

    # --- 2. Load Wall (Templates) ---
    Wall_np = np.load(os.path.join(export_dir, "Wall.npy"))
    Wall_h = os.path.join(result_dir, "Wall.h")
    write_c_float_header(Wall_np, "Wall", Wall_h)

    # --- 3. Export nm[] — one scalar per template ---
    #nm = (Wall_np ** 2).sum(axis=-1).sum(axis=-1).astype(np.float32)  # (K,)
    Wall_nm_np = np.load(os.path.join(export_dir, "Wall_nm.npy"))
    nm_h = os.path.join(result_dir, "Wall_nm.h")
    write_c_float_header(Wall_nm_np, "Wall_nm", nm_h)

    # --- 4. Load Ops ---
    ops_path = os.path.join(offline_dir, "ops.npy")
    ops = np.load(ops_path, allow_pickle=True).item()

    # --- 5. Export Th_learned (detection threshold) ---
    Th_learned = float(ops.get("Th_learned", 8.0))
    print(f"Th_learned = {Th_learned}  (compare scores against {Th_learned**2:.2f} in C)")
    
    # --- 6. Load 'wPCA' (PC templates) ---
    WPCA_np = np.load(os.path.join(export_dir, "wPCA.npy"))
    WPCA_h = os.path.join(result_dir, "wPCA.h")
    write_c_float_header(WPCA_np, "wPCA", WPCA_h)

    # --- 7. Extract and Write Wrot (Whitening Matrix) ---
    Wrot = ops.get("Wrot", None)
    if Wrot is not None:
        if isinstance(Wrot, torch.Tensor):
            Wrot = Wrot.detach().cpu().numpy()
        Wrot_h = os.path.join(result_dir, "Wrot.h")
        write_c_float_header(Wrot, "Wrot", Wrot_h)
    else:
        print("Wrot not found in ops.")

    # --- 8. Extract and Write Mul (Drift Shift * whitening) ---
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

    # --- 9. Get IIR filter weights 
    get_iir_weights(result_dir)

if __name__ == "__main__":
    main()
