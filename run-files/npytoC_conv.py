import numpy as np
import torch
import os
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


# --- Filter Generation Logic ---

def generate_unscaled_fir_taps(fs=20000, cutoff=300, num_taps=71):
    """Generates the exact unscaled time-domain impulse response."""
    NT = 30122
    
    # 1. Kilosort's exact Butterworth definition
    b, a = butter(3, cutoff, fs=fs, btype='high')
    x = np.zeros(NT)
    x[NT//2] = 1
    
    # 2. Get the massive symmetrical filter
    hp_filter = filtfilt(b, a, x).copy()
    
    # 3. Automatically find the center peak of the impulse response
    center = np.argmax(np.abs(hp_filter))
    
    # 4. Slice out only the core taps needed for the SoC
    half_taps = num_taps // 2
    core_filter = hp_filter[center - half_taps : center + half_taps + 1]
    
    # Return raw floats (NO scaling)
    return core_filter


# --- Main Execution ---

def main():
    # --- 1. Set your directory ---
    offline_dir = "/data/results/mS51_res/flows_final_shift/offline_3600"
    result_directory = "~/kilosort/run-files"
    # --- 2. Load Wall (Templates) ---
    Wall_np = np.load(os.path.join(offline_dir, "Wall.npy"))
    Wall_h = os.path.join(result_dir, "Wall_templates.h")
    write_c_float_header(Wall_np, "Wall_templates", Wall_h)

    # --- 3. Load Ops ---
    ops_path = os.path.join(offline_dir, "ops.npy")
    ops = np.load(ops_path, allow_pickle=True).item()

    # --- 4. Extract and Write Wrot (Whitening Matrix) ---
    Wrot = ops.get("Wrot", None)
    if Wrot is not None:
        if isinstance(Wrot, torch.Tensor):
            Wrot = Wrot.detach().cpu().numpy()
        Wrot_h = os.path.join(result_dir, "Wrot.h")
        write_c_float_header(Wrot, "Wrot_matrix", Wrot_h)
    else:
        print("Wrot not found in ops.")

    # --- 5. Extract and Write dshift (Drift Shift) ---
    dshift = ops.get("dshift", None)
    if dshift is not None:
        if isinstance(dshift, torch.Tensor):
            dshift = dshift.detach().cpu().numpy()
        dshift_h = os.path.join(result_dir, "dshift.h")
        write_c_float_header(dshift, "dshift_array", dshift_h)
    else:
        print("dshift not found in ops.")

    # --- 6. Generate and Write the Unscaled Time-Domain FIR Filter ---
    try:
        # Note: Set fs to 30000 here if your specific dataset is 30kHz!
        fir_taps_float = generate_unscaled_fir_taps(fs=20000, cutoff=300, num_taps=71)
        fir_h = os.path.join(result_dir, "fir_filter_taps.h")
        write_c_float_header(fir_taps_float, "fir_weights", fir_h)
    except Exception as e:
        print(f"Failed to generate FIR filter: {e}")

if __name__ == "__main__":
    main()
