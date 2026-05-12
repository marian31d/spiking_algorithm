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

# --- Main Execution ---

def main():
    # --- 1. Set your directory ---
    #
    offline_dir = "/data/results/mS51_res/flows_final_shift/batch32768/offline_3600"
    result_dir = "/home/mariandawud/kilosort/ms51"
    # --- 2. Load Wall (Templates) ---
    Wall_np = np.load(os.path.join(offline_dir, "Wall_Normalized.npy"))
    Wall_h = os.path.join(result_dir, "Wall_Normal.h")
    write_c_float_header(Wall_np, "Wall_Normal", Wall_h)

    # --- 3. Load 'wPCA' (PC templates) ---
    WPCA_np = np.load(os.path.join(offline_dir, "wPCA.npy"))
    WPCA_h = os.path.join(result_dir, "wPCA.h")
    write_c_float_header(WPCA_np, "wPCA", WPCA_h)

    # --- 4. Load Ops ---
    ops_path = os.path.join(offline_dir, "ops.npy")
    ops = np.load(ops_path, allow_pickle=True).item()

    # --- 5. Extract and Write Wrot (Whitening Matrix) ---
    Wrot = ops.get("Wrot", None)
    if Wrot is not None:
        if isinstance(Wrot, torch.Tensor):
            Wrot = Wrot.detach().cpu().numpy()
        Wrot_h = os.path.join(result_dir, "Wrot.h")
        write_c_float_header(Wrot, "Wrot_matrix", Wrot_h)
    else:
        print("Wrot not found in ops.")

    # --- 6. Extract and Write Mul (Drift Shift * whitening) ---
    Wsh = np.load(os.path.join(offline_dir, "Wsh.npy"))
    if Wsh is not None:
        if isinstance(Wsh, torch.Tensor):
            Wsh = Wsh.detach().cpu().numpy()
        Wsh_h = os.path.join(result_dir, "Wsh.h")
        write_c_float_header(Wsh, "Wsh_array", Wsh_h)
    else:
        print("Wsh not found in directory.")

if __name__ == "__main__":
    main()
