import numpy as np
import torch
import os

#This script converts Wall, dshift and Wrot from npy arrays to header files .h

def write_c_float_header(array_data, array_name, filename):
    """Flattens a numpy array and writes it as a C float header file."""
    flat_data = array_data.flatten()
    
    with open(filename, "w") as f:
        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")
        
        # Declare as const float (single precision)
        f.write(f"const float {array_name}[{len(flat_data)}] = {{\n")
        
        for i, val in enumerate(flat_data):
            # Format with 6 decimal places and the 'f' suffix to enforce 32-bit float in C
            f.write(f"{val:.6f}f, ")
            
            # Newline every 10 values for readability
            if (i + 1) % 10 == 0:
                f.write("\n")
                
        f.write("\n};\n\n")
        f.write(f"#endif // {array_name.upper()}_H\n")
        
    print(f"Generated {filename} ({len(flat_data)} elements)")

def main():
    # --- 1. Set your directory ---
    # Update this path to where your offline results are saved
    offline_dir = "/data/results/mS51_res/flows_final_shift/offline_3600"
    
    # --- 2. Load Wall (Templates) ---
    Wall_np = np.load(os.path.join(offline_dir, "Wall.npy"))
    
    # TIP: For your first test, you might want to slice Wall so it fits in L2 RAM.
    # To export just the first template, uncomment the next line:
    # Wall_np = Wall_np[0:1] 
    Wall_h = os.path.join(offline_dir, "Wall_templates.h")
    write_c_float_header(Wall_np, "Wall_templates", Wall_h)

    # --- 3. Load Ops (Contains Wrot and dshift) ---
    ops_path = os.path.join(offline_dir, "ops.npy")
    # Kilosort ops are saved as a pickled dictionary
    ops = np.load(ops_path, allow_pickle=True).item()

    # --- 4. Extract and Write Wrot (Whitening Matrix) ---
    Wrot = ops.get("Wrot", None)
    if Wrot is not None:
        # If it is a PyTorch tensor, convert it to a NumPy array
        if isinstance(Wrot, torch.Tensor):
            Wrot = Wrot.detach().cpu().numpy()
        Wrot_h = os.path.join(offline_dir, "Wrot.h")
        write_c_float_header(Wrot, "Wrot_matrix", Wrot_h)
    else:
        print("Wrot not found in ops.")

    # --- 5. Extract and Write dshift (Drift Shift) ---
    dshift = ops.get("dshift", None)
    if dshift is not None:
        if isinstance(dshift, torch.Tensor):
            dshift = dshift.detach().cpu().numpy()
        dshift_h = os.path.join(offline_dir, "dshift.h")
        write_c_float_header(dshift, "dshift_array", dshift_h)
    else:
        print("dshift not found in ops.")

    # --- 6. Extract and Write fwav (HP Filter) ---
    fwav = ops.get("fwav", None)
    if fwav is not None:
        if isinstance(fwav, torch.Tensor):
            fwav = fwav.detach().cpu().numpy()
        fwav_h = os.path.join(offline_dir, "fwav.h")
        write_c_float_header(fwav, "fwav_array", fwav_h)
    else:
        print("fwav not found in ops.")

if __name__ == "__main__":
    main()
