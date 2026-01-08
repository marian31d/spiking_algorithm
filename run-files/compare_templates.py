import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def main():
    # 1. Setup the terminal argument parser
    parser = argparse.ArgumentParser(description="Compare Kilosort templates between GUI and Two-Flow runs.")
    
    # Define the two required path arguments
    parser.add_argument("--gui_path", type=str, required=True, 
                        help="Path to the Wall.npy or templates.npy from the FULL GUI run")
    parser.add_argument("--flows_path", type=str, required=True, 
                        help="Path to the Wall.npy from the Two-Flows offline output")
    
    args = parser.parse_args()

    # 2. Load the templates
    try:
        wall_gui = np.load(args.gui_path)
        wall_flows = np.load(args.flows_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"✅ Loaded GUI templates: {wall_gui.shape}")
    print(f"✅ Loaded Flows templates: {wall_flows.shape}")

    # 3. Flatten the 3D templates (K, Channels, PCs) into 2D vectors (K, Features)
    # This allows us to compare the entire "shape" of the spike at once
    gui_flat = wall_gui.reshape(wall_gui.shape[0], -1)
    flows_flat = wall_flows.reshape(wall_flows.shape[0], -1)

    # 4. Compute the Similarity Matrix (every flow cluster vs every gui cluster)
    sim_matrix = cosine_similarity(flows_flat, gui_flat)

    # 5. Find and display the 1:1 matches
    print("\n" + "="*50)
    print(f"{'Flow Cluster':<15} | {'GUI Cluster':<15} | {'Match Score'}")
    print("-" * 50)
    
    for i in range(len(wall_flows)):
        best_match_idx = np.argmax(sim_matrix[i])
        score = sim_matrix[i][best_match_idx]
        
        # Display as a percentage (1.0 = 100%)
        print(f"{i:<15} | {best_match_idx:<15} | {score*100:.2f}%")
    print("="*50 + "\n")

    # 6. Visualization: Heatmap of similarities
    plt.figure(figsize=(12, 10))
    plt.imshow(sim_matrix, aspect='auto', cmap='magma')
    plt.colorbar(label='Similarity (1.0 = 100% Match)')
    plt.xlabel('GUI Cluster IDs (Full Data Run)')
    plt.ylabel('Two-Flows Cluster IDs (Offline-Online)')
    plt.title('Template Similarity Matrix: GUI vs. SoC Simulation')
    plt.show()

if __name__ == "__main__":
    main()
