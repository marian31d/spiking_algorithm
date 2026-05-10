import numpy as np

parser = argparse.ArgumentParser(description="Normalize Wall templates")
parser.add_argument("results_path", type=str, help="Path to results")
args = parser.parse_args()

results_path = Path(args.results_path)
if not results_path.exists():
    raise FileNotFoundError(f"{results_path} not found")

# 1. Load your existing exported Kilosort arrays
# Make sure the filenames match what is in your directory
wPCA = np.load(results_path / 'wPCA.npy')  # Shape: (3, 61)
Wall = np.load(results_path / 'Wall.npy')  # Shape: (N_TEMPLATES, 126, 3)

# Create a copy so we don't overwrite the original Kilosort output
Wall_normalized = np.copy(Wall)

print(f"Normalizing {Wall.shape[0]} templates...")

# 2. Loop through every template and normalize it
for i in range(Wall_normalized.shape[0]):

    # Reconstruct the full 61-sample voltage waveform for this template
    # Math: (126, 3) dot (3, 61) -> (126, 61) matrix
    full_template = np.dot(Wall_normalized[i], wPCA)

    # Calculate the L2 norm (the mathematical 'length' or amplitude of the spike)
    norm = np.linalg.norm(full_template)

    # Divide the spatial weights by the norm to shrink the template to exactly 1.0
    # We check > 0 just to prevent divide-by-zero errors on empty/deleted templates
    if norm > 0:
        Wall_normalized[i] = Wall_normalized[i] / norm

# 3. Quick Verification
test_template = np.dot(Wall_normalized[0], wPCA)
print(f"New Template 0 Norm: {np.linalg.norm(test_template):.6f} (Should be 1.000000)")

# 4. Save the corrected array to disk
np.save('Wall_normalized.npy', Wall_normalized)
print("Success! Saved 'Wall_normalized.npy'.")
