import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def cosine_similarity_matrix(A, B, eps=1e-12):
    """A: (G, D), B: (F, D) -> S: (G, F)"""
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    A_unit = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), eps)
    B_unit = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), eps)
    return A_unit @ B_unit.T


def main():
	"""
	Calculates template similarity between templates in the same Wall
	"""
	ap = argparse.ArgumentParser()
	ap.add_argument("--k_path", required=True, type=str)
	args = ap.parse_args()

	k_path = Path(args.k_path)

	ops = np.load(k_path / "ops.npy", allow_pickle=True).item()  # dict
	Wall = np.load(k_path / "Wall.npy")        # (nclu, nchan, npc)
	wPCA = ops["wPCA"]                          # (npc, nt)

    # reconstruct each cluster's spatiotemporal template -> (nclu, nchan, nt)
	waveforms = np.einsum("cnp,pt->cnt", Wall, wPCA)

	# flatten each template to one vector, then a single similarity call
    
	flat = waveforms.reshape(waveforms.shape[0], -1)   # (nclu, nchan*nt)
	sim_mat = cosine_similarity_matrix(flat, flat)     # (nclu, nclu)

	plt.figure(figsize=(9, 8))
	plt.imshow(sim_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
	plt.colorbar(label="Cosine similarity")
	plt.title("Template similarity")
	plt.xlabel("Cluster")
	plt.ylabel("Cluster")
	plt.savefig(k_path / "template_similarity.png", dpi=300, bbox_inches="tight")
	plt.close()


if __name__ == "__main__":
    main()