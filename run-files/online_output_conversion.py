import numpy as np
from pathlib import Path

out_dir = Path("/path/to/your/online_output")  # <- change this
st = np.load(out_dir / "st.npy")

# safety: handle torch-saved arrays etc
st = np.asarray(st)

spike_times = st[:, 0].astype(np.int64, copy=False)
spike_clusters = st[:, 1].astype(np.int64, copy=False)
spike_clusters = np.load(out_dir / "spike_clusters.npy")

np.save(out_dir / "spike_times.npy", spike_times)
np.save(out_dir / "spike_clusters.npy", spike_clusters)
np.save(out_dir / "spike_templates.npy", spike_clusters.astype(np.int64, copy=False))

print("Wrote:", out_dir/"spike_times.npy")
print("Wrote:", out_dir/"spike_clusters.npy")
print("Wrote:", out_dir/"spike_templates.npy")
print("times min/max:", int(spike_times.min()), int(spike_times.max()))
print("clusters:", int(np.unique(spike_clusters).size))
