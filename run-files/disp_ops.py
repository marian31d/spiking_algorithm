#!/usr/bin/env python3
import numpy as np
import argparse
from pathlib import Path


# Parameters grouped like the GUI layout
PARAM_GROUPS = {
    "data": [
        "nt", "shift", "scale", "batch_downsampling"
    ],
    "preprocessing": [
        "artifact_threshold", "nskip", "whitening_range",
        "highpass_cutoff", "binning_depth", "sig_interp",
        "drift_smoothing"
    ],
    "spike_detection": [
        "nt0min", "dmin", "dminx", "min_template_size",
        "template_sizes", "nearest_chans", "nearest_templates",
        "max_channel_distance", "max_peels",
        "templates_from_data", "n_templates",
        "n_pcs", "Th_universal", "Th_learned"
    ],
    "clustering": [
        "acg_threshold", "ccg_threshold",
        "cluster_neighbors", "cluster_downsampling",
        "max_cluster_subset", "x_centers",
        "cluster_init_seed"
    ],
    "postprocessing": [
        "duplicate_spike_ms", "position_limit"
    ]
}


def print_group(name, ops):
    print(f"\n{name}")
    print("-" * len(name))
    for key in PARAM_GROUPS[name]:
        val = ops.get(key, "NOT FOUND")
        print(f"{key:25s}: {val}")


def main():
    parser = argparse.ArgumentParser(description="Extract parameters from ops.npy")
    parser.add_argument("ops_path", type=str, help="Path to ops.npy")
    args = parser.parse_args()

    ops_path = Path(args.ops_path)
    if not ops_path.exists():
        raise FileNotFoundError(f"{ops_path} not found")

    ops = np.load(ops_path, allow_pickle=True).item()

    print(f"\nLoaded ops from: {ops_path}")

    for group in PARAM_GROUPS:
        print_group(group, ops)

    print("tmin:", ops.get("tmin"))
    print("tmax:", ops.get("tmax"))


if __name__ == "__main__":
    main()



