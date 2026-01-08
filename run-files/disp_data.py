import numpy as np

bin_file = "/media/mariandawud/int_ssd/data/mS51_49.dat"  # CHANGE THIS
n_channels = 64
n_timepoints = 5                     # number of time samples to print

# read first samples
data = np.fromfile(
    bin_file,
    dtype=np.int16,
    count=n_channels * n_timepoints
)

# reshape to (time, channels)
data = data.reshape(n_timepoints, n_channels)

print(data)
