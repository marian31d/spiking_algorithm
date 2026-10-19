import numpy as np

# .dat is int16, shape (n_samples, n_chans)
n_chans = 136
data = np.memmap('/data/mP79/dat/mP79_17/mP79_17.dat', dtype='int16', mode='r')
data = data.reshape(-1, n_chans)
cropped = data[2600:3600, :]          # 1000 samples x 128 channels

# --- save for Python ---
np.save('/home/mariandawud/cropped.npy', cropped)

# --- save ONE channel for MATLAB (plain text, readable with readmatrix) ---
# kilosort main channel 45 (0-based) -> column index 45 in Python.
main_ch0 = 4
one_channel = cropped[:, main_ch0]                 # 1000-sample vector
np.savetxt(f"/home/mariandawud/cropped_ch{main_ch0}.csv", one_channel, fmt='%d')

# NOTE on indexing for the MATLAB script:
#   crop = data[2000:3000]  -> crop row k (0-based) is global sample 2000+k
#   spike at global sample 2267 -> crop row 267 (0-based) -> MATLAB sample 268 (1-based)
#   kilosort main channel 45 (0-based)                    -> MATLAB column  46 (1-based)