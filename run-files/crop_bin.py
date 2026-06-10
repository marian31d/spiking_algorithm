import numpy as np

# temp_wh.dat is int16, shape (n_samples, n_chans)
n_chans = 128
data = np.memmap('temp_wh.dat', dtype='int16', mode='r')
data = data.reshape(-1, n_chans)
last_99 = data[-99:, :]
np.save('/tmp/ks4_last99.npy', last_99)
