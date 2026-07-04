import numpy as np
import gc, torch
from pathlib import Path
from kilosort import run_kilosort
from kilosort.io import load_probe
from kilosort.parameters import DEFAULT_SETTINGS

# --- recover the exact settings from a reference GUI run ---
ref = np.load('/data/results/mds2_res/IIR/matching_pursuit/100peels/ops.npy', allow_pickle=True).item()

# ops holds every merged setting; pull the ones run_kilosort takes as `settings`
# (n_chan_bin, fs, Th_universal, Th_learned, batch_size, nt, dmin, dminx, ... )
settings = {k: ref[k] for k in DEFAULT_SETTINGS if k in ref}
# if your version doesn't nest 'settings', copy the DEFAULT_SETTINGS keys out of ops directly.

probe = load_probe('/home/mariandawud/kilosort/mds2/probe_mds2.json') #make sure that the probe kilosort uses is the json one

def gui_equivalent_run(data_file, results_dir, settings, probe, ref):
	return run_kilosort(
		settings=settings,
		probe=probe,
		filename=data_file,
		results_dir=results_dir,
		# pull the GUI_SETTINGS extras from the reference run so they match exactly:
		data_dtype=ref.get('data_dtype', 'int16'),
		do_CAR=ref.get('do_CAR', True),
		invert_sign=ref.get('invert_sign', False),
		save_preprocessed_copy=ref.get('save_preprocessed_copy', False),
	)
#defining the iterations parameter 
param = [1, 2, 4, 10, 20, 100]

#parameters for dataset
data_file = '/data/mDS2/dat/mDS2_02/mDS2_02.dat'
base_results_dir = '/data/results/mds2_res/IIR/matching_pursuit/'

# --- now loop over datasets / parameter variations ---
for x in param:
	settings['max_peels'] = x
	resdir_name = f"{x}peels"
	results_dir = Path(base_results_dir) / Path(resdir_name)
	results_dir.mkdir(parents=True, exist_ok=True)
	gui_equivalent_run(data_file, str(results_dir), settings, probe, ref)

	gc.collect()
	torch.cuda.empty_cache()
