import numpy as np
import gc, torch
from pathlib import Path
from kilosort import run_kilosort
from kilosort.io import load_probe
from kilosort.parameters import DEFAULT_SETTINGS


class data:
    def __init__(self, BIN, PROBE, REF, RES_BASE):
        self.BIN = BIN
        self.PROBE=PROBE
        self.REF=REF
        self.RES_BASE=RES_BASE

# if your version doesn't nest 'settings', copy the DEFAULT_SETTINGS keys out of ops directly.

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
param = [1, 2, 4, 10, 20, 50, 70, 100]

#parameters for dataset
ms51 = data("/data/mS51/dat/mS51_49/mS51_49.dat", \
        "/home/mariandawud/kilosort/ms51/probe_ms51_new.json", \
		'/data/results/mS51_res/IIR/ms51-det-3pc-2peels-IIR-28.6/ops.npy', \
		'/data/results/mS51_res/IIR/matching_pursuit/')

mp79 = data("/data/mP79/dat/mP79_17/mP79_17.dat", \
	"/home/mariandawud/kilosort/mp79/probe_mp79.json", \
	'/data/results/mP79_res/IIR/mp79-det-3pc-2peels-IIR-28.6/ops.npy', \
	'/data/results/mP79_res/IIR/matching_pursuit/')

mds2 = data("/data/mDS2/dat/mDS2_02/mDS2_02.dat", \
	"/home/mariandawud/kilosort/mds2/probe_mds2.json", \
	'/data/results/mds2_res/IIR/mds2-det-3pc-2peels-IIR-29.6/ops.npy', \
	'/data/results/mds2_res/IIR/matching_pursuit/')

data_l = [ms51, mp79, mds2]

# --- now loop over datasets / parameter variations ---
for d in data_l:
	# --- recover the exact settings from a reference GUI run ---
	ref = np.load(d.REF, allow_pickle=True).item()

	# ops holds every merged setting; pull the ones run_kilosort takes as `settings`
	# (n_chan_bin, fs, Th_universal, Th_learned, batch_size, nt, dmin, dminx, ... )
	settings = {k: ref[k] for k in DEFAULT_SETTINGS if k in ref}

	#make sure that the probe kilosort uses is the json one
	probe = load_probe(d.PROBE)

	for x in param:
		settings['max_peels'] = x
		resdir_name = f"{x}peels"
		results_dir = Path(d.RES_BASE) / resdir_name
		results_dir.mkdir(parents=True, exist_ok=True)
		gui_equivalent_run(d.BIN, str(results_dir), settings, probe, ref)

		gc.collect()
		torch.cuda.empty_cache()
