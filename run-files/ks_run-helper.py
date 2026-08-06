import numpy as np
import gc, torch, traceback
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

#parameters for dataset
ms51 = data("/data/mS51/dat/mS51_49/mS51_49.dat", \
        "/home/mariandawud/kilosort/ms51/probe_ms51.json", \
		'/data/results/mS51_res/IIR/ms51-det-3pc-2peels-IIR-28.6/ops.npy', \
		'/data/results/mS51_res/IIR/time/')

mp79 = data("/data/mP79/dat/mP79_17/mP79_17.dat", \
	"/home/mariandawud/kilosort/mp79/probe_mp79.json", \
	'/data/results/mP79_res/IIR/mp79-det-3pc-2peels-IIR-28.6/ops.npy', \
	'/data/results/mP79_res/IIR/time/')

mds2 = data("/data/mDS2/dat/mDS2_02/mDS2_02.dat", \
	"/home/mariandawud/kilosort/mds2/probe_mds2.json", \
	'/data/results/mds2_res/IIR/mds2-det-3pc-2peels-IIR-29.6/ops.npy', \
	'/data/results/mds2_res/IIR/time/')

data_l = [ms51, mp79, mds2]

#defining the iterations parameter 
param = [300, 900, 1800, 2400, 3600, 5400, 10800, 12600, 13800, 14400, 20400]
failed_runs = []
for p in param:
	if p != 3600:
		continue
	for d in data_l:
		if d != mp79:
			continue
		
		# --- now loop over datasets / parameter variations ---
		# --- recover the exact settings from a reference GUI run ---
		ref = np.load(d.REF, allow_pickle=True).item()

		# ops holds every merged setting; pull the ones run_kilosort takes as `settings`
		# (n_chan_bin, fs, Th_universal, Th_learned, batch_size, nt, dmin, dminx, ... )
		settings = {k: ref[k] for k in DEFAULT_SETTINGS if k in ref}

		settings['nblocks'] = 0 # for no drift correction 
		settings['freeze_sec'] = p
		
		#make sure that the probe kilosort uses is the json one
		probe = ref['probe']	#load_probe(d.PROBE)
		settings['max_peels'] = 1

		resdir_name = str(p)
		results_dir = Path(d.RES_BASE) / resdir_name / "1peel" / "2ndOrd-filt"
		results_dir.mkdir(parents=True, exist_ok=True)

		try:
			gui_equivalent_run(d.BIN, str(results_dir), settings, probe, ref)
		except Exception:
			print(f'[FAILED] freeze_sec= {p} s, dataset={d.BIN}')
			traceback.print_exc()
			#failed_runs.append((1, d.BIN))
		finally:
			gc.collect()
			torch.cuda.empty_cache()

print()
if failed_runs:
	print(f'{len(failed_runs)} run(s) failed:')
	for p, bin_path in failed_runs:
		print(f'  freeze_sec= {p}  dataset={bin_path}')
else:
	print('All runs completed successfully.')