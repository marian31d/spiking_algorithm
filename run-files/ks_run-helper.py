import multiprocessing as mp
import numpy as np, gc, torch, time
from pathlib import Path
from kilosort import run_kilosort
from kilosort.io import load_probe
from kilosort.parameters import DEFAULT_SETTINGS


class data:
	def __init__(self, BIN, PROBE, REF, RES_BASE, PARAMS, SHANKIDX):
		self.BIN = BIN
		self.PROBE=PROBE
		self.REF=REF
		self.RES_BASE=RES_BASE
		self.PARAMS = PARAMS
		self.SHANKIDX = SHANKIDX

#parameters for dataset
ms51 = data("/data/mS51/dat/mS51_49/mS51_49.dat", \
		"/home/mariandawud/kilosort/ms51/probe_ms51.json", \
		'/data/results/mS51_res/IIR/ms51-det-3pc-IIR-gui-9.9/ops.npy', \
		'/data/results/mS51_res/IIR/matching_pursuit', \
		[300, 900, 1800, 2400, 3600, 5400, 10800, 14400, 20400],
		[5])

mp79 = data("/data/mP79/dat/mP79_17/mP79_17.dat", \
		"/home/mariandawud/kilosort/mp79/probe_mp79.json", \
		'/data/results/mP79_res/IIR/mp79-det-3pc-IIR-gui-9.9/ops.npy', \
		'/data/results/mP79_res/IIR/matching_pursuit', \
		[300, 900, 1800, 2400, 3600, 5400, 10800, 12600],
		[1])

mds2 = data("/data/mDS2/dat/mDS2_02/mDS2_02.dat", \
		"/home/mariandawud/kilosort/mds2/probe_mds2.json", \
		'/data/results/mds2_res/IIR/mds2-det-3pc-IIR-gui-9.9/ops.npy', \
		'/data/results/mds2_res/IIR/matching_pursuit', \
		[300, 900, 1800, 3600, 2400, 5400, 10800, 13800],
		[3])

data_l = [ms51, mp79, mds2]

#defining the iterations parameter 
peels = [1, 2, 4, 7, 10, 20, 50, 70, 90, 100]


def one_run(d, p):
	# --- recover the exact settings from a reference GUI run ---
	ref = np.load(d.REF, allow_pickle=True).item()

	# ops holds every merged setting; pull the ones run_kilosort takes as `settings`
	settings = {k: ref[k] for k in DEFAULT_SETTINGS if k in ref}

	#settings['nblocks'] = 0 # for no drift correction 
	#settings['freeze_sec'] = p
	settings['max_peels'] = p
	#settings['probe_path'] = d.PROBE
	probe = ref['probe']

	results_dir = Path(d.RES_BASE) / "pershank-2ordf" / f"{p}" #/ "2ndOrd-filt"
	if (results_dir / "ops.npy").exists():
		print(f"[skip] {results_dir}", flush=True); return
	results_dir.mkdir(parents=True, exist_ok=True)

	run_kilosort(settings=settings, probe=probe, filename=d.BIN,
					results_dir=str(results_dir),
					data_dtype=ref.get('data_dtype', 'int16'),
					do_CAR=ref.get('do_CAR', True),
					invert_sign=ref.get('invert_sign', False),
					save_preprocessed_copy=ref.get('save_preprocessed_copy', False), shank_idx=d.SHANKIDX)

l=[1]
if __name__ == "__main__":
	ctx = mp.get_context("spawn")     # fresh interpreter + CUDA context per job
	failed = []
	for d in data_l:
		for p in peels:
			#print(f"=== freeze-sec={p} dataset={d.BIN} ===", flush=True)
			proc = ctx.Process(target=one_run, args=(d, p))
			proc.start()
			proc.join()
			if proc.exitcode != 0:
				print(f"[FAILED] freeze-sec={p} dataset={d.BIN} (exit {proc.exitcode})", flush=True)
				failed.append((p, d.BIN))
			time.sleep(3600)   # optional cooldown, between guaranteed-fresh contexts
	print("\nfailed:", failed if failed else "none")

