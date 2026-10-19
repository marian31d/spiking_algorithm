"""ISI check for the max_peels sweep.

For every dataset/shank, match units between a low-peel and a high-peel run by
spike-time coincidence, keep the units that are 'good' at low peels but 'mua'
at high peels, and plot their ISI histograms. A peak at ~0.3-2 ms that appears
only in the high-peel run points to matching-pursuit re-detection of residuals.

Usage:
    python peel_isi_check.py                 # lowest vs highest peel found
    python peel_isi_check.py --low 1 --high 100
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASES = {
	"ms51": "/data/results/mS51_res/IIR/updated_clu/matching_pursuit/",
	"mp79": "/data/results/mP79_res/IIR/updated_clu/matching_pursuit/",
	"mds2": "/data/results/mds2_res/IIR/updated_clu/matching_pursuit/",
}
OUT = Path("/data/results/peel_isi_check")

MATCH_TOL_MS = 0.5      # spikes within this are "the same spike"
MIN_OVERLAP = 0.5       # fraction of low-unit spikes that must land in the matched unit
ISI_MAX_MS = 10.0
ISI_BIN_MS = 0.1
BAND_MS = (0.3, 2.0)    # re-detection band
MAX_PER_FIG = 24


def find_runs(base):
	"""{group: {peel: run_dir}} for every completed run anywhere under base.

	The peel number is the closest ancestor directory with an integer name, e.g.
	base/pershank/100/shank_5/ops.npy -> peel 100, group "pershank/shank_5".
	Runs are only compared within the same group.
	"""
	base = Path(base)
	runs = {}
	for ops in sorted(base.rglob("ops.npy")):
		run_dir = ops.parent
		peel_dir = next((d for d in [run_dir, *run_dir.parents]
						 if d != base and base in d.parents and d.name.isdigit()), None)
		if peel_dir is None:
			continue
		prefix = peel_dir.parent.relative_to(base)
		suffix = run_dir.relative_to(peel_dir)
		group = "/".join(s for s in (str(prefix), str(suffix)) if s != ".") or "."
		runs.setdefault(group, {})[int(peel_dir.name)] = run_dir
	return runs


def load_run(d):
	st = np.load(d / "spike_times.npy").astype(np.int64).ravel()
	clu = np.load(d / "spike_clusters.npy").astype(np.int64).ravel()
	labels = {}
	with open(d / "cluster_KSLabel.tsv") as f:
		next(f)
		for line in f:
			cid, lab = line.split()
			labels[int(cid)] = lab
	fs = 30000.0
	for line in open(d / "params.py"):
		if line.startswith("sample_rate"):
			fs = float(line.split("=")[1])
	order = np.argsort(st, kind="stable")
	return st[order], clu[order], labels, fs


def match_unit(t_low, st_high, clu_high, tol):
	"""Best-matching high-run cluster for spike times t_low, and overlap fraction."""
	lo = np.searchsorted(st_high, t_low - tol, side="left")
	hi = np.searchsorted(st_high, t_low + tol, side="right")
	has = hi > lo
	if not has.any():
		return None, 0.0
	# nearest high spike in the window
	cand = []
	for a, b, t in zip(lo[has], hi[has], t_low[has]):
		j = a + np.argmin(np.abs(st_high[a:b] - t))
		cand.append(clu_high[j])
	ids, n = np.unique(cand, return_counts=True)
	k = np.argmax(n)
	return int(ids[k]), n[k] / len(t_low)


def isi_ms(t, fs):
	return np.diff(np.sort(t)) / fs * 1000.0


def band_frac(isi):
	return np.mean((isi >= BAND_MS[0]) & (isi < BAND_MS[1])) if len(isi) else np.nan


def plot_units(rows, title, path):
	bins = np.arange(0, ISI_MAX_MS + ISI_BIN_MS, ISI_BIN_MS)
	n = len(rows)
	ncol = 4
	nrow = int(np.ceil(n / ncol))
	fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False)
	for ax, r in zip(axes.flat, rows):
		ax.axvspan(*BAND_MS, color="0.9", zorder=0)
		ax.hist(r["isi_low"], bins, histtype="step", lw=1.2, color="C0",
				label=f"p={r['p_low']} ({r['n_low']} sp)")
		ax.hist(r["isi_high"], bins, histtype="step", lw=1.2, color="C3",
				label=f"p={r['p_high']} ({r['n_high']} sp)")
		ax.set_title(f"low {r['u_low']} -> high {r['u_high']}  "
					 f"band {r['band_low']:.3f}->{r['band_high']:.3f}", fontsize=8)
		ax.set_xlim(0, ISI_MAX_MS)
		ax.tick_params(labelsize=7)
		ax.legend(fontsize=6, frameon=False)
	for ax in axes.flat[n:]:
		ax.axis("off")
	for ax in axes[-1]:
		ax.set_xlabel("ISI (ms)", fontsize=8)
	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(path, dpi=120)
	plt.close(fig)


def plot_pooled(rows, title, path):
	bins = np.arange(0, ISI_MAX_MS + ISI_BIN_MS, ISI_BIN_MS)
	lo = np.concatenate([r["isi_low"] for r in rows])
	hi = np.concatenate([r["isi_high"] for r in rows])
	fig, ax = plt.subplots(figsize=(6, 3.5))
	ax.axvspan(*BAND_MS, color="0.9", zorder=0)
	# normalise by ISIs in the whole 0-ISI_MAX window so the two runs are comparable
	ax.hist(lo, bins, histtype="step", lw=1.5, color="C0", density=True,
			label=f"p={rows[0]['p_low']}")
	ax.hist(hi, bins, histtype="step", lw=1.5, color="C3", density=True,
			label=f"p={rows[0]['p_high']}")
	ax.set_xlim(0, ISI_MAX_MS)
	ax.set_xlabel("ISI (ms)")
	ax.set_ylabel("density (ISIs < 10 ms)")
	ax.set_title(title, fontsize=9)
	ax.legend(frameon=False)
	fig.tight_layout()
	fig.savefig(path, dpi=120)
	plt.close(fig)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--low", type=int, default=None)
	ap.add_argument("--high", type=int, default=None)
	args = ap.parse_args()
	OUT.mkdir(parents=True, exist_ok=True)

	summary = []
	for name, base in BASES.items():
		runs = find_runs(base)
		if not runs:
			print(f"[{name}] no runs with an integer peel directory under {base}")
		for shank, by_p in runs.items():
			peels = sorted(by_p)
			if len(peels) < 2:
				print(f"[{name} {shank}] only peels {peels}, need two runs; skipping")
				continue
			p_low = args.low if args.low is not None else peels[0]
			p_high = args.high if args.high is not None else peels[-1]
			if p_low not in by_p or p_high not in by_p:
				print(f"[{name} {shank}] missing p={p_low} or p={p_high} (have {peels})")
				continue

			st_l, clu_l, lab_l, fs = load_run(by_p[p_low])
			st_h, clu_h, lab_h, _ = load_run(by_p[p_high])
			tol = int(round(MATCH_TOL_MS * fs / 1000))

			good_low = [c for c, l in lab_l.items() if l == "good"]
			n_good_high = sum(l == "good" for l in lab_h.values())
			rows, n_unmatched, n_stay_good = [], 0, 0
			for u in good_low:
				t = st_l[clu_l == u]
				if len(t) < 10:
					continue
				v, frac = match_unit(t, st_h, clu_h, tol)
				if v is None or frac < MIN_OVERLAP:
					n_unmatched += 1
					continue
				if lab_h.get(v) == "good":
					n_stay_good += 1
					continue
				th = st_h[clu_h == v]
				il, ih = isi_ms(t, fs), isi_ms(th, fs)
				rows.append(dict(u_low=u, u_high=v, p_low=p_low, p_high=p_high,
								 n_low=len(t), n_high=len(th), overlap=frac,
								 isi_low=il, isi_high=ih,
								 band_low=band_frac(il), band_high=band_frac(ih)))

			print(f"[{name} {shank}] p={p_low}: {len(good_low)} good | p={p_high}: "
				  f"{n_good_high} good | good->mua {len(rows)}, stayed good "
				  f"{n_stay_good}, unmatched {n_unmatched}")
			if not rows:
				continue

			tag = f"{name}_{shank.replace('/', '_')}_p{p_low}_vs_p{p_high}"
			rows.sort(key=lambda r: r["band_high"] - r["band_low"], reverse=True)
			for i in range(0, len(rows), MAX_PER_FIG):
				plot_units(rows[i:i + MAX_PER_FIG],
						   f"{name} {shank}: good at p={p_low}, mua at p={p_high} "
						   f"(grey = {BAND_MS[0]}-{BAND_MS[1]} ms)",
						   OUT / f"{tag}_units_{i // MAX_PER_FIG}.png")
			plot_pooled(rows, f"{name} {shank}: pooled over {len(rows)} good->mua units",
						OUT / f"{tag}_pooled.png")

			for r in rows:
				summary.append((name, shank, p_low, p_high, r["u_low"], r["u_high"],
								r["n_low"], r["n_high"], r["overlap"],
								r["band_low"], r["band_high"]))

	if summary:
		hdr = ("dataset,shank,p_low,p_high,unit_low,unit_high,n_spk_low,n_spk_high,"
			   "overlap,band_frac_low,band_frac_high")
		np.savetxt(OUT / "summary.csv", np.array(summary, dtype=object), fmt="%s",
				   delimiter=",", header=hdr, comments="")
		bl = np.array([s[-2] for s in summary], float)
		bh = np.array([s[-1] for s in summary], float)
		print(f"\n{len(summary)} good->mua units. Fraction of ISIs in "
			  f"{BAND_MS[0]}-{BAND_MS[1]} ms, median: low {np.nanmedian(bl):.4f} "
			  f"-> high {np.nanmedian(bh):.4f}; increased in "
			  f"{np.mean(bh > bl) * 100:.0f}% of units")
	print(f"output: {OUT}")


if __name__ == "__main__":
	main()
