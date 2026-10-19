"""Does the waveform fit in a shorter `nt`? Learned templates vs. mean waveforms in the data.

For the 'good' units of each run: peak-channel template (the 61-sample model Kilosort
matches with) and the mean waveform in the whitened, drift-corrected data over a wider
window, with the windows that nt=41 / nt=51 would keep marked. Also the fraction of
template energy that would fall outside each shorter window.

Usage: python nt_window_check.py
"""
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kilosort.run_kilosort  # determinism flags
from kilosort import io

RUNS = {
	"mS51": "/data/results/mS51_res/IIR/updated_clu/baseline/det/1",
	"mP79": "/data/results/mP79_res/IIR/updated_clu/baseline/det/1",
	"mDS2": "/data/results/mds2_res/IIR/updated_clu/baseline/det/1",
}
OUT = Path("/data/results/nt_check")
N_BATCHES = 40              # batches sampled for the data mean waveforms
WIDE = (-40, 80)            # data window in samples around the spike time (trough)

# palette (reference instance, light surface)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def window(nt, nt0min=None):
	"""Samples [start, stop) relative to the trough that Kilosort keeps for nt / nt0min.

	nt0min=None is Kilosort's automatic value, int(20 * nt / 61).
	"""
	if nt0min is None:
		nt0min = int(20 * nt / 61)
	return -nt0min, nt - nt0min


# candidate windows: (nt, nt0min, color, linestyle, label)
CANDIDATES = [
	(41, None, ORANGE, "-", "nt=41 (auto nt0min=13)"),
	(51, 10, AQUA, "--", "nt=51, nt0min=10"),
]
TABLE_ONLY = [(51, None), (45, 8), (45, None)]
YLIM = (-1.4, 1.6)


def load_labels(d):
	with open(d / "cluster_KSLabel.tsv") as f:
		next(f)
		return {int(a): b for a, b in (l.split() for l in f)}


def data_waveforms(d, ops, units, peak_ch):
	"""Mean whitened waveform on each unit's peak channel over WIDE, from N_BATCHES batches."""
	dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ops_t = io.load_ops(d / "ops.npy", device=dev)
	bfile = io.bfile_from_ops(ops_t, device=dev)
	nt, NT = ops["nt"], ops["batch_size"]
	st = np.load(d / "spike_times.npy").ravel().astype(np.int64)
	clu = np.load(d / "spike_clusters.npy").ravel()
	keep = np.isin(clu, units)
	st, clu = st[keep], clu[keep]
	row = {u: i for i, u in enumerate(units)}
	acc = np.zeros((len(units), WIDE[1] - WIDE[0]))
	cnt = np.zeros(len(units))
	offs = np.arange(*WIDE)
	for ib in np.linspace(1, bfile.n_batches - 2, N_BATCHES).astype(int):
		X = bfile.padded_batch_to_torch(int(ib), ops_t).cpu().numpy()
		m = (st >= ib * NT) & (st < (ib + 1) * NT)
		col = st[m] - ib * NT + nt
		ok = (col + WIDE[0] >= 0) & (col + WIDE[1] <= X.shape[1])
		for c, u in zip(col[ok], clu[m][ok]):
			i = row[u]
			acc[i] += X[peak_ch[i], c + offs]
			cnt[i] += 1
	return acc / np.maximum(cnt, 1)[:, None], cnt


def normalize(w, zero, half=5):
	"""Scale so the largest deflection within +-half samples of `zero` is -1."""
	seg = w[:, zero - half:zero + half + 1]
	pk = seg[np.arange(len(w)), np.abs(seg).argmax(1)]
	return w / -pk[:, None]


def style(ax):
	ax.grid(True, color=GRID, lw=0.6)
	ax.set_axisbelow(True)
	for s in ("top", "right"):
		ax.spines[s].set_visible(False)
	for s in ("left", "bottom"):
		ax.spines[s].set_color(MUTED)
	ax.tick_params(colors=INK2, labelsize=8)


def mark_windows(ax, fs):
	for nt, nt0, c, ls, _ in [(61, 20, INK2, ":", "")] + CANDIDATES:
		a, b = window(nt, nt0)
		for x in (a, b - 1):
			ax.axvline(1000 * x / fs, color=c, ls=ls, lw=1.2, zorder=1)


def main():
	OUT.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(3, len(RUNS), figsize=(5 * len(RUNS), 10.5), squeeze=False)
	lines = []
	for j, (name, path) in enumerate(RUNS.items()):
		d = Path(path)
		ops = np.load(d / "ops.npy", allow_pickle=True).item()
		fs, nt, nt0min = ops["fs"], ops["nt"], ops["nt0min"]
		lab = load_labels(d)
		T = np.load(d / "templates.npy")                     # (units, nt, chans), whitened
		units = np.array(sorted(u for u, l in lab.items() if l == "good" and u < len(T)))
		Tg = T[units]
		energy_ch = (Tg ** 2).sum(1)
		peak_ch = energy_ch.argmax(1)
		trough_idx = np.abs(Tg[np.arange(len(units)), :, peak_ch]).argmax(1)

		# 1) learned templates, peak channel
		tw = normalize(Tg[np.arange(len(units)), :, peak_ch], nt0min)
		t = 1000 * (np.arange(nt) - nt0min) / fs
		ax = axes[0, j]
		style(ax)
		ax.plot(t, tw.T, color=MUTED, lw=0.6, alpha=0.35)
		ax.plot(t, np.median(tw, 0), color=BLUE, lw=2, label="median")
		mark_windows(ax, fs)
		ax.set_title(f"{name}: learned templates ({len(units)} good units)", color=INK, fontsize=10)
		ax.set_xlim(1000 * WIDE[0] / fs, 1000 * WIDE[1] / fs)
		ax.set_ylim(*YLIM)

		# 2) mean waveform in the data, wider window
		W, cnt = data_waveforms(d, ops, units, peak_ch)
		wd = normalize(W[cnt >= 20], -WIDE[0])
		td = 1000 * np.arange(*WIDE) / fs
		ax = axes[1, j]
		style(ax)
		ax.plot(td, wd.T, color=MUTED, lw=0.6, alpha=0.35)
		ax.plot(td, np.median(wd, 0), color=BLUE, lw=2)
		mark_windows(ax, fs)
		ax.set_title(f"{name}: mean waveform in data ({len(wd)} units, >=20 spikes)",
					 color=INK, fontsize=10)
		ax.set_xlim(td[0], td[-1])
		ax.set_ylim(*YLIM)
		n_out = int((np.abs(wd) > YLIM[1]).any(1).sum())
		if n_out:
			ax.text(0.99, 0.02, f"{n_out} unit(s) exceed the y-range (clipped)", transform=ax.transAxes,
					ha="right", va="bottom", fontsize=7, color=INK2)

		# 3) template energy (all channels) that a shorter window would drop
		ax = axes[2, j]
		style(ax)
		E = (Tg ** 2).sum((1, 2))
		bins = np.linspace(0, 30, 31)
		print(f"\n{name}: {len(units)} good units, fs={fs:.0f}, nt={nt}, nt0min={nt0min}; "
			  f"template peak at index {np.median(trough_idx):.0f} "
			  f"(IQR {np.percentile(trough_idx, 25):.0f}-{np.percentile(trough_idx, 75):.0f})")
		def lost_pct(ntn, nt0):
			a, b = window(ntn, nt0)
			inside = (Tg[:, nt0min + a:nt0min + b] ** 2).sum((1, 2))
			return a, b, 100 * (1 - inside / E)
		for ntn, nt0, c, ls, label in CANDIDATES:
			a, b, lost = lost_pct(ntn, nt0)
			ax.hist(lost, bins, histtype="step", lw=1.8, color=c, ls=ls, label=label)
		for ntn, nt0 in [(c[0], c[1]) for c in CANDIDATES] + TABLE_ONLY:
			a, b, lost = lost_pct(ntn, nt0)
			print(f"   nt={ntn:<3} nt0min={-a:<3} keeps {1000*a/fs:+.2f} to {1000*(b-1)/fs:+.2f} ms: "
				  f"template energy lost median {np.median(lost):4.1f}%, 90th pct {np.percentile(lost, 90):4.1f}%, "
				  f"units losing >5%: {(lost > 5).sum()}/{len(lost)}")
		# data: energy of the mean waveform between the nt=41 edge and the current edge, per side
		i0 = -WIDE[0]
		a41, b41 = window(41)
		a61, b61 = window(61)
		wtot = (wd[:, i0 + a61:i0 + b61] ** 2).sum(1)
		pre = (wd[:, i0 + a61:i0 + a41] ** 2).sum(1) / wtot
		post = (wd[:, i0 + b41:i0 + b61] ** 2).sum(1) / wtot
		beyond = (wd[:, i0 + b61:] ** 2).sum(1) / wtot
		print(f"   data mean waveform, share of energy (relative to current window): "
			  f"before -0.65 ms {100*np.median(pre):.2f}%, after +1.35 ms {100*np.median(post):.1f}%, "
			  f"beyond current +2.0 ms {100*np.median(beyond):.1f}% (medians)")
		ax.set_title(f"{name}: template energy lost by a shorter nt", color=INK, fontsize=10)
		ax.set_xlabel("energy outside window (% of template)", color=INK2, fontsize=9)
		ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

	for i, yl in enumerate(["normalized amplitude", "normalized amplitude", "units"]):
		axes[i, 0].set_ylabel(yl, color=INK2, fontsize=9)
	for ax in axes[1]:
		ax.set_xlabel("time from trough (ms)", color=INK2, fontsize=9)
	for ax in axes[0]:
		ax.set_xlabel("time from trough (ms)", color=INK2, fontsize=9)
	handles = [plt.Line2D([], [], color=BLUE, lw=2, label="median unit"),
			   plt.Line2D([], [], color=INK2, ls=":", lw=1.2, label="nt=61 window (current)"),
			   ] + [plt.Line2D([], [], color=c, ls=ls, lw=1.2, label=label) for _, _, c, ls, label in CANDIDATES]
	fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9,
			   labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
	fig.tight_layout(rect=(0, 0, 1, 0.97))
	fig.patch.set_facecolor("#fcfcfb")
	out = OUT / "nt_window_check.png"
	fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
	print(f"\nsaved {out}")


if __name__ == "__main__":
	main()
