import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_cf_by_sample(cf_lines, nclu):
	# cf_trace.txt is written in fixed nclu-line blocks ("%u, %f\n" = sample, cf),
	# but only for samples where confirm_peak_evt fired -- NOT one block per
	# sample index. So key each block by its own leading sample number instead
	# of assuming a fixed (time-offset)*nclu stride.
	cf_by_sample = {}
	for k in range(0, len(cf_lines) - nclu + 1, nclu):
		block = cf_lines[k:k+nclu]
		samp = int(block[0].strip().split(",")[0])
		vals = [float(l.strip().split(",")[1]) for l in block]
		cf_by_sample[samp] = vals
	return cf_by_sample


def plot_batch(data, cf_by_sample, ch, nchan):
	#iterates over each channel and plots its signal as a function of time
	data.seek(0)
	times, signal = [], []
	i=0
	for l in data:
		if i == ch:
			t, d = l.strip().split(",")
			times.append(int(t))
			signal.append(float(d))
		i+=1
		if i == nchan:
			i = 0
	z=1
	start=0
	end=-1
	while z:
		inter = input("Enter range of interest in format s1,s2 (absolute):")
		start, end = (int(x) for x in inter.strip().split(","))  # split on comma
		fig, (ax_main, ax_sub) = plt.subplots(
				1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]}
			)
		ax_main.plot(times[start:end], signal[start:end])
		ax_main.set_title(f"Signal on channel- {ch}")
		ax_main.set_xlabel("Time (samples)")
		ax_main.set_ylabel("Amplitude (whitened)")

		fig.canvas.mpl_connect(
			"button_press_event",
			lambda event: on_click(event, cf_by_sample, ax_sub),
		)
		plt.show()
		z = int(input("If zoom is required press 1 else 0:"))   # cast to int
	return times, signal


def on_click(event, cf_by_sample, ax_sub):
	if event.inaxes is None or event.xdata is None:
		return
	time = int(round(event.xdata))
	vals = cf_by_sample.get(time)
	if vals is None:
		ax_sub.clear()
		ax_sub.set_title("no cf dumped here")
		plt.draw()
		return
	templates = list(range(len(vals)))   # template id = position in block
	ax_sub.clear()
	ax_sub.bar(templates, vals)
	ax_sub.set_title(f"cf at sample {time}")
	ax_sub.set_xlabel("Template")
	ax_sub.set_ylabel("cf")
	plt.draw()

def zoom(times, signal):
	#zoom in on relevant areas if required
	z = int(input("If zoom is required press 1 else 0:"))   # cast to int
	while z:                                                 # loop on the int
		inter = input("Enter range of interest in format s1,s2 (absolute):")
		start, end = (int(x) for x in inter.strip().split(","))  # split on comma
		plt.figure(figsize=(9, 8))
		plt.plot(times[start:end], signal[start:end])
		plt.show()
		z = int(input("Zoom again? 1/0:"))

def main():
	"""
	plots filtered binary data in C and returns the point where spike is detected
	"""
	ap = argparse.ArgumentParser()
	ap.add_argument("--k_path", required=True, type=str)
	ap.add_argument("--c_path", required=True, type=str)
	ap.add_argument("--nchan", required=True, type=str)
	args = ap.parse_args()
	#read dump files
	c_path = Path(args.c_path)
	k_path = Path(args.k_path)
	nchan = int(args.nchan)
	nclu = len(np.load(k_path / "Wall.npy"))
	cf = open(c_path / "cf_trace.txt", "r")
	data = open(c_path / "data_trace.txt", "r")
	cf_lines = cf.readlines()
	cf_by_sample = load_cf_by_sample(cf_lines, nclu)

	#plot the whole batch
	channel = int(input("If you know which channel to display press its number else press -1 to iterate:"))
	if channel == -1:
		for ch in range(nchan):          # use nchan, not the unloaded Wall
			times, signal = plot_batch(data, cf_by_sample, ch, nchan)
	else:
		times, signal = plot_batch(data, cf_by_sample, channel, nchan)

if __name__ == "__main__":               # so main() actually runs
	main()
