#___________________________________________
#example 1 from kilosort github library
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb' 
import matplotlib
from pathlib import Path
matplotlib.use("qt5agg")   # or "TkAgg"
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
import numpy as np
import pandas as pd

# outputs saved to results_dir
results_dir = Path('/home/mariandawud/kilosort/ms51/ms51_res')
ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
chan_map =  np.load(results_dir / 'channel_map.npy')
templates =  np.load(results_dir / 'templates.npy')
chan_best_idx = (templates**2).sum(axis=1).argmax(axis=-1)
chan_best = chan_map[chan_best_idx]
amplitudes = np.load(results_dir / 'amplitudes.npy')
st = np.load(results_dir / 'spike_times.npy')
clu = np.load(results_dir / 'spike_clusters.npy')
firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
dshift = ops['dshift']

#-------------------------------------------------------------

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
gray = .5 * np.ones(3)

fig = plt.figure(figsize=(10,10), dpi=100)
grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

ax = fig.add_subplot(grid[0,0])
ax.plot(np.arange(0, ops['Nbatches'])*2, dshift);
ax.set_xlabel('time (sec.)')
ax.set_ylabel('drift (um)')

ax = fig.add_subplot(grid[0,1:])
t0 = 0
t1 = np.nonzero(st > ops['fs']*5)[0][0]
ax.scatter(st[t0:t1]/30000., chan_best[clu[t0:t1]], s=0.5, color='k', alpha=0.25)
ax.set_xlim([0, 5])
ax.set_ylim([chan_map.max(), 0])
ax.set_xlabel('time (sec.)')
ax.set_ylabel('channel')
ax.set_title('spikes from units')

ax = fig.add_subplot(grid[1,0])
nb=ax.hist(firing_rates, 20, color=gray)
ax.set_xlabel('firing rate (Hz)')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1,1])
nb=ax.hist(camps, 20, color=gray)
ax.set_xlabel('amplitude')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1,2])
nb=ax.hist(np.minimum(100, contam_pct), np.arange(0,105,5), color=gray)
ax.plot([10, 10], [0, nb[0].max()], 'k--')
ax.set_xlabel('% contamination')
ax.set_ylabel('# of units')
ax.set_title('< 10% = good units')

for k in range(2):
    ax = fig.add_subplot(grid[2,k])
    is_ref = contam_pct<10.
    ax.scatter(firing_rates[~is_ref], camps[~is_ref], s=3, color='r', label='mua', alpha=0.25)
    ax.scatter(firing_rates[is_ref], camps[is_ref], s=3, color='b', label='good', alpha=0.25)
    ax.set_ylabel('amplitude (a.u.)')
    ax.set_xlabel('firing rate (Hz)')
    ax.legend()
    if k==1:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('loglog')

#plt.tight_layout()
#plt.show()
#------------------------------------------------------

probe = ops['probe']
# x and y position of probe sites
#__________________________________UPDATED______________________________________-


# --- geometry & selection ---
xc_all, yc_all = probe['xc'], probe['yc']
xc, yc = xc_all[chan_map], yc_all[chan_map]   # geometry in templates' channel order

nc = 16  # channels to show around best channel
good_units = np.nonzero(contam_pct <= 0.1)[0]
mua_units  = np.nonzero(contam_pct  > 0.1)[0]

# best channel index per unit (within templates' channel axis)
chan_best_idx = (templates**2).sum(axis=1).argmax(axis=-1).astype(int)  # (K,)

# estimate vertical pitch (µm) for scaling; fallback if degenerate
uniq_y = np.unique(yc)
pitch_y = float(np.median(np.diff(np.sort(uniq_y)))) if uniq_y.size > 1 else 20.0
if not np.isfinite(pitch_y) or pitch_y <= 0: pitch_y = 20.0

# ---------------- knobs you can tweak ----------------
WIDTH_MULT  = 2.0   # horizontal width (in pitch units) -> larger = wider waveforms
HEIGHT_MULT = 0.65  # target vertical peak-to-peak (as fraction of pitch)
PAD_MULT    = 1.0   # extra padding in relative plot frame
# -----------------------------------------------------

# fixed horizontal width in micrometers (centered around 0)
W_um = WIDTH_MULT * pitch_y
xpad = PAD_MULT * 0.2 * W_um

# fixed vertical frame based on a window of nc channels
# (we’ll compute it once using a generic centered window)
half = nc // 2
# relative y-frame: from -half*pitch to +half*pitch (with small padding)
Ymin_rel = -(half + 0.4) * pitch_y
Ymax_rel = +(half + 0.4) * pitch_y

gstr = ['good', 'mua']
for j in range(2):
    print(f'~~~~~~~~~~~~~~ {gstr[j]} units ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('title = number of spikes from each unit')
    units = good_units if j == 0 else mua_units
    if units.size == 0:
        print('(no units in this group)')
        continue

    fig = plt.figure(figsize=(12, 3), dpi=150)
    grid = gridspec.GridSpec(2, 20, figure=fig, hspace=0.25, wspace=0.2)

    for k in range(40):
        wi = int(units[np.random.randint(units.size)])
        wv = templates[wi]                     # (T, C)
        cb = int(chan_best_idx[wi])            # center index in templates
        nsp = int((clu == wi).sum())

        ax = fig.add_subplot(grid[k // 20, k % 20])

        # choose a centered channel window of exactly nc (clamped at edges)
        C = wv.shape[-1]
        start = max(0, min(cb - nc // 2, C - nc))
        stop  = start + min(nc, C - start)
        wvw = wv[:, start:stop]                # (T, nc_eff)
        x0, y0 = xc[start:stop], yc[start:stop]

        # ---- CENTER EACH TILE on the best site ----
        x_center, y_center = xc[cb], yc[cb]
        x_rel = x0 - x_center      # relative to best channel
        y_rel = y0 - y_center

        # horizontal “time” span (wider as requested)
        T = wvw.shape[0]
        t = np.linspace(-0.5 * W_um, 0.5 * W_um, T, dtype='float32')

        # robust amplitude scaling to target HEIGHT_MULT * pitch_y p2p
        p2p_ch = (np.nanmax(wvw, axis=0) - np.nanmin(wvw, axis=0))
        p2p_ref = np.nanpercentile(p2p_ch[np.isfinite(p2p_ch)], 90) if np.any(np.isfinite(p2p_ch)) else 0.0
        if p2p_ref <= 1e-9:
            amp = 0.25 * pitch_y
        else:
            amp = (HEIGHT_MULT * pitch_y) / p2p_ref

        # draw each channel waveform in the centered frame
        for xi, yi, trace in zip(x_rel, y_rel, wvw.T):
                ax.plot(xi + t, ytrace, lw=0.7, color='k')

        # ----- FIXED, SHARED FRAME (centered) -----
        ax.set_xlim(-0.5 * W_um - xpad, 0.5 * W_um + xpad)
        ax.set_ylim(Ymax_rel, Ymin_rel)   # invert Y so depth points downward
        #ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'{nsp}', fontsize='x-small')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
