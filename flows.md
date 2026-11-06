# flows: online template matching in C (phase1)

`flows.c` is a C version of Kilosort's template-matching step. It runs on a
recording one sample at a time. It does not learn anything: it uses the
templates and whitening that Kilosort learned from the start of the recording
(the `freeze_sec` window). It starts at the end of that window and detects
spikes from there on.

Each dataset has two versions:

```
<dataset>/phase1/origin/     the original version: the whole probe at once
<dataset>/phase1/pershank/   new version: every shank on its own, ready for a RISC-V chip
```
`<dataset>` is `ms51`, `mp79` or `mds2`.

## What the code does

For every new sample (one value per channel):

1. **Preprocess:** CAR (subtract the median across channels), then a
   3rd-order 300 Hz high-pass IIR filter, then whitening with `Wrot`. This is
   Kilosort's order, and both versions use it (so do `phase2/flows.c` and
   `mp79/phase3/flows_scaled.c`).
2. **Gate:** template matching runs only when a whitened channel has a peak
   larger than `V_TH`.
3. **Score:** the last 61 samples are projected on the PCs (`wPCA`) and
   compared with every template (`Wall`), at lags of -3 to +3 samples.
   Score = (positive part of the projection)² / `Wall_nm`.
4. **Pick:** the template with the highest score wins if:
   - its score is above Th_learned² = 64;
   - its score is above its own threshold `TH`, when TH is on;
   - it didn't fire in the last 20 samples.
5. **Output and peel:** the spike is recorded (time, template, amplitude),
   and the template is subtracted from the data, so overlapping spikes can be
   found next.

## origin vs pershank

| | origin | pershank |
|---|---|---|
| Channels | the whole probe at once | each shank separately: its own CAR, whitening, templates and peeling |
| Headers made from | whole-probe run `time/3600/peel_specified` | per-shank runs `time/module/5400/shank_<N>` |
| First sample processed | 72,000,000 (3600 s) | 108,000,000 (5400 s) |
| Memory | `malloc` for all spikes; PC only | all static, no `malloc`; also builds for bare metal (`HOST_IO=0`) |
| Output folder | `/data/results/<m>_res/flows_c/` | `/data/results/<m>_res/flows_c/pershank/` |

In the output folders, `<m>` is `mS51`, `mP79` or `mds2`. mS51 writes one
level deeper, into `no_th/`, because TH is off for it.

## Header files (what is exported from Kilosort)

`run-files/npytoC_conv.py` writes these from a Kilosort results folder. Use
`per_shank = False` for origin and `per_shank = True` for pershank.

| Header | Holds | origin shape | pershank shape |
|---|---|---|---|
| `Wall.h` | templates in PC space (masked to the 10 nearest channels) | templates × channels × 3 | all shanks' templates × `NCHAN_MAX` × 3 |
| `Wall_nm.h` | squared norm of each template | templates | all templates |
| `TH.h` | per-template threshold (5th percentile of its scores) | templates | all templates |
| `Wrot.h` | whitening matrix | channels × channels | shanks × `NCHAN_MAX` × `NCHAN_MAX` |
| `wPCA.h` | temporal PCs | 3 × 61 | shanks × 3 × 61 (each shank has its own) |
| `iir_weights.h` | high-pass filter coefficients | 7 | 7 |
| `shank_info.h` | pershank only: number of shanks, channels per shank, which templates belong to which shank, channel map | – | – |

In pershank every shank is padded with zeros to the size of the largest shank
(`NCHAN_MAX`). Shank `s` owns templates `shank_offset[s]` to
`shank_offset[s+1]-1`.

## Inputs and outputs

**Input:** the raw `.dat` file (int16, `NCHAN_BIN` channels per sample). The
first `NCHAN` channels are used.

**Output** (in the output folder):

| File | Content |
|---|---|
| `spike_times.npy` | spike peak time, absolute sample (int64) |
| `spike_templates.npy` | template id of each spike (int32); called `spike_clusters.npy` for mS51 |
| `amplitudes.npy` | spike amplitude (float64) |
| `cf_trace.txt` | debug: scores of all templates at every trigger, first 14,000 samples |
| `data_trace.txt` | debug: preprocessed signal of every channel, first 14,000 samples |

In pershank the template ids count across all shanks. The Kilosort id inside
shank `s` is `id - shank_offset[s]`.

**On a chip** (pershank built with `HOST_IO=0`) there are no files. You provide
two functions:
- `platform_read_frame()` gives one sample of `NCHAN` channels;
- `platform_emit_spike()` receives each detected spike.

## Build and run

```
# origin
cd <dataset>/phase1/origin   && gcc -O2 -o flows_online flows.c -lm && ./flows_online

# pershank, on the PC
cd <dataset>/phase1/pershank && gcc -O2 -o flows_online flows.c && ./flows_online

# pershank, for a RISC-V chip
riscv32-unknown-elf-gcc -O2 -march=rv32imf -mabi=ilp32f -DHOST_IO=0 -ffp-contract=off -c flows.c
```
The output folder must exist before you run. Pershank needs about 85–126 KB
of RAM and 67–80 KB of constant data.

## Settings per dataset

| | mS51 | mP79 | mDS2 |
|---|---|---|---|
| `NCHAN` / `NCHAN_BIN` | 126 / 135 | 124 / 136 | 62 / 70 |
| `V_TH` (gate) | 3.5 | 3.7 | 3.5 |
| TH check | off | off | on |
| Shanks (pershank) | 8 | 12 | 4 |

## Other files in `origin/`

- `flows_online`: the built program.
- `process_log.txt`: console output of the last full run.
- `probe_*`: probe files.
- `flows_fir.c`, `fir_weights.h`, `fir_filter_taps.h`: an older version with
  an FIR filter, kept for reference only. It doesn't build as is, and it still
  filters before CAR.
