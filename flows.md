# flows: online template matching in C

`flows.c` is a C version of Kilosort's template-matching step. It runs on a
recording one sample at a time. It does not learn anything: it uses the
templates and whitening that Kilosort learned from the start of the recording
(the `freeze_sec` window). It starts at the end of that window and detects
spikes from there on.

Each dataset has four versions:

```
<dataset>/phase1/     whole probe, floating point (the original version)
<dataset>/phase2/     whole probe, floating point, headers from the older old_clu_assignement runs
<dataset>/phase3/     whole probe, fixed point (integers only): flows_scaled.c
<dataset>/pershank/   every shank on its own, floating point, ready for a RISC-V chip
```
`<dataset>` is `ms51`, `mp79` or `mds2`. phase1–3 are the whole-probe versions.

## What the code does

For every new sample (one value per channel):

1. **Preprocess:** CAR (subtract the median across channels), then a
   3rd-order 300 Hz high-pass IIR filter, then whitening with `Wrot`. This is
   Kilosort's order, and all four versions use it.
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

## Versions

| | phase1 | phase2 | phase3 | pershank |
|---|---|---|---|---|
| Channels | whole probe | whole probe | whole probe | each shank separately: its own CAR, whitening, templates and peeling |
| Numbers | float | float | integers (fixed point) | float |
| First sample processed | 72,000,000 (3600 s) | 72,000,000 (3600 s) | 72,000,000 (3600 s) | 108,000,000 (5400 s) |
| Memory | `malloc` for all spikes; PC only | static arrays for all spikes; PC only | `malloc` for all spikes; PC only | all static, no `malloc`; also builds for bare metal (`HOST_IO=0`) |

## Header files (what is exported from Kilosort)

In every version:
- `Wall` keeps only each template's 10 nearest channels (Kilosort's
  `nearest_chans`); the other channels are zero.
- The export scripts print `Th_learned` but don't export it. The C code has
  Th_learned² written by hand (`TH_LEARNED_SQ`, 64).

### Whole probe (phase1, phase2, phase3)

All headers come from one whole-probe Kilosort run. At the top of the export
script's `main()`, set the run to read (`export_dir`) and the phase folder to
write into (`result_dir`).

- **phase1 and phase2:** `run-files/npytoC_conv.py` with `per_shank = False`.
  phase1's headers come from the `updated_clu/time/3600/peel_specified` runs,
  phase2's from the older `old_clu_assignement/time/3600` runs.
- **phase3:** `run-files/scaled/export_scaled.py`. The same arrays, stored as
  integers for fixed-point math. `Wrot`, `wPCA` and the filter coefficients are
  multiplied by 2¹⁵ (Q15) and stored as int16; the C code shifts back by 15
  bits after each multiply. The files are named with `_SCALED`: `Wall_SCALED.h`,
  `Wall_nm_SCALED.h`, `TH_SCALED.h`, `Wrot_SCALED.h`, `wPCA_SCALED.h` (the
  filter is still `iir_weights.h`). Note: `export_scaled.py` still writes a
  2nd-order filter (5 coefficients), while `flows_scaled.c` expects 7.

In C, template `t` is `Wall[t*(NCHAN*NPCS) + c*NPCS + p]`.

### Per shank (pershank)

`run-files/npytoC_conv.py` with `per_shank = True`. Reads every `shank_<N>`
folder in `flow_dir` (`time/module/5400`, one Kilosort run per shank). It
combines them into one header per array and writes into `<dataset>/pershank/`.

Shanks have different numbers of channels, so every shank is padded with zeros
to the size of the largest shank (`NCHAN_MAX`). The padded channels are zero in
`Wall` and `Wrot`, so they add nothing. Templates are stacked shank by shank:
shank `s` owns templates `shank_offset[s]` to `shank_offset[s+1]-1`.

In C, the macros `WALL(t, c, p)`, `WROT(s, w, c)` and `WPCA(s, p, k)` do the
indexing.

### All headers

| Header | Holds | Whole probe shape | Pershank shape |
|---|---|---|---|
| `Wall.h` | templates in PC space | templates × channels × 3 | all shanks' templates × `NCHAN_MAX` × 3 |
| `Wall_nm.h` | squared norm of each template | templates | all templates |
| `TH.h` | per-template threshold (5th percentile of its scores) | templates | all templates |
| `Wrot.h` | whitening matrix | channels × channels | shanks × `NCHAN_MAX` × `NCHAN_MAX` (one per shank) |
| `wPCA.h` | temporal PCs | 3 × 61 | shanks × 3 × 61 (each shank has its own) |
| `iir_weights.h` | high-pass filter coefficients | 7 | 7 |
| `shank_info.h` | number of shanks, `NCHAN_MAX` and number of templates; per shank: its Kilosort shank number, channel count, first template (`shank_offset`) and channel map | – | yes |

In phase3 the first five have `_SCALED` in their names and hold integers.

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
# phase1 or phase2
cd <dataset>/phase1   && gcc -O2 -o flows_online flows.c -lm && ./flows_online

# phase3
cd <dataset>/phase3   && gcc -O2 -o flows_scaled flows_scaled.c -lm && ./flows_scaled

# pershank, on the PC
cd <dataset>/pershank && gcc -O2 -o flows_online flows.c && ./flows_online

# pershank, for a RISC-V chip
riscv32-unknown-elf-gcc -O2 -march=rv32imf -mabi=ilp32f -DHOST_IO=0 -ffp-contract=off -c flows.c
```
The output folder must exist before you run. Pershank needs about 85–126 KB
of RAM and 67–80 KB of constant data.

## Settings per dataset

| | mS51 | mP79 | mDS2 |
|---|---|---|---|
| `NCHAN` / `NCHAN_BIN` | 126 / 135 | 124 / 136 | 62 / 70 |
| Shanks (pershank) | 8 | 12 | 4 |

## Other files

- `flows_online` / `flows_scaled`: the built programs.
- `process_log.txt`: console output of the last full run.
- `<dataset>/probe_*`: probe files.
- `phase1/flows_fir.c`, `fir_weights.h`, `fir_filter_taps.h`: an older version
  with an FIR filter, kept for reference only. It doesn't build as is, and it
  still filters before CAR.
