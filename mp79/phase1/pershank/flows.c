// Per-shank online template matching (Kilosort-compatible): every shank is preprocessed,
// gated, scored and peeled on its own, with the per-shank headers from npytoC_conv.py
// (per_shank=True).
//
// All memory is static (no malloc). The core (everything above the I/O section at the
// end) uses only <stdint.h>: no files, no printf, no libm, single-precision floats only.
//
// Linux host (reads the .dat file, writes .npy files to RESULTS_DIR):
//     gcc -O2 -o flows_online flows.c
// Bare-metal RISC-V target (no stdio; provide platform_read_frame / platform_emit_spike):
//     riscv32-unknown-elf-gcc -O2 -march=rv32imf -mabi=ilp32f -DHOST_IO=0 -c flows.c
// The target needs memset/memcpy (any bare-metal C runtime has them), because the
// compiler may turn the reset loops into those calls.

#include <stdint.h>

#ifndef HOST_IO
#define HOST_IO 1      // 1: Linux host, .dat input and .npy output; 0: bare-metal target, no stdio
#endif

// Configuration constants (all can be overridden with -D)
#ifndef START_SAMPLE
#define START_SAMPLE 108000000u   // 5400 s * 20 kHz: first sample after the freeze_sec learning window
#endif
#ifndef USE_TH
#define USE_TH 0                  // 1: also require cf > TH[t] (per-template threshold, TH.h)
#endif

// Include array hearder files for configuration
// Per-shank headers: every shank was sorted on its own, templates are stacked shank by
// shank and every shank is zero-padded to NCHAN_MAX channels.
#include "shank_info.h"   // N_SHANKS, NCHAN_MAX, NCLU_ALL, shank_nchan, shank_offset, shank_chanMap
#include "Wall.h"         // [NCLU_ALL][NCHAN_MAX][NPCS]
#include "Wall_nm.h"      // [NCLU_ALL]
#include "Wrot.h"         // [N_SHANKS][NCHAN_MAX][NCHAN_MAX]
#include "iir_weights.h"
#include "wPCA.h"         // [N_SHANKS][NPCS][NT]
#if USE_TH
#include "TH.h"           // [NCLU_ALL]
#endif

#define RELU(x) ((x) > 0.0f ? (x) : 0.0f)
#define ABSF(x) ((x) < 0.0f ? -(x) : (x))   // fabsf without libm

#define NCLU NCLU_ALL   //Number of clusters over all shanks (shank_info.h)
#define NCHAN 124       //channels per time step used from the recording (the first NCHAN of the .dat)
#define NCHAN_BIN 136   //channels per time step in the .dat file
#define NCH_PAD (N_SHANKS * NCHAN_MAX)   //ring-buffer width: shank s owns columns s*NCHAN_MAX .. s*NCHAN_MAX + NCHAN_MAX-1
#define TH_LEARNED_SQ 64.00f   //Th_learned^2 (single precision: no double math on the target)
//Constants
#define NT 61
#define NPCS 3
#define FS 20000
#define V_TH 3.7f                // magnitude trigger on whitened data — TUNE
//Algorithm specific constants
#define LAG 3
#define NT_TRIG 61               // 20 pre-peak + peak + 2  (peak at nt0min=20)
#define ENABLE_PEEL_SUBTRACT 1
#define RB 128       //ring buffer length prev 64
#define RING_MASK (RB - 1)
#define PEAK_OFFSET 20           // nt0min: peak sits 20 samples into the template
#define REFRACTORY_SAMPLES 20
#define WIN_END_OFFSET (NT_TRIG - 1 - PEAK_OFFSET)   // 61-1-20 = 40
#define CC 16   // cyclic cache of cf-arrays keyed by absolute window-end time
#define CC_MASK (CC - 1)

//DEBUG (host only): traces of the first CF_DUMP_LEN samples
#define CF_DUMP_N START_SAMPLE
#define CF_DUMP_LEN 14000u

// Per-shank header indexing: t = global template (shank_offset[s] + local id),
// s = shank, c/w = channel inside the shank, p = PC, k = time sample
#define WALL(t, c, p)  Wall[(t) * (NCHAN_MAX * NPCS) + (c) * NPCS + (p)]
#define WROT(s, w, c)  Wrot[(s) * (NCHAN_MAX * NCHAN_MAX) + (w) * NCHAN_MAX + (c)]
#define WPCA(s, p, k)  wPCA[(s) * (NPCS * NT) + (p) * NT + (k)]

// the headers must match the layout in shank_info.h
_Static_assert(sizeof(Wall)    / sizeof(Wall[0])    == NCLU_ALL * NCHAN_MAX * NPCS,      "Wall.h does not match shank_info.h");
_Static_assert(sizeof(Wall_nm) / sizeof(Wall_nm[0]) == NCLU_ALL,                         "Wall_nm.h does not match shank_info.h");
_Static_assert(sizeof(Wrot)    / sizeof(Wrot[0])    == N_SHANKS * NCHAN_MAX * NCHAN_MAX, "Wrot.h does not match shank_info.h");
_Static_assert(sizeof(wPCA)    / sizeof(wPCA[0])    == N_SHANKS * NPCS * NT,             "wPCA.h does not match shank_info.h");
#if USE_TH
_Static_assert(sizeof(TH)      / sizeof(TH[0])      == NCLU_ALL,                         "TH.h does not match shank_info.h");
#endif
_Static_assert(NCLU_ALL <= 65535, "template ids are stored as uint16_t");
_Static_assert((RB & (RB - 1)) == 0 && (CC & (CC - 1)) == 0, "RB and CC must be powers of two");

// Provided by the I/O section at the end of the file (host: files, target: platform hooks)
static void emit_spike(uint32_t t, uint16_t template_id, float amp);
static void debug_cf_block(uint32_t t, const float *cf);
static void debug_data_row(uint32_t t, const float *row);


/* ===================== core: static memory only, runs on the target ===================== */

//those structures should be located in TCM
typedef struct {
    float raw_data[RB][NCH_PAD];   // per-shank layout, padding columns stay 0
    int32_t head;
}SampleBuffer;

static SampleBuffer sample;
static uint32_t num_spikes = 0;

static int32_t last_commit_i[NCLU];  // per-neuron refractory clock
static int32_t last_any_commit; static float last_any_cf; //per nt timewindow refractory for each peel

//filter history variables - should be located in TCM for speed (per-shank layout)
static float iir_x1[NCH_PAD];  // x[n-1] raw input
static float iir_x2[NCH_PAD];  // x[n-2] raw input
static float iir_x3[NCH_PAD];  // x[n-3] raw input
static float iir_y1[NCH_PAD];  // y[n-1] per channel
static float iir_y2[NCH_PAD];  // y[n-2] per channel
static float iir_y3[NCH_PAD];  // y[n-3] per channel

// cf cache: one valid-time per shank (scoring a shank reads only its own channels);
// the cf/dot rows are shared, each shank fills only its own template range
static int32_t cc_time[N_SHANKS][CC];
static float cc_cf[CC][NCLU];
static float cc_dot[CC][NCLU];

// scratch buffers (static so the stack stays small)
static float car_tmp[NCHAN_MAX];
static float whitened_temp[NCHAN_MAX];
static float pc[NCHAN_MAX][NPCS];
static float cf_best[NCLU], dot_best[NCLU];
static int lag_best[NCLU];


// place one time step (NCHAN channels) in the ring buffer, shank by shank (shank_chanMap)
static void push_frame(SampleBuffer *sample, const int16_t *frame) {
    sample->head = (sample->head + 1) & RING_MASK;
    float *row = sample->raw_data[sample->head];
    for (int s = 0; s < N_SHANKS; s++)
        for (int c = 0; c < NCHAN_MAX; c++) {
            int src = shank_chanMap[s * NCHAN_MAX + c];   // .dat channel, -1 = padding
            row[s * NCHAN_MAX + c] = (src >= 0) ? (float)frame[src] : 0.0f;
        }
}

// k-th smallest (0-indexed), reorders buf in place
static float kth_smallest(float *buf, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float p = buf[(lo + hi) >> 1];
        int i = lo, j = hi;
        while (i <= j) {
            while (buf[i] < p) i++;
            while (buf[j] > p) j--;
            if (i <= j) { float t = buf[i]; buf[i] = buf[j]; buf[j] = t; i++; j--; }
        }
        if (k <= j) hi = j; else if (k >= i) lo = i; else break;
    }
    return buf[k];
}

// Kilosort's order (io.BinaryFiltered.filter): CAR -> high-pass filter -> whitening, per shank
static void filter_sample(SampleBuffer *sample) {
    int32_t h = sample->head;

    for (int s = 0; s < N_SHANKS; s++) {
        float *x = &sample->raw_data[h][s * NCHAN_MAX];   // this shank's channels
        const int n = shank_nchan[s];

        // 1. CAR - median over this shank's channels only (each shank run did its own CAR)
        for (int c = 0; c < n; c++) car_tmp[c] = x[c];
        float med = kth_smallest(car_tmp, n, (n - 1) / 2);   // lower median, index (n-1)/2 (torch.median)
        for (int c = 0; c < n; c++) x[c] -= med;

        // 2. Filter each sample (causal IIR, as Kilosort's lfilter for hp_filter_mode 'iir3')
        for (int c = 0; c < n; c++){
            int g = s * NCHAN_MAX + c;                    // filter-history slot of this channel
            float x0 = x[c];
            float y = (iir_weights[0] * x0)
                + (iir_weights[1] * iir_x1[g])
                + (iir_weights[2] * iir_x2[g])
                + (iir_weights[3] * iir_x3[g])
                - (iir_weights[4] * iir_y1[g])
                - (iir_weights[5] * iir_y2[g])
                - (iir_weights[6] * iir_y3[g]);
            iir_x3[g] = iir_x2[g];
            iir_x2[g] = iir_x1[g];
            iir_x1[g] = x0;
            iir_y3[g] = iir_y2[g];
            iir_y2[g] = iir_y1[g];
            iir_y1[g] = y;
            x[c] = y;
        }

        // 3. Whitening with this shank's Wrot
        for (int w = 0; w < n; w++) {
            float sum = 0.0f;
            for (int c = 0; c < n; c++) {
                sum += x[c] * WROT(s, w, c);
            }
            whitened_temp[w] = sum;
        }
        for (int w = 0; w < n; w++) {
            x[w] = whitened_temp[w];
        }
    }
}

// scores the templates of shank s only, on that shank's channels
static inline void score_at(SampleBuffer *sample, int s, int32_t head_at, float *cf_all, float *dot_all) {
    const int base = s * NCHAN_MAX, n = shank_nchan[s];
    // 1. Temporal Dot Product - walk backwards from ring_head
    for (int c = 0; c < n; c++)
        for (int p = 0; p < NPCS; p++) {
            float acc = 0.0f;
            for (int k = 0; k < NT_TRIG; k++)
                acc += sample->raw_data[(head_at - (NT_TRIG-1-k)) & RING_MASK][base + c] * WPCA(s, p, k);
            pc[c][p] = acc;
        }
    // 2. Spatial Collapse & Winner Selection
    for (int t = shank_offset[s]; t < shank_offset[s + 1]; t++) {
        float dot = 0.0f;
        for (int c = 0; c < n; c++)
            for (int p = 0; p < NPCS; p++)
                dot += pc[c][p] * WALL(t, c, p);
        dot_all[t] = dot;
        float r = RELU(dot);
        cf_all[t] = (r*r) / Wall_nm[t];
    }
}

// returns cf array for abs time T; computes only on a miss (per shank)
static inline void cf_for(SampleBuffer *sample, int s, int32_t head_at, int32_t T, float **cf_out, float **dot_out) {
    int32_t slot = T & CC_MASK;
    if (cc_time[s][slot] != T) {
        score_at(sample, s, head_at, cc_cf[slot], cc_dot[slot]);   // both, no scratch discard
        cc_time[s][slot] = T;
    }
    *cf_out  = cc_cf[slot];
    *dot_out = cc_dot[slot];
}


// returns 1 and sets *end_center if a magnitude peak is CONFIRMED 3 samples back on shank s
static int confirm_peak_evt(SampleBuffer *sample, int s, int32_t *end_center) {
    int32_t h = sample->head;
    int32_t pk = (h - (WIN_END_OFFSET + 1 + LAG)) & RING_MASK;   // was h-(3+LAG)
    int32_t a = (pk + 1) & RING_MASK;     // after peak
    int32_t b = (pk - 1) & RING_MASK;     // before peak
    const int base = s * NCHAN_MAX;
    for (int c = 0; c < shank_nchan[s]; c++) {
        float vpk = ABSF(sample->raw_data[pk][base + c]);
        if (vpk <= V_TH) continue;
        float va = ABSF(sample->raw_data[a][base + c]);
        float vb = ABSF(sample->raw_data[b][base + c]);
        if (vpk >= va && vpk >= vb) {
            *end_center = (pk + WIN_END_OFFSET) & RING_MASK;       // window ends 40 past peak
            return 1;
        }
    }
    return 0;
}

//The core of the code - template matching and spike detection, for shank s
static void template_match(SampleBuffer *sample, int s, uint32_t i) {
    const int t_first = shank_offset[s], t_end = shank_offset[s + 1];   // this shank's templates

    int32_t end_center;
    // stage 1: gate
    if (!confirm_peak_evt(sample, s, &end_center)) return;
    int32_t T0 = (i >= (uint32_t)(LAG + 1)) ? (int32_t)(i - (LAG + 1)) : 0; //min end of spike time
    int32_t base_peak = (i >= (uint32_t)(WIN_END_OFFSET + 1 + LAG)) ? (int32_t)(i - (WIN_END_OFFSET + 1 + LAG)) : 0; //min estimated peak time

    // stage 2: lag search — recomputed each peel because data changed after subtraction
    for (int t = 0; t < NCLU; t++) {cf_best[t] = -1.0f; }   // other shanks' templates stay -1
    for (int d = -LAG; d <= LAG; d++) {
        int32_t head_at = (end_center + d) & RING_MASK;
        float *cf, *dot;
        cf_for(sample, s, head_at, T0 + (int32_t)d, &cf, &dot);  // computed once across triggers/peels
        for (int t = t_first; t < t_end; t++)
            if (cf[t] > cf_best[t]) { cf_best[t]=cf[t]; dot_best[t]=dot[t]; lag_best[t]=d; }
    }

    //DEBUG: one block of NCLU lines per trigger; templates of other shanks are -1
    if (i >= CF_DUMP_N && i <= CF_DUMP_N + CF_DUMP_LEN) debug_cf_block((uint32_t)base_peak, cf_best);

    // best eligible template: above threshold AND not refractory (per-template peak time)
    int best_t = -1; float best_cf = 0.0f, best_dot = 0.0f; int best_lag = 0;
    int32_t best_peak = base_peak;
    for (int t = t_first; t < t_end; t++) {
        if (cf_best[t] <= TH_LEARNED_SQ) continue;
#if USE_TH
        if (cf_best[t] <= TH[t]) continue;
#endif
        int32_t peak_t = base_peak + lag_best[t];
        int32_t local_diff = peak_t - last_commit_i[t];
        //int32_t global_diff = peak_t - last_any_commit;
        if (local_diff < REFRACTORY_SAMPLES) continue;
        //if (global_diff < NT && cf_best[t] < last_any_cf) continue; //
        if (lag_best[t] == LAG) continue;
        if (cf_best[t] > best_cf) {
            best_cf=cf_best[t]; best_dot=dot_best[t]; best_lag=lag_best[t];
            best_t=t; best_peak=peak_t;
        }
    }
    if (best_t < 0) return;                       // nothing eligible left at this location

    float amp = best_dot / Wall_nm[best_t];
    emit_spike((uint32_t)best_peak, (uint16_t)best_t, amp);   // spikes leave the core right away
    num_spikes++;
    last_commit_i[best_t] = best_peak;
    last_any_commit = best_peak;
    last_any_cf = best_cf;

#if ENABLE_PEEL_SUBTRACT
    // subtract at the SAME ring slots scoring read: head-relative window-end of the
    // WINNING lag, not absolute best_peak. This cancels exactly the matched-filter energy.
    // Only this shank's channels change.
    int32_t best_end = (end_center + best_lag) & RING_MASK;
    const int base = s * NCHAN_MAX;
    for (int c = 0; c < shank_nchan[s]; c++)
        for (int ts = 0; ts < NT; ts++) {
            float recon = 0.0f;
            for (int p = 0; p < NPCS; p++)
                recon += WALL(best_t, c, p) * WPCA(s, p, ts);
            sample->raw_data[(best_end - (NT-1-ts)) & RING_MASK][base + c] -= recon * amp;
        }
    for (int k = 0; k < CC; k++) cc_time[s][k] = 0;    // this shank's data changed -> invalidate its cf/dot cache
#endif
    // without subtraction: one detection per location
}

// reset all state; returns -1 if a shank channel is not among the NCHAN channels of a frame
int flows_init(void) {
    for (int k = 0; k < NCH_PAD; k++)
        if (shank_chanMap[k] >= NCHAN) return -1;
    sample.head = 0;
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < NCH_PAD; c++) sample.raw_data[r][c] = 0.0f;
    for (int c = 0; c < NCH_PAD; c++)
        iir_x1[c] = iir_x2[c] = iir_x3[c] = iir_y1[c] = iir_y2[c] = iir_y3[c] = 0.0f;
    for (int t = 0; t < NCLU; t++) last_commit_i[t] = 0;
    for (int s = 0; s < N_SHANKS; s++)
        for (int k = 0; k < CC; k++) cc_time[s][k] = 0;
    last_any_commit = 0; last_any_cf = 0.0f;
    num_spikes = 0;
    return 0;
}

// process one time step: frame = NCHAN int16 channels, i = samples since START_SAMPLE
void flows_step(const int16_t *frame, uint32_t i) {
    uint32_t abs_i = i + START_SAMPLE;  // true absolute sample index
    push_frame(&sample, frame);
    filter_sample(&sample);
    if (abs_i >= CF_DUMP_N && abs_i <= CF_DUMP_N + CF_DUMP_LEN)
        debug_data_row(abs_i, sample.raw_data[sample.head]);
    if (i < (NT - 1)) return;
    for (int s = 0; s < N_SHANKS; s++)   // every shank is matched on its own
        template_match(&sample, s, abs_i);
}


/* ===================== I/O ===================== */
#if HOST_IO
/* Linux host: read the .dat file, stream spikes into .npy files, write the debug traces */
#include <stdio.h>
#include <string.h>

#ifndef DAT_FILE
#define DAT_FILE "/data/mP79/dat/mP79_17/mP79_17.dat"
#endif
#ifndef RESULTS_DIR
#define RESULTS_DIR "/data/results/mP79_res/flows_c/pershank"
#endif
#define TEMPLATES_FILE "spike_templates.npy"
#ifndef N_SAMPLES
#define N_SAMPLES 0u      // samples to process from START_SAMPLE; 0 = to the end of the file
#endif

static FILE *f_times, *f_templates, *f_amps, *cf_trace, *data_trace;

// .npy header with a fixed-width shape, so it can be rewritten with the final count
static void npy_header(FILE *f, const char *dtype, uint32_t n) {
    char dict[128];
    int len = snprintf(dict, sizeof(dict),
                       "{'descr': '%s', 'fortran_order': False, 'shape': (%10u,), }", dtype, n);
    int pad = 64 - ((10 + len + 1) % 64);
    if (pad == 64) pad = 0;
    uint16_t hlen = (uint16_t)(len + pad + 1);
    fseek(f, 0, SEEK_SET);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);            // little endian on x86 and RISC-V
    fwrite(dict, 1, (size_t)len, f);
    for (int k = 0; k < pad; k++) fputc(' ', f);
    fputc('\n', f);
}

static FILE *npy_open(const char *path, const char *dtype) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", path); return NULL; }
    npy_header(f, dtype, 0);
    return f;
}

static void npy_close(FILE *f, const char *dtype, const char *path) {
    npy_header(f, dtype, num_spikes);
    fclose(f);
    printf("Saved %u spikes to %s\n", num_spikes, path);
}

static void emit_spike(uint32_t t, uint16_t template_id, float amp) {
    // Kilosort types: int64 time, int32 template, float64 amplitude
    int64_t tt = t; int32_t k = template_id; double a = amp;
    fwrite(&tt, sizeof(tt), 1, f_times);
    fwrite(&k, sizeof(k), 1, f_templates);
    fwrite(&a, sizeof(a), 1, f_amps);
}

static void debug_cf_block(uint32_t t, const float *cf) {
    for (int k = 0; k < NCLU; k++) fprintf(cf_trace, "%u, %f\n", t, (double)cf[k]);
}

static void debug_data_row(uint32_t t, const float *row) {   // real channels only, shank by shank
    for (int s = 0; s < N_SHANKS; s++)
        for (int c = 0; c < shank_nchan[s]; c++)
            fprintf(data_trace, "%u, %f\n", t, (double)row[s * NCHAN_MAX + c]);
}

static int16_t frame_buf[NCHAN_BIN], frame[NCHAN_BIN];

int main(void) {
    if (flows_init() != 0) {
        fprintf(stderr, "shank_chanMap has a channel outside the first %d channels\n", NCHAN);
        return -1;
    }
    printf("Reading data from %s...\n", DAT_FILE);
    FILE *fptr = fopen(DAT_FILE, "rb");
    if (!fptr) { fprintf(stderr, "Error opening binary file\n"); return -1; }

    // Compute total sample count from file size (64-bit offsets: host only)
    fseeko(fptr, 0, SEEK_END);
    uint64_t total_samples = (uint64_t)ftello(fptr) / (NCHAN_BIN * sizeof(int16_t));
    if (total_samples <= START_SAMPLE) { fprintf(stderr, "file is shorter than START_SAMPLE\n"); return -1; }
    uint64_t n_samples = total_samples - START_SAMPLE;
    if (N_SAMPLES > 0 && N_SAMPLES < n_samples) n_samples = N_SAMPLES;
    fseeko(fptr, (off_t)((uint64_t)START_SAMPLE * NCHAN_BIN * sizeof(int16_t)), SEEK_SET);

    f_times     = npy_open(RESULTS_DIR "/spike_times.npy", "<i8");
    f_templates = npy_open(RESULTS_DIR "/" TEMPLATES_FILE, "<i4");
    f_amps      = npy_open(RESULTS_DIR "/amplitudes.npy", "<f8");
    cf_trace    = fopen(RESULTS_DIR "/cf_trace.txt", "w");
    data_trace  = fopen(RESULTS_DIR "/data_trace.txt", "w");
    if (!f_times || !f_templates || !f_amps || !cf_trace || !data_trace) {
        fprintf(stderr, "cannot write to %s\n", RESULTS_DIR); return -1;
    }

    for (uint32_t i = 0; i < (uint32_t)n_samples; i++) {
        size_t got = fread(frame_buf, sizeof(int16_t), NCHAN_BIN, fptr);
        if (got == NCHAN_BIN) memcpy(frame, frame_buf, sizeof(frame));
        else if (i == 0) { printf("Error reading first sample, number of channels is: %zu.\n", got); return 1; }
        // else: incomplete last time step -> repeat the last complete one
        flows_step(frame, i);
        if (i % 100000 == 0) printf("Processed %u samples... out of %u\n", i, (uint32_t)n_samples);
    }
    fclose(fptr);

    printf("Processing complete. Detected %u spikes.\nSaving results in Kilosort format...\n", num_spikes);
    npy_close(f_times,     "<i8", RESULTS_DIR "/spike_times.npy");
    npy_close(f_templates, "<i4", RESULTS_DIR "/" TEMPLATES_FILE);
    npy_close(f_amps,      "<f8", RESULTS_DIR "/amplitudes.npy");
    fclose(cf_trace);
    fclose(data_trace);
    printf("Done!\n");
    return 0;
}

#else
/* Bare-metal target: the platform code provides these two (ADC/DMA input, UART/DMA/memory output) */
extern int  platform_read_frame(int16_t frame[NCHAN]);   // next time step, NCHAN channels; 0 = a frame was read
extern void platform_emit_spike(uint32_t t, uint16_t template_id, float amp);

static void emit_spike(uint32_t t, uint16_t template_id, float amp) { platform_emit_spike(t, template_id, amp); }
static void debug_cf_block(uint32_t t, const float *cf) { (void)t; (void)cf; }
static void debug_data_row(uint32_t t, const float *row) { (void)t; (void)row; }

static int16_t frame[NCHAN];

int main(void) {
    if (flows_init() != 0) return -1;
    for (uint32_t i = 0; platform_read_frame(frame) == 0; i++)
        flows_step(frame, i);
    return 0;
}
#endif
