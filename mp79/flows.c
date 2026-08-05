#include <sys/stat.h> //does not exist in bare metal
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// Include array hearder files for configuration 
#include "Wall.h"
#include "Wall_nm.h"  //#include "Wall_nm.h" 
#include "Wrot.h"
#include "iir_weights.h"
#include "wPCA.h"

#define RELU(x) ((x) > 0.0f ? (x) : 0.0f) //check if good for bare metal

#define NCHAN_BIN 136
#define NCHAN 124
#define NCLU 93        //Number of clusters, should be configured on the RAM
#define TH_LEARNED_SQ 64.00    //check if it's constant

#define LAG 3
#define FS 20000
#define nt 61
#define RB 128       //ring buffer length prev 64
#define RING_MASK (RB - 1)
#define NPCS 3
#define MAX_PEELS 1 //matching pursuit iteration number, if disabled set to 1
#define BATCH_SIZE 60000
#define REFRACTORY_SAMPLES 20

#define NT_TRIG 61               // 20 pre-peak + peak + 2  (peak at nt0min=20)
#define PEAK_OFFSET 20           // nt0min: peak sits 20 samples into the template
#define WIN_END_OFFSET (NT_TRIG - 1 - PEAK_OFFSET)   // 61-1-20 = 40 
#define V_TH 4.0f                // magnitude trigger on whitened data — TUNE
#define ENABLE_PEEL_SUBTRACT 0

// cyclic cache of cf-arrays keyed by absolute window-end time
#define CC 16
#define CC_MASK (CC - 1)
static uint32_t cc_time[CC];
static float cc_cf[CC][NCLU];
static float cc_dot[CC][NCLU];     // ADD: cache dot too

//DEBUG
#define CF_DUMP_N 72000000
//static float *cf_trace = NULL;     // one win_cf per processed sample
/*
1. need to change the file pointer size, riscv doesn't support 64bit altho the compiler can deal with it.
2. change the implementation of filtering the unused channel to better fit the Soc.
3. storing the batch should be done by linker or memory specific storing.
4. check if the mean variables needs 32 or 64 bits
5. need to know how to configure the number of clusters in Wall
*/

//those structures should be located in TCM
typedef struct {
    float raw_data[RB][NCHAN]; 
    int32_t head;
}SampleBuffer;

typedef struct {
    uint32_t absolute_time;
    uint16_t template_id;
    float score;
}SpikeEvent;

static SampleBuffer sample;
static uint32_t num_spikes = 0;

static int32_t last_commit_i[NCLU];  // per-neuron refractory clock
static int32_t last_any_commit; static float last_any_cf; //per nt timewindow refractory for each peel


//filter history variables - should be located in TCM for speed
static float iir_x1[NCHAN];  // x[n-1] raw input
static float iir_x2[NCHAN];  // x[n-2] raw input
static float iir_y1[NCHAN];  // y[n-1] per channel
static float iir_y2[NCHAN];  // y[n-2] per channel


// Function to read batchs of binary data from the file (to be changed to read the data from other source)
int dma_read_block(FILE *fptr, uint32_t i, SampleBuffer *sample) {
    // Implementation of data fetching logic
    // This function should read the data from the specified filename
    // and store it in a suitable data structure for further processing.
    
    sample->head = (sample->head + 1) & RING_MASK;
    /////////////Logic for discarding unused channels - not needed for Soc/////////////
    int h = sample->head;
    int16_t temp_int_buffer[NCHAN];     //temporary buffer to read as int16 and convert to float 
    size_t samples_read = fread(temp_int_buffer, sizeof(int16_t), NCHAN, fptr);
    if (samples_read == NCHAN) {
        //float conversion loop
        for (int c = 0; c < NCHAN; c++) {
            sample->raw_data[h][c] = (float)temp_int_buffer[c];
        }
        // 2. Jump the file pointer forward past the 7 dead channels
        fseek(fptr, (NCHAN_BIN - NCHAN) * sizeof(int16_t), SEEK_CUR);
    }
    //handle an incomplete batch at the end of the file by padding with the last valid sample for each channel
    else {
        // SAFETY CHECK: Did the file end before even giving us 1 full time step?
        if (i == 0) {
            printf("Error reading first batch, number of channels is: %ld.\n", samples_read);
            return 1; 
        }
        // 3. The Padding Loop
        for (int c = 0; c < NCHAN; c++) {
            // we pad the current timestep if it's missing less than NCHAN channels
            sample->raw_data[h][c] = sample->raw_data[(h - 1) & RING_MASK][c];    
        }   
        return 0; // Success! The batch is fully padded and ready for math.
    }
    /* printf("Read batch %u: value =\n", i);
    for(int c=0; c<NCHAN; c++){printf("  %f", sample->raw_data[h][c]); if(c%10==0){printf("\n");}}
    printf("\n"); */
    return 0; // Success! Reached the end of the 61 rows normally.
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

int filter_sample(SampleBuffer *sample, uint32_t i) {
    int32_t h = sample->head;

    // 1. Filter each sample
    for (int c = 0; c < NCHAN; c++){    
        float x0 = sample->raw_data[h][c];
        float y = (iir_weights[0] * x0) 
            + (iir_weights[1] * iir_x1[c])
            + (iir_weights[2] * iir_x2[c])
            - (iir_weights[3] * iir_y1[c])
            - (iir_weights[4] * iir_y2[c]);
        iir_x2[c] = iir_x1[c];
        iir_x1[c] = x0;
        iir_y2[c] = iir_y1[c];
        iir_y1[c] = y;
        sample->raw_data[h][c] = y;
    }
    /* printf("Filtered batch %u: value =\n", i);
    for(int c=0; c<NCHAN; c++){printf("  %f", sample->raw_data[h][c]); if(c%10==0){printf("\n");}}
    printf("\n"); */

    // 2. CAR - median
    float car_tmp[NCHAN];
    for (int c = 0; c < NCHAN; c++) car_tmp[c] = sample->raw_data[h][c];
    float med = kth_smallest(car_tmp, NCHAN, (NCHAN - 1) / 2);   // lower median, n=124 -> index 61
    for (int c = 0; c < NCHAN; c++) sample->raw_data[h][c] -= med;

    // 3. Whitening
    int batch_idx = i / BATCH_SIZE; //debug: dshift for all batches
    //const float *Wsh_curr = &Wrot[batch_idx * NCHAN * NCHAN]; //debug: dshift for all batches
    float whitened_temp[NCHAN];
    for (int w = 0; w < NCHAN; w++) {
        float sum = 0.0f; 
        for (int c = 0; c < NCHAN; c++) {
            sum += sample->raw_data[h][c] * Wrot[w * NCHAN + c];
        }
        whitened_temp[w] = sum;
    }
    for (int w = 0; w < NCHAN; w++) {
        sample->raw_data[h][w] = whitened_temp[w];
    }
    /* printf("Batch after whitening %u: value =\n", i);
    for(int c=0; c<NCHAN; c++) {printf("  %f", sample->raw_data[h][c]); if(c%10==0){printf("\n");}}
    printf("\n"); */
    return(0);
}
static inline void score_at(SampleBuffer *sample, int32_t head_at,
                            float *cf_all, float *dot_all) {
    float pc[NCHAN][NPCS];
    for (int c = 0; c < NCHAN; c++)
        for (int p = 0; p < NPCS; p++) {
            float s = 0.0f;
            for (int k = 0; k < NT_TRIG; k++)
                s += sample->raw_data[(head_at - (NT_TRIG-1-k)) & RING_MASK][c] * wPCA[p*nt + k];
            pc[c][p] = s;
        }
    for (int t = 0; t < NCLU; t++) {
        float dot = 0.0f;
        for (int c = 0; c < NCHAN; c++)
            for (int p = 0; p < NPCS; p++)
                dot += pc[c][p] * Wall[t*(NCHAN*NPCS) + c*NPCS + p];
        dot_all[t] = dot;
        float r = RELU(dot);
        cf_all[t] = (r*r) / Wall_nm[t];
    }
}

// returns cf array for abs time T; computes only on a miss
static inline void cf_for(SampleBuffer *sample, int32_t head_at, int32_t T,
                          float **cf_out, float **dot_out) {
    int32_t s = T & CC_MASK;
    if (cc_time[s] != T) {
        score_at(sample, head_at, cc_cf[s], cc_dot[s]);   // both, no scratch discard
        cc_time[s] = T;
    }
    *cf_out  = cc_cf[s];
    *dot_out = cc_dot[s];
}

// static inline suggests the compiler inline the assembly to avoid call overhead
// returns 1 and sets *peak_end_head if a magnitude peak is CONFIRMED 3 samples back
static inline int confirm_peak_evt(SampleBuffer *sample, int32_t *end_center) {
    int32_t h = sample->head;
    int32_t pk = (h - (WIN_END_OFFSET + 1 + LAG)) & RING_MASK;   // was h-(3+LAG)
    int32_t a = (pk + 1) & RING_MASK;     // after peak
    int32_t b = (pk - 1) & RING_MASK;     // before peak
    for (int c = 0; c < NCHAN; c++) {
        float vpk = fabsf(sample->raw_data[pk][c]);
        if (vpk < V_TH) continue;
        float va = fabsf(sample->raw_data[a][c]);
        float vb = fabsf(sample->raw_data[b][c]);
        if (vpk >= va && vpk >= vb) {
            *end_center = (pk + WIN_END_OFFSET) & RING_MASK;       // window ends 40 past peak
            printf("___________________________________________________________________\n");
            printf("Confirmed peak: %u\n", pk);
            return 1;
        }
    }
    return 0;
}

int template_match(SampleBuffer *sample, SpikeEvent *spike_results,
                   uint32_t *spike_count, uint32_t *data_size, int match, uint32_t i, FILE *cf_trace) {
    static float cf_best[NCLU], dot_best[NCLU];
    static int lag_best[NCLU];

    int32_t end_center;
    if (!confirm_peak_evt(sample, &end_center)) return 0;          // stage 1: gate
    printf("peak at sample: %u, end_center: %u\n", i, end_center);
    int32_t T0 = (i >= (int32_t)(LAG + 1)) ? (i - (LAG + 1)) : 0;
    int32_t base_peak = (i >= (int32_t)(WIN_END_OFFSET + 1 + LAG))
                         ? (i - (WIN_END_OFFSET + 1 + LAG)) : 0;
    
    // matching-pursuit: detect strongest, subtract, re-detect on residual, up to MAX_PEELS
    for (int peel = 0; peel < MAX_PEELS; peel++) {
        // stage 2: lag search — recomputed each peel because data changed after subtraction
        for (int t = 0; t < NCLU; t++) cf_best[t] = -1.0f;
        for (int d = -LAG; d <= LAG; d++) {
            int32_t head_at = (end_center + d) & RING_MASK;
            float *cf, *dot;
            cf_for(sample, head_at, T0 + (int32_t)d, &cf, &dot);  // computed once across triggers/peels
            printf("template 15 cf score: %f, lag: %d\n", cf[15], d);
            for (int t = 0; t < NCLU; t++)
                if (cf[t] > cf_best[t]) { cf_best[t]=cf[t]; dot_best[t]=dot[t]; lag_best[t]=d; }
        }

        /////////////////////DEBUG (first peel pass only)/////////////////////
        if (peel == 0 && i >= CF_DUMP_N && i <= CF_DUMP_N + 1000u) {
            float win_cf = 0.0f; int win_temp = -1;
            for (int t = 0; t < NCLU; t++) if (cf_best[t] > win_cf) { win_cf = cf_best[t]; win_temp = t; }
            for (int t = 0; t < NCLU; t++)
                fprintf(cf_trace, "%u, %f\n", base_peak, cf_best[t]);
            //printf("sample: %u, win_cf: %f, template_id: %d, lag: %d\n",
            //       i, win_cf, win_temp, win_temp >= 0 ? lag_best[win_temp] : 0);
        } 
        //////////////////////////////////////////////////////////////////////

        // best eligible template: above threshold AND not refractory (per-template peak time)
        int best_t = -1; float best_cf = 0.0f, best_dot = 0.0f; int best_lag = 0;
        int32_t best_peak = base_peak;
        for (int t = 0; t < NCLU; t++) {
            if (cf_best[t] <= TH_LEARNED_SQ) continue;
            int32_t peak_t = base_peak + lag_best[t];
            int32_t local_diff = peak_t - last_commit_i[t];
            int32_t global_diff = peak_t - last_any_commit;
            if (local_diff < REFRACTORY_SAMPLES) continue;       //maybe useless
            if (global_diff < nt && cf_best[t] < last_any_cf) continue; //
            printf("peak_t: %u, last_any_commit: %u\n", peak_t, last_any_commit);
            if (cf_best[t] > best_cf) {
                best_cf=cf_best[t]; best_dot=dot_best[t]; best_lag=lag_best[t];
                best_t=t; best_peak=peak_t;
            }
            //fprintf(cf_trace, "%u, %f, %u\n", i, cf_best[t], t);
            if (i < CF_DUMP_N + 72000000) printf("sample: %u, iteration t: %d, win_cf: %f, template_id: %d, lag: %d, peel number: %d\n",
            i, t, best_cf, best_t, best_t >= 0 ? lag_best[best_t] : 0, peel);
        }
        if (best_t < 0) return 0;                       // nothing eligible left at this location

        float amp = best_dot / Wall_nm[best_t];          // single multiply: no sqrt, no divide
        uint64_t size = 2*(*data_size);
        if (*spike_count < size) {
            spike_results[*spike_count].absolute_time = best_peak;
            spike_results[*spike_count].template_id   = (uint16_t)best_t;
            spike_results[*spike_count].score         = amp;
            (*spike_count)++;
        }
        last_commit_i[best_t] = best_peak;
        last_any_commit = best_peak;
        printf("last any commit: %u\n", last_any_commit);
        last_any_cf = best_cf;

#if ENABLE_PEEL_SUBTRACT
        // subtract at the SAME ring slots scoring read: head-relative window-end of the
        // WINNING lag, not absolute best_peak. This cancels exactly the matched-filter energy.
        int32_t best_end = (end_center + best_lag) & RING_MASK;
        for (int c = 0; c < NCHAN; c++)
            for (int ts = 0; ts < nt; ts++) {
                float recon = 0.0f;
                for (int p = 0; p < NPCS; p++)
                    recon += Wall[best_t*(NCHAN*NPCS) + c*NPCS + p] * wPCA[p*nt + ts];
                sample->raw_data[(best_end - (nt-1-ts)) & RING_MASK][c] -= recon * amp;
            }
        for (int s = 0; s < CC; s++) cc_time[s] = 0;    // data changed -> invalidate cf/dot cache
#else
        break;                                          // no subtraction -> one detection per location
#endif
    }
    return 0;
}

// Function to write 1D arrays into Python-compatible .npy files
void save_npy(const char* filename, void* data, size_t num_elements, size_t element_size, const char* dtype) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    
    char header_dict[256];
    // Create the python dictionary header string required by the NPY format
    snprintf(header_dict, sizeof(header_dict), "{'descr': '%s', 'fortran_order': False, 'shape': (%zu,), }", dtype, num_elements);
    
    size_t dict_len = strlen(header_dict);
    size_t header_len = dict_len + 1; // +1 for the newline character at the end
    
    // The total header length (magic + versions + len + dict + padding) must be evenly divisible by 64
    // Magic string length is 6, versions is 2, length indicator is 2 (total 10 bytes)
    size_t padding = 64 - ((10 + header_len) % 64);
    if (padding == 64) padding = 0;
    
    uint16_t total_header_len = (uint16_t)(header_len + padding);
    
    // Write NPY Magic string and version (1.0)
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    // Write total header length (little endian)
    fwrite(&total_header_len, 2, 1, f);
    // Write the dictionary string
    fwrite(header_dict, 1, dict_len, f);
    // Pad with spaces
    for(size_t i = 0; i < padding; i++) fputc(' ', f);
    fputc('\n', f);
    
    // Write the actual binary data
    fwrite(data, element_size, num_elements, f);
    fclose(f);
    
    printf("Saved %u spikes to %s\n", (uint32_t)num_elements, filename);
}

int main() {
    char filename[] = "/data/mP79/dat/mP79_17/mP79_17.dat";
    char results_dir[] = "/data/results/mP79_res/flows_c/1peel/";

    /////////////////////DEBUG/////////////////////////
    FILE *cf_trace = fopen("/data/results/mP79_res/flows_c/1peel/cf_trace.txt", "w");
    FILE *data_trace = fopen("/data/results/mP79_res/flows_c/1peel/data_trace.txt", "w");

    printf("Reading data from %s...\n", filename);
    FILE *fptr = fopen(filename, "rb");
    if (!fptr) { fprintf(stderr, "Error opening binary file\n"); return -1; }

    // Compute total sample count from file size
    fseeko(fptr, 0, SEEK_END);
    uint64_t file_bytes = (uint64_t)ftello(fptr);
    uint64_t total_samples = file_bytes / (NCHAN_BIN * sizeof(int16_t));
    fseeko(fptr, 0, SEEK_SET);
    
    sample.head = 0;
    num_spikes = 0;
    memset(&sample.raw_data, 0, sizeof(sample.raw_data));

    for (int t = 0; t < NCLU; t++) last_commit_i[t] = 0;
    uint64_t start_sample = 72000000;
    uint64_t samples_to_process = total_samples - start_sample;
    uint32_t binary_size = (uint32_t)samples_to_process;
    SpikeEvent *spike_results = malloc(2 * binary_size * sizeof(SpikeEvent));
    if (!spike_results) { fprintf(stderr, "malloc failed\n"); return -1; } 

    uint64_t byte_offset = start_sample * NCHAN_BIN * sizeof(int16_t);
    fseeko(fptr, (int64_t)byte_offset, SEEK_SET);
    
    //cf_trace = calloc(CF_DUMP_N, sizeof(float));     // calloc zeroes the warmup gap
    //if (!cf_trace) { fprintf(stderr, "cf_trace alloc failed\n"); return -1; }
//120010
    for (uint32_t i = 0; i < binary_size; i++) {
        uint32_t abs_i = i + (uint32_t)start_sample;  // true absolute sample index
        int r_ret = dma_read_block(fptr, i, &sample);
        if (r_ret == 1) { // Check for error loading batch
            printf("last batch is too small");
            break; // Skip processing this batch it's too small
        }
        int f_ret = filter_sample(&sample, i);
        if (f_ret == -1) { // Check for error filtering sample
            printf("Error while filtering batch %d\n", i);
            break;
        }
        // DEBUG: export last 100 preprocessed samples
        if (abs_i >= CF_DUMP_N && abs_i <= CF_DUMP_N + 1000u) {
            for (int c = 0; c < NCHAN; c++) {
                fprintf(data_trace, "%u, %f\n", abs_i, sample.raw_data[sample.head][c]);  // debug: dump one template's cf for inspection
            }
        }
        if (i < (nt - 1)) {continue;}
        int m_ret = template_match(&sample, spike_results, &num_spikes, &binary_size, 1, abs_i, cf_trace);
        if (i % 100000 == 0) {
            printf("Processed %u batches... out of %u\n", i, (uint32_t)(binary_size));
        }
    }
    fclose(fptr);
    // ---------- END OF BATCH PROCESSING LOOP ----------
    
    printf("Processing complete. Detected %u spikes.\nSaving results in Kilosort format...\n", num_spikes);
    
    
    // 1. Allocate flat arrays for Kilosort output types
    // Note: Kilosort prefers int64 for time, int32 for templates, and float64 for amplitudes
    int64_t *spike_times    = (int64_t *)malloc(num_spikes * sizeof(int64_t));
    int32_t *spike_templates = (int32_t *)malloc(num_spikes * sizeof(int32_t));
    double  *amplitudes     = (double *)malloc(num_spikes * sizeof(double));


    if (!spike_times || !spike_templates || !amplitudes) {
        fprintf(stderr, "Memory allocation failed for output arrays.\n");
        return -1;
    }

    // 2. Unpack the structs into the flat arrays
    for (uint32_t k = 0; k < num_spikes; k++) {
        spike_times[k]     = (int64_t)spike_results[k].absolute_time;
        spike_templates[k] = (int32_t)spike_results[k].template_id;
        amplitudes[k]      = (double)spike_results[k].score;
    }

    // 3. Format the file paths
    char times_file[256], temp_file[256], amp_file[256];
    snprintf(times_file, sizeof(times_file), "%s/spike_times.npy", results_dir);
    snprintf(temp_file,  sizeof(temp_file),  "%s/spike_templates.npy", results_dir);
    snprintf(amp_file,   sizeof(amp_file),   "%s/amplitudes.npy", results_dir);

    // 4. Save as .npy using Little-Endian NumPy types (<i8, <i4, <f8)
    save_npy(times_file, spike_times,     num_spikes, sizeof(int64_t), "<i8");
    save_npy(temp_file,  spike_templates, num_spikes, sizeof(int32_t), "<i4");
    save_npy(amp_file,   amplitudes,      num_spikes, sizeof(double),  "<f8");

    fclose(cf_trace);
    fclose(data_trace);
    // 5. Clean up RAM
    free(spike_times);
    free(spike_templates);
    free(amplitudes);

    printf("Done!\n");
    return 0;
}