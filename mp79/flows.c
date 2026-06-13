#include <sys/stat.h> //does not exist in bare metal
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Include array hearder files for configuration 
#include "Wall.h"
#include "Wall_nm.h" 
#include "Wsh_per_batch.h"
#include "iir_weights.h"
#include "wPCA.h"

#define RELU(x) ((x) > 0.0f ? (x) : 0.0f) //check if good for bare metal


#define NCHAN_BIN 128
#define NCHAN 124
#define FS 20000
#define nt 61
#define RB 128       //ring buffer length prev 64
#define RING_MASK (RB - 1)
#define NPCS 3
#define NCLU 131        //Number of clusters, should be configured on the RAM
#define TH_LEARNED_SQ 64.00    //check if it's constant
#define MAX_PEELS 2 //matching pursuit iteration number, if disabled set to 1
#define BATCH_SIZE 60000
#define PEAK_CONFIRM 4            // confirmation window in samples
#define REFRACTORY_SAMPLES 20

#define SPIKE_TIME_OFFSET 40
#define ENABLE_PEEL_SUBTRACT 1
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
    uint32_t head;
}SampleBuffer;

typedef struct {
    uint32_t absolute_time;
    uint16_t template_id;
    float score;
}SpikeEvent;

static SampleBuffer sample;
static uint32_t num_spikes = 0;

// excluded MUA templates — pulled out so the peak scan and the re-scan agree
#define SKIP_TEMPLATE(t) ((t)==1||(t)==2||(t)==3||(t)==41||(t)==46||(t)==75|| \
                          (t)==78||(t)==80||(t)==90||(t)==95||(t)==113|| \
                          (t)==121||(t)==126)

// ONE peak detector: it only locates the peak in time
static float    wta_cand_cf = 0.0f;   // running max of the best template's cost
static uint32_t wta_cand_i  = 0;      // sample index of that max  == peak location
static int      wta_since   = 0;
static int      wta_armed   = 1;
static uint32_t last_commit_i[NCLU];  // per-neuron refractory clock

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
    uint32_t h = sample->head;

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
    const float *Wsh_curr = &Wsh_per_batch[batch_idx * NCHAN * NCHAN]; //debug: dshift for all batches
    float whitened_temp[NCHAN];
    for (int w = 0; w < NCHAN; w++) {
        float sum = 0.0f; 
        for (int c = 0; c < NCHAN; c++) {
            sum += sample->raw_data[h][c] * Wsh_curr[w * NCHAN + c];
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

// static inline suggests the compiler inline the assembly to avoid call overhead
static inline void score_all_templates(SampleBuffer *sample, uint32_t ring_head,
                                       float *cf_all, float *dot_all) {
    float pc_scores[NCHAN][NPCS];

    // Temporal projection — shared across all templates (the expensive part, unchanged)
    for(int c = 0; c < NCHAN; c++){
        for(int p = 0; p < NPCS; p++){
            float score = 0.0f;
            #pragma GCC unroll 61
            for(int s = 0; s < nt; s++){
                score += sample->raw_data[(ring_head - (nt - 1 - s)) & RING_MASK][c] * wPCA[p*nt + s];
            }
            pc_scores[c][p] = score;
        }
    }

    // Per-template cost function — keep ALL of them, do NOT collapse to a winner
    for(int t = 0; t < NCLU; t++){
        if (Wall_nm[t] < 1e-6f) {
            cf_all[t] = 0.0f; dot_all[t] = 0.0f; continue;
        }
        float dot = 0.0f;
        for(int c = 0; c < NCHAN; c++)
            for(int p = 0; p < NPCS; p++)
                dot += pc_scores[c][p] * Wall[t*(NCHAN*NPCS) + c*NPCS + p];
        dot_all[t] = dot;
        float r = RELU(dot);
        cf_all[t] = (r * r) / Wall_nm[t];
    }
}

int template_match(SampleBuffer *sample, SpikeEvent *spike_results,
                   uint32_t *spike_count, uint32_t *data_size, int match, uint32_t i){
    static float cf_all[NCLU];
    static float dot_all[NCLU];

    // score every template at the current sample
    score_all_templates(sample, sample->head, cf_all, dot_all);

    // detection signal = the SINGLE best template's cost (one max, not per-template)
    float win_cf = 0.0f;
    for (int t = 0; t < NCLU; t++) {
        if (SKIP_TEMPLATE(t)) continue;
        if (cf_all[t] > win_cf) win_cf = cf_all[t];
    }

    // locate the temporal peak of that best cost (arm / track-max / confirm)
    if (win_cf <= TH_LEARNED_SQ) wta_armed = 1;
    if (wta_armed && win_cf > wta_cand_cf) { wta_cand_cf = win_cf; wta_cand_i = i; wta_since = 0; }
    else                                     { wta_since++; }

    // confirmed peak -> matching pursuit at the peak location
    if (wta_armed && wta_cand_cf > TH_LEARNED_SQ && wta_since >= PEAK_CONFIRM) {

        uint32_t peak_head = (sample->head - wta_since) & RING_MASK;
        uint32_t peak_time = (wta_cand_i >= SPIKE_TIME_OFFSET)
                             ? (wta_cand_i - SPIKE_TIME_OFFSET) : 0;

        for (int peel = 0; peel < MAX_PEELS; peel++) {

            // (1) re-score ALL templates on the CURRENT residual at the peak
            score_all_templates(sample, peak_head, cf_all, dot_all);

            // (2) take the single best remaining template
            int   best_t = -1; float best_cf = 0.0f, best_dot = 0.0f;
            for (int t = 0; t < NCLU; t++) {
                if (SKIP_TEMPLATE(t)) continue;
                if (cf_all[t] > best_cf) { best_cf = cf_all[t]; best_dot = dot_all[t]; best_t = t; }
            }

            // (3) residual clean -> stop
            if (best_t < 0 || best_cf <= TH_LEARNED_SQ) break;

            // (4) per-neuron refractory
            if (wta_cand_i - last_commit_i[best_t] < REFRACTORY_SAMPLES) break;

            // (5) commit at the least-squares amplitude
            float amp = best_dot / Wall_nm[best_t];
            if (*spike_count < 2*(*data_size)) {
                spike_results[*spike_count].absolute_time = peak_time;
                spike_results[*spike_count].template_id   = (uint16_t)best_t;
                spike_results[*spike_count].score         = amp;
                (*spike_count)++;
            }
            last_commit_i[best_t] = wta_cand_i;

            // (6) subtract the reconstructed waveform from the residual
            for (int c = 0; c < NCHAN; c++)
                for (int ts = 0; ts < nt; ts++) {
                    float recon = 0.0f;
                    for (int p = 0; p < NPCS; p++)
                        recon += Wall[best_t*(NCHAN*NPCS) + c*NPCS + p] * wPCA[p*nt + ts];
                    sample->raw_data[(peak_head - (nt-1-ts)) & RING_MASK][c] -= recon * amp;
                }
        }

        wta_armed = 0; wta_cand_cf = 0.0f; wta_since = 0;
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
    char results_dir[] = "/data/results/mP79_res/flows_c";

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

    wta_armed = 1; wta_cand_cf = 0.0f; wta_cand_i = 0; wta_since = 0;
    for (int t = 0; t < NCLU; t++) last_commit_i[t] = 0;
    uint64_t start_sample = 72000000;
    uint64_t samples_to_process = total_samples - start_sample;
    uint32_t binary_size = (uint32_t)samples_to_process;
    SpikeEvent *spike_results = malloc(2 * binary_size * sizeof(SpikeEvent));
    if (!spike_results) { fprintf(stderr, "malloc failed\n"); return -1; } 

    uint64_t byte_offset = start_sample * NCHAN_BIN * sizeof(int16_t);
    fseeko(fptr, (int64_t)byte_offset, SEEK_SET);

    // Add temporarily in main, before the loop:
    for (int t = 125; t < NCLU; t++)
        printf("Wall_nm[%d] = %f\n", t, Wall_nm[t]);

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
        if (i == 600*60000 + 30000) {
            uint32_t abs_i = i + (uint32_t)start_sample;
            for (int c = 0; c < NCHAN; c++) {
                printf("SAMPLE %u CHAN %d: %f\n", abs_i, c,
                    sample.raw_data[sample.head][c]);
            }
        } 
        if (i < (nt - 1)) {continue;}
        int m_ret = template_match(&sample, spike_results, &num_spikes, &binary_size, 1, abs_i);
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
    snprintf(times_file, sizeof(times_file), "%s/debug/spike_times.npy", results_dir);
    snprintf(temp_file,  sizeof(temp_file),  "%s/debug/spike_templates.npy", results_dir);
    snprintf(amp_file,   sizeof(amp_file),   "%s/debug/amplitudes.npy", results_dir);

    // 4. Save as .npy using Little-Endian NumPy types (<i8, <i4, <f8)
    save_npy(times_file, spike_times,     num_spikes, sizeof(int64_t), "<i8");
    save_npy(temp_file,  spike_templates, num_spikes, sizeof(int32_t), "<i4");
    save_npy(amp_file,   amplitudes,      num_spikes, sizeof(double),  "<f8");

    // 5. Clean up RAM
    free(spike_times);
    free(spike_templates);
    free(amplitudes);

    printf("Done!\n");
    return 0;
}