#include <sys/stat.h> //does not exist in bare metal
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Include array hearder files for configuration 
#include "iir_weights.h"

#include "Wall.h"
#include "Wall_nm.h" 
#include "Wrot.h"
#include "wPCA.h"

#define RELU(x) ((x) > 0.0f ? (x) : 0.0f) //check if good for bare metal


#define FS 20000
#define nt 61
#define NPCS 3
#define RB 128 
#define RING_MASK (RB - 1)
#define MAX_PEELS 2 //matching pursuit iteration number, if disabled set to 1

#define NCHAN_BIN 63
#define NCHAN 62      //ring buffer length prev 64
#define NCLU 310        //Number of clusters, should be configured on the RAM
#define TH_LEARNED_SQ 64.00    //check if it's constant

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

static float prev_max_cf    = 0.0f;   // Cf = relu(dot)^2 / nm  (for threshold)
static float prev_max_dot   = 0.0f;   // raw dot product         (for amp = dot/nm)
static int   prev_best_t_id = -1;

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

int filter_sample(SampleBuffer *sample, uint32_t i){
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

    // 2. Apply CAR (Common Average Referencing)
    float spatial_mean = 0.0f;
    for (int c = 0; c < NCHAN; c++) {
        spatial_mean += sample->raw_data[h][c];
    }
    spatial_mean /= NCHAN;
    for (int c = 0; c < NCHAN; c++) {
        sample->raw_data[h][c] -= spatial_mean;
    }

    // 3. Whitening
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

// static inline suggests the compiler inline the assembly to avoid call overhead
static inline void get_max_score_at_time(SampleBuffer *sample, uint32_t ring_head, float *out_max_cf, float *out_max_dot, int *out_best_t_id) {
    float pc_scores[NCHAN][NPCS];

    // 1. Temporal Dot Product - walk backwards from ring_head
    for(int c = 0; c < NCHAN; c++){
        for(int p = 0; p < NPCS; p++){
            float score = 0;
            #pragma GCC unroll 61
            for(int s = 0; s < nt; s++){
                score += sample->raw_data[(ring_head - (nt - 1 - s)) & RING_MASK][c] * wPCA[p * nt + s];
            }
            pc_scores[c][p] = score;
        }
    }

    // 2. Spatial Collapse & Winner Selection - unchanged
    *out_max_cf    = 0.0f;
    *out_max_dot   = 0.0f;
    *out_best_t_id = -1;
    for(int t = 0; t < NCLU; t++){
        if (Wall_nm[t] < 1e-6f) continue;
        float dot = 0.0f;
        for(int c = 0; c < NCHAN; c++){
            for(int p = 0; p < NPCS; p++){
                dot += pc_scores[c][p] * Wall[t * (NCHAN * NPCS) + c * NPCS + p];
            }
        }
        float cf = (RELU(dot) * RELU(dot)) / Wall_nm[t];
        if (cf > *out_max_cf) {
            *out_max_cf    = cf;
            *out_max_dot   = dot;
            *out_best_t_id = t;
        }
    }
}

int template_match (SampleBuffer  *sample, SpikeEvent *spike_results, uint32_t *spike_count, uint32_t *batch_size, int match, uint32_t i){
        float current_cf, current_dot;
        int   current_t_id;

        // --- SCANNING PHASE ---
        // 1. Calculate the score for the current microsecond
        get_max_score_at_time(sample, sample->head, &current_cf, &current_dot, &current_t_id);

        //printf("Time index %u | Max Score: %f (Template %d)\n", i, current_dot, current_t_id);

        // --- PEAK DETECTION & PEELING PHASE ---
        if (prev_max_cf > TH_LEARNED_SQ && prev_max_cf > current_cf) {

            // peak_head is the ring buffer position one step ago, where the actual spike peak sits
            uint32_t peak_head = (sample->head - 1) & RING_MASK;

            // absolute_time: the template peak is at wPCA index 20, which is (nt-1-20)=40 samples
            // before the end of the window, so subtract 40 from the current absolute sample index
            uint32_t peak_time = (i >= 41) ? (i - 41) : 0;  // i-1 for the lag, then -40 for template offset

            float peak_cf   = prev_max_cf;
            float peak_dot  = prev_max_dot;
            int   best_t_id = prev_best_t_id;
            // --- ORIGINAL PEELING LOOP ---
            for(int iter = 0; iter < MAX_PEELS; iter++) {
                
                // 1. Record the Spike (using the peak time)
                if(*spike_count < 2*(*batch_size)){
                    spike_results[*spike_count].absolute_time = peak_time;
                    spike_results[*spike_count].template_id = best_t_id;
                    spike_results[*spike_count].score = peak_dot / Wall_nm[best_t_id];;
                    (*spike_count)++;
                    //printf("spike detected at time %u with template %d and score %f iter No. %d.\n", peak_time, best_t_id, spike_results[*spike_count-1].score, iter);
                }
                
                // 2. Reconstruct and Subtract (Erase the spike)
                if (Wall_nm[best_t_id] < 1e-6f) break;  // degenerate template, skip
                float amp = peak_dot / Wall_nm[best_t_id];
                for (int c = 0; c < NCHAN; c++) {
                    for (int ts = 0; ts < nt; ts++) {
                        float reconstructed = 0.0f;
                        for (int p = 0; p < NPCS; p++)
                            reconstructed += Wall[best_t_id*(NCHAN*NPCS) + c*NPCS + p]* wPCA[p*nt + ts];
                        reconstructed *= amp;
                        //printf("Subtracting from channel %d at time index %u (original value: %f, subtracted value: %f, after subtraction: %f)\n", c, (peak_head - (nt-1-ts)) & RING_MASK, sample->raw_data[(peak_head - (nt-1-ts)) & RING_MASK][c], reconstructed, sample->raw_data[(peak_head - (nt-1-ts)) & RING_MASK][c] - reconstructed);
                        sample->raw_data[(peak_head - (nt - 1 - ts)) & RING_MASK][c] -= reconstructed;
                    }
                }
                // 3. Recalculate to see if another spike was hiding underneath!
                // NOTE: We must rescan at the PEAK time, not the current time
                get_max_score_at_time(sample, peak_head, &peak_cf, &peak_dot, &best_t_id);
                
                // 4. Break Condition
                if (peak_cf <= TH_LEARNED_SQ) break;

            } // end iter loop
        // set the previous max to the peeled current max cause that's the closes peeled score we can get without forcing an off time for next spike
        prev_max_cf    = 0.0f;//peak_cf;
        prev_max_dot   = 0.0f;//peak_dot;
        prev_best_t_id = -1;//best_t_id;
        return 0;
        }
    // --- TRACKING PHASE ---
    // If we haven't hit a peak yet, save the current score for the next loop iteration
    prev_max_cf = current_cf;
    prev_max_dot = current_dot;
    prev_best_t_id = current_t_id;
    return(0);
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

    char filename[] = "/data/mDS2/dat/mDS2_02/mDS2_02.dat";
    char results_dir[] = "/data/results/mds2_res/flows_c";

    printf("Reading data from %s...\n", filename);
    FILE *fptr = fopen(filename, "rb");
    if (!fptr) { fprintf(stderr, "Error opening binary file\n"); return -1; }

    // Compute total sample count from file size
    fseek(fptr, 0, SEEK_END);
    uint64_t file_bytes  = (uint64_t)ftell(fptr);
    uint64_t total_samples = file_bytes / (NCHAN_BIN * sizeof(int16_t));
    fseek(fptr, 0, SEEK_SET);
    
    sample.head = 0;
    num_spikes = 0;
    memset(&sample.raw_data, 0, sizeof(sample.raw_data));

    uint32_t    binary_size = (uint32_t)total_samples;
    SpikeEvent *spike_results = malloc(2 * binary_size * sizeof(SpikeEvent));
    if (!spike_results) { fprintf(stderr, "malloc failed\n"); return -1; } 

    uint64_t start_sample = 72000000;
    uint64_t byte_offset = start_sample * NCHAN_BIN * sizeof(int16_t);
    fseek(fptr, (long)byte_offset, SEEK_SET);

    // Add temporarily in main, before the loop:
    printf("Wall_nm[0]=%f Wall_nm[1]=%f Wall_nm[2]=%f\n", 
        Wall_nm[0], Wall_nm[1], Wall_nm[2]);
    printf("Wall[0]=%f Wall[1]=%f wPCA[0]=%f\n",
        Wall[0], Wall[1], wPCA[0]);
//(nt+3)
    for (uint32_t i = 0; i <total_samples; i++) {
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
    snprintf(times_file, sizeof(times_file), "%s/spike_times.npy", results_dir);
    snprintf(temp_file,  sizeof(temp_file),  "%s/spike_templates.npy", results_dir);
    snprintf(amp_file,   sizeof(amp_file),   "%s/amplitudes.npy", results_dir);

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