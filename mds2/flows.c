#include <sys/stat.h> //does not exist in bare metal
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Include array hearder files for configuration 
#include "Wall_Normal.h"
#include "Wsh.h"
#include "fir_weights.h"
#include "wPCA.h"

#define NCHAN_BIN 135
#define NCHAN 126
#define FS 20000
#define NT 32768
#define nt 61
#define NT0MIN 20 //probably not needed
#define NEAREST_CHANS 8
#define FB 35       //0.5 * filter buffer size 
#define NPCS 3
#define NCLU 266        //Number of clusters, should be configured on the RAM
#define TH 8    //check if it's constant
#define MAX_SPIKES_PER_BATCH 10000      //ask Eran about this
#define MAX_PEELS 2 //matching pursuit iteration number, if disabled set to 1

/*
1. need to change the file pointer size, riscv doesn't support 64bit altho the compiler can deal with it.
2. change the implementation of filtering the unused channel to better fit the Soc.
3. storing the batch should be done by linker or memory pecific storing.
4. check if the mean variables needs 32 or 64 bits
5. need to know how to configure the number of clusters in Wall
*/

//those structures should be located in TCM
struct batch {
    uint32_t start_time_index;
    int16_t raw_batch[NT][NCHAN]; 
};
struct SpikeEvent {
    uint32_t absolute_time;
    uint16_t template_id;
    float score;
};
int16_t temp[NT+2*FB][NCHAN]; //temporary array for storing padded batch 
int16_t overlap_tail[FB][NCHAN]; // Holds the last 35 samples for the next batch!



// Function to get the file size using POSIX stat
uint64_t get_file_size_posix(const char *filename) {
    // Use stat to get the file size
    struct stat st;
    if (stat(filename, &st) == 0) {
        return (uint64_t)st.st_size; // st_size is of type off_t
    }
    return -1;
}

// Function to read batchs of binary data from the file (to be changed to read the data from other source)
int dma_read_block(FILE *fptr, int i, struct batch *batch_data) {
    // Implementation of data fetching logic
    // This function should read the data from the specified filename
    // and store it in a suitable data structure for further processing.
    
    //filling up ibatch with the start time for the batch
    batch_data->start_time_index = i;
    /////////////Logic for discarding unused channels - not needed for Soc/////////////
    for (uint16_t s = 0; s < NT; s++) {
        size_t samples_read = fread(&batch_data->raw_batch[s][0], sizeof(int16_t), NCHAN, fptr);
        
        if (samples_read == NCHAN) {
            // 2. Jump the file pointer forward past the 7 dead channels
            fseek(fptr, (NCHAN_BIN - NCHAN) * sizeof(int16_t), SEEK_CUR);
        }
        //handle an incomplete batch at the end of the file by padding with the last valid sample for each channel
        else {
            printf("Partial batch detected at index: %d. Padding the rest...\n", s);
            // SAFETY CHECK: Did the file end before even giving us 1 full time step?
            if (s == 0) {
                printf("Error reading first batch, size is: %d.\n", samples_read);
                return 1; 
            }
            printf("Padding batch from time step %d to %d...\n", s, NT);
            // 3. The Padding Loop
            
            // We start 't' at the first MISSING or CORRUPTED time step
                for (int t = s; t < NT; t++) {
                    for (int c = 0; c < NCHAN; c++) {
                        // we pad the current timestep if it's missing less than NCHAN channels
                        if ((samples_read > 0) && (t == s)){
                            c = samples_read;
                            samples_read = 0;
                        }
                        batch_data->raw_batch[t][c] = batch_data->raw_batch[s - 1][c];    
                    }   
                }
            return 0; // Success! The batch is fully padded and ready for math.
        }
    }
    return 0; // Success! Reached the end of the 61 rows normally.
}

int filter_batch(struct batch *batch_data){
    //In kilosort each batch was padded with nt in the first and last batch but I shoud pad it with FB for the
    //filter. Regarding the last batch I'm not sure how the logic should be but for now I'll use batch index
    
    int64_t b_mean[NCHAN] = {0}; //variable for storing the mean for each channel -
    for (int i = 0; i < NT+2*FB; i++){
        for (int j = 0; j < NCHAN; j++){
            if (i < FB){
                if(batch_data->start_time_index > 0){temp[i][j] = overlap_tail[i][j];}
                else {temp[i][j] = batch_data->raw_batch[0][j];}
            }
            else if (i < (NT + FB)){
                temp[i][j] = batch_data->raw_batch[i-FB][j];
            }
            else {
                temp[i][j] = batch_data->raw_batch[NT-1][j];
            }
            b_mean[j] += temp[i][j]; //sum over all samples for each channel after padding
        }
    }
    printf("Batch padded succesfuly, size of temp memory is: %zu\n", sizeof(temp)/sizeof(temp[0][0]));

    // 1.5 SAVE THE TAIL FOR THE NEXT BATCH
    // We must grab the last FB (35) samples of THIS raw batch before we 
    // run the memset() and overwrite it all!
    for (int i = 0; i < FB; i++) {
        for (int j = 0; j < NCHAN; j++) {
            overlap_tail[i][j] = batch_data->raw_batch[NT - FB + i][j];
        }
    }
    printf("updated overlap_tail buffer for next batch");

    //subtract the mean
    for (int i = 0; i < NT+2*FB; i++){
        for (int j = 0; j < NCHAN; j++) {
        temp[i][j] -= (int16_t)(b_mean[j]/(NT+2*FB)); //we need to divide by the number of samples for each channel
        }
    }
    printf("\nMean subtracted\n");

    //reset memory in order to do MAC 
    memset(batch_data->raw_batch, 0, (NT*NCHAN) * sizeof(int16_t)); //for reseting the array
    printf("Raw data buffer is set to 0\n");

    //Filter each batch
    for (int b = 0; b < NT; b++){   //batch loop
        for (int n = 0; n < NCHAN; n++){    //channel loop
            float filtered_val = 0;
            for (int i = 0; i < 2*FB; i++){     //filter loop
                filtered_val += temp[i+b][n] * fir_weights[i];
            }
            batch_data->raw_batch[b][n] = (int16_t)filtered_val;
        }
    }
    printf("Batch filtered\n");


    //multiply by Wsh for drift correction and whitening
    for (int t = 0; t < NT; t++){
        float temp_clean_row[NCHAN] = {0}; 
        for (int w = 0; w < NCHAN; w++) {
            // Accumulate as a float to prevent integer overflow
            // and preserve decimal precision from the Wsh matrix
            float sum = 0; 
            for (int c = 0; c < NCHAN; c++) {
                sum += (float)batch_data->raw_batch[t][c] * Wsh[c][w]; 
            }
            temp_clean_row[w] = sum;
        }
        for (int w = 0; w < NCHAN; w++) {
            batch_data->raw_batch[t][w] = (int16_t)temp_clean_row[w];
        }
    }
    printf("Applied drift correction and whitening\n");
    return(0);
}

// static inline suggests the compiler inline the assembly to avoid call overhead
static inline void get_max_score_at_time(struct batch *batch_data, int time_idx, float *out_max_score, int *out_best_t_id) {
    float pc_scores[NCHAN][NPCS]; 
    //raw_batch[61][128] || wPCA[3][61] 
    //we calculate the temporal scores and eleminate the temporal dimension
    // 1. Temporal Dot Product
    for(int c = 0; c < NCHAN; c++){
        for(int p = 0; p < NPCS; p++){
            float score = 0;
            for(int s = 0; s < nt; s++){
                score += batch_data->raw_batch[time_idx + s][c] * wPCA[p][s];
            }
            pc_scores[c][p] = score;
        }
    }
    // 2. Spatial Collapse & Winner Selection
    *out_max_score = 0.0f;
    *out_best_t_id = -1;
    for(int t = 0; t < NCLU; t++){
        float c_score = 0.0f;
        for(int c = 0; c < NCHAN; c++){
            for(int p = 0; p < NPCS; p++){
                c_score += pc_scores[c][p] * Wall_Normal[t][c][p];
            }
        }
        if (c_score > *out_max_score){
            *out_max_score = c_score;
            *out_best_t_id = t;
        }
    }
}

int template_match (struct batch *batch_data, struct SpikeEvent *st, int *spike_count, int match){
    *spike_count = 0;
    // 1-Sample Delay variables to find the physical peak
    float prev_max_score = 0.0f;
    int prev_best_t_id = -1;
    for(int b = 0; b < NT - nt; b++){
        float max_score;
        int best_t_id;
        // --- SCANNING PHASE ---
        // Calculate the score for the current microsecond
        get_max_score_at_time(batch_data, b, &max_score, &best_t_id);

        // --- PEAK DETECTION & PEELING PHASE ---
        // If the PREVIOUS microsecond was above threshold AND was higher than the current...
        if(prev_max_score > TH && prev_max_score > max_score){
            // We lock onto the exact peak of the spike!
            int target_b = b - 1;
            float target_score = prev_max_score;
            int target_t_id = prev_best_t_id;
            
            for(int iter = 0; iter < MAX_PEELS; iter++) {
                
                // 1. Record the Spike
                if(*spike_count < MAX_SPIKES_PER_BATCH){
                    st[*spike_count].absolute_time = (batch_data->start_time_index * NT) + target_b;
                    st[*spike_count].template_id = target_t_id;
                    st[*spike_count].score = target_score;
                    (*spike_count)++;
                }
                // 2. Reconstruct and Subtract (Erase the spike)
                for(int c = 0; c < NCHAN; c++){
                    float scaled_wall[NPCS];
                    for(int p = 0; p < NPCS; p++){
                        scaled_wall[p] = target_score * Wall_Normal[target_t_id][c][p];
                    }
                    for(int s = 0; s < nt; s++){
                        float reconstructed_voltage = 0.0f;
                        for(int p = 0; p < NPCS; p++){
                            reconstructed_voltage += scaled_wall[p] * wPCA[p][s];
                        }
                        // Erase from target_b
                        batch_data->raw_batch[target_b + s][c] -= (int16_t)reconstructed_voltage;
                    }
                }
                // 3. Recalculate to see if another spike was hiding underneath!
                get_max_score_at_time(batch_data, target_b, &target_score, &target_t_id);
                
                // 4. Break Condition
                if (target_score <= TH) {
                    break; // No more spikes crossing the threshold here. Stop peeling!
                }
            } // end iter loop
            
            // Reset max_score so we don't accidentally trigger again as the wave falls
            max_score = 0.0f;
        }
        // Save state for the next time step
        prev_max_score = max_score;
        prev_best_t_id = best_t_id;
    }
    return(0);
}

int main() {

    char filename[] = "/data/mDs2/.../mds2.dat"; // Edit this path to the actual binary file location
    char results_dir[] = "/data/results/mds2_results/flows_c";
    uint64_t binary_size = get_file_size_posix(filename);

    // loop over binary file in batch_size for binary_size/batch_size times 
    // and process each batch.
    printf("Reading %d time steps from %s...\n", NT, filename);
    FILE *fptr;
    fptr = fopen(filename, "rb");
    if (!fptr) {
        fprintf(stderr, "Error opening binary file");
        return -1; // Return error batch
    }

    struct batch b;
    struct SpikeEvent st[MAX_SPIKES_PER_BATCH];
    int num_spikes = 0;

    for (int i = 0; i < binary_size / NT; i++) {
        int r_ret = dma_read_block(fptr, i, &b);
        if (r_ret == 1) { // Check for error loading batch
            printf("last batch is too small");
            break; // Skip processing this batch it's too small
        }
        int f_ret = filter_batch(&b);
        if (f_ret == -1) { // Check for error filtering batch
            printf("Error while filtering batch %d\n", i);
            break;
        }
        int m_ret = template_match(&b, st, &num_spikes, 1);
    }
    fclose(fptr);
}