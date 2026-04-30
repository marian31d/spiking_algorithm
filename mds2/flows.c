#include <sys/stat.h>
#include <stdint.h>
#include <stdio.h>

// Include array hearder files for configuration 
#include "Wall_templates.h"
#include "Wrot.h"
#include "dshift.h"
#include "fwav.h"

#define NCHAN_BIN 135
#define NCHAN 128
#define FS 20000
#define NT 20480 //easier than 20000 for addrss math 
#define nt 61
#define NT0MIN 20
#define NEAREST_CHANS 8
#define BUS_SIZE 4


/*
1. need to change the file pointer size, riscv doesn't support 64bit altho the compiler can deal with it.
2. change the implementation of filtering the unused channel to better fit the Soc
*/


struct batch { uint32_t ibatch[NT]; int16_t raw_batch[NT][NCHAN]; }; //Edit for the data types of both of the arrays

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
    
    //size_t samples_read = fread(batch_data->raw_batch, sizeof(int16_t), NT, fptr);

    /////////////Logic for discarding unused channels - not needed for Soc/////////////
    for (uint16_t s = 0; s < NT; s++) {
        size_t samples_read = fread(&batch_data->raw_batch[s][0], sizeof(int16_t), 128, fptr);
        // 2. Jump the file pointer forward past the 7 dead channels
        fseek(fptr, NCHAN_BIN - NCHAN, SEEK_CUR);
    }

    //filling up ibatch with the corresponding indices for the batch
    //NOTE: this might be very inefficient to run on processor, check if actually needed
    for (int j = 0; j < NT; j++) {
        batch_data->ibatch[j] = i*NT + j;
    }

    if (samples_read == NT) {
        return 0;
    }
    //handle an incomplete batch at the end of the file by padding with the last valid sample for each channel
    if (samples_read > 0) {
        printf("Partial batch detected at end of file. Padding the rest...\n");
        // 1. How many PERFECT time steps did we get?
        int valid_time_steps = samples_read / NCHAN;
        // SAFETY CHECK: Did the file end before even giving us 1 full time step?
        // If so, we don't have a "last valid sample" to copy. Throw it away!
        if (valid_time_steps == 0) {
            printf("End fragment too small. Discarding.\n");
            return 1; 
        }
        // 2. Find the starting index of the LAST perfect time step
        // Example: If we have 5 valid steps (0 to 4), we want step 4.
        int last_valid_time_start_idx = (valid_time_steps - 1) * NCHAN;
        printf("Padding batch from time step %d to %d...\n", valid_time_steps, NT);
        // 3. The Padding Loop
        // We start 't' at the first MISSING or CORRUPTED time step
        for (int t = valid_time_steps; t < NT; t++) {
            
            // Copy all 135 channels from the last valid step to the missing step
            for (int c = 0; c < NCHAN; c++) {
                int dest_idx = (t * NCHAN) + c;                 // Where we write
                int src_idx  = last_valid_time_start_idx + c;   // Where we read from
                
                batch_data->ibatch[dest_idx] = batch_data->raw_batch[src_idx];    
            }
        }
        return 0; // Success! The batch is fully padded and ready for math.
    }
}

int filter_batch(const char *results_dir, struct batch *batch_data){
    //In kilosort each batch was padded with nt in the first and last batch
    //for the last batch I'm not sure how the logic should be but for now I'll use batch index
    int *ptr = (int*) malloc(sizeof(int) * 10); 
    int16_t p_batch[NT+2*nt][NCHAN]; //temporary array for storing padded batch 
    int32_t b_mean[NCHAN]; //variable for storing the mean for each channel
    for (j=0; j<NT+2*nt; j++){
        if (j<nt){
            p_batch[j] = &batch_data->raw_batch[0];
        }
        else if (j<(NT+nt)){
            p_batch[j] = &batch_data->raw_batch[j];
        }
        else {
            p_batch[j] = &batch_data->raw_batch[NT+nt];
        }
        b_mean += p_btach[j]; //sum over all samples for each channel after padding
    }
    for (i=0; i<NT+2*nt; i++){
        for (j=0; j<NCHAN; j++) {
        p_batch[i][j] - b_mean[j];
        }
    }
    //Filter each batch
    int32_t f_batch 

}

int __main__() {

    char filename[] = "/data/mDs2/.../mds2.dat"; // Edit this path to the actual binary file location
    char results_dir[] = "/data/results/mds2_results/flows_c";
    uint64_t binary_size = get_file_size_posix(filename);

    
    // loop over binary file in batch_size for binary_size/batch_size times 
    // and process each batch.
    printf("Reading %d time steps from %s...\n", NT, filename);
    FILE *fptr;
    fptr = fopen(filename, "r");
    if (!fptr) {
        fprintf(stderr, "Error opening binary file");
        return -1; // Return error batch
        }
    for (int i = 0; i < binary_size / NT; i++) {
        struct batch *b = (batch) malloc(sizeof(batch)); //memory allocation for each batch 
        int r_ret = dma_read_block(fptr, results_dir, i, &b);
        if (ret == 1) { // Check for error batch
            break; // Skip processing this batch it's too small
        }
        int f_ret = filter_batch()

        free(b);
    }
    fclose(fptr);
}

    


