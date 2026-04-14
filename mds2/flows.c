#include <sys/stat.h>
#include <stdint.h>
#include <stdio.h>

// Include array hearder files for configuration 
#include "Wall_templates.h"
#include "Wrot.h"
#include "dshift.h"
#include "fwav.h"


#define NCHAN 135
#define FS 20000
#define NT 16384
#define nt 61
#define NT0MIN 20
#define NEAREST_CHANS 8
#define NSAMPLES 138240     // 1024 time steps * 135 channels


struct batch { uint32_t ibatch[NSAMPLES]; int16_t raw_batch[NSAMPLES]; }; //Edit for the data types of both of the arrays

// Function to get the file size using POSIX stat
uint32_t get_file_size_posix(const char *filename) {
    // Use stat to get the file size
    struct stat st;
    if (stat(filename, &st) == 0) {
        return (uint32_t)st.st_size; // st_size is of type off_t
    }
    return -1;
}

// Function to read batchs of binary data from the file (to be changed to read the data from other source)
int dma_read_block(FILE *fptr, const char *results_dir, int i, struct batch *batch_data) {
    // Implementation of data fetching logic
    // This function should read the data from the specified filename
    // and store it in a suitable data structure for further processing.
    
    size_t samples_read = fread(batch_data->raw_batch, sizeof(int16_t), NSAMPLES, fptr);

    //filling up ibatch with the corresponding indices for the batch
    //NOTE: this might be very inefficient to run on processor, check if actually needed
    for (int i = 0; i < NSAMPLES; i++) {
        batch_data->ibatch[i] = i*NSAMPLES + i;
    }

    if (samples_read ==NSAMPLES) {
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
            return 0; 
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
        return 1; // Success! The batch is fully padded and ready for math.
    }

    fclose(fptr);
    return 0; // Return successful batch
}

int __main__() {

    char filename[] = "/data/mDs2/.../mds2.dat"; // Edit this path to the actual binary file location
    char results_dir[] = "/data/results/mds2_results/flows_c";
    uint32_t binary_size = get_file_size_posix(filename);

    
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
        struct batch b;
        int ret = dma_read_block(fptr, results_dir, i, &b);
        if (ret == 0) { // Check for error batch
            break; // Skip processing this batch
        }
    } 
}

    


