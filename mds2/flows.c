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
#define NT 20000
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

    if (elements_read != NSAMPLES) {
        if (feof(fptr)) {
            printf("Reached End of File at batch %d.\n", batch_index);
            //padd the remaining data with zeros if needed
        } else {
            fprintf(stderr, "Error reading data for batch %d.\n", batch_index);
        }
        return 0; // Return 0 to indicate we should stop
    }
    //filling up ibatch with the corresponding indices for the batch
    //NOTE: this might be very inefficient to run on processor, check if actually needed
    for (int i = 0; i < NSAMPLES; i++) {
        batch_data->ibatch[i] = i*NSAMPLES + i;
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
    print(f"Reading {batch_size} time steps from {bin_path}...")
    FILE *fptr;
    fptr = fopen(filename, "r");
    if (!fptr) {
            fprintf(stderr, "Error fetching data for batch %d\n", i);
            return (struct batch){-1, -1}; // Return error batch
        }
    for (int i = 0; i < binary_size / NT; i++) {
        struct batch b;
        int ret = dma_read_block(fptr, results_dir, i, &b);
        if (ret == 0) { // Check for error batch
            break; // Skip processing this batch
        }
        
}
    


