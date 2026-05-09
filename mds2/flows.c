#include <sys/stat.h> //does not exist in bare metal
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Include array hearder files for configuration 
#include "Wall_templates.h"
#include "Wsh.h"
#include "fir_weights.h"
#include "wPCA.h"

#define NCHAN_BIN 135
#define NCHAN 126
#define FS 20000
#define NT 61 //easier than 20480 for addrss math 
#define nt 61
#define NT0MIN 20
#define NEAREST_CHANS 8
#define BUS_SIZE 4
#define FB 35 //0.5 * filter buffer size 

/*
1. need to change the file pointer size, riscv doesn't support 64bit altho the compiler can deal with it.
2. change the implementation of filtering the unused channel to better fit the Soc.
3. storing the batch should be done by linker or memory pecific storing.
*/

//this structure should be located in TCM
struct batch { uint32_t start_time_index; int16_t raw_batch[NT][NCHAN]; }; //Edit for the data types of both of the arrays
int16_t temp[NT+2*FB][NCHAN]; //temporary array for storing padded batch 

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
    batch_data->start_time_index = i*NT;
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
    
    int32_t b_mean[NCHAN] = {0}; //variable for storing the mean for each channel
    for (int i=0; i<NT+2*FB; i++){
        for (int j=0; j<NCHAN; j++){
            if (i < FB){
                temp[i][j] = batch_data->raw_batch[0][j];
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

    //subtract the mean
    for (int i=0; i<NT+2*FB; i++){
        for (int j=0; j<NCHAN; j++) {
        temp[i][j] -= (int16_t)(b_mean[j]/(NT+2*FB)); //we need to divide by the number of samples for each channel
        }
    }
    printf("\nMean subtracted\n");

    //reset memory in order to do MAC 
    memset(batch_data->raw_batch, 0, (NT*NCHAN) * sizeof(int16_t)); //for reseting the array
    printf("Raw data buffer is set to 0\n");

    //Filter each batch
    for (int b=0; b<NT; b++){   //batch loop
        for (int n=0; n<NCHAN; n++){    //channel loop
            float filtered_val = 0;
            for (int i=0; i<2*FB; i++){     //filter loop
                filtered_val += temp[i+b][n] * fir_weights[i];
            }
            batch_data->raw_batch[b][n] = (int16_t)filtered_val;
        }
    }
    printf("Batch filtered\n");


    //multiply by Wsh for drift correction and whitening
    for (int t=0; t<NT; t++){
        float temp_clean_row[NCHAN] = {0}; 
        for (int w = 0; w < NCHAN; w++) {
            // Accumulate as a float to prevent integer overflow
            // and preserve decimal precision from the Wsh matrix
            float sum = 0; 
            for (int j = 0; j < NCHAN; j++) {
                sum += (float)batch_data->raw_batch[t][j] * Wsh[j][w]; 
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
int template_match (struct batch *batch_data){
    //we need a 3D array to hold the scores of the dot product
    //raw_batch[128][20480]
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
        int m_ret = template_match(&b);
        //here we need logic to move the current last Fb samples (careful not the padding)
        //to the first FB samples of the next batch

    }
    fclose(fptr);
}