#include <stdlib.h>

int main() {
    //remember to create the configured files in the data directory before running this code
    system("cd /data/results/spiking_algorithm/ms51"); 
    system("gcc -O3 -o flows_c flows.c");
    system("time ./flows_online >process_log.txt");
    system("cd /data/results/spiking_algorithm/mp79"); 
    system("gcc -O3 -o flows_c flows.c");
    system("time ./flows_online >process_log.txt");
    system("cd /data/results/spiking_algorithm/mds2"); 
    system("gcc -O3 -o flows_c flows.c");
    system("time ./flows_online >process_log.txt");
    return 0;
}