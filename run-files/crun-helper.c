#include <stdlib.h>

int main() {
    system("cd /data/results/spiking_algorithm/ms51 && gcc -O3 -o flows_online flows.c && time ./flows_online > process_log.txt");
    system("cd /data/results/spiking_algorithm/mp79 && gcc -O3 -o flows_online flows.c && time ./flows_online > process_log.txt");
    system("cd /data/results/spiking_algorithm/mds2 && gcc -O3 -o flows_online flows.c && time ./flows_online > process_log.txt");
    return 0;
}