// Header file for /data/results/mP79_res/IIR/updated_clu/time/module/5400
// Per-shank layout of Wall.h, Wall_nm.h, TH.h, Wrot.h, wPCA.h
#ifndef SHANK_INFO_H
#define SHANK_INFO_H

#define N_SHANKS 12
#define NCHAN_MAX 11
#define NCLU_ALL 392

// shank number N of each shank_<N> folder (the probe's kcoords value)
const int shank_id[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
// real channels per shank; local channels nchan..NCHAN_MAX-1 are zero padding
const int shank_nchan[12] = {10, 10, 11, 10, 10, 10, 11, 10, 11, 11, 10, 10};
// templates of shank s are rows shank_offset[s] .. shank_offset[s+1]-1
const int shank_offset[13] = {0, 36, 63, 100, 151, 177, 199, 238, 277, 311, 337, 372, 392};
// .dat channel of local channel c on shank s: shank_chanMap[s*NCHAN_MAX + c], -1 = padding
const int shank_chanMap[132] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, -1,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, -1,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, -1,
    82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
    93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112, 113, -1,
    114, 115, 116, 117, 118, 119, 120, 121, 122, 123, -1,
};

#endif // SHANK_INFO_H
