// Header file for /data/results/mds2_res/IIR/updated_clu/time/module/5400
// Per-shank layout of Wall.h, Wall_nm.h, TH.h, Wrot.h, wPCA.h
#ifndef SHANK_INFO_H
#define SHANK_INFO_H

#define N_SHANKS 4
#define NCHAN_MAX 16
#define NCLU_ALL 363

// shank number N of each shank_<N> folder (the probe's kcoords value)
const int shank_id[4] = {0, 1, 2, 3};
// real channels per shank; local channels nchan..NCHAN_MAX-1 are zero padding
const int shank_nchan[4] = {16, 16, 14, 16};
// templates of shank s are rows shank_offset[s] .. shank_offset[s+1]-1
const int shank_offset[5] = {0, 104, 198, 275, 363};
// .dat channel of local channel c on shank s: shank_chanMap[s*NCHAN_MAX + c], -1 = padding
const int shank_chanMap[64] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, -1, -1,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
};

#endif // SHANK_INFO_H
