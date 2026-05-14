#pragma once
// NOTE: copied same structure as the mwa one

#include "gpu_common.cuh"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
       //
typedef enum ANALYTIC_TYPE { SKA } ANALYTIC_TYPE;

const char *ska_gpu_analytic_calc_jones(
    const ANALYTIC_TYPE at, const FLOAT *d_azs, const FLOAT d_zas,
    int num_directions, const unsigned int *d_freqs_hz, const int num_freqs,
    const int num_stations, const FLOAT *station_coordinates,
    const FLOAT *station_angles, const FLOAT lst_rad,
    const uint8_t norm_to_zenith, void *d_results);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

// To allow bindgen to run on this file we have to hide a bunch of stuff behind
// a macro.
#ifndef BINDGEN

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Kernel goes in this block
__global__ void ska_analytic_kernel() {}

extern "C" const char *ska_gpu_analytic_calc_jones() {}

#endif // BINDGEN
       //
#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
