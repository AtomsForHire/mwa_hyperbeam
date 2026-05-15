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
    const FLOAT pc_ra, const FLOAT pc_dec, const int num_stations,
    const FLOAT *station_coordinates, const FLOAT *station_angles,
    const int *num_elems_per_station, const FLOAT lst_rad,
    const FLOAT site_latitude_rad, const uint8_t norm_to_zenith,
    void *d_results);

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
__global__ void
ska_analytic_kernel(const ANALYTIC_TYPE at, const FLOAT *d_azs,
                    const FLOAT d_zas, int num_directions,
                    const unsigned int *d_freqs_hz, const int num_freqs,
                    const int num_stations, const FLOAT *station_coordinates,
                    const FLOAT *station_angles, const FLOAT lst_rad,
                    const uint8_t norm_to_zenith, void *d_results) {
  // NOTE: Mostly copy the loop structure from MWA

  // Each thread works on a direction, if there are more directions than threads
  // then take a stride by gridDim.x * blockDim.x
  for (int i_direction = blockIdx.x * blockDim.x + threadIdx.x;
       i_direction < num_directions; i_direction += gridDim.x * blockDim.x) {
    const FLOAT az = azs[i_direction];
    const FLOAT za = zas[i_direction];

    // First, let's convert azza into dl, dm for array factor
    // 1.1 Get El from Za
    const FLOAT el = M_PI_2 - za;

    // 1.2 Convert AzEl to HaDec
    FLOAT s_az, c_az, s_el, c_el, s_site, c_site;
    SINCOS(az, &s_az, &c_az);
    SINCOS(el, &s_el, &c_el);
    SINCOS(site_latitude_rad, &s_site, &c_site);

    FLOAT x = -c_az * c_el * s_site + s_el * c_site;
    FLOAT y = -s_az * c_el;
    FLOAT z = c_az * c_el * c_site + s_el * s_site;

    FLOAT r = SQRT(x * x + y * y);
    FLOAT ha = 0.0;
    if r
      != 0.0 { ha = ATAN2(y, x); }
    FLOAT dec = ATAN2(z, r);

    // 1.2 Now calculate beam Ra from Ha and lst_rad (beam_dec == dec)
    FLOAT beam_ra = lst_rad - ha;

    // 1.3 Now find beam_lmn from beam_ra and zenith (lst_rad,
    // site_latitude_rad)
    FLOAT d_ra = beam_ra - lst_rad;
    FLOAT s_d_ra, c_d_ra, s_dec, c_dec, s_pc_dec, c_pc_dec;
    SINCOS(d_ra, &s_d_ra, &c_d_ra);
    SINCOS(dec, &s_dec, &c_dec);
    SINCOS(site_latitude_rad, &s_z_dec, &c_z_dec);

    FLOAT beam_l = c_dec * s_d_ra;
    FLOAT beam_m = s_dec * c_z_dec - c_dec * s_z_dec * c_d_ra;
    FLOAT beam_n = s_dec * s_z_dec + c_dec * c_z_dec * c_d_ra;

    // 1.4 Now find cent_lmn from pc_ra, pc_dec and zenith (lst_rad,
    // site_latitude_rad)
    // NOTE: Re-using some variable names here, sorry future debugger
    FLOAT d_ra = pc_ra - lst_rad;
    FLOAT s_d_ra, c_d_ra, s_pc_dec, c_pc_dec, s_pc_dec, c_pc_dec;
    SINCOS(d_ra, &s_d_ra, &c_d_ra);
    SINCOS(pc_dec, &s_pc_dec, &c_pc_dec);

    FLOAT cent_l = c_pc_dec * s_d_ra;
    FLOAT cent_m = s_pc_dec * c_z_dec - c_pc_dec * s_z_dec * c_d_ra;
    FLOAT cent_n = s_pc_dec * s_z_dec + c_pc_dec * c_z_dec * c_d_ra;

    FLOAT dl = beam_l - cent_l;
    FLOAT dm = beam_m - cent_m;

    // 2. Calculate array factor
    array_factor = MAKE_COMPLEX(0.0, 0.0);

    // 2.1 Loop over each station
    for (i_station = 0; i_station < num_stations; i_station++) {
      // Get number of elements/antennas for this station
      int num_elems = num_elems_per_station[i_station];
      for (i_elem = 0; i_elem < num_elems; i_elem++) {
      }
    }
  }
}

extern "C" const char *ska_gpu_analytic_calc_jones(
    const ANALYTIC_TYPE at, const FLOAT *d_azs, const FLOAT d_zas,
    int num_directions, const unsigned int *d_freqs_hz, const int num_freqs,
    const int num_stations, const FLOAT *station_coordinates,
    const FLOAT *station_angles, const FLOAT lst_rad,
    const uint8_t norm_to_zenith, void *d_results) {}

#endif // BINDGEN
       //
#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
