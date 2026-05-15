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
    const FLOAT *d_station_coordinates, const FLOAT *d_station_angles,
    const int *d_num_elems_per_station, const FLOAT lst_rad,
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
ska_analytic_kernel(const ANALYTIC_TYPE at, const FLOAT *azs, const FLOAT *zas,
                    int num_directions, const unsigned int *freqs_hz,
                    const int num_freqs, const FLOAT pc_ra, const FLOAT pc_dec,
                    const int num_stations, const FLOAT *station_coordinates,
                    const FLOAT *station_angles,
                    const FLOAT *num_elems_per_station, const FLOAT lst_rad,
                    const FLOAT site_latitude_rad, const uint8_t norm_to_zenith,
                    JONES *results) {
  // NOTE: Mostly copy the loop structure from MWA

  // Each thread works on a direction, if there are more directions than threads
  // then take a stride by gridDim.x * blockDim.x
  for (int i_direction = blockIdx.x * blockDim.x + threadIdx.x;
       i_direction < num_directions; i_direction += gridDim.x * blockDim.x) {
    const FLOAT az = azs[i_direction];
    const FLOAT za = zas[i_direction];
    const FLOAT phi = M_PI_2 - az;
    const FLOAT theta = za;

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
    if (r != 0.0) {
      ha = ATAN2(y, x);
    }
    FLOAT dec = ATAN2(z, r);

    // 1.2 Now calculate beam Ra from Ha and lst_rad (beam_dec == dec)
    FLOAT beam_ra = lst_rad - ha;

    // 1.3 Now find beam_lmn from beam_ra and zenith (lst_rad,
    // site_latitude_rad)
    FLOAT d_ra = beam_ra - lst_rad;
    FLOAT s_d_ra, c_d_ra, s_dec, c_dec, s_pc_dec, c_pc_dec, s_z_dec, c_z_dec;
    SINCOS(d_ra, &s_d_ra, &c_d_ra);
    SINCOS(dec, &s_dec, &c_dec);
    SINCOS(site_latitude_rad, &s_z_dec, &c_z_dec);

    FLOAT beam_l = c_dec * s_d_ra;
    FLOAT beam_m = s_dec * c_z_dec - c_dec * s_z_dec * c_d_ra;
    FLOAT beam_n = s_dec * s_z_dec + c_dec * c_z_dec * c_d_ra;

    // 1.4 Now find cent_lmn from pc_ra, pc_dec and zenith (lst_rad,
    // site_latitude_rad)
    // NOTE: Re-using some variable names here, sorry future debugger
    // BUG:
    d_ra = pc_ra - lst_rad;
    SINCOS(d_ra, &s_d_ra, &c_d_ra);
    SINCOS(pc_dec, &s_pc_dec, &c_pc_dec);

    FLOAT cent_l = c_pc_dec * s_d_ra;
    FLOAT cent_m = s_pc_dec * c_z_dec - c_pc_dec * s_z_dec * c_d_ra;
    FLOAT cent_n = s_pc_dec * s_z_dec + c_pc_dec * c_z_dec * c_d_ra;

    FLOAT dl = beam_l - cent_l;
    FLOAT dm = beam_m - cent_m;

    // 2. Calculate Jones matrix for all stations at each frequency

    // 2.1 Loop over each station
    int prev_num_elems = 0;
    for (int i_station = 0; i_station < num_stations; i_station++) {
      // 2.2 Loop over all frequencies
      for (int i_freq = 0; i_freq < num_freqs; i_freq++) {
        FLOAT lambda_m = VEL_C / freqs_hz[i_freq];

        COMPLEX array_factor = MAKE_COMPLEX(0.0, 0.0);

        // 2.3 Loop over individual elems to calculate array factor for this
        // station
        int num_elems = num_elems_per_station[i_station];
        for (int i_elem = 0; i_elem < num_elems; i_elem++) {
          int idx = (i_elem * 3) + (i_station * prev_num_elems * 3);
          FLOAT x_loc = station_coordinates[idx];
          FLOAT y_loc = station_coordinates[idx + 1];

          FLOAT tot_phase = (-x_loc * dl + y_loc * dm) / lambda_m;
          FLOAT s_phase, c_phase;
          SINCOS(tot_phase, &s_phase, &c_phase);

          array_factor += MAKE_COMPLEX(c_phase, s_phase);
        }
        prev_num_elems = num_elems;

        // Normalise array factor
        array_factor *= 1.0 / num_elems;

        // Half-wavelength dipole element pattern calculation from here
        FLOAT phi_p = phi;
        FLOAT phi_q = phi + PI_2;

        FLOAT denom_p = gpu_calc_half_wavelength_dipole_denom(theta, phi_p);
        FLOAT denom_q = gpu_calc_half_wavelength_dipole_denom(theta, phi_q);

        FLOAT kl = PI_2;

        FLOAT s_phi_p, c_phi_p, s_phi_q, c_phi_q, s_theta, c_theta, s_kl, c_kl;
        SINCOS(phi_p, &s_phi_p, &c_phi_p);
        SINCOS(phi_q, &s_phi_q, &c_phi_q);
        SINCOS(theta, &s_theta, &c_theta);
        SINCOS(kl, &s_kl, &c_kl);

        FLOAT numer_p_inner = (kl * c_phi_p * s_theta);
        FLOAT numer_q_inner = (kl * c_phi_q * s_theta);
        FLOAT s_numer_p_inner, c_numer_p_inner, s_numer_q_inner,
            c_numer_q_inner;
        SINCOS(numer_p_inner, &s_numer_p_inner, &c_numer_p_innner);
        SINCOS(numer_q_inner, &s_numer_q_inner, &c_numer_q_inner);

        FLOAT numer_p = c_numer_p_inner - c_kl;
        FLOAT numer_q = c_numer_q_inner - c_kl;

        COMPLEX e_p_theta =
            array_factor * (-c_phi_p * c_theta * numer_p) / denom_p;
        COMPLEX e_p_phi = array_factor * (s_phi_p * numer_p) / denom_p;

        COMPLEX e_q_theta =
            array_factor * (-c_phi_q * c_theta * numer_q) / denom_q;
        COMPLEX e_q_phi = array_factor * (s_phi_q * numer_q) / denom_q;

        // Form the Jones matrix for this station, at this frequency, at this
        // direction.
        JONES jones = JONES{
            .j00 = e_p_theta, .j01 = e_p_phi, .j10 = e_q_theta, .j11 = e_q_phi};

        // Copy Jones matrix to global memory
        results[((num_directions * num_freqs * i_station) +
                 num_directions * i_freq) +
                i_direction] = jones;
      }
    }
  }
}

__device__ FLOAT gpu_calc_half_wavelength_dipole_denom(FLOAT theta, FLOAT phi) {
  FLOAT s_theta, c_theta, s_phi, c_phi;
  SINCOS(theta, &s_theta, &c_theta);
  SINCOS(phi, &s_phi, &c_phi);

  FLOAT result = 1.0 + c_phi * c_phi * (c_theta * c_theta - 1.0);
  return result;
}

extern "C" const char *ska_gpu_analytic_calc_jones(
    const ANALYTIC_TYPE at, const FLOAT *d_azs, const FLOAT d_zas,
    int num_directions, const unsigned int *d_freqs_hz, const int num_freqs,
    const FLOAT pc_ra, const FLOAT pc_dec, const int num_stations,
    const FLOAT *d_station_coordinates, const FLOAT *d_station_angles,
    const int *d_num_elems_per_station, const FLOAT lst_rad,
    const FLOAT site_latitude_rad, const uint8_t norm_to_zenith,
    void *d_results) {
  dim3 gridDim, blockDim;
  blockDim.x = warpSize;
  gridDim.x = (int)ceil((double)num_directions / (double)blockDim.x);
  ska_analytic_kernel(at, d_azs, d_zas, num_directions, d_freqs_hz, num_freqs,
                      pc_ra, pc_dec, num_stations, d_station_coordinates,
                      d_station_angles, d_num_elems_per_station, lst_rad,
                      site_latitude_rad, norm_to_zenith, (JONES *)d_results);
  gpuError_t error_id;
#ifdef DEBUG
  error_id = gpuDeviceSynchronize();
  if (error_id != gpuSuccess) {
    return gpuGetErrorString(error_id);
  }
#endif
  error_id = gpuGetLastError();
  if (error_id != gpuSuccess) {
    return gpuGetErrorString(error_id);
  }

  return NULL;
}

#endif // BINDGEN
       //
#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
