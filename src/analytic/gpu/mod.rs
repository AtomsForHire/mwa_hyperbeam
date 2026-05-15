// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! GPU code to implement the MWA analytic beam.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#[cfg(test)]
mod tests;

mod mwa_rts;
mod ska;

use mwa_rts::MwaRtsInner;
use ska::SkaInner;

use std::{
    collections::hash_map::DefaultHasher,
    convert::TryInto,
    ffi::CStr,
    hash::{Hash, Hasher},
};

use marlu::{AzEl, Jones};
use ndarray::prelude::*;

use super::{delay_ints_to_floats, reorder_to_rts, AnalyticBeam, AnalyticBeamError};
use crate::{
    analytic::AnalyticType,
    gpu::{DevicePointer, GpuError, GpuFloat},
};

/// A trait to be used in the mwa_rts and ska submodules. The original calc_jones_pair_inner
trait CalcJones {
    /// Originally, this was the host side function to handle de-duplication and copying arrays to
    /// device. With the introduction of SKA logic, this function will handle SKA differently.
    fn calc_jones_pair_inner(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        freqs_hz: &[u32],
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
        results: ArrayViewMut3<Jones<GpuFloat>>,
    ) -> Result<(), AnalyticBeamError>;

    /// Originally a top level function akin to `calc_jones` and `calc_jones_pair`, but needs to be
    /// a trait because it calls `calc_jones_device_pair_inner` which for MWA has moved to the
    /// mwa_rts submodule.
    fn calc_jones_device(
        &self,
        azels: &[AzEl],
        freqs_hz: &[u32],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<DevicePointer<Jones<GpuFloat>>, AnalyticBeamError>;
}

pub(crate) enum AnalyticTypeInner {
    MwaRts(MwaRtsInner),
    Ska(SkaInner),
}

/// A GPU beam object ready to calculate beam responses.
pub struct AnalyticBeamGpu {
    /// This is now an enum variant, which holds relevant information for MWA/SKA calculations
    /// Allows us to split MWA/SKA into their own submodules.
    pub(super) analytic_type: AnalyticTypeInner,
}

impl AnalyticBeamGpu {
    /// Prepare a GPU-capable device for beam-response computations given the
    /// frequencies, delays and amps to be used. The resulting object takes
    /// directions and computes the beam responses on the device.
    ///
    /// This function is intentionally kept private. Use
    /// [`AnalyticBeam::gpu_prepare`] to create a `AnalyticBeamGpu`.
    pub(super) unsafe fn new(
        analytic_beam: &AnalyticBeam,
        delays_array: ArrayView2<u32>,
        amps_array: ArrayView2<f64>,
    ) -> Result<AnalyticBeamGpu, AnalyticBeamError> {
        let analytic_type =
            match analytic_beam.beam_type {
                AnalyticType::MwaPb | AnalyticType::Rts => AnalyticTypeInner::MwaRts(
                    MwaRtsInner::new(analytic_beam, delays_array, amps_array)?,
                ),
                AnalyticType::Ska | AnalyticType::SkaMean => {
                    AnalyticTypeInner::Ska(SkaInner::new(analytic_beam)?)
                }
            };

        Ok(AnalyticBeamGpu { analytic_type })
    }

    pub fn calc_jones_device(
        &self,
        azels: &[AzEl],
        freqs_hz: &[u32],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<DevicePointer<Jones<GpuFloat>>, AnalyticBeamError> {
        match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => {
                inner.calc_jones_device(azels, freqs_hz, latitude_rad, norm_to_zenith)
            }
            AnalyticTypeInner::Ska(inner) => {
                inner.calc_jones_device(azels, freqs_hz, latitude_rad, norm_to_zenith)
            }
        }
    }

    /// Given directions, calculate beam-response Jones matrices
    /// into the supplied pre-allocated device pointer. This buffer
    /// should have a shape of (`num_unique_tiles`, `num_freqs`,
    /// `az_rad_length`). The number of unique tiles can be accessed with
    /// [`AnalyticBeamGpu::get_num_unique_tiles`]. `d_latitude_rad` is
    /// populated with the array latitude, if the caller wants the parallactic-
    /// angle correction to be applied. If the pointer is null, then no
    /// correction is applied.
    ///
    /// # Safety
    ///
    /// If `d_results` is too small (correct size described above), then
    /// undefined behaviour looms.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn calc_jones_device_pair_inner(
        &self,
        d_az_rad: *const GpuFloat,
        d_za_rad: *const GpuFloat,
        num_directions: i32,
        d_freqs_hz: *const u32,
        num_freqs: i32,
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
        d_results: *mut std::ffi::c_void,
    ) -> Result<(), AnalyticBeamError> {
        match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => inner.calc_jones_device_pair_inner(
                d_az_rad,
                d_za_rad,
                num_directions,
                d_freqs_hz,
                num_freqs,
                latitude_rad,
                norm_to_zenith,
                d_results,
            ),
            AnalyticTypeInner::Ska(inner) => inner.calc_jones_device_pair_inner(
                d_az_rad,
                d_za_rad,
                num_directions,
                d_freqs_hz,
                num_freqs,
                latitude_rad,
                norm_to_zenith,
                d_results,
            ),
        }
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. The returned array
    /// is "expanded"; tile and frequency de-duplication is undone to give
    /// an array with the same number of tiles as was specified when this
    /// [`AnalyticBeamGpu`] was created and frequencies specified to this
    /// function.
    ///
    /// Note that this function needs to allocate two vectors for azimuths and
    /// zenith angles from the supplied `azels`.
    pub fn calc_jones(
        &self,
        azels: &[AzEl],
        freqs_hz: &[u32],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<Array3<Jones<GpuFloat>>, AnalyticBeamError> {
        let mut results: Array3<Jones<GpuFloat>> = match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => Array3::from_elem(
                (inner.tile_map.len(), freqs_hz.len(), azels.len()),
                Jones::default(),
            ),
            AnalyticTypeInner::Ska(inner) => Array3::from_elem(
                (inner.num_stations as usize, freqs_hz.len(), azels.len()),
                Jones::default(),
            ),
        };

        let (azs, zas): (Vec<GpuFloat>, Vec<GpuFloat>) = azels
            .iter()
            .map(|&azel| (azel.az as GpuFloat, azel.za() as GpuFloat))
            .unzip();
        self.calc_jones_pair_inner(
            &azs,
            &zas,
            freqs_hz,
            latitude_rad as GpuFloat,
            norm_to_zenith,
            results.view_mut(),
        )?;
        Ok(results)
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. The returned array
    /// is "expanded"; tile and frequency de-duplication is undone to give
    /// an array with the same number of tiles as was specified when this
    /// [`AnalyticBeamGpu`] was created and frequencies specified to this
    /// function.
    pub fn calc_jones_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        freqs_hz: &[u32],
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
    ) -> Result<Array3<Jones<GpuFloat>>, AnalyticBeamError> {
        let mut results = match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => Array3::from_elem(
                (inner.tile_map.len(), freqs_hz.len(), az_rad.len()),
                Jones::default(),
            ),
            AnalyticTypeInner::Ska(inner) => Array3::from_elem(
                (inner.num_stations as usize, freqs_hz.len(), az_rad.len()),
                Jones::default(),
            ),
        };

        self.calc_jones_pair_inner(
            az_rad,
            za_rad,
            freqs_hz,
            latitude_rad,
            norm_to_zenith,
            results.view_mut(),
        )?;
        Ok(results)
    }

    /// Given directions, calculate beam-response Jones matrices on the device,
    /// copy them to the host, and free the device memory. This function is
    /// the same as [`AnalyticBeamGpu::calc_jones_pair`], but the results are
    /// stored in a pre-allocated array. This array should have a shape of
    /// (`total_num_tiles`, `total_num_freqs`, `az_rad_length`). The first
    /// dimension can be accessed with `AnalyticBeamGpu::get_total_num_tiles`.
    pub fn calc_jones_pair_inner(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        freqs_hz: &[u32],
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
        mut results: ArrayViewMut3<Jones<GpuFloat>>,
    ) -> Result<(), AnalyticBeamError> {
        match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => inner.calc_jones_pair_inner(
                az_rad,
                za_rad,
                freqs_hz,
                latitude_rad,
                norm_to_zenith,
                results,
            ),
            AnalyticTypeInner::Ska(inner) => inner.calc_jones_pair_inner(
                az_rad,
                za_rad,
                freqs_hz,
                latitude_rad,
                norm_to_zenith,
                results,
            ),
        }
    }

    /// Get the number of tiles that this [`AnalyticBeamGpu`] applies to.
    pub fn get_total_num_tiles(&self) -> usize {
        match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => inner.tile_map.len(),
            AnalyticTypeInner::Ska(inner) => inner.num_stations as usize,
        }
    }

    // TODO: Update these functions below
    /// Get a pointer to the tile map associated with this
    /// [`AnalyticBeamGpu`]. This is necessary to access de-duplicated beam
    /// Jones matrices.
    pub fn get_tile_map(&self) -> *const i32 {
        match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => inner.tile_map.as_ptr(),
            AnalyticTypeInner::Ska(inner) => todo!(),
        }
    }

    /// Get a pointer to the device tile map associated with this
    /// [`AnalyticBeamGpu`]. This is necessary to access de-duplicated beam
    /// Jones matrices on the device.
    pub fn get_device_tile_map(&self) -> *const i32 {
        match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => inner.d_tile_map.get(),
            AnalyticTypeInner::Ska(inner) => todo!(),
        }
    }

    /// Get the number of de-duplicated tiles associated with this
    /// [`AnalyticBeamGpu`].
    pub fn get_num_unique_tiles(&self) -> i32 {
        match &self.analytic_type {
            AnalyticTypeInner::MwaRts(inner) => inner.num_unique_tiles,
            AnalyticTypeInner::Ska(inner) => todo!(),
        }
    }
}
