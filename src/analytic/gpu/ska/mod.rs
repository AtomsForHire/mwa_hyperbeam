// Include Rust bindings to the GPU code, depending on the precision used.
#[cfg(feature = "gpu-single")]
include!("single.rs");
#[cfg(not(feature = "gpu-single"))]
include!("double.rs");

use marlu::{AzEl, Jones};
use ndarray::prelude::*;
use ndarray::ArrayView2;
use std::{
    collections::hash_map::DefaultHasher,
    convert::TryInto,
    ffi::CStr,
    hash::{Hash, Hasher},
};

use crate::{
    analytic::{AnalyticBeam, AnalyticBeamError, AnalyticType},
    gpu::{DevicePointer, GpuError, GpuFloat},
};

/// A struct for holding relavent data for SKA analytic beam
pub(crate) struct SkaInner {
    pub num_stations: i32,
    pub d_feed_coordinates: DevicePointer<GpuFloat>,
    pub d_feed_angles: DevicePointer<GpuFloat>,
    pub d_num_elems_per_station: DevicePointer<i32>,
    pub d_phase_centre_ra: GpuFloat,
    pub d_phase_centre_dec: GpuFloat,
    pub d_site_latitude_rad: GpuFloat,
}

impl SkaInner {
    pub(super) unsafe fn new(analytic_beam: &AnalyticBeam) -> Result<Self, AnalyticBeamError> {
        let ska_config = analytic_beam
            .ska_config
            .clone()
            .expect("SKA config is empty in SkaInner");

        let flat_coords_vec: Vec<GpuFloat> = ska_config
            .clone()
            .feed_coordinates
            .unwrap()
            .iter()
            .flat_map(|a| a.iter().copied())
            .collect();

        let d_feed_coordinates = DevicePointer::copy_to_device(&flat_coords_vec)?;

        let flat_angles_vec: Vec<GpuFloat> = ska_config
            .clone()
            .feed_angles_rad
            .unwrap()
            .iter()
            .flat_map(|a| a.iter().copied())
            .collect();

        let d_feed_angles = DevicePointer::copy_to_device(&flat_angles_vec)?;

        let d_num_elems_per_station = DevicePointer::copy_to_device(
            &ska_config
                .clone()
                .num_elems_per_station
                .unwrap()
                .into_iter()
                .map(|x| x as i32)
                .collect::<Vec<i32>>(),
        )?;

        Ok(SkaInner {
            num_stations: ska_config.number_of_stations as i32,
            d_feed_coordinates,
            d_feed_angles,
            d_num_elems_per_station,
            d_phase_centre_ra: ska_config.phase_centre.ra as GpuFloat,
            d_phase_centre_dec: ska_config.phase_centre.dec as GpuFloat,
            d_site_latitude_rad: ska_config.site_latitude_rad as GpuFloat,
        })
    }

    /// Function to do stuff on the host first, before passing to a separate function to handle the
    /// kernel call
    pub fn calc_jones_device_pair(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        freqs_hz: &[u32],
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
    ) -> Result<DevicePointer<Jones<GpuFloat>>, AnalyticBeamError> {
        unsafe {
            // Allocate a buffer on the device for results.
            let d_results = DevicePointer::malloc(
                self.num_stations as usize
                    * freqs_hz.len()
                    * az_rad.len()
                    * std::mem::size_of::<Jones<GpuFloat>>(),
            )?;

            // Also copy the directions to the device.
            let d_azs = DevicePointer::copy_to_device(az_rad)?;
            let d_zas = DevicePointer::copy_to_device(za_rad)?;
            let d_freqs = DevicePointer::copy_to_device(freqs_hz)?;

            self.calc_jones_device_pair_inner(
                d_azs.get(),
                d_zas.get(),
                az_rad.len().try_into().expect("much fewer than i32::MAX"),
                d_freqs.get(),
                freqs_hz.len().try_into().expect("much fewer than i32::MAX"),
                latitude_rad,
                norm_to_zenith,
                d_results.get_mut() as *mut std::ffi::c_void,
            )?;
            Ok(d_results)
        }
    }

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
        // Don't do anything if there aren't any directions.
        if num_directions == 0 {
            return Ok(());
        }

        // The return value is a pointer to a CUDA/HIP error string. If it's
        // null then everything is fine.
        let error_message_ptr = ska_gpu_analytic_calc_jones(
            ANALYTIC_TYPE_SKA,
            d_az_rad,
            d_za_rad,
            num_directions,
            d_freqs_hz,
            num_freqs,
            self.d_phase_centre_ra,
            self.d_phase_centre_dec,
            self.num_stations,
            self.d_feed_coordinates.get(),
            self.d_feed_angles.get(),
            self.d_num_elems_per_station.get(),
            latitude_rad, // NOTE: This should hopefully be lst_rad
            self.d_site_latitude_rad,
            norm_to_zenith as _,
            d_results,
        );
        if error_message_ptr.is_null() {
            Ok(())
        } else {
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read GPU error string>");
            let our_error_str =
                format!("ska_analytic.h:ska_analytic_calc_jones_gpu failed with: {error_message}");
            Err(AnalyticBeamError::Gpu(GpuError::Kernel {
                msg: our_error_str.into(),
                file: file!(),
                line: line!(),
            }))
        }
    }
}

impl super::CalcJones for SkaInner {
    fn calc_jones_pair_inner(
        &self,
        az_rad: &[GpuFloat],
        za_rad: &[GpuFloat],
        freqs_hz: &[u32],
        latitude_rad: GpuFloat,
        norm_to_zenith: bool,
        mut results: ArrayViewMut3<marlu::Jones<GpuFloat>>,
    ) -> Result<(), AnalyticBeamError> {
        // NOTE: Trying to follow the steps in the `calc_jones_pair_inner` function of the MWA
        // analytic beam, found in ../mwa_rts/mod.rs

        // 1. Allocate memory on host matching memory on device
        let mut temp_results: Array3<Jones<GpuFloat>> = Array3::from_elem(
            (self.num_stations as usize, freqs_hz.len(), az_rad.len()),
            Jones::default(),
        );

        // 2. Calculate beam responses
        let device_ptr =
            self.calc_jones_device_pair(az_rad, za_rad, freqs_hz, latitude_rad, norm_to_zenith)?;

        // 3. Copy from device to host
        unsafe {
            device_ptr.copy_from_device(temp_results.as_slice_mut().expect("is contiguous"))?;
        }
        drop(device_ptr);

        results
            .outer_iter_mut() // iterate over Axis(0)
            .enumerate()
            .for_each(|(row, mut jones_row)| {
                jones_row.assign(&temp_results.slice(s![row, .., ..]))
            });
        Ok(())
    }

    fn calc_jones_device(
        &self,
        azels: &[marlu::AzEl],
        freqs_hz: &[u32],
        latitude_rad: f64,
        norm_to_zenith: bool,
    ) -> Result<DevicePointer<marlu::Jones<GpuFloat>>, AnalyticBeamError> {
        todo!();
    }
}
