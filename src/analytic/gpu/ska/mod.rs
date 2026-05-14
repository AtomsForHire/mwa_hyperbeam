use crate::{
    analytic::{AnalyticBeam, AnalyticBeamError},
    gpu::DevicePointer,
    GpuFloat,
};

use marlu::Jones;
use ndarray::prelude::*;

/// A struct for holding relavent data for SKA analytic beam
pub(super) struct SkaInner {
    pub num_stations: i32,
    pub d_feed_coordinates: DevicePointer<GpuFloat>,
    pub d_feed_angles: DevicePointer<GpuFloat>,
    pub d_phase_centre_ra: GpuFloat,
    pub d_phase_centre_dec: GpuFloat,
    pub d_site_latitude_rad: GpuFloat,
}

impl SkaInner {
    pub(super) unsafe fn new(analytic_beam: &AnalyticBeam) -> Result<Self, AnalyticBeamError> {
        let ska_config = analytic_beam
            .ska_config
            .expect("SKA config is empty in SkaInner");

        let d_feed_coordinates =
            DevicePointer::copy_to_device(&ska_config.feed_coordinates.unwrap().into_raw_vec())?;

        let d_feed_angles =
            DevicePointer::copy_to_device(&ska_config.feed_angles_rad.unwrap().into_raw_vec())?;

        Ok(SkaInner {
            d_feed_coordinates,
            d_feed_angles,
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
        let error_message_ptr = gpu_analytic_calc_jones(
            match self.analytic_type {
                super::AnalyticType::MwaPb => ANALYTIC_TYPE_MWA_PB,
                super::AnalyticType::Rts => ANALYTIC_TYPE_RTS,
                _ => unreachable!(), // NOTE: Should be unreachable, since this submodule is
                                     // only for MwaPb or Rts types.
            },
            self.dipole_height,
            d_az_rad,
            d_za_rad,
            num_directions,
            d_freqs_hz,
            num_freqs,
            self.d_delays.get(),
            self.d_amps.get(),
            self.num_unique_tiles,
            latitude_rad,
            norm_to_zenith as _,
            self.bowties_per_row,
            d_results,
        );
        if error_message_ptr.is_null() {
            Ok(())
        } else {
            let error_message = CStr::from_ptr(error_message_ptr)
                .to_str()
                .unwrap_or("<cannot read GPU error string>");
            let our_error_str =
                format!("analytic.h:analytic_calc_jones_gpu failed with: {error_message}");
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
        let results: Array3<Jones<GpuFloat>> = Array3::from_elem(
            (self.num_stations as usize, freqs_hz.len(), az_rad.len()),
            elem,
        );

        // 2. Calculate beam responses
        let device_ptr = 0;
        todo!();
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
