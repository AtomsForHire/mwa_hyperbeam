use crate::{
    analytic::{AnalyticBeam, AnalyticBeamError},
    gpu::DevicePointer,
    GpuFloat,
};

use ndarray::prelude::*;

/// A struct for holding relavent data for SKA analytic beam
pub(super) struct SkaInner {
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
        todo!();
    }
}
