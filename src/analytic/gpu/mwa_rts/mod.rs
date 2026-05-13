use ndarray::ArrayView2;

use crate::{
    analytic::{AnalyticBeam, AnalyticBeamError},
    gpu::DevicePointer,
    GpuFloat,
};

/// A struct for holding relavent data for MWA analytic beams (both MwaPb and Rts)
pub(super) struct MwaRtsInner {
    pub dipole_height: GpuFloat,
    pub bowties_per_row: u8,
    pub d_delays: DevicePointer<GpuFloat>,
    pub d_amps: DevicePointer<GpuFloat>,
    pub(super) num_unique_tiles: i32,
    tile_map: Vec<i32>,
    d_tile_map: DevicePointer<i32>,
}

impl MwaRtsInner {
    pub(super) unsafe fn new(
        analytic_beam: &AnalyticBeam,
        delays_array: ArrayView2<u32>,
        amps_array: ArrayView2<f64>,
    ) -> Result<Self, AnalyticBeamError> {
        let num_bowties =
            usize::from(analytic_beam.bowties_per_row * analytic_beam.bowties_per_row);
        if delays_array.len_of(Axis(1)) != num_bowties {
            return Err(AnalyticBeamError::IncorrectDelaysArrayColLength {
                rows: delays_array.len_of(Axis(0)),
                num_delays: delays_array.len_of(Axis(1)),
                expected: num_bowties,
            });
        }
        if amps_array.len_of(Axis(1)) != num_bowties
            && amps_array.len_of(Axis(1)) != num_bowties * 2
        {
            return Err(AnalyticBeamError::IncorrectAmpsLength {
                got: amps_array.len_of(Axis(1)),
                expected1: num_bowties,
                expected2: num_bowties * 2,
            });
        }

        // Determine the unique tiles according to the gains and delays. Unlike
        // FEE, all frequencies give different results, so there's no need to
        // consider them.
        let mut unique_tiles = vec![];
        let mut tile_map = vec![];
        let mut i_tile = 0;
        let mut unique_delays = vec![];
        let mut unique_amps = vec![];
        for (delays, amps) in delays_array.outer_iter().zip(amps_array.outer_iter()) {
            let mut unique_tile_hasher = DefaultHasher::new();
            delays.hash(&mut unique_tile_hasher);
            // We can't hash f64 values, but we can hash their bits.
            for amp in amps {
                amp.to_bits().hash(&mut unique_tile_hasher);
            }
            let unique_tile_hash = unique_tile_hasher.finish();

            let (amps, delays) = fix_amps_ndarray(amps, delays);
            let (amps, delays) = if matches!(analytic_beam.beam_type, super::AnalyticType::Rts) {
                reorder_to_rts(&amps, &delays)
            } else {
                (amps.to_vec(), delay_ints_to_floats(&delays))
            };

            let this_tile_index = if let Some((index, _)) = unique_tiles
                .iter()
                .enumerate()
                .find(|(_, t)| **t == unique_tile_hash)
            {
                index.try_into().expect("smaller than i32::MAX")
            } else {
                unique_tiles.push(unique_tile_hash);
                unique_delays.extend(delays.iter().copied().map(|d| d as GpuFloat));
                unique_amps.extend(amps.iter().map(|&f| f as GpuFloat));
                i_tile += 1;
                i_tile - 1
            };
            tile_map.push(this_tile_index);
        }

        let d_tile_map = DevicePointer::copy_to_device(&tile_map)?;
        Ok(MwaRtsInner {
            dipole_height: analytic_beam.dipole_height as GpuFloat,
            bowties_per_row: analytic_beam.bowties_per_row,
            d_delays: DevicePointer::copy_to_device(&unique_delays)?,
            d_amps: DevicePointer::copy_to_device(&unique_amps)?,
            num_unique_tiles: unique_tiles
                .len()
                .try_into()
                .expect("smaller than i32::MAX"),
            tile_map,
            d_tile_map,
        })
    }
}
