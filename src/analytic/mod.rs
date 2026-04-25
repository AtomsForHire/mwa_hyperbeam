// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Code for the analytic MWA beam.

mod error;
mod ffi;
#[cfg(any(feature = "cuda", feature = "hip"))]
mod gpu;
#[cfg(test)]
mod tests;

pub use error::AnalyticBeamError;

#[cfg(any(feature = "cuda", feature = "hip"))]
pub use gpu::AnalyticBeamGpu;
use ndarray::Array2;

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use marlu::{c64, constants::VEL_C, rayon, AzEl, Jones, RADec};
use rayon::prelude::*;

use crate::constants::{DELAY_STEP, MWA_DPL_SEP};
use num_complex::Complex;

#[cfg(any(feature = "cuda", feature = "hip"))]
use ndarray::prelude::*;

/// Which analytic beam code are we emulating?
#[derive(Clone, Copy, Debug)]
pub enum AnalyticType {
    /// Behaviour derived from [mwa_pb](https://github.com/MWATelescope/mwa_pb).
    MwaPb,

    /// Behaviour derived from the RTS.
    Rts,

    /// SKA-Low array factor beam
    Ska,

    ///
    SkaMean,
}

impl AnalyticType {
    /// Different analytic beam types use different MWA dipole heights by
    /// default. This method returns the default height given a analytic beam
    /// type.
    pub fn get_default_dipole_height(self) -> f64 {
        match self {
            AnalyticType::MwaPb => 0.278,
            AnalyticType::Rts => 0.30,
            AnalyticType::Ska => 0.00, // Array factor does not need height
            AnalyticType::SkaMean => 0.00, // Array factor does not need height
        }
    }
}

/// A struct for specifically holding SKA information
#[derive(Clone)]
pub struct SkaConfig {
    /// Number of stations in array
    pub number_of_stations: usize,

    /// Rotation angle for each station
    pub feed_angles_rad: Option<Vec<Vec<f64>>>,

    /// Coordinates of feeds within each station
    pub feed_coordinates: Option<Vec<Array2<f64>>>,

    /// Needed for SKA logic
    pub phase_centre: RADec,

    pub site_latitude_rad: f64,
}

/// The main struct to be used for calculating analytic pointings.
pub struct AnalyticBeam {
    /// The height of the MWA dipoles we're simulating \[metres\].
    ///
    /// The RTS uses an old value, presumably derived from early MWA dipoles.
    /// The up-to-date value is 0.278m, and is used by default.
    dipole_height: f64,

    /// Which analytic beam code are we emulating?
    beam_type: AnalyticType,

    /// The number of bowties in a row of an MWA tile. Almost all MWA tiles
    /// have 4 bowties per row, for a total of 16 bowties. As of October 2023,
    /// the only exception is the CRAM tile, which has 8 bowties per row, for a
    /// total of 64 bowties.
    // ERIC NOTE: Probably just set this to default 16, for the SKA case. Don't want to break it
    // by making this an option.
    pub(crate) bowties_per_row: u8,

    // ERIC NOTE: The information I need for the ska beam.
    pub(crate) ska_config: Option<SkaConfig>,
}

impl Default for AnalyticBeam {
    fn default() -> Self {
        let beam_type = AnalyticType::MwaPb;
        AnalyticBeam {
            dipole_height: beam_type.get_default_dipole_height(),
            beam_type,
            bowties_per_row: 4,
            ska_config: None,
        }
    }
}

impl AnalyticBeam {
    /// Create a new [`AnalyticBeam`] struct using mwa_pb analytic beam code and
    /// the [default](MWA_DPL_HGT) MWA dipole height.
    pub fn new() -> AnalyticBeam {
        AnalyticBeam::default()
    }

    /// Create a new [`AnalyticBeam`] struct using RTS analytic beam code and
    /// the MWA dipole height from the [RTS](MWA_DPL_HGT_RTS).
    pub fn new_rts() -> AnalyticBeam {
        let beam_type = AnalyticType::Rts;
        AnalyticBeam {
            dipole_height: beam_type.get_default_dipole_height(),
            beam_type,
            bowties_per_row: 4,
            ska_config: None,
        }
    }

    pub fn new_ska(ska_config: SkaConfig) -> AnalyticBeam {
        let beam_type = AnalyticType::Ska;
        AnalyticBeam {
            dipole_height: beam_type.get_default_dipole_height(),
            beam_type,
            bowties_per_row: 0,
            ska_config: Some(ska_config),
        }
    }

    /// Create a new [`AnalyticBeam`] struct with custom behaviour, MWA
    /// dipole height and variable bowties per row (you want this to be 4 for
    /// normal MWA tiles, 8 for the CRAM).
    pub fn new_custom(
        beam_type: AnalyticType,
        dipole_height_metres: f64,
        bowties_per_row: u8,
    ) -> AnalyticBeam {
        if bowties_per_row == 0 {
            panic!("bowties_per_row was 0, why would you do that?");
        }
        // We want the number of bowties to fit in a u8; complain if there are
        // too many damn bowties.
        if bowties_per_row >= 16 {
            panic!("bowties_per_row is restricted to be less than 16");
        }
        AnalyticBeam {
            dipole_height: dipole_height_metres,
            beam_type,
            bowties_per_row,
            ska_config: None,
        }
    }

    /// Calculate the beam-response Jones matrix for a given direction, pointing
    /// and latitude.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    pub fn calc_jones(
        &self,
        azel: AzEl,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        tile_index: Option<usize>,
    ) -> Result<Jones<f64>, AnalyticBeamError> {
        self.calc_jones_pair(
            azel.az,
            azel.za(),
            freq_hz,
            delays,
            amps,
            latitude_rad,
            norm_to_zenith,
            tile_index,
        )
    }

    /// Calculate the beam-response Jones matrix for a given direction and
    /// pointing.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_pair(
        &self,
        az_rad: f64,
        za_rad: f64,
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        tile_index: Option<usize>,
    ) -> Result<Jones<f64>, AnalyticBeamError> {
        if za_rad > FRAC_PI_2 {
            return Err(AnalyticBeamError::BelowHorizon { za: za_rad });
        }

        // Validate delay length only if NOT Ska analytic beam type
        if !matches!(self.beam_type, AnalyticType::Ska) {
            let num_bowties = usize::from(self.bowties_per_row * self.bowties_per_row);
            if delays.len() != num_bowties {
                return Err(AnalyticBeamError::IncorrectDelaysLength {
                    got: delays.len(),
                    expected: num_bowties,
                });
            }
            if amps.len() != num_bowties && amps.len() != num_bowties * 2 {
                return Err(AnalyticBeamError::IncorrectAmpsLength {
                    got: amps.len(),
                    expected1: num_bowties,
                    expected2: num_bowties * 2,
                });
            }
        }

        let amps = fix_amps(amps, delays);
        // let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
        //     reorder_to_rts(&amps, delays)
        // } else {
        //     (amps.to_vec(), delay_ints_to_floats(delays))
        // };
        let (amps, delays) = match self.beam_type {
            AnalyticType::Rts => reorder_to_rts(&amps, delays),
            AnalyticType::MwaPb => (amps.to_vec(), delay_ints_to_floats(delays)),
            AnalyticType::Ska => (vec![], vec![]), // Don't need amps or delays for SKA array
            // factor logic
            AnalyticType::SkaMean => (vec![], vec![]),
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        let jones = self.calc_jones_inner(
            az_rad,
            za_rad,
            lambda_m,
            latitude_rad,
            s_lat,
            c_lat,
            &delays,
            &amps,
            norm_to_zenith,
            tile_index,
        );

        Ok(jones)
    }

    /// Calculate the beam-response Jones matrices for many directions
    /// given a pointing and latitude. This is basically a wrapper around
    /// `calc_jones` that efficiently calculates the Jones matrices in
    /// parallel. The number of parallel threads used can be controlled by
    /// setting `RAYON_NUM_THREADS`.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    pub fn calc_jones_array(
        &self,
        azels: &[AzEl],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        tile_index: Option<usize>,
    ) -> Result<Vec<Jones<f64>>, AnalyticBeamError> {
        let mut results = vec![Jones::default(); azels.len()];
        self.calc_jones_array_inner(
            azels,
            freq_hz,
            delays,
            amps,
            latitude_rad,
            norm_to_zenith,
            &mut results,
            tile_index,
        )?;

        Ok(results)
    }

    /// Calculate the Jones matrices for many directions given a pointing and
    /// latitude. This is the same as `calc_jones_array` but uses pre-allocated
    /// memory.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_inner(
        &self,
        azels: &[AzEl],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        results: &mut [Jones<f64>],
        tile_index: Option<usize>,
    ) -> Result<(), AnalyticBeamError> {
        for azel in azels {
            let za = azel.za();
            if za > FRAC_PI_2 {
                return Err(AnalyticBeamError::BelowHorizon { za });
            }
        }

        match self.beam_type {
            AnalyticType::MwaPb | AnalyticType::Rts => {
                let num_bowties = usize::from(self.bowties_per_row * self.bowties_per_row);
                if delays.len() != num_bowties {
                    return Err(AnalyticBeamError::IncorrectDelaysLength {
                        got: delays.len(),
                        expected: num_bowties,
                    });
                }
                if amps.len() != num_bowties && amps.len() != num_bowties * 2 {
                    return Err(AnalyticBeamError::IncorrectAmpsLength {
                        got: amps.len(),
                        expected1: num_bowties,
                        expected2: num_bowties * 2,
                    });
                }
            }
            AnalyticType::Ska => {
                // Do nothing
                ();
            }
            AnalyticType::SkaMean => {
                // Do nothing
                ();
            }
        };

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps.to_vec(), delay_ints_to_floats(delays))
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        azels
            .par_iter()
            .zip(results.par_iter_mut())
            .try_for_each(|(&azel, result)| {
                if azel.za() > FRAC_PI_2 {
                    return Err(AnalyticBeamError::BelowHorizon { za: azel.za() });
                }

                let j = self.calc_jones_inner(
                    azel.az,
                    azel.za(),
                    lambda_m,
                    latitude_rad,
                    s_lat,
                    c_lat,
                    &delays,
                    &amps,
                    norm_to_zenith,
                    tile_index,
                );

                *result = j;

                Ok(())
            })
    }

    /// Calculate the beam-response Jones matrices for many directions given a
    /// pointing. This is basically a wrapper around `calc_jones` that
    /// efficiently calculates the Jones matrices in parallel. The number of
    /// parallel threads used can be controlled by setting `RAYON_NUM_THREADS`.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_pair(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        tile_index: Option<usize>,
    ) -> Result<Vec<Jones<f64>>, AnalyticBeamError> {
        for &za in za_rad {
            if za > FRAC_PI_2 {
                return Err(AnalyticBeamError::BelowHorizon { za });
            }
        }

        match self.beam_type {
            AnalyticType::MwaPb | AnalyticType::Rts => {
                let num_bowties = usize::from(self.bowties_per_row * self.bowties_per_row);
                if delays.len() != num_bowties {
                    return Err(AnalyticBeamError::IncorrectDelaysLength {
                        got: delays.len(),
                        expected: num_bowties,
                    });
                }
                if amps.len() != num_bowties && amps.len() != num_bowties * 2 {
                    return Err(AnalyticBeamError::IncorrectAmpsLength {
                        got: amps.len(),
                        expected1: num_bowties,
                        expected2: num_bowties * 2,
                    });
                }
            }
            AnalyticType::Ska => (),
            AnalyticType::SkaMean => (),
        }

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps.to_vec(), delay_ints_to_floats(delays))
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        let out = az_rad
            .par_iter()
            .zip(za_rad.par_iter())
            .map(|(&az, &za)| {
                self.calc_jones_inner(
                    az,
                    za,
                    lambda_m,
                    latitude_rad,
                    s_lat,
                    c_lat,
                    &delays,
                    &amps,
                    norm_to_zenith,
                    tile_index,
                )
            })
            .collect();
        Ok(out)
    }

    /// Calculate the Jones matrices for many directions given a pointing. This
    /// is the same as `calc_jones_array_pair` but uses pre-allocated memory.
    ///
    /// `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    /// order; see
    /// <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    /// `delays` *must* have `bowties_per_row * bowties_per_row` elements (which
    /// was declared when `AnalyticBeam` was created), whereas `amps` can have
    /// this number or double elements; if the former is given, then these map
    /// 1:1 with bowties. If double are given, then the *smallest* of the two
    /// amps corresponding to a bowtie's dipoles is used.
    ///
    /// e.g. A normal MWA tile has 4 bowties per row. `delays` must then have
    /// 16 elements, and `amps` can have 16 or 32 elements. A CRAM tile has 8
    /// bowties per row; `delays` must have 64 elements, and `amps` can have 64
    /// or 128 elements.
    #[allow(clippy::too_many_arguments)]
    pub fn calc_jones_array_pair_inner(
        &self,
        az_rad: &[f64],
        za_rad: &[f64],
        freq_hz: u32,
        delays: &[u32],
        amps: &[f64],
        latitude_rad: f64,
        norm_to_zenith: bool,
        results: &mut [Jones<f64>],
        tile_index: Option<usize>,
    ) -> Result<(), AnalyticBeamError> {
        for &za in za_rad {
            if za > FRAC_PI_2 {
                return Err(AnalyticBeamError::BelowHorizon { za });
            }
        }
        let num_bowties = usize::from(self.bowties_per_row * self.bowties_per_row);

        match self.beam_type {
            AnalyticType::MwaPb | AnalyticType::Rts => {
                if delays.len() != num_bowties {
                    return Err(AnalyticBeamError::IncorrectDelaysLength {
                        got: delays.len(),
                        expected: num_bowties,
                    });
                }
                if amps.len() != num_bowties && amps.len() != num_bowties * 2 {
                    return Err(AnalyticBeamError::IncorrectAmpsLength {
                        got: amps.len(),
                        expected1: num_bowties,
                        expected2: num_bowties * 2,
                    });
                }
            }
            AnalyticType::Ska => (),
            AnalyticType::SkaMean => (),
        };

        let amps = fix_amps(amps, delays);
        let (amps, delays) = if matches!(self.beam_type, AnalyticType::Rts) {
            reorder_to_rts(&amps, delays)
        } else {
            (amps.to_vec(), delay_ints_to_floats(delays))
        };

        let lambda_m = VEL_C / freq_hz as f64;
        let (s_lat, c_lat) = latitude_rad.sin_cos();
        az_rad
            .par_iter()
            .zip(za_rad.par_iter())
            .zip(results.par_iter_mut())
            .try_for_each(|((&az, &za), result)| {
                if za > FRAC_PI_2 {
                    return Err(AnalyticBeamError::BelowHorizon { za });
                }

                let j = self.calc_jones_inner(
                    az,
                    za,
                    lambda_m,
                    latitude_rad,
                    s_lat,
                    c_lat,
                    &delays,
                    &amps,
                    norm_to_zenith,
                    tile_index,
                );

                *result = j;

                Ok(())
            })
    }

    /// Helper function.
    // The code here was derived with the help of primary_beam.py in mwa_pb,
    // commit 8619797, and Jack's WODEN.
    #[allow(clippy::too_many_arguments)]
    fn calc_jones_inner(
        &self,
        az_rad: f64,
        za_rad: f64,
        lambda_m: f64,
        latitude_rad: f64,
        sin_latitude: f64,
        cos_latitude: f64,
        delays: &[f64],
        amps: &[f64],
        norm_to_zenith: bool,
        tile_index: Option<usize>,
    ) -> Jones<f64> {
        // The following logic could probably be significantly cleaned up, but
        // I'm out of time.

        let (s_az, c_az) = az_rad.sin_cos();
        let (s_za, c_za) = za_rad.sin_cos();

        match self.beam_type {
            AnalyticType::Rts | AnalyticType::MwaPb => {
                let mut jones = match self.beam_type {
                    AnalyticType::MwaPb => Jones::from([
                        c64::new(c_za * s_az, 0.0),
                        c64::new(c_az, 0.0),
                        c64::new(c_za * c_az, 0.0),
                        c64::new(-s_az, 0.0),
                    ]),
                    AnalyticType::Rts => {
                        let hadec =
                            AzEl::from_radians(az_rad, FRAC_PI_2 - za_rad).to_hadec(latitude_rad);
                        let (s_ha, c_ha) = hadec.ha.sin_cos();
                        let (s_dec, c_dec) = hadec.dec.sin_cos();

                        Jones::from([
                            c64::new(cos_latitude * c_dec + sin_latitude * s_dec * c_ha, 0.0),
                            c64::new(-sin_latitude * s_ha, 0.0),
                            c64::new(s_dec * s_ha, 0.0),
                            c64::new(c_ha, 0.0),
                        ])
                    }
                    AnalyticType::Ska => {
                        unreachable!("This should be unreachable");
                    }
                    AnalyticType::SkaMean => {
                        unreachable!("This should be unreachable");
                    }
                };

                let proj_e = s_za * s_az;
                let proj_n = s_za * c_az;
                // The RTS code uses proj_z as below, but dip_z is always set to 0.0, so
                // we don't actually need proj_z. lmao
                // let proj_z = c_za;

                let multiplier = -TAU / lambda_m;

                // Loop over each dipole.
                let mut array_factor = c64::new(0.0, 0.0);
                for (k, (&delay, &amp)) in delays.iter().zip(amps.iter()).enumerate() {
                    let col = k % usize::from(self.bowties_per_row);
                    let row = k / usize::from(self.bowties_per_row);
                    let (dip_e, dip_n) = match self.beam_type {
                        AnalyticType::MwaPb => (
                            (col as f64 - 1.5) * MWA_DPL_SEP,
                            (row as f64 - 1.5) * MWA_DPL_SEP,
                        ),
                        AnalyticType::Rts => (
                            (row as f64 - 1.5) * MWA_DPL_SEP,
                            (col as f64 - 1.5) * MWA_DPL_SEP,
                        ),
                        AnalyticType::Ska => {
                            unreachable!("This should be unreachable");
                        }
                        AnalyticType::SkaMean => {
                            unreachable!("This should be unreachable");
                        }
                    };
                    // let dip_z = 0.0;

                    let phase = match self.beam_type {
                        AnalyticType::MwaPb => {
                            -multiplier
                                * (dip_e * proj_e
                         + dip_n * proj_n
                         // + dip_z * proj_z
                         - delay)
                        }
                        AnalyticType::Rts => {
                            multiplier
                                * (dip_e * proj_e
                         + dip_n * proj_n
                         // + dip_z * proj_z
                         - delay)
                        }
                        AnalyticType::Ska => {
                            unreachable!("This should be unreachable");
                        }
                        AnalyticType::SkaMean => {
                            unreachable!("This should be unreachable");
                        }
                    };
                    let (s_phase, c_phase) = phase.sin_cos();
                    array_factor += amp * c64::new(c_phase, s_phase);
                }

                let mut ground_plane = 2.0 * (TAU * self.dipole_height / lambda_m * c_za).sin()
                    / usize::from(self.bowties_per_row).pow(2) as f64;
                if norm_to_zenith {
                    ground_plane /= 2.0 * (TAU * self.dipole_height / lambda_m).sin();
                }

                jones[0] *= ground_plane * array_factor;
                jones[1] *= ground_plane * array_factor;
                jones[2] *= ground_plane * array_factor;
                jones[3] *= ground_plane * array_factor;

                // The RTS deliberately sets the imaginary parts to 0.
                if matches!(self.beam_type, AnalyticType::Rts) {
                    for j in jones.iter_mut() {
                        *j = c64::new(j.re, 0.0);
                    }
                }

                jones
            }
            AnalyticType::Ska => {
                let ska_config = self
                    .ska_config
                    .clone()
                    .expect("Somehow AnalyticType::Ska has ended up without needed SKA data!");

                // latitude_rad should be lst_rad
                let lst_rad = latitude_rad;

                let index =
                    tile_index.expect("Error! tile_index is needed for array factor beam forming");
                // let index = tile_index.unwrap_or(0 as usize); // Uncomment this for debugging, lets
                // program run all the way through

                // Feed angles, euler angles, azimutal angles from x to y, N of E. Two elements [x, y]
                let feed_angles_rad = &ska_config
                    .feed_angles_rad
                    .expect("Somehow ended up with no feed_angles_rad in Ska logic");
                let phi_pq: &Vec<f64> = &feed_angles_rad[index];

                // get element coordinates and transformation matrix for station 'index'
                let feed_coordinates = &ska_config
                    .feed_coordinates
                    .expect("Somehow ended up without feed coordinates in Ska logic");
                let coordinates: &Array2<f64> = &feed_coordinates[index];

                let num_elems = coordinates.nrows();

                // NOTE: Some hack fixes =====================================
                // TODO: These were taken from LLMs, was really frustrated, just needed something.
                // NEED TO CHECK LATER
                let site_latitude_rad = ska_config.site_latitude_rad;
                let zenith_radec = RADec {
                    ra: lst_rad,
                    dec: site_latitude_rad,
                };

                let hadec =
                    AzEl::from_radians(az_rad, FRAC_PI_2 - za_rad).to_hadec(site_latitude_rad);

                let beam_radec = hadec.to_radec(lst_rad);

                let beam_lmn = beam_radec.to_lmn(zenith_radec);
                let cent_lmn = ska_config.phase_centre.to_lmn(zenith_radec);

                let dl = beam_lmn.l - cent_lmn.l;
                let dm = beam_lmn.m - cent_lmn.m;

                // NOTE: End of hack fixes ====================================

                // 1. Station rotation
                // The station rotation information, when using the array factor method, is already
                // implicitly included in the coordinates of the elements. We do not need to apply extra
                // rotation for it.
                // NOTE: The array factor is a *scalar* complex quantity, multiply this array factor by the
                // element factor
                // Notation is a bit confusing:
                // 1. We form the array factor with (l, m) coordinates not (theta, phi)
                // 2. station_beam_x_theta is the voltage pattern for the array of x-dipoles
                //    It describes the array's whole x-dipole response to a signal coming from (l, m)
                //    NOTE: But how does it know to describe the response to (x, y) or (theta, phi)
                //    components of the electric field?
                let mut array_factor = Complex::from(0.0);

                for i in 0..num_elems {
                    let x_loc = coordinates[[i, 0]];
                    let y_loc = coordinates[[i, 1]];
                    assert!(
                        coordinates[[i, 2]].abs() < 1e-10,
                        "z-coordinate of station coordinates is not close to 0: {:?}, {:?}, {:?}",
                        coordinates[[i, 0]].abs(),
                        coordinates[[i, 1]].abs(),
                        coordinates[[i, 2]].abs()
                    );

                    // Add up phases
                    // let tot_phase = (-x_loc / lambda_m * (beam_l - cent_l)
                    //     + y_loc / lambda_m * (beam_m - cent_m));
                    let tot_phase = (-x_loc * dl + y_loc * dm) / lambda_m;

                    let angle = -2.0 * PI * tot_phase;
                    array_factor += Complex::from_polar(1.0, angle);
                }

                // Normalise complex Array Factor
                let af_norm = array_factor / num_elems as f64;

                // 1.1 Embedded Element Pattern for crossed dipoles
                // This is assuming the dipoles are aligned with the x and y axis. i.e. NO ROTATION!
                let phi = FRAC_PI_2 - az_rad;
                let theta = za_rad;

                // The phi angle is different for both p and q dipoles because q is rotated 90 degrees
                // (usually). In Hyperdrive, use MWA-convention i.e. x/p dipole is aligned EW and q is
                // aligned NS.
                let phi_p = phi;
                let phi_q = phi + PI / 2.0;

                let denom_p = self.calc_half_wavelength_dipole_denom(theta, phi_p);
                let denom_q = self.calc_half_wavelength_dipole_denom(theta, phi_q);

                let kl: f64 = PI / 2.0; // By default OSKAR uses a dipole length of 0.5 wavelengths, so
                                        // the expression for kL simplifies to pi/2.0

                //let dipole_length = 0.5; // Default dipole length used in OSKAR
                //let kl: f64 = dipole_length * PI * (freq_hz / SPEED_OF_LIGHT);
                let numer_p = (kl * phi_p.cos() * theta.sin()).cos() - kl.cos();
                let numer_q = (kl * (phi_q).cos() * theta.sin()).cos() - kl.cos();

                let e_p_theta = (-phi_p.cos() * theta.cos() * numer_p) / denom_p * af_norm;
                let e_p_phi = (phi_p.sin() * numer_p) / denom_p * af_norm;

                let e_q_theta = (-(phi_q).cos() * theta.cos() * numer_q) / denom_q * af_norm;
                let e_q_phi = ((phi_q).sin() * numer_q) / denom_q * af_norm;

                // Steps outlined in hyperbeam fee_pols.pdf is specifically made for the FEE MWA beam.
                // We do not use the FEE mwa beam here, so don't follow it.
                // 1. Construct Jones matrix 'B'
                let b = Jones::from([e_p_theta, e_p_phi, e_q_theta, e_q_phi]);

                return b;
            }
            AnalyticType::SkaMean => {
                let ska_config = self
                    .ska_config
                    .clone()
                    .expect("Somehow AnalyticType::Ska has ended up without needed SKA data!");

                let lst_rad = latitude_rad;

                let index =
                    tile_index.expect("Error! tile_index is needed for array factor beam forming");

                let feed_angles_rad = &ska_config
                    .feed_angles_rad
                    .expect("Somehow ended up with no feed_angles_rad in Ska logic");
                let phi_pq: &Vec<f64> = &feed_angles_rad[index];

                let feed_coordinates = &ska_config
                    .feed_coordinates
                    .expect("Somehow ended up without feed coordinates in Ska logic");

                let num_elems = coordinates.nrows();

                let site_latitude_rad = ska_config.site_latitude_rad;
                let zenith_radec = RADec {
                    ra: lst_rad,
                    dec: site_latitude_rad,
                };

                let hadec =
                    AzEl::from_radians(az_rad, FRAC_PI_2 - za_rad).to_hadec(site_latitude_rad);

                let beam_radec = hadec.to_radec(lst_rad);

                let beam_lmn = beam_radec.to_lmn(zenith_radec);
                let cent_lmn = ska_config.phase_centre.to_lmn(zenith_radec);

                let dl = beam_lmn.l - cent_lmn.l;
                let dm = beam_lmn.m - cent_lmn.m;

                let mut array_factor_mean = Complex::from(0.0);
                let num_stations = ska_config.number_of_stations;

                for j in 0..num_stations {
                    let mut array_factor_station = Complex::from(0.0);
                    let coordinates: &Array2<f64> = &feed_coordinates[j];
                    for i in 0..num_elems {
                        let x_loc = coordinates[[i, 0]];
                        let y_loc = coordinates[[i, 1]];
                        assert!(
                        coordinates[[i, 2]].abs() < 1e-10,
                        "z-coordinate of station coordinates is not close to 0: {:?}, {:?}, {:?}",
                        coordinates[[i, 0]].abs(),
                        coordinates[[i, 1]].abs(),
                        coordinates[[i, 2]].abs()
                    );

                        let tot_phase = (-x_loc * dl + y_loc * dm) / lambda_m;

                        let angle = -2.0 * PI * tot_phase;
                        array_factor_station += Complex::from_polar(1.0, angle);
                    }

                    let af_norm = array_factor_station / num_elems as f64;

                    array_factor_mean += array_factor_station;
                }

                array_factor_mean /= num_stations as f64;

                let phi = FRAC_PI_2 - az_rad;
                let theta = za_rad;

                let phi_p = phi;
                let phi_q = phi + PI / 2.0;

                let denom_p = self.calc_half_wavelength_dipole_denom(theta, phi_p);
                let denom_q = self.calc_half_wavelength_dipole_denom(theta, phi_q);

                let kl: f64 = PI / 2.0;

                let numer_p = (kl * phi_p.cos() * theta.sin()).cos() - kl.cos();
                let numer_q = (kl * (phi_q).cos() * theta.sin()).cos() - kl.cos();

                let e_p_theta =
                    (-phi_p.cos() * theta.cos() * numer_p) / denom_p * array_factor_mean;
                let e_p_phi = (phi_p.sin() * numer_p) / denom_p * array_factor_mean;

                let e_q_theta =
                    (-(phi_q).cos() * theta.cos() * numer_q) / denom_q * array_factor_mean;
                let e_q_phi = ((phi_q).sin() * numer_q) / denom_q * array_factor_mean;

                // Steps outlined in hyperbeam fee_pols.pdf is specifically made for the FEE MWA beam.
                // We do not use the FEE mwa beam here, so don't follow it.
                // 1. Construct Jones matrix 'B'
                let b = Jones::from([e_p_theta, e_p_phi, e_q_theta, e_q_phi]);

                return b;
            }
        }
    }

    /// Calculate the denominator that is common to both E_phi and E_theta components, when using a
    /// half-wavelength dipole (as OSKAR does)
    fn calc_half_wavelength_dipole_denom(&self, theta: f64, phi: f64) -> f64 {
        return 1.0 + phi.cos() * phi.cos() * (theta.cos() * theta.cos() - 1.0);
    }

    /// Prepare a compute-capable GPU device for beam-response computations
    /// given the delays and amps to be used. The resulting object takes
    /// directions and frequencies to compute the beam responses on the device.
    ///
    /// `delays_array` and `amps_array` must have the same number of rows;
    /// these correspond to tile configurations (i.e. each tile is allowed
    /// to have distinct delays and amps). The number of elements per row of
    /// `delays_array` and `amps_array` have the same restrictions as `delays`
    /// and `amps` in `calc_jones`.
    ///
    /// The code will automatically de-duplicate tile configurations so that no
    /// redundant calculations are done.
    ///
    /// # Safety
    ///
    /// This function interfaces directly with the CUDA/HIP API. Rust errors
    /// attempt to catch problems but there are no guarantees.
    #[cfg(any(feature = "cuda", feature = "hip"))]
    pub unsafe fn gpu_prepare(
        &self,
        delays: ArrayView2<u32>,
        amps: ArrayView2<f64>,
    ) -> Result<gpu::AnalyticBeamGpu, AnalyticBeamError> {
        // This function is deliberately kept thin to keep the focus of this
        // module on the CPU code.
        gpu::AnalyticBeamGpu::new(self, delays, amps)
    }
}

/// Ensure that any delays of 32 have an amplitude (dipole gain) of 0. The
/// results are bad otherwise! Also potentially halve the number of amps (e.g.
/// if 32 are given for a 16-bowtie tile, yield 16); we use the smaller of the
/// two gains associated with a bowtie.
fn fix_amps(amps: &[f64], delays: &[u32]) -> Vec<f64> {
    // The lengths of `amps` and `delays` should be checked before calling this
    // functions; the asserts are a last resort guard.
    assert!(amps.len() == delays.len() || amps.len() == delays.len() * 2);

    let mut fixed_amps = vec![0.0; delays.len()];
    fixed_amps
        .iter_mut()
        .zip(amps.iter())
        .zip(delays.iter())
        .for_each(|((fixed, &amp), &delay)| *fixed = if delay == 32 { 0.0 } else { amp });
    if amps.len() == delays.len() * 2 {
        fixed_amps
            .iter_mut()
            .zip(amps.iter().skip(delays.len()))
            .for_each(|(fixed, &amp)| {
                *fixed = fixed.min(amp);
            });
    }
    fixed_amps
}

/// The RTS doesn't use the M&C order. This function takes in the M&C-ordered
/// amps and delays, and returns RTS-ordered amps and delays. It also does
/// extra... things.
// Several thousand upside down emojis.
fn reorder_to_rts(amps: &[f64], delays: &[u32]) -> (Vec<f64>, Vec<f64>) {
    // Assume that the number of delays is the number of bowties.
    let num_bowties = delays.len();
    // Get the number of bowties per row from the number of bowties. This
    // assumes that the number is a perfect square.
    let bowties_per_row = (num_bowties as f64).sqrt().round() as usize;
    assert_eq!(bowties_per_row * bowties_per_row, num_bowties);

    let mut indices = Vec::with_capacity(num_bowties);
    for i_col in 0..bowties_per_row {
        for i_row in (0..bowties_per_row).rev() {
            indices.push(i_row * bowties_per_row + i_col);
        }
    }

    // Convert to "RTS order".
    let mut rts_amps = vec![0.0; num_bowties];
    let mut rts_delays = vec![0.0; num_bowties];
    indices
        .into_iter()
        .zip(rts_amps.iter_mut())
        .zip(rts_delays.iter_mut())
        .for_each(|((i, rts_amp), rts_delay)| {
            *rts_amp = amps[i];
            *rts_delay = f64::from(delays[i]);
        });

    // Do this crazy stuff.
    let delay_0 = rts_delays.iter().sum::<f64>() * VEL_C * DELAY_STEP / num_bowties as f64;
    rts_delays.iter_mut().for_each(|d| {
        *d = *d * VEL_C * DELAY_STEP - delay_0;
    });
    (rts_amps, rts_delays)
}

fn delay_ints_to_floats(delays: &[u32]) -> Vec<f64> {
    delays
        .iter()
        .copied()
        .map(|d| d as f64 * VEL_C * DELAY_STEP)
        .collect()
}
