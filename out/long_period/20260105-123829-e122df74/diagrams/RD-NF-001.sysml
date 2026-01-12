package RD_NF_001_PerformanceNFR_SensitivityAndAngularResolution {
    import ScalarValues::*;
    import Quantities::*;

    doc /* diagram_id: RD-NF-001
         * type: RD
         * title: Performance NFR: Sensitivity & Angular Resolution (Optical/IR)
         * goal: Define top-level performance non-functional requirements for sensitivity and angular resolution drivers.
         * dependencies: RD-CTX-003
         */

    // --- Assumptions / Observing Conditions (constraints + notes) ---
    requirement def ObservingAssumptionsAndConditionsDef {
        doc /* Assumptions for performance requirements; values may be refined per instrument/mode.
             * Includes observing bands and environmental/operational conditions that bound sensitivity and angular resolution.
             */

        attribute observingBand : String;                 // e.g., "optical", "near-IR", "mid-IR" (TBD)
        attribute wavelengthEffective : Real;              // [m] TBD
        attribute zenithAngleMax : Real;                   // [deg] TBD
        attribute seeingFWHM : Real;                       // [arcsec] TBD (site + conditions)
        attribute skyBackground : Real;                    // [mag/arcsec^2] TBD
        attribute airmassMax : Real;                       // [1] TBD
        attribute pwv : Real;                              // [mm] precipitable water vapor, TBD
        attribute moonIlluminationMax : Real;              // [%] TBD
        attribute windSpeedMax : Real;                     // [m/s] TBD
        attribute temperatureRange : Real;                 // [K] TBD (representative span)

        require constraint {
            wavelengthEffective > 0 and
            zenithAngleMax >= 0 and zenithAngleMax <= 90 and
            airmassMax >= 1 and
            seeingFWHM > 0 and
            pwv >= 0 and
            moonIlluminationMax >= 0 and moonIlluminationMax <= 100 and
            windSpeedMax >= 0
        }
    }

    // --- Enabling Functional Requirement Sets (trace targets) ---
    requirement def FR_PhasedSegmentedApertureDef {
        doc /* Enabling FR set: alignment & phasing of segmented primary mirror (APS/WFS/actuators)
             * needed to preserve wavefront quality for diffraction-limited capability and sensitivity.
             */
        attribute phasingResidualWFE : Real;               // [nm_rms] TBD
        attribute phasingUpdateCadence : Real;             // [Hz] TBD
        require constraint { phasingResidualWFE > 0 and phasingUpdateCadence > 0 }
    }

    requirement def FR_PointingTrackingAndStabilityDef {
        doc /* Enabling FR set: pointing, tracking, guiding, and stabilization to support long exposures
             * and prevent image motion / smear that degrades sensitivity and angular resolution.
             */
        attribute trackingJitterRMS : Real;                // [mas_rms] TBD
        attribute guidingErrorRMS : Real;                  // [mas_rms] TBD
        attribute vibrationLineOfSightRMS : Real;          // [mas_rms] TBD
        attribute maxAllowableImageSmear : Real;           // [mas] TBD
        require constraint {
            trackingJitterRMS > 0 and guidingErrorRMS > 0 and vibrationLineOfSightRMS >= 0 and maxAllowableImageSmear > 0
        }
    }

    requirement def FR_ObservationPhasingAndExposureControlDef {
        doc /* Enabling FR set: observation sequencing & exposure control (OCS/instruments)
             * to support long-period planning, coherent phasing steps, calibrations, and repeatable performance.
             */
        attribute maxExposureTime : Real;                  // [s] TBD
        attribute dutyCycleMin : Real;                     // [%] TBD
        attribute calibrationCadence : Real;               // [1/hr] TBD
        require constraint {
            maxExposureTime > 0 and dutyCycleMin >= 0 and dutyCycleMin <= 100 and calibrationCadence >= 0
        }
    }

    requirement def FR_AdaptiveOpticsOptionalDef {
        doc /* Enabling FR set (optional): adaptive optics capability for approaching diffraction-limited PSF.
             * Model as optional: requirements may be conditional on AO-enabled observing mode.
             */
        attribute aoEnabled : Boolean;
        attribute residualWFE_AO : Real;                   // [nm_rms] TBD
        attribute strehlRatio : Real;                      // [1] TBD (0..1)
        attribute aoLoopRate : Real;                       // [Hz] TBD
        require constraint {
            (aoEnabled implies (residualWFE_AO > 0 and strehlRatio >= 0 and strehlRatio <= 1 and aoLoopRate > 0))
        }
    }

    // --- Performance NFRs (top-level) ---
    requirement def SensitivityPerformanceNFRDef {
        doc /* Top-level NFR: Sensitivity (Optical/IR).
             * Defines measurable performance targets with units; placeholders allowed (TBD).
             * Sensitivity drivers include throughput, background, stability, calibration, and observing conditions.
             */

        // Target metrics (placeholders allowed)
        attribute limitingMagnitudePointSource : Real;      // [mag] TBD (band-dependent)
        attribute snrTarget : Real;                         // [1] TBD
        attribute exposureTimeForSNR : Real;                // [s] TBD
        attribute spectralResolution : Real;                // [1] TBD (if spectroscopy mode)
        attribute throughputEndToEnd : Real;                // [%] TBD
        attribute photometricStability : Real;              // [%] TBD (over exposure)
        attribute encircledEnergyRadius : Real;             // [mas] TBD (PSF metric driver)

        require constraint {
            snrTarget > 0 and
            exposureTimeForSNR > 0 and
            throughputEndToEnd >= 0 and throughputEndToEnd <= 100 and
            photometricStability >= 0 and photometricStability <= 100 and
            encircledEnergyRadius > 0
        }
    }

    requirement def AngularResolutionPerformanceNFRDef {
        doc /* Top-level NFR: Angular Resolution (Optical/IR).
             * Defines measurable performance targets with units; placeholders allowed (TBD).
             * Angular resolution depends on wavelength, effective aperture, wavefront error, jitter, and AO (optional).
             */

        // Target metrics (placeholders allowed)
        attribute diffractionLimited : Boolean;             // indicates goal to approach diffraction limit in mode
        attribute wavelengthForResolution : Real;           // [m] TBD
        attribute angularResolutionFWHM : Real;             // [mas] TBD
        attribute psfCoreFWHM : Real;                       // [mas] TBD (alternative metric)
        attribute totalWFERMS : Real;                       // [nm_rms] TBD (budget aggregate)
        attribute imageMotionRMS : Real;                    // [mas_rms] TBD
        attribute strehlRatioAchieved : Real;               // [1] TBD (0..1)

        require constraint {
            wavelengthForResolution > 0 and
            angularResolutionFWHM > 0 and
            psfCoreFWHM > 0 and
            totalWFERMS > 0 and
            imageMotionRMS >= 0 and
            strehlRatioAchieved >= 0 and strehlRatioAchieved <= 1
        }
    }

    // --- RD Package Content / Instances for trace links ---
    package Requirements_RD_NF_001 {
        doc /* Requirements instances for RD-NF-001 with trace to enabling FR sets and observing assumptions. */

        requirement ObservingAssumptionsAndConditions : ObservingAssumptionsAndConditionsDef {
            observingBand = "TBD";
            // wavelengthEffective [m] TBD
            // seeingFWHM [arcsec] TBD
        }

        requirement FR_PhasedSegmentedAperture : FR_PhasedSegmentedApertureDef { }
        requirement FR_PointingTrackingAndStability : FR_PointingTrackingAndStabilityDef { }
        requirement FR_ObservationPhasingAndExposureControl : FR_ObservationPhasingAndExposureControlDef { }
        requirement FR_AdaptiveOpticsOptional : FR_AdaptiveOpticsOptionalDef { aoEnabled = false; }

        requirement NFR_Sensitivity_OpticalIR : SensitivityPerformanceNFRDef {
            // limitingMagnitudePointSource [mag] TBD
            // snrTarget [1] TBD
            // exposureTimeForSNR [s] TBD
            // throughputEndToEnd [%] TBD
        }

        requirement NFR_AngularResolution_OpticalIR : AngularResolutionPerformanceNFRDef {
            diffractionLimited = true;     // goal placeholder; actual per-mode TBD
            // wavelengthForResolution [m] TBD
            // angularResolutionFWHM [mas] TBD
            // totalWFERMS [nm_rms] TBD
        }

        // --- Trace links (NFRs -> enabling FR sets, and bounded by assumptions) ---
        satisfy NFR_Sensitivity_OpticalIR by FR_ObservationPhasingAndExposureControl;
        satisfy NFR_Sensitivity_OpticalIR by FR_PointingTrackingAndStability;
        satisfy NFR_Sensitivity_OpticalIR by FR_PhasedSegmentedAperture;

        satisfy NFR_AngularResolution_OpticalIR by FR_PhasedSegmentedAperture;
        satisfy NFR_AngularResolution_OpticalIR by FR_PointingTrackingAndStability;
        satisfy NFR_AngularResolution_OpticalIR by FR_AdaptiveOpticsOptional;

        // Assumption bindings (treated as governing constraints/notes for performance evaluation)
        refine NFR_Sensitivity_OpticalIR by ObservingAssumptionsAndConditions;
        refine NFR_AngularResolution_OpticalIR by ObservingAssumptionsAndConditions;
    }
}
