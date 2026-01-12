package RD_F_003_PointingTrackingStabilization_FunctionalRequirementsSet {
    import ScalarValues::*;
    import Quantities::*;
    
    // =========================
    // Requirement Definitions
    // =========================
    
    requirement def PointToTargetReqDef {
        doc /* The TMT shall compute and command telescope pointing to the requested astronomical target coordinates for long-exposure observations. */
        attribute pointingErrorArcsec : Real;
        attribute pointingErrorToleranceArcsec : Real;
        attribute pointingLatencyMs : Real;
        attribute pointingLatencyMaxMs : Real;
        require constraint {
            (pointingErrorArcsec <= pointingErrorToleranceArcsec) and
            (pointingLatencyMs <= pointingLatencyMaxMs)
        }
    }
    
    requirement def SlewAndAcquireReqDef {
        doc /* The TMT shall execute slews and achieve acquisition-ready pointing for the requested target. */
        attribute acquisitionTimeS : Real;
        attribute acquisitionTimeMaxS : Real;
        attribute postSlewResidualErrorArcsec : Real;
        attribute postSlewResidualToleranceArcsec : Real;
        require constraint {
            (acquisitionTimeS <= acquisitionTimeMaxS) and
            (postSlewResidualErrorArcsec <= postSlewResidualToleranceArcsec)
        }
    }
    
    requirement def TrackSiderealAndNonSiderealReqDef {
        doc /* The TMT shall track targets using sidereal and non-sidereal rates as provided by the observation plan. */
        attribute trackingRateErrorArcsecPerHr : Real;
        attribute trackingRateErrorToleranceArcsecPerHr : Real;
        attribute rateUpdateLatencyMs : Real;
        attribute rateUpdateLatencyMaxMs : Real;
        require constraint {
            (trackingRateErrorArcsecPerHr <= trackingRateErrorToleranceArcsecPerHr) and
            (rateUpdateLatencyMs <= rateUpdateLatencyMaxMs)
        }
    }
    
    requirement def MaintainTrackingStabilityReqDef {
        doc /* The TMT shall maintain tracking stability during long exposures within the specified jitter and drift limits. */
        attribute jitterRmsArcsec : Real;
        attribute jitterRmsToleranceArcsec : Real;
        attribute driftArcsecPerMin : Real;
        attribute driftToleranceArcsecPerMin : Real;
        require constraint {
            (jitterRmsArcsec <= jitterRmsToleranceArcsec) and
            (driftArcsecPerMin <= driftToleranceArcsecPerMin)
        }
    }
    
    requirement def SettleAfterSlewReqDef {
        doc /* The TMT shall settle to within the required pointing and stability limits after a slew or disturbance before starting exposure. */
        attribute settlingTimeS : Real;
        attribute settlingTimeMaxS : Real;
        attribute settledErrorArcsec : Real;
        attribute settledErrorToleranceArcsec : Real;
        require constraint {
            (settlingTimeS <= settlingTimeMaxS) and
            (settledErrorArcsec <= settledErrorToleranceArcsec)
        }
    }
    
    requirement def GuideOnReferenceReqDef {
        doc /* The TMT shall support guiding using a provided reference source to reduce low-frequency pointing and tracking errors during exposures. */
        attribute guideCorrectionBandwidthHz : Real;
        attribute guideCorrectionBandwidthMinHz : Real;
        attribute guideLoopLatencyMs : Real;
        attribute guideLoopLatencyMaxMs : Real;
        require constraint {
            (guideCorrectionBandwidthHz >= guideCorrectionBandwidthMinHz) and
            (guideLoopLatencyMs <= guideLoopLatencyMaxMs)
        }
    }
    
    requirement def AcceptGuidanceSensorInputsReqDef {
        doc /* The TMT shall accept guidance sensor measurements and time tags needed to support pointing, tracking, and stabilization control. */
        attribute sensorUpdateRateHz : Real;
        attribute sensorUpdateRateMinHz : Real;
        attribute sensorTimestampErrorMs : Real;
        attribute sensorTimestampToleranceMs : Real;
        require constraint {
            (sensorUpdateRateHz >= sensorUpdateRateMinHz) and
            (sensorTimestampErrorMs <= sensorTimestampToleranceMs)
        }
    }
    
    requirement def StabilizeLineOfSightReqDef {
        doc /* The TMT shall stabilize the line-of-sight to meet long-exposure image motion limits under expected disturbances. */
        attribute losMotionRmsArcsec : Real;
        attribute losMotionToleranceArcsec : Real;
        attribute stabilizationLatencyMs : Real;
        attribute stabilizationLatencyMaxMs : Real;
        require constraint {
            (losMotionRmsArcsec <= losMotionToleranceArcsec) and
            (stabilizationLatencyMs <= stabilizationLatencyMaxMs)
        }
    }
    
    requirement def CompensateDisturbancesReqDef {
        doc /* The TMT shall compensate predictable disturbances (e.g., wind/structural excitation) impacting pointing and tracking performance. */
        attribute disturbanceRejectionDb : Real;
        attribute disturbanceRejectionMinDb : Real;
        attribute disturbanceResponseTimeS : Real;
        attribute disturbanceResponseTimeMaxS : Real;
        require constraint {
            (disturbanceRejectionDb >= disturbanceRejectionMinDb) and
            (disturbanceResponseTimeS <= disturbanceResponseTimeMaxS)
        }
    }
    
    requirement def ProvidePointingStatusTelemetryReqDef {
        doc /* The TMT shall provide telemetry of pointing, tracking, and stabilization state sufficient for monitoring long exposures. */
        attribute telemetryUpdateRateHz : Real;
        attribute telemetryUpdateRateMinHz : Real;
        attribute telemetryLatencyMs : Real;
        attribute telemetryLatencyMaxMs : Real;
        require constraint {
            (telemetryUpdateRateHz >= telemetryUpdateRateMinHz) and
            (telemetryLatencyMs <= telemetryLatencyMaxMs)
        }
    }
    
    requirement def HandleLossOfGuidanceReqDef {
        doc /* The TMT shall detect loss or degradation of guidance sensor inputs and transition to a defined degraded tracking behavior. */
        attribute detectionTimeMs : Real;
        attribute detectionTimeMaxMs : Real;
        attribute allowableDriftDuringDegradedArcsecPerMin : Real;
        attribute allowableDriftMaxArcsecPerMin : Real;
        require constraint {
            (detectionTimeMs <= detectionTimeMaxMs) and
            (allowableDriftDuringDegradedArcsecPerMin <= allowableDriftMaxArcsecPerMin)
        }
    }
    
    requirement def CoordinateReferenceFrameConsistencyReqDef {
        doc /* The TMT shall maintain consistency of coordinate frames and time references used for pointing and tracking computations. */
        attribute frameAlignmentErrorArcsec : Real;
        attribute frameAlignmentToleranceArcsec : Real;
        attribute timeSyncErrorMs : Real;
        attribute timeSyncToleranceMs : Real;
        require constraint {
            (frameAlignmentErrorArcsec <= frameAlignmentToleranceArcsec) and
            (timeSyncErrorMs <= timeSyncToleranceMs)
        }
    }
    
    requirement def InterfaceToMountAndEncodersReqDef {
        doc /* The TMT shall interface with the mount and encoder information sources to obtain position, rate, and time-tagged feedback needed for pointing and tracking. */
        attribute encoderSampleRateHz : Real;
        attribute encoderSampleRateMinHz : Real;
        attribute encoderLatencyMs : Real;
        attribute encoderLatencyMaxMs : Real;
        require constraint {
            (encoderSampleRateHz >= encoderSampleRateMinHz) and
            (encoderLatencyMs <= encoderLatencyMaxMs)
        }
    }
    
    requirement def TrackDuringExposureReqDef {
        doc /* The TMT shall maintain continuous pointing and tracking control throughout a long exposure without violating stability limits. */
        attribute maxContinuousExposureS : Real;
        attribute maxContinuousExposureMinS : Real;
        attribute stabilityViolationCount : Real;
        attribute stabilityViolationMaxCount : Real;
        require constraint {
            (maxContinuousExposureS >= maxContinuousExposureMinS) and
            (stabilityViolationCount <= stabilityViolationMaxCount)
        }
    }
    
    // =========================
    // External Interface Placeholders (no design decisions)
    // =========================
    
    interface def MountAndEncodersIF {
        doc /* Abstract interface representing mount command/feedback and encoder-derived position/rate/time information. */
        attribute feedbackSampleRateHz : Real;
        attribute feedbackLatencyMs : Real;
        attribute positionResolutionArcsec : Real;
    }
    
    interface def GuidanceSensorsIF {
        doc /* Abstract interface representing guidance sensor measurements and time-tagged observables used for guiding/stabilization. */
        attribute measurementUpdateRateHz : Real;
        attribute measurementLatencyMs : Real;
        attribute measurementNoiseRmsArcsec : Real;
    }
    
    // =========================
    // Requirement Instances (RD view content)
    // =========================
    
    package RD_F_003_Requirements {
        
        requirement FR_003_001_PointToTarget : PointToTargetReqDef {
            doc /* FR-003-001: The TMT shall compute and command telescope pointing to the requested astronomical target coordinates for long exposures. */
            pointingErrorToleranceArcsec = 0.0;      // TBD
            pointingLatencyMaxMs = 0.0;              // TBD
        }
        
        requirement FR_003_002_SlewAndAcquire : SlewAndAcquireReqDef {
            doc /* FR-003-002: The TMT shall execute slews and achieve acquisition-ready pointing for the requested target. */
            acquisitionTimeMaxS = 0.0;               // TBD
            postSlewResidualToleranceArcsec = 0.0;   // TBD
        }
        
        requirement FR_003_003_TrackSiderealNonSidereal : TrackSiderealAndNonSiderealReqDef {
            doc /* FR-003-003: The TMT shall track targets using sidereal and non-sidereal rates as provided by the observation plan. */
            trackingRateErrorToleranceArcsecPerHr = 0.0; // TBD
            rateUpdateLatencyMaxMs = 0.0;                // TBD
        }
        
        requirement FR_003_004_MaintainTrackingStability : MaintainTrackingStabilityReqDef {
            doc /* FR-003-004: The TMT shall maintain tracking stability during long exposures within specified jitter and drift limits. */
            jitterRmsToleranceArcsec = 0.0;          // TBD
            driftToleranceArcsecPerMin = 0.0;        // TBD
        }
        
        requirement FR_003_005_SettleAfterSlew : SettleAfterSlewReqDef {
            doc /* FR-003-005: The TMT shall settle after slews/disturbances before exposure start. */
            settlingTimeMaxS = 0.0;                  // TBD
            settledErrorToleranceArcsec = 0.0;       // TBD
        }
        
        requirement FR_003_006_GuideOnReference : GuideOnReferenceReqDef {
            doc /* FR-003-006: The TMT shall support guiding using a provided reference source to reduce low-frequency errors. */
            guideCorrectionBandwidthMinHz = 0.0;     // TBD
            guideLoopLatencyMaxMs = 0.0;             // TBD
        }
        
        requirement FR_003_007_AcceptGuidanceSensorInputs : AcceptGuidanceSensorInputsReqDef {
            doc /* FR-003-007: The TMT shall accept guidance sensor measurements and time tags for control. */
            sensorUpdateRateMinHz = 0.0;             // TBD
            sensorTimestampToleranceMs = 0.0;        // TBD
        }
        
        requirement FR_003_008_StabilizeLineOfSight : StabilizeLineOfSightReqDef {
            doc /* FR-003-008: The TMT shall stabilize line-of-sight to meet long-exposure image motion limits. */
            losMotionToleranceArcsec = 0.0;          // TBD
            stabilizationLatencyMaxMs = 0.0;         // TBD
        }
        
        requirement FR_003_009_CompensateDisturbances : CompensateDisturbancesReqDef {
            doc /* FR-003-009: The TMT shall compensate predictable disturbances impacting pointing/tracking. */
            disturbanceRejectionMinDb = 0.0;         // TBD
            disturbanceResponseTimeMaxS = 0.0;       // TBD
        }
        
        requirement FR_003_010_ProvideTelemetry : ProvidePointingStatusTelemetryReqDef {
            doc /* FR-003-010: The TMT shall provide pointing/tracking/stabilization telemetry for monitoring. */
            telemetryUpdateRateMinHz = 0.0;          // TBD
            telemetryLatencyMaxMs = 0.0;             // TBD
        }
        
        requirement FR_003_011_HandleLossOfGuidance : HandleLossOfGuidanceReqDef {
            doc /* FR-003-011: The TMT shall detect loss/degradation of guidance sensor inputs and transition to degraded tracking. */
            detectionTimeMaxMs = 0.0;                // TBD
            allowableDriftMaxArcsecPerMin = 0.0;     // TBD
        }
        
        requirement FR_003_012_FrameAndTimeConsistency : CoordinateReferenceFrameConsistencyReqDef {
            doc /* FR-003-012: The TMT shall maintain coordinate frame and time-reference consistency for pointing/tracking computations. */
            frameAlignmentToleranceArcsec = 0.0;     // TBD
            timeSyncToleranceMs = 0.0;               // TBD
        }
        
        requirement FR_003_013_InterfaceToMountEncoders : InterfaceToMountAndEncodersReqDef {
            doc /* FR-003-013: The TMT shall interface to mount/encoders for time-tagged feedback needed for pointing/tracking. */
            encoderSampleRateMinHz = 0.0;            // TBD
            encoderLatencyMaxMs = 0.0;               // TBD
        }
        
        requirement FR_003_014_TrackDuringExposure : TrackDuringExposureReqDef {
            doc /* FR-003-014: The TMT shall maintain continuous control throughout long exposures without stability violations. */
            maxContinuousExposureMinS = 0.0;         // TBD
            stabilityViolationMaxCount = 0.0;        // TBD
        }
        
        // =========================
        // Requirement-level interface references (no design decisions)
        // =========================
        
        interface MountEncoders_InterfaceRef : MountAndEncodersIF {
            doc /* Interface reference used by requirements FR-003-001..014 where mount/encoder feedback is needed. */
            feedbackSampleRateHz = 0.0;              // TBD
            feedbackLatencyMs = 0.0;                 // TBD
            positionResolutionArcsec = 0.0;          // TBD
        }
        
        interface GuidanceSensors_InterfaceRef : GuidanceSensorsIF {
            doc /* Interface reference used by requirements FR-003-006..011 where guidance sensor inputs are needed. */
            measurementUpdateRateHz = 0.0;           // TBD
            measurementLatencyMs = 0.0;              // TBD
            measurementNoiseRmsArcsec = 0.0;         // TBD
        }
    }
}
