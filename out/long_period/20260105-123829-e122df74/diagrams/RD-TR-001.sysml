// diagram_id: RD-TR-001
// type: RD
// title: Traceability Backbone: Goals → Capabilities → Requirements
// goal: Create the RD traceability structure linking science goals to capabilities and to functional/NFR requirement sets.
// dependencies: RD-CTX-003, RD-F-001, RD-NF-001

package RD_TR_001_TraceabilityBackbone {
    import ScalarValues::*;
    import Quantities::*;

    // =========================
    // Core element definitions
    // =========================

    requirement def ScienceGoalDef {
        doc /* A top-level science goal for TMT (Thirty Meter Telescope, 三十米望远镜). */
        attribute id : String;
    }

    requirement def CapabilityDef {
        doc /* A system capability enabling one or more science goals. */
        attribute id : String;
    }

    requirement def RequirementDef {
        doc /* A requirement statement (functional or non-functional). */
        attribute id : String;
        attribute kind : String; // "FR" or "NFR"
        attribute priority : String; // "P1" | "P2" | "P3"
    }

    // =========================
    // Trace relationship defs
    // =========================

    relationship def GoalToCapabilityTrace {
        doc /* Trace link: a Capability supports a Science Goal. */
        end goal : ScienceGoalDef[1];
        end capability : CapabilityDef[1];
    }

    relationship def CapabilityToRequirementTrace {
        doc /* Trace link: Requirements realize a Capability. */
        end capability : CapabilityDef[1];
        end req : RequirementDef[1];
    }

    // =========================
    // Goals (>= 6)
    // =========================

    package Goals {
        requirement G_001_EarlyUniverse : ScienceGoalDef {
            id = "G-001";
            doc /* Enable studies of the early universe (first light, reionization-era galaxies) via high sensitivity and resolution. */
        }

        requirement G_002_GalaxyEvolution : ScienceGoalDef {
            id = "G-002";
            doc /* Enable galaxy formation and evolution science across cosmic time with deep spectroscopy and imaging. */
        }

        requirement G_003_Exoplanets : ScienceGoalDef {
            id = "G-003";
            doc /* Enable detection and characterization of exoplanets (imaging/spectroscopy) with high contrast and stability. */
        }

        requirement G_004_BlackHoleGrowth : ScienceGoalDef {
            id = "G-004";
            doc /* Enable measurements of black hole growth and dynamics in galaxies using high angular resolution and kinematic spectroscopy. */
        }

        requirement G_005_FirstStarsAndChemistry : ScienceGoalDef {
            id = "G-005";
            doc /* Enable studies of first stars, stellar populations, and chemical enrichment with faint-object capability. */
        }

        requirement G_006_TimeDomainAndTransients : ScienceGoalDef {
            id = "G-006";
            doc /* Enable time-domain astronomy and transient follow-up with responsive operations and reliable observing execution. */
        }
    }

    // =========================
    // Capabilities (>= 10)
    // =========================

    package Capabilities {
        requirement C_001_LightCollection : CapabilityDef {
            id = "C-001";
            doc /* Collect optical/IR photons with a 30m-class segmented aperture and high end-to-end throughput. */
        }

        requirement C_002_PointingTrackingStability : CapabilityDef {
            id = "C-002";
            doc /* Provide precise pointing, tracking, and line-of-sight stability for long exposures. */
        }

        requirement C_003_SegmentedPhasingAndAlignment : CapabilityDef {
            id = "C-003";
            doc /* Maintain segmented primary mirror alignment and phasing within wavefront error tolerances. */
        }

        requirement C_004_WavefrontSensingAndMetrology : CapabilityDef {
            id = "C-004";
            doc /* Sense wavefront/alignment state and provide metrology inputs for correction loops. */
        }

        requirement C_005_AdaptiveOpticsOptional : CapabilityDef {
            id = "C-005";
            doc /* Provide optional adaptive optics correction to approach diffraction-limited performance. */
        }

        requirement C_006_InstrumentHostingAndInterfaces : CapabilityDef {
            id = "C-006";
            doc /* Host science instruments and provide standardized power/data/thermal/control interfaces. */
        }

        requirement C_007_ObservationSequencingAndCalibration : CapabilityDef {
            id = "C-007";
            doc /* Execute observation sequences and manage calibrations across instruments and subsystems. */
        }

        requirement C_008_DataCaptureAndMetadata : CapabilityDef {
            id = "C-008";
            doc /* Capture raw science data with complete metadata and support quick-look quality assessment. */
        }

        requirement C_009_DataProductsAndArchiveDelivery : CapabilityDef {
            id = "C-009";
            doc /* Deliver data products and archive interfaces for downstream pipelines and users. */
        }

        requirement C_010_SafetyHealthMonitoringAndFaultResponse : CapabilityDef {
            id = "C-010";
            doc /* Provide safety/interlocks, health monitoring, automatic safe-state transitions, and operator alerting. */
        }
    }

    // =========================
    // Requirements (>= 40)
    // 40 requirements mapped across 10 capabilities (4 each)
    // =========================

    package Requirements {

        // ---- FR set: Collect light / throughput / optical path ----
        package FR_OpticsAndCollection {
            requirement FR_001_ApertureCollectingArea : RequirementDef {
                id = "FR-001";
                kind = "FR";
                priority = "P1";
                doc /* The system shall provide a segmented primary mirror aperture supporting 30m-class light collection. */
            }

            requirement FR_002_OpticalBandSupport : RequirementDef {
                id = "FR-002";
                kind = "FR";
                priority = "P1";
                doc /* The system shall support optical and infrared observing bands as defined by the observatory configuration. */
            }

            requirement FR_003_EndToEndThroughputControl : RequirementDef {
                id = "FR-003";
                kind = "FR";
                priority = "P2";
                doc /* The system shall maintain and report end-to-end throughput performance within allocated budgets over time. */
            }

            requirement FR_004_OpticalPathConfiguration : RequirementDef {
                id = "FR-004";
                kind = "FR";
                priority = "P2";
                doc /* The system shall configure optical path elements (e.g., M3 routing) to support instrument selection. */
            }
        }

        // ---- FR set: Pointing / tracking / stabilization ----
        package FR_PointingTracking {
            requirement FR_005_TargetAcquisition : RequirementDef {
                id = "FR-005";
                kind = "FR";
                priority = "P1";
                doc /* The system shall acquire astronomical targets and confirm acquisition state prior to science exposure start. */
            }

            requirement FR_006_PointingModelUse : RequirementDef {
                id = "FR-006";
                kind = "FR";
                priority = "P2";
                doc /* The system shall use a configurable pointing model for slews and tracking to improve accuracy. */
            }

            requirement FR_007_TrackingControlLoop : RequirementDef {
                id = "FR-007";
                kind = "FR";
                priority = "P1";
                doc /* The system shall provide a tracking control loop to maintain target position during exposures. */
            }

            requirement FR_008_ImageStabilization : RequirementDef {
                id = "FR-008";
                kind = "FR";
                priority = "P2";
                doc /* The system shall provide image stabilization functionality to reduce line-of-sight motion during exposures. */
            }
        }

        // ---- FR set: Alignment / phasing for segmented M1 ----
        package FR_AlignmentAndPhasing {
            requirement FR_009_SegmentActuation : RequirementDef {
                id = "FR-009";
                kind = "FR";
                priority = "P1";
                doc /* The system shall actuate primary mirror segments to correct piston/tip/tilt and maintain alignment. */
            }

            requirement FR_010_PhasingSequenceExecution : RequirementDef {
                id = "FR-010";
                kind = "FR";
                priority = "P1";
                doc /* The system shall execute an alignment and phasing sequence to achieve phased operation for observing. */
            }

            requirement FR_011_PhasingStateMonitoring : RequirementDef {
                id = "FR-011";
                kind = "FR";
                priority = "P2";
                doc /* The system shall monitor and report phasing state and quality metrics throughout operations. */
            }

            requirement FR_012_PhasingRecovery : RequirementDef {
                id = "FR-012";
                kind = "FR";
                priority = "P2";
                doc /* The system shall provide procedures to recover phasing after disturbances or faults. */
            }
        }

        // ---- FR set: Wavefront sensing / metrology ----
        package FR_WavefrontSensing {
            requirement FR_013_WFSMeasurement : RequirementDef {
                id = "FR-013";
                kind = "FR";
                priority = "P1";
                doc /* The system shall measure wavefront and alignment error signals using wavefront sensing instrumentation. */
            }

            requirement FR_014_MetrologyAcquisition : RequirementDef {
                id = "FR-014";
                kind = "FR";
                priority = "P2";
                doc /* The system shall acquire metrology inputs needed for alignment/phasing and maintain time synchronization. */
            }

            requirement FR_015_WFSCalibration : RequirementDef {
                id = "FR-015";
                kind = "FR";
                priority = "P2";
                doc /* The system shall support calibration routines for wavefront sensors to maintain measurement integrity. */
            }

            requirement FR_016_WFSDataPublication : RequirementDef {
                id = "FR-016";
                kind = "FR";
                priority = "P3";
                doc /* The system shall publish wavefront sensing measurements to observatory control and logging services. */
            }
        }

        // ---- FR set: Adaptive Optics optional ----
        package FR_AdaptiveOptics {
            requirement FR_017_AOAcquisition : RequirementDef {
                id = "FR-017";
                kind = "FR";
                priority = "P2";
                doc /* When AO is enabled, the system shall acquire guide sources and transition to closed-loop AO operation. */
            }

            requirement FR_018_AORealTimeControl : RequirementDef {
                id = "FR-018";
                kind = "FR";
                priority = "P2";
                doc /* When AO is enabled, the system shall provide real-time control for wavefront correction elements. */
            }

            requirement FR_019_AOLoopMonitoring : RequirementDef {
                id = "FR-019";
                kind = "FR";
                priority = "P3";
                doc /* When AO is enabled, the system shall monitor loop status and report residual error metrics. */
            }

            requirement FR_020_AOFallbackMode : RequirementDef {
                id = "FR-020";
                kind = "FR";
                priority = "P3";
                doc /* The system shall support a fallback observing mode when AO cannot be acquired or maintained. */
            }
        }

        // ---- FR set: Instrument hosting / interfaces ----
        package FR_InstrumentHosting {
            requirement FR_021_InstrumentPortSelection : RequirementDef {
                id = "FR-021";
                kind = "FR";
                priority = "P1";
                doc /* The system shall support selection and activation of an instrument port for an observing configuration. */
            }

            requirement FR_022_InstrumentPowerDataInterfaces : RequirementDef {
                id = "FR-022";
                kind = "FR";
                priority = "P1";
                doc /* The system shall provide standardized power and data interfaces to hosted instruments. */
            }

            requirement FR_023_InstrumentThermalServices : RequirementDef {
                id = "FR-023";
                kind = "FR";
                priority = "P2";
                doc /* The system shall provide thermal services or constraints needed for instrument performance and safety. */
            }

            requirement FR_024_InstrumentControlIntegration : RequirementDef {
                id = "FR-024";
                kind = "FR";
                priority = "P2";
                doc /* The system shall integrate instrument control with observatory sequencing and configuration management. */
            }
        }

        // ---- FR set: Observation sequencing / calibration management ----
        package FR_SequencingAndCalibration {
            requirement FR_025_SequenceExecution : RequirementDef {
                id = "FR-025";
                kind = "FR";
                priority = "P1";
                doc /* The system shall execute observation sequences including slews, acquisitions, exposures, and readouts. */
            }

            requirement FR_026_CalibrationPlanning : RequirementDef {
                id = "FR-026";
                kind = "FR";
                priority = "P2";
                doc /* The system shall plan and schedule calibrations (e.g., flats/darks/arcs) within observing sequences. */
            }

            requirement FR_027_CalibrationTraceability : RequirementDef {
                id = "FR-027";
                kind = "FR";
                priority = "P2";
                doc /* The system shall associate calibration data with science exposures via metadata and configuration state. */
            }

            requirement FR_028_OperatorWorkflowSupport : RequirementDef {
                id = "FR-028";
                kind = "FR";
                priority = "P3";
                doc /* The system shall support operator workflows for startup, shutdown, and mode transitions. */
            }
        }

        // ---- NFR set: Performance (sensitivity / angular resolution / stability) ----
        package NFR_Performance {
            requirement NFR_001_SensitivityLimitingMagnitude : RequirementDef {
                id = "NFR-001";
                kind = "NFR";
                priority = "P1";
                doc /* The system shall meet allocated sensitivity performance targets (e.g., limiting magnitude / SNR) under defined conditions. */
            }

            requirement NFR_002_AngularResolution : RequirementDef {
                id = "NFR-002";
                kind = "NFR";
                priority = "P1";
                doc /* The system shall support high angular resolution operation consistent with phased segmented aperture performance. */
            }

            requirement NFR_003_LineOfSightJitter : RequirementDef {
                id = "NFR-003";
                kind = "NFR";
                priority = "P1";
                doc /* The system shall limit line-of-sight jitter contributions to remain within the image quality error budget. */
            }

            requirement NFR_004_ExposureDutyCycle : RequirementDef {
                id = "NFR-004";
                kind = "NFR";
                priority = "P2";
                doc /* The system shall achieve a minimum observing duty cycle for science exposure loops and readout overheads. */
            }
        }

        // ---- NFR set: Data integrity / metadata / delivery ----
        package NFR_DataQualityAndDelivery {
            requirement NFR_005_MetadataCompleteness : RequirementDef {
                id = "NFR-005";
                kind = "NFR";
                priority = "P1";
                doc /* The system shall capture complete and consistent metadata needed for downstream processing and provenance. */
            }

            requirement NFR_006_DataIntegrity : RequirementDef {
                id = "NFR-006";
                kind = "NFR";
                priority = "P1";
                doc /* The system shall ensure data integrity from acquisition through archival handoff with detectable corruption handling. */
            }

            requirement NFR_007_ArchiveInterfaceAvailability : RequirementDef {
                id = "NFR-007";
                kind = "NFR";
                priority = "P2";
                doc /* The system shall provide archive delivery interfaces with defined availability targets during operations. */
            }

            requirement NFR_008_QuickLookLatency : RequirementDef {
                id = "NFR-008";
                kind = "NFR";
                priority = "P3";
                doc /* The system shall provide quick-look data access within a defined latency to support on-sky decisions. */
            }
        }

        // ---- NFR set: Reliability / availability / maintainability / safety ----
        package NFR_RAMSAndSafety {
            requirement NFR_009_ObservatoryAvailability : RequirementDef {
                id = "NFR-009";
                kind = "NFR";
                priority = "P1";
                doc /* The system shall meet observatory availability targets for scheduled observing time as allocated by subsystem budgets. */
            }

            requirement NFR_010_FaultDetectionCoverage : RequirementDef {
                id = "NFR-010";
                kind = "NFR";
                priority = "P2";
                doc /* The system shall provide fault detection coverage for critical subsystems sufficient to support safe operations. */
            }

            requirement NFR_011_MaintainabilitySegmentHandling : RequirementDef {
                id = "NFR-011";
                kind = "NFR";
                priority = "P2";
                doc /* The system shall support maintainability for segment handling/replacement/cleaning with defined operational procedures. */
            }

            requirement NFR_012_PersonnelSafetyInterlocks : RequirementDef {
                id = "NFR-012";
                kind = "NFR";
                priority = "P1";
                doc /* The system shall implement personnel safety interlocks and emergency stop behavior for hazardous operations. */
            }
        }

        // ---- NFR set: Site / environmental / controversial operations constraints ----
        package NFR_SiteAndEnvironmental {
            requirement NFR_013_EnvironmentalOperatingEnvelope : RequirementDef {
                id = "NFR-013";
                kind = "NFR";
                priority = "P2";
                doc /* The system shall operate within defined site environmental envelopes (wind/temperature/seeing assumptions as allocated). */
            }

            requirement NFR_014_OperationalConstraintCompliance : RequirementDef {
                id = "NFR-014";
                kind = "NFR";
                priority = "P2";
                doc /* The system shall support compliance with operational constraints required by site governance and safety policy. */
            }

            requirement NFR_015_NoiseAndVibrationControl : RequirementDef {
                id = "NFR-015";
                kind = "NFR";
                priority = "P2";
                doc /* The system shall control noise and vibration contributions to preserve instrument performance and site constraints. */
            }

            requirement NFR_016_TelemetryAndAuditability : RequirementDef {
                id = "NFR-016";
                kind = "NFR";
                priority = "P3";
                doc /* The system shall provide telemetry and audit logs sufficient for diagnostics, compliance, and post-run analysis. */
            }
        }
    }

    // =========================
    // Trace instances
    // =========================

    package Traceability {

        // ---- Goals → Capabilities (no orphan goals, no orphan capabilities) ----

        GoalToCapabilityTrace TR_G001_C001 { goal = Goals::G_001_EarlyUniverse;          capability = Capabilities::C_001_LightCollection; }
        GoalToCapabilityTrace TR_G001_C005 { goal = Goals::G_001_EarlyUniverse;          capability = Capabilities::C_005_AdaptiveOpticsOptional; }

        GoalToCapabilityTrace TR_G002_C001 { goal = Goals::G_002_GalaxyEvolution;        capability = Capabilities::C_001_LightCollection; }
        GoalToCapabilityTrace TR_G002_C009 { goal = Goals::G_002_GalaxyEvolution;        capability = Capabilities::C_009_DataProductsAndArchiveDelivery; }

        GoalToCapabilityTrace TR_G003_C005 { goal = Goals::G_003_Exoplanets;             capability = Capabilities::C_005_AdaptiveOpticsOptional; }
        GoalToCapabilityTrace TR_G003_C002 { goal = Goals::G_003_Exoplanets;             capability = Capabilities::C_002_PointingTrackingStability; }
        GoalToCapabilityTrace TR_G003_C006 { goal = Goals::G_003_Exoplanets;             capability = Capabilities::C_006_InstrumentHostingAndInterfaces; }

        GoalToCapabilityTrace TR_G004_C003 { goal = Goals::G_004_BlackHoleGrowth;        capability = Capabilities::C_003_SegmentedPhasingAndAlignment; }
        GoalToCapabilityTrace TR_G004_C004 { goal = Goals::G_004_BlackHoleGrowth;        capability = Capabilities::C_004_WavefrontSensingAndMetrology; }

        GoalToCapabilityTrace TR_G005_C001 { goal = Goals::G_005_FirstStarsAndChemistry; capability = Capabilities::C_001_LightCollection; }
        GoalToCapabilityTrace TR_G005_C008 { goal = Goals::G_005_FirstStarsAndChemistry; capability = Capabilities::C_008_DataCaptureAndMetadata; }

        GoalToCapabilityTrace TR_G006_C007 { goal = Goals::G_006_TimeDomainAndTransients; capability = Capabilities::C_007_ObservationSequencingAndCalibration; }
        GoalToCapabilityTrace TR_G006_C010 { goal = Goals::G_006_TimeDomainAndTransients; capability = Capabilities::C_010_SafetyHealthMonitoringAndFaultResponse; }

        // Ensure remaining capabilities are also supported by at least one goal (no orphan capability nodes)
        GoalToCapabilityTrace TR_G002_C008 { goal = Goals::G_002_GalaxyEvolution;        capability = Capabilities::C_008_DataCaptureAndMetadata; }
        GoalToCapabilityTrace TR_G001_C004 { goal = Goals::G_001_EarlyUniverse;          capability = Capabilities::C_004_WavefrontSensingAndMetrology; }
        GoalToCapabilityTrace TR_G004_C002 { goal = Goals::G_004_BlackHoleGrowth;        capability = Capabilities::C_002_PointingTrackingStability; }
        GoalToCapabilityTrace TR_G003_C009 { goal = Goals::G_003_Exoplanets;             capability = Capabilities::C_009_DataProductsAndArchiveDelivery; }
        GoalToCapabilityTrace TR_G006_C006 { goal = Goals::G_006_TimeDomainAndTransients; capability = Capabilities::C_006_InstrumentHostingAndInterfaces; }
        GoalToCapabilityTrace TR_G006_C008 { goal = Goals::G_006_TimeDomainAndTransients; capability = Capabilities::C_008_DataCaptureAndMetadata; }

        // ---- Capabilities → Requirements (40 total, 4 per capability) ----

        // C-001 Light collection
        CapabilityToRequirementTrace TR_C001_FR001 { capability = Capabilities::C_001_LightCollection; req = Requirements::FR_OpticsAndCollection::FR_001_ApertureCollectingArea; }
        CapabilityToRequirementTrace TR_C001_FR002 { capability = Capabilities::C_001_LightCollection; req = Requirements::FR_OpticsAndCollection::FR_002_OpticalBandSupport; }
        CapabilityToRequirementTrace TR_C001_NFR001 { capability = Capabilities::C_001_LightCollection; req = Requirements::NFR_Performance::NFR_001_SensitivityLimitingMagnitude; }
        CapabilityToRequirementTrace TR_C001_NFR002 { capability = Capabilities::C_001_LightCollection; req = Requirements::NFR_Performance::NFR_002_AngularResolution; }

        // C-002 Pointing/tracking/stability
        CapabilityToRequirementTrace TR_C002_FR005 { capability = Capabilities::C_002_PointingTrackingStability; req = Requirements::FR_PointingTracking::FR_005_TargetAcquisition; }
        CapabilityToRequirementTrace TR_C002_FR007 { capability = Capabilities::C_002_PointingTrackingStability; req = Requirements::FR_PointingTracking::FR_007_TrackingControlLoop; }
        CapabilityToRequirementTrace TR_C002_FR008 { capability = Capabilities::C_002_PointingTrackingStability; req = Requirements::FR_PointingTracking::FR_008_ImageStabilization; }
        CapabilityToRequirementTrace TR_C002_NFR003 { capability = Capabilities::C_002_PointingTrackingStability; req = Requirements::NFR_Performance::NFR_003_LineOfSightJitter; }

        // C-003 Segmented phasing/alignment
        CapabilityToRequirementTrace TR_C003_FR009 { capability = Capabilities::C_003_SegmentedPhasingAndAlignment; req = Requirements::FR_AlignmentAndPhasing::FR_009_SegmentActuation; }
        CapabilityToRequirementTrace TR_C003_FR010 { capability = Capabilities::C_003_SegmentedPhasingAndAlignment; req = Requirements::FR_AlignmentAndPhasing::FR_010_PhasingSequenceExecution; }
        CapabilityToRequirementTrace TR_C003_FR011 { capability = Capabilities::C_003_SegmentedPhasingAndAlignment; req = Requirements::FR_AlignmentAndPhasing::FR_011_PhasingStateMonitoring; }
        CapabilityToRequirementTrace TR_C003_NFR002 { capability = Capabilities::C_003_SegmentedPhasingAndAlignment; req = Requirements::NFR_Performance::NFR_002_AngularResolution; }

        // C-004 Wavefront sensing/metrology
        CapabilityToRequirementTrace TR_C004_FR013 { capability = Capabilities::C_004_WavefrontSensingAndMetrology; req = Requirements::FR_WavefrontSensing::FR_013_WFSMeasurement; }
        CapabilityToRequirementTrace TR_C004_FR014 { capability = Capabilities::C_004_WavefrontSensingAndMetrology; req = Requirements::FR_WavefrontSensing::FR_014_MetrologyAcquisition; }
        CapabilityToRequirementTrace TR_C004_FR015 { capability = Capabilities::C_004_WavefrontSensingAndMetrology; req = Requirements::FR_WavefrontSensing::FR_015_WFSCalibration; }
        CapabilityToRequirementTrace TR_C004_NFR016 { capability = Capabilities::C_004_WavefrontSensingAndMetrology; req = Requirements::NFR_SiteAndEnvironmental::NFR_016_TelemetryAndAuditability; }

        // C-005 Adaptive optics optional
        CapabilityToRequirementTrace TR_C005_FR017 { capability = Capabilities::C_005_AdaptiveOpticsOptional; req = Requirements::FR_AdaptiveOptics::FR_017_AOAcquisition; }
        CapabilityToRequirementTrace TR_C005_FR018 { capability = Capabilities::C_005_AdaptiveOpticsOptional; req = Requirements::FR_AdaptiveOptics::FR_018_AORealTimeControl; }
        CapabilityToRequirementTrace TR_C005_FR019 { capability = Capabilities::C_005_AdaptiveOpticsOptional; req = Requirements::FR_AdaptiveOptics::FR_019_AOLoopMonitoring; }
        CapabilityToRequirementTrace TR_C005_NFR002 { capability = Capabilities::C_005_AdaptiveOpticsOptional; req = Requirements::NFR_Performance::NFR_002_AngularResolution; }

        // C-006 Instrument hosting/interfaces
        CapabilityToRequirementTrace TR_C006_FR021 { capability = Capabilities::C_006_InstrumentHostingAndInterfaces; req = Requirements::FR_InstrumentHosting::FR_021_InstrumentPortSelection; }
        CapabilityToRequirementTrace TR_C006_FR022 { capability = Capabilities::C_006_InstrumentHostingAndInterfaces; req = Requirements::FR_InstrumentHosting::FR_022_InstrumentPowerDataInterfaces; }
        CapabilityToRequirementTrace TR_C006_FR023 { capability = Capabilities::C_006_InstrumentHostingAndInterfaces; req = Requirements::FR_InstrumentHosting::FR_023_InstrumentThermalServices; }
        CapabilityToRequirementTrace TR_C006_NFR015 { capability = Capabilities::C_006_InstrumentHostingAndInterfaces; req = Requirements::NFR_SiteAndEnvironmental::NFR_015_NoiseAndVibrationControl; }

        // C-007 Observation sequencing/calibration
        CapabilityToRequirementTrace TR_C007_FR025 { capability = Capabilities::C_007_ObservationSequencingAndCalibration; req = Requirements::FR_SequencingAndCalibration::FR_025_SequenceExecution; }
        CapabilityToRequirementTrace TR_C007_FR026 { capability = Capabilities::C_007_ObservationSequencingAndCalibration; req = Requirements::FR_SequencingAndCalibration::FR_026_CalibrationPlanning; }
        CapabilityToRequirementTrace TR_C007_FR027 { capability = Capabilities::C_007_ObservationSequencingAndCalibration; req = Requirements::FR_SequencingAndCalibration::FR_027_CalibrationTraceability; }
        CapabilityToRequirementTrace TR_C007_NFR004 { capability = Capabilities::C_007_ObservationSequencingAndCalibration; req = Requirements::NFR_Performance::NFR_004_ExposureDutyCycle; }

        // C-008 Data capture/metadata
        CapabilityToRequirementTrace TR_C008_NFR005 { capability = Capabilities::C_008_DataCaptureAndMetadata; req = Requirements::NFR_DataQualityAndDelivery::NFR_005_MetadataCompleteness; }
        CapabilityToRequirementTrace TR_C008_NFR006 { capability = Capabilities::C_008_DataCaptureAndMetadata; req = Requirements::NFR_DataQualityAndDelivery::NFR_006_DataIntegrity; }
        CapabilityToRequirementTrace TR_C008_NFR008 { capability = Capabilities::C_008_DataCaptureAndMetadata; req = Requirements::NFR_DataQualityAndDelivery::NFR_008_QuickLookLatency; }
        CapabilityToRequirementTrace TR_C008_FR028 { capability = Capabilities::C_008_DataCaptureAndMetadata; req = Requirements::FR_SequencingAndCalibration::FR_028_OperatorWorkflowSupport; }

        // C-009 Data products/archive delivery
        CapabilityToRequirementTrace TR_C009_NFR007 { capability = Capabilities::C_009_DataProductsAndArchiveDelivery; req = Requirements::NFR_DataQualityAndDelivery::NFR_007_ArchiveInterfaceAvailability; }
        CapabilityToRequirementTrace TR_C009_NFR006 { capability = Capabilities::C_009_DataProductsAndArchiveDelivery; req = Requirements::NFR_DataQualityAndDelivery::NFR_006_DataIntegrity; }
        CapabilityToRequirementTrace TR_C009_FR003 { capability = Capabilities::C_009_DataProductsAndArchiveDelivery; req = Requirements::FR_OpticsAndCollection::FR_003_EndToEndThroughputControl; }
        CapabilityToRequirementTrace TR_C009_NFR016 { capability = Capabilities::C_009_DataProductsAndArchiveDelivery; req = Requirements::NFR_SiteAndEnvironmental::NFR_016_TelemetryAndAuditability; }

        // C-010 Safety/health monitoring/fault response
        CapabilityToRequirementTrace TR_C010_NFR012 { capability = Capabilities::C_010_SafetyHealthMonitoringAndFaultResponse; req = Requirements::NFR_RAMSAndSafety::NFR_012_PersonnelSafetyInterlocks; }
        CapabilityToRequirementTrace TR_C010_NFR010 { capability = Capabilities::C_010_SafetyHealthMonitoringAndFaultResponse; req = Requirements::NFR_RAMSAndSafety::NFR_010_FaultDetectionCoverage; }
        CapabilityToRequirementTrace TR_C010_NFR009 { capability = Capabilities::C_010_SafetyHealthMonitoringAndFaultResponse; req = Requirements::NFR_RAMSAndSafety::NFR_009_ObservatoryAvailability; }
        CapabilityToRequirementTrace TR_C010_NFR014 { capability = Capabilities::C_010_SafetyHealthMonitoringAndFaultResponse; req = Requirements::NFR_SiteAndEnvironmental::NFR_014_OperationalConstraintCompliance; }
    }
}
