package RD_F_001_BlackBoxFunctionsOverview {
    import ScalarValues::*;
    import Quantities::*;

    // Diagram metadata
    doc /* diagram_id: RD-F-001
         * type: RD
         * title: Black-Box Functions Overview
         * goal: Establish the system-level black-box functions and their intent as the basis for functional requirements.
         * dependencies: RD-CTX-003
         * target_system: Thirty Meter Telescope (TMT, 三十米望远镜)
         */

    // ----------------------------
    // Common value types (high-level I/O tokens)
    // ----------------------------
    value type TargetDesignation;
    value type SkyCoordinates;
    value type TrackingEphemeris;
    value type EnvironmentalConditions;
    value type OpticalIRPhotons;
    value type WavefrontSignal;
    value type SegmentState;
    value type AlignmentState;
    value type PointingCommand;
    value type AOControlCommand;
    value type InstrumentConfig;
    value type SequencePlan;
    value type CalibrationCommand;
    value type SafetyStatus;
    value type InterlockStatus;
    value type HealthTelemetry;
    value type ScienceData;
    value type Metadata;
    value type OperatorCommand;
    value type AlertNotification;

    // ----------------------------
    // Black-box function pattern
    // ----------------------------
    action def BlackBoxFunction {
        doc /* Abstract black-box function template for TMT system-level behavior. */
        in  primaryInputs  : Any;
        out primaryOutputs : Any;
    }

    // ----------------------------
    // 1) Collect light (optical/IR)
    // ----------------------------
    action def F_CollectLight : BlackBoxFunction {
        doc /* Collect optical/IR photons from designated astronomical targets and deliver usable light to downstream optical paths/instruments. */
        in  target : TargetDesignation;
        in  sky     : SkyCoordinates;
        in  env     : EnvironmentalConditions;
        out photons : OpticalIRPhotons;
    }

    // ----------------------------
    // 2) Pointing and tracking
    // ----------------------------
    action def F_PointingAndTracking : BlackBoxFunction {
        doc /* Point the telescope to a target and track it with required stability for long exposures and acquisition activities. */
        in  target       : TargetDesignation;
        in  ephemeris    : TrackingEphemeris;
        in  env          : EnvironmentalConditions;
        out pointingCmd  : PointingCommand;
        out alignmentRef : AlignmentState;
    }

    // ----------------------------
    // 3) Stabilization for long exposures (guiding/jitter control at high level)
    // ----------------------------
    action def F_StabilizeImage : BlackBoxFunction {
        doc /* Stabilize the delivered image during observing by minimizing jitter and drift at the system level to support long exposures. */
        in  env           : EnvironmentalConditions;
        in  pointingCmd   : PointingCommand;
        in  wfsSignal     : WavefrontSignal;
        out stabilizedCmd : PointingCommand;
    }

    // ----------------------------
    // 4) Wavefront sensing (WFS)
    // ----------------------------
    action def F_WavefrontSensing : BlackBoxFunction {
        doc /* Sense optical wavefront errors at a high level to enable alignment/phasing and (optionally) adaptive optics correction. */
        in  photons   : OpticalIRPhotons;
        in  env       : EnvironmentalConditions;
        out wfsSignal : WavefrontSignal;
    }

    // ----------------------------
    // 5) Segment alignment and phasing (APS)
    // ----------------------------
    action def F_AlignmentAndPhasing : BlackBoxFunction {
        doc /* Maintain segmented primary mirror alignment and phasing within tolerances by estimating and commanding segment states. */
        in  wfsSignal     : WavefrontSignal;
        in  segmentState  : SegmentState;
        out phasedState   : SegmentState;
        out alignmentState: AlignmentState;
    }

    // ----------------------------
    // 6) Adaptive optics correction (optional scope)
    // ----------------------------
    action def F_AdaptiveOpticsCorrection : BlackBoxFunction {
        doc /* Apply high-level adaptive optics correction to approach diffraction-limited performance when enabled by the observing mode. */
        in  wfsSignal    : WavefrontSignal;
        in  aoModeEnable : Boolean;
        out aoCmd        : AOControlCommand;
    }

    // ----------------------------
    // 7) Instrument hosting and interface services
    // ----------------------------
    action def F_InstrumentHosting : BlackBoxFunction {
        doc /* Host science instruments and provide high-level interface services (configuration/control/handshake) required for observations. */
        in  instrumentCfg : InstrumentConfig;
        in  operatorCmd   : OperatorCommand;
        out instReady     : Boolean;
        out instStatus    : HealthTelemetry;
    }

    // ----------------------------
    // 8) Observation sequencing and execution
    // ----------------------------
    action def F_ObservationSequencing : BlackBoxFunction {
        doc /* Orchestrate observation execution from plans into coordinated subsystem actions (slew, acquire, calibrate, expose, readout). */
        in  sequencePlan : SequencePlan;
        in  operatorCmd  : OperatorCommand;
        out execStatus   : HealthTelemetry;
        out alerts       : AlertNotification;
    }

    // ----------------------------
    // 9) Calibration management (high level)
    // ----------------------------
    action def F_CalibrationManagement : BlackBoxFunction {
        doc /* Execute and manage calibration activities (e.g., flats/darks/arcs/WFS calibration) and report readiness/results. */
        in  calCmd     : CalibrationCommand;
        in  instrumentCfg : InstrumentConfig;
        out calStatus  : HealthTelemetry;
        out calReady   : Boolean;
    }

    // ----------------------------
    // 10) Science data capture (raw acquisition)
    // ----------------------------
    action def F_DataCapture : BlackBoxFunction {
        doc /* Capture raw science (and engineering) data generated during exposures and readouts at the system level. */
        in  instrumentCfg : InstrumentConfig;
        in  operatorCmd   : OperatorCommand;
        out rawData       : ScienceData;
    }

    // ----------------------------
    // 11) Metadata generation and association
    // ----------------------------
    action def F_MetadataManagement : BlackBoxFunction {
        doc /* Generate and associate observation metadata (configuration, timing, conditions, states) required for downstream processing and traceability. */
        in  execStatus : HealthTelemetry;
        in  env        : EnvironmentalConditions;
        in  alignment  : AlignmentState;
        out metadata   : Metadata;
    }

    // ----------------------------
    // 12) Data product delivery (handoff to pipelines/archive at high level)
    // ----------------------------
    action def F_DataDelivery : BlackBoxFunction {
        doc /* Deliver captured data and metadata to downstream processing/archiving interfaces at a high level. */
        in  rawData   : ScienceData;
        in  metadata  : Metadata;
        out delivered : Boolean;
    }

    // ----------------------------
    // 13) Safety monitoring and interlocks
    // ----------------------------
    action def F_SafetyAndInterlocks : BlackBoxFunction {
        doc /* Monitor safety status, enforce interlocks, and support emergency actions to protect personnel, equipment, and site constraints. */
        in  health     : HealthTelemetry;
        in  operatorCmd: OperatorCommand;
        out safety     : SafetyStatus;
        out interlocks : InterlockStatus;
    }

    // ----------------------------
    // 14) Health monitoring and fault response (system-level)
    // ----------------------------
    action def F_HealthMonitoringAndFaultResponse : BlackBoxFunction {
        doc /* Detect anomalies, raise alerts, and coordinate transitions to safe states or recovery workflows at the system level. */
        in  health     : HealthTelemetry;
        in  interlocks : InterlockStatus;
        out alerts     : AlertNotification;
        out safeState  : Boolean;
    }

    // ----------------------------
    // System black-box context (ties inputs/outputs together at top level)
    // ----------------------------
    part def TMT_SystemBlackBox {
        doc /* System black-box context for TMT to anchor functional requirements. */

        // External / environment inputs
        in  target      : TargetDesignation;
        in  sky         : SkyCoordinates;
        in  ephemeris   : TrackingEphemeris;
        in  env         : EnvironmentalConditions;
        in  operatorCmd : OperatorCommand;
        in  seqPlan     : SequencePlan;
        in  instCfg     : InstrumentConfig;
        in  calCmd      : CalibrationCommand;
        in  aoEnable    : Boolean;

        // High-level system outputs
        out deliveredData : Boolean;
        out alerts        : AlertNotification;
        out safety        : SafetyStatus;

        // Black-box functions (>= 10) with consistent prefix "F_"
        action collectLight              : F_CollectLight;
        action pointingAndTracking       : F_PointingAndTracking;
        action stabilizeImage            : F_StabilizeImage;
        action wavefrontSensing          : F_WavefrontSensing;
        action alignmentAndPhasing       : F_AlignmentAndPhasing;
        action adaptiveOpticsCorrection  : F_AdaptiveOpticsCorrection;
        action instrumentHosting         : F_InstrumentHosting;
        action observationSequencing     : F_ObservationSequencing;
        action calibrationManagement     : F_CalibrationManagement;
        action dataCapture               : F_DataCapture;
        action metadataManagement        : F_MetadataManagement;
        action dataDelivery              : F_DataDelivery;
        action safetyAndInterlocks       : F_SafetyAndInterlocks;
        action healthAndFaultResponse    : F_HealthMonitoringAndFaultResponse;

        // High-level I/O bindings (abstract; detailed flows refined in later IBD/AD views)
        // Collect light -> WFS + instruments
        connect target to collectLight.target;
        connect sky    to collectLight.sky;
        connect env    to collectLight.env;

        connect collectLight.photons to wavefrontSensing.photons;
        connect env                 to wavefrontSensing.env;

        // WFS -> Alignment/Phasing and Stabilization and (optional) AO
        connect wavefrontSensing.wfsSignal to alignmentAndPhasing.wfsSignal;
        connect wavefrontSensing.wfsSignal to stabilizeImage.wfsSignal;
        connect wavefrontSensing.wfsSignal to adaptiveOpticsCorrection.wfsSignal;

        connect aoEnable to adaptiveOpticsCorrection.aoModeEnable;

        // Pointing/Tracking + Stabilization
        connect target    to pointingAndTracking.target;
        connect ephemeris to pointingAndTracking.ephemeris;
        connect env       to pointingAndTracking.env;

        connect pointingAndTracking.pointingCmd to stabilizeImage.pointingCmd;
        connect env                             to stabilizeImage.env;

        // Sequencing orchestrates observation execution
        connect seqPlan     to observationSequencing.sequencePlan;
        connect operatorCmd to observationSequencing.operatorCmd;

        // Instrument hosting
        connect instCfg     to instrumentHosting.instrumentCfg;
        connect operatorCmd to instrumentHosting.operatorCmd;

        // Calibration management
        connect calCmd   to calibrationManagement.calCmd;
        connect instCfg  to calibrationManagement.instrumentCfg;

        // Data capture and metadata
        connect instCfg     to dataCapture.instrumentCfg;
        connect operatorCmd to dataCapture.operatorCmd;

        connect observationSequencing.execStatus to metadataManagement.execStatus;
        connect env                              to metadataManagement.env;
        connect alignmentAndPhasing.alignmentState to metadataManagement.alignment;

        // Delivery
        connect dataCapture.rawData         to dataDelivery.rawData;
        connect metadataManagement.metadata to dataDelivery.metadata;

        connect dataDelivery.delivered to deliveredData;

        // Safety + health/fault response
        connect observationSequencing.execStatus to safetyAndInterlocks.health;
        connect operatorCmd                      to safetyAndInterlocks.operatorCmd;

        connect observationSequencing.execStatus to healthAndFaultResponse.health;
        connect safetyAndInterlocks.interlocks   to healthAndFaultResponse.interlocks;

        connect healthAndFaultResponse.alerts to alerts;
        connect safetyAndInterlocks.safety    to safety;
    }

    // Instantiate the context
    part tmtBlackBox : TMT_SystemBlackBox;
}
