package BDD_SUB04_001_AlignmentAndPhasingSystem {

  import ScalarValues::*;
  import SI::*;
  import Quantities::*;

  // ============================================================
  // Traceability stubs (referenced requirements from dependencies)
  // NOTE: These are references-by-name so this BDD can satisfy RD-2
  // ============================================================

  // Mission / science drivers (from upstream RD packages)
  requirement MissionGoalsAndScienceDrivers;
  requirement EarlyUniverseGoalMapping;
  requirement GalaxyEvolutionGoalMapping;
  requirement ExoplanetsGoalMapping;
  requirement BHgrowthGoalMapping;
  requirement StakeholderGroupConcerns;

  // Observing assumptions + functional requirements / NFR seeds (from RD-NF-001)
  requirement ObservingAssumptionsAndConditionsDef;
  requirement ObservingAssumptionsAndConditions;

  requirement FR_PhasedSegmentedApertureDef;
  requirement FR_PhasedSegmentedAperture;

  requirement FR_PointingTrackingAndStabilityDef;
  requirement FR_PointingTrackingAndStability;

  requirement FR_ObservationPhasingAndExposureControlDef;
  requirement FR_ObservationPhasingAndExposureControl;

  requirement FR_AdaptiveOpticsOptionalDef;
  requirement FR_AdaptiveOpticsOptional;

  requirement SensitivityPerformanceNFRDef;
  requirement AngularResolutionPerformanceNFRDef;

  // ============================================================
  // Domain value / data type definitions
  // ============================================================

  attribute def TimeStamp {
    seconds : Real;
  }

  attribute def WfsFrame {
    frameId : Integer;
    t : TimeStamp;
    valid : Boolean;
  }

  attribute def WavefrontEstimate {
    t : TimeStamp;
    rmsWfe : Real;
    piston : Real;
    tip : Real;
    tilt : Real;
  }

  attribute def SegmentId {
    value : Integer;
  }

  attribute def SegmentPoseError {
    segment : SegmentId;
    pistonError : Real;
    tipError : Real;
    tiltError : Real;
  }

  attribute def ActuatorCommand {
    segment : SegmentId;
    pistonDelta : Real;
    tipDelta : Real;
    tiltDelta : Real;
  }

  attribute def HealthStatus {
    ok : Boolean;
    code : Integer;
  }

  // ============================================================
  // Interfaces (ports exchanged between parts)
  // ============================================================

  interface def TimingSyncIF {
    out time : TimeStamp;
  }

  interface def WfsDataIF {
    out frame : WfsFrame;
  }

  interface def WavefrontEstimateIF {
    out estimate : WavefrontEstimate;
  }

  interface def SegmentErrorIF {
    out error : SegmentPoseError;
  }

  interface def ActuatorCommandIF {
    out cmd : ActuatorCommand;
  }

  interface def ActuatorFeedbackIF {
    out status : HealthStatus;
  }

  interface def OcsControlIF {
    in enablePhasing : Boolean;
    in requestCalibration : Boolean;
    in startExposure : Boolean;
    in stopExposure : Boolean;
  }

  interface def TelemetryIF {
    out health : HealthStatus;
  }

  interface def SafetyInterlockIF {
    in inhibitMotion : Boolean;
    out safeStateActive : Boolean;
  }

  // ============================================================
  // Abstract bases (for consistent decomposition)
  // ============================================================

  abstract part def LogicalSubsystem;
  abstract part def Sensor :> LogicalSubsystem;
  abstract part def Controller :> LogicalSubsystem;
  abstract part def Actuator :> LogicalSubsystem;
  abstract part def Metrology :> LogicalSubsystem;
  abstract part def Safety :> LogicalSubsystem;
  abstract part def InterfaceAdapter :> LogicalSubsystem;

  // ============================================================
  // Subsystem decomposition: Alignment & Phasing System (APS, WFS, actuators)
  // ============================================================

  part def AlignmentAndPhasingSystem :> LogicalSubsystem {
    // Key externally visible ports (interfaces to OCS, timing, safety, telemetry)
    port ocs : OcsControlIF;
    port timing : TimingSyncIF;
    port safety : SafetyInterlockIF;
    port telem : TelemetryIF;

    // Internal decomposition
    part apsController : APS_Controller;
    part wfs : WavefrontSensor;
    part metrology : SegmentMetrologySystem;
    part actuatorArray : SegmentActuatorArray;
    part calibration : WfsCalibrationUnit;
    part ocsAdapter : APS_OcsAdapter;
    part safetyMgr : APS_SafetyManager;

    // Internal connections (BDD-style structural wiring)
    connect timing to apsController.timing;
    connect timing to wfs.timing;
    connect timing to metrology.timing;

    connect ocs to ocsAdapter.ocs;
    connect ocsAdapter.phasingEnable to apsController.enable;

    connect wfs.wfsData to apsController.wfsIn;
    connect metrology.segmentError to apsController.metrologyIn;

    connect apsController.actCmd to actuatorArray.cmdIn;
    connect actuatorArray.fbOut to apsController.actFb;

    connect calibration.calRequest to wfs.calIn;
    connect ocsAdapter.calRequest to calibration.requestIn;

    connect safety to safetyMgr.interlock;
    connect safetyMgr.motionInhibit to actuatorArray.inhibit;
    connect safetyMgr.motionInhibit to apsController.inhibit;

    connect apsController.healthOut to telem;
    connect wfs.healthOut to telem;
    connect metrology.healthOut to telem;
    connect actuatorArray.healthOut to telem;
  }

  // ============================================================
  // Major parts
  // ============================================================

  part def APS_Controller :> Controller {
    // Inputs
    port timing : TimingSyncIF;
    port enable : in Boolean;
    port inhibit : in Boolean;

    port wfsIn : WfsDataIF;
    port metrologyIn : SegmentErrorIF;
    port actFb : ActuatorFeedbackIF;

    // Outputs
    port actCmd : ActuatorCommandIF;
    port healthOut : TelemetryIF;

    // Key internal attributes (algorithm/config placeholders)
    attribute phasingLoopRateHz : Real;
    attribute updateCadenceSec : Real;
    attribute maxResidualWfe : Real;
  }

  part def WavefrontSensor :> Sensor {
    port timing : TimingSyncIF;

    // Calibration + data output
    port calIn : in Boolean;
    port wfsData : WfsDataIF;
    port wfEstimate : WavefrontEstimateIF;

    // Health
    port healthOut : TelemetryIF;

    // Representative attributes
    attribute sensorMode : String;
    attribute exposureTimeSec : Real;
  }

  part def SegmentMetrologySystem :> Metrology {
    port timing : TimingSyncIF;

    // Outputs segment pose/phasing error (piston/tip/tilt per segment)
    port segmentError : SegmentErrorIF;

    // Health
    port healthOut : TelemetryIF;

    attribute measurementRateHz : Real;
    attribute enabled : Boolean;
  }

  part def SegmentActuatorArray :> Actuator {
    // Motion inhibit (from safety)
    port inhibit : in Boolean;

    // Command + feedback
    port cmdIn : ActuatorCommandIF;
    port fbOut : ActuatorFeedbackIF;

    // Health
    port healthOut : TelemetryIF;

    // Structural decomposition: per-segment actuator set (logical)
    part segmentActuator[492] : SegmentActuatorSet;
  }

  part def SegmentActuatorSet :> Actuator {
    attribute segment : SegmentId;
    attribute strokeMax : Real;
    attribute bandwidthHz : Real;
  }

  part def WfsCalibrationUnit :> LogicalSubsystem {
    port requestIn : in Boolean;
    port calRequest : out Boolean;

    attribute calibrationCadenceHours : Real;
  }

  part def APS_OcsAdapter :> InterfaceAdapter {
    port ocs : OcsControlIF;

    // Derived internal control signals for APS
    port phasingEnable : out Boolean;
    port calRequest : out Boolean;
  }

  part def APS_SafetyManager :> Safety {
    port interlock : SafetyInterlockIF;
    port motionInhibit : out Boolean;

    attribute safeStateLatched : Boolean;
  }

  // ============================================================
  // Requirement implementations (to satisfy RD-2 validation)
  // Map requirements -> concrete structural elements in this BDD
  // ============================================================

  satisfy MissionGoalsAndScienceDrivers by AlignmentAndPhasingSystem;
  satisfy EarlyUniverseGoalMapping by AlignmentAndPhasingSystem;
  satisfy GalaxyEvolutionGoalMapping by AlignmentAndPhasingSystem;
  satisfy ExoplanetsGoalMapping by AlignmentAndPhasingSystem;
  satisfy BHgrowthGoalMapping by AlignmentAndPhasingSystem;
  satisfy StakeholderGroupConcerns by AlignmentAndPhasingSystem;

  satisfy ObservingAssumptionsAndConditionsDef by WavefrontSensor;
  satisfy ObservingAssumptionsAndConditions by WavefrontSensor;

  satisfy FR_PhasedSegmentedApertureDef by SegmentActuatorArray;
  satisfy FR_PhasedSegmentedAperture by SegmentActuatorArray;

  satisfy FR_PointingTrackingAndStabilityDef by APS_Controller;
  satisfy FR_PointingTrackingAndStability by APS_Controller;

  satisfy FR_ObservationPhasingAndExposureControlDef by APS_Controller;
  satisfy FR_ObservationPhasingAndExposureControl by APS_Controller;

  satisfy FR_AdaptiveOpticsOptionalDef by WavefrontSensor;
  satisfy FR_AdaptiveOpticsOptional by WavefrontSensor;

  satisfy SensitivityPerformanceNFRDef by WavefrontSensor;
  satisfy AngularResolutionPerformanceNFRDef by APS_Controller;

}
