package BDD_SUB10_002_OperationsMaintenance {
  import ScalarValues::*;
  import SI::*;
  import Quantities::*;

  //============================================================
  // Requirement reference stubs (to enable RD-2 traceability)
  //============================================================

  package RD_F_001_RequirementRefs {
    requirement def MissionGoalsAndScienceDrivers;
    requirement def EarlyUniverseGoalMapping;
    requirement def GalaxyEvolutionGoalMapping;
    requirement def ExoplanetsGoalMapping;
    requirement def BHgrowthGoalMapping;
    requirement def StakeholderGroupConcerns;
  }

  package RD_NF_001_RequirementRefs {
    requirement def ObservingAssumptionsAndConditionsDef;
    requirement def ObservingAssumptionsAndConditions;

    requirement def FR_PhasedSegmentedApertureDef;
    requirement def FR_PhasedSegmentedAperture;

    requirement def FR_PointingTrackingAndStabilityDef;
    requirement def FR_PointingTrackingAndStability;

    requirement def FR_ObservationPhasingAndExposureControlDef;
    requirement def FR_ObservationPhasingAndExposureControl;

    requirement def FR_AdaptiveOpticsOptionalDef;
    requirement def FR_AdaptiveOpticsOptional;

    requirement def SensitivityPerformanceNFRDef;
    requirement def AngularResolutionPerformanceNFRDef;
  }

  // Instantiate requirement usages so satisfy links are well-formed
  requirement MissionGoalsAndScienceDrivers : RD_F_001_RequirementRefs::MissionGoalsAndScienceDrivers;
  requirement EarlyUniverseGoalMapping      : RD_F_001_RequirementRefs::EarlyUniverseGoalMapping;
  requirement GalaxyEvolutionGoalMapping    : RD_F_001_RequirementRefs::GalaxyEvolutionGoalMapping;
  requirement ExoplanetsGoalMapping         : RD_F_001_RequirementRefs::ExoplanetsGoalMapping;
  requirement BHgrowthGoalMapping           : RD_F_001_RequirementRefs::BHgrowthGoalMapping;
  requirement StakeholderGroupConcerns      : RD_F_001_RequirementRefs::StakeholderGroupConcerns;

  requirement ObservingAssumptionsAndConditionsDef : RD_NF_001_RequirementRefs::ObservingAssumptionsAndConditionsDef;
  requirement ObservingAssumptionsAndConditions    : RD_NF_001_RequirementRefs::ObservingAssumptionsAndConditions;

  requirement FR_PhasedSegmentedApertureDef : RD_NF_001_RequirementRefs::FR_PhasedSegmentedApertureDef;
  requirement FR_PhasedSegmentedAperture    : RD_NF_001_RequirementRefs::FR_PhasedSegmentedAperture;

  requirement FR_PointingTrackingAndStabilityDef : RD_NF_001_RequirementRefs::FR_PointingTrackingAndStabilityDef;
  requirement FR_PointingTrackingAndStability    : RD_NF_001_RequirementRefs::FR_PointingTrackingAndStability;

  requirement FR_ObservationPhasingAndExposureControlDef : RD_NF_001_RequirementRefs::FR_ObservationPhasingAndExposureControlDef;
  requirement FR_ObservationPhasingAndExposureControl    : RD_NF_001_RequirementRefs::FR_ObservationPhasingAndExposureControl;

  requirement FR_AdaptiveOpticsOptionalDef : RD_NF_001_RequirementRefs::FR_AdaptiveOpticsOptionalDef;
  requirement FR_AdaptiveOpticsOptional    : RD_NF_001_RequirementRefs::FR_AdaptiveOpticsOptional;

  requirement SensitivityPerformanceNFRDef      : RD_NF_001_RequirementRefs::SensitivityPerformanceNFRDef;
  requirement AngularResolutionPerformanceNFRDef : RD_NF_001_RequirementRefs::AngularResolutionPerformanceNFRDef;

  //============================================================
  // Shared interface/port definitions
  //============================================================

  interface def CommandIF {
    attribute cmdName : String;
    attribute cmdId   : Integer;
    attribute cmdData : String;
  }

  interface def StatusIF {
    attribute state        : String;
    attribute health       : String;
    attribute timestampUtc : String;
  }

  interface def TelemetryIF {
    attribute topic     : String;
    attribute payload   : String;
    attribute sampleHz  : Real;
  }

  interface def AlarmIF {
    attribute severity  : String;
    attribute code      : String;
    attribute message   : String;
  }

  interface def MaintAccessIF {
    attribute accessMode : String; // e.g., "local", "remote", "restricted"
    attribute requestId  : String;
  }

  interface def PowerControlIF {
    attribute railName : String;
    attribute enable   : Boolean;
  }

  //============================================================
  // Common abstract types
  //============================================================

  abstract part def LogicalComponent;
  abstract part def ControlService :> LogicalComponent;
  abstract part def ManagedSubsystem :> LogicalComponent;

  //============================================================
  // Operations/Maintenance subsystem structural decomposition
  //============================================================

  part def OpsMaintSubsystem :> LogicalComponent {
    attribute subsystemName : String;
    attribute mode          : String;   // e.g., "Startup", "Operational", "Shutdown", "Maintenance", "Safe"
    attribute isReady       : Boolean;

    // External interaction points
    port cmdIn    : in CommandIF;
    port statusOut: out StatusIF;
    port tmOut    : out TelemetryIF;
    port alarmOut : out AlarmIF;

    port maintIn  : in MaintAccessIF;
    port pwrOut   : out PowerControlIF;

    // Internal decomposition
    part startupShutdownMgr : StartupShutdownManager;
    part faultMgr           : FaultHandlingManager;
    part maintModeMgr       : MaintenanceModeManager;

    part healthMon          : HealthMonitor;
    part safeStateCtrl      : SafeStateController;
    part logAndTrace        : LoggingAndTraceability;

    part opsPlanner         : OperationsPlannerAndSequencer;
    part observingCondMgr   : ObservingConditionsManager;

    part performanceAssure  : PerformanceAssuranceService;
    part scienceEnablement  : ScienceOpsEnablement;

    // Key internal connections (BDD-level structural intent)
    connect cmdIn to startupShutdownMgr.cmdIn;
    connect cmdIn to faultMgr.cmdIn;
    connect cmdIn to maintModeMgr.cmdIn;
    connect cmdIn to opsPlanner.cmdIn;

    connect healthMon.tmOut to tmOut;
    connect logAndTrace.tmOut to tmOut;

    connect faultMgr.alarmOut to alarmOut;
    connect safeStateCtrl.alarmOut to alarmOut;

    connect startupShutdownMgr.statusOut to statusOut;
    connect maintModeMgr.statusOut to statusOut;
    connect safeStateCtrl.statusOut to statusOut;

    connect startupShutdownMgr.pwrOut to pwrOut;
    connect safeStateCtrl.pwrOut to pwrOut;

    connect maintIn to maintModeMgr.maintIn;
    connect maintModeMgr.statusOut to logAndTrace.statusIn;
    connect faultMgr.statusOut to logAndTrace.statusIn;
  }

  //============================================================
  // Core managers/services
  //============================================================

  part def StartupShutdownManager :> ControlService {
    attribute startupProfile : String; // e.g., "cold", "warm"
    attribute shutdownProfile: String; // e.g., "park", "storm-safe"

    port cmdIn     : in CommandIF;
    port statusOut : out StatusIF;
    port pwrOut    : out PowerControlIF;
    port alarmOut  : out AlarmIF;
    port tmOut     : out TelemetryIF;
  }

  part def FaultHandlingManager :> ControlService {
    attribute faultPolicy     : String; // e.g., "fail-operational", "fail-safe"
    attribute autoRecover     : Boolean;

    port cmdIn     : in CommandIF;
    port statusOut : out StatusIF;
    port alarmOut  : out AlarmIF;
    port tmOut     : out TelemetryIF;

    part anomalyDetector : AnomalyDetector;
    part diagnoser       : FaultDiagnoser;
    part recoveryExec    : RecoveryExecutor;
  }

  part def MaintenanceModeManager :> ControlService {
    attribute maintenanceWindowId : String;
    attribute lockoutEnabled      : Boolean;

    port cmdIn     : in CommandIF;
    port maintIn   : in MaintAccessIF;
    port statusOut : out StatusIF;
    port alarmOut  : out AlarmIF;
    port tmOut     : out TelemetryIF;

    part accessCtrl  : MaintenanceAccessController;
    part procedures  : MaintenanceProcedureLibrary;
    part toolingIF   : ToolingAndHandlingInterface;
  }

  part def HealthMonitor :> ControlService {
    attribute heartbeatHz : Real;
    attribute healthRollup: String;

    port statusOut : out StatusIF;
    port tmOut     : out TelemetryIF;
    port alarmOut  : out AlarmIF;
  }

  part def SafeStateController :> ControlService {
    attribute safeStateName : String; // e.g., "Parked", "E-Stop", "StormSafe"

    port cmdIn     : in CommandIF;
    port statusOut : out StatusIF;
    port alarmOut  : out AlarmIF;
    port pwrOut    : out PowerControlIF;
    port tmOut     : out TelemetryIF;
  }

  part def LoggingAndTraceability :> ControlService {
    attribute logLevel : String; // e.g., "INFO", "DEBUG", "AUDIT"
    attribute traceId  : String;

    port statusIn : in StatusIF;
    port tmOut    : out TelemetryIF;
  }

  part def OperationsPlannerAndSequencer :> ControlService {
    attribute planId       : String;
    attribute stepIndex    : Integer;

    port cmdIn     : in CommandIF;
    port statusOut : out StatusIF;
    port tmOut     : out TelemetryIF;

    part startupSequence : SequenceLibrary;
    part shutdownSequence: SequenceLibrary;
    part faultPlaybooks  : SequenceLibrary;
    part maintenanceRuns : SequenceLibrary;
  }

  part def ObservingConditionsManager :> ControlService {
    attribute observingBand : String;
    attribute airmassMax    : Real;
    attribute windSpeedMax  : Real;

    port statusOut : out StatusIF;
    port tmOut     : out TelemetryIF;
  }

  part def PerformanceAssuranceService :> ControlService {
    attribute kpiName  : String;
    attribute kpiValue : Real;

    port statusOut : out StatusIF;
    port tmOut     : out TelemetryIF;

    part alignmentPhasingAssure : SubsystemAssurance;
    part pointingTrackingAssure : SubsystemAssurance;
    part exposureControlAssure  : SubsystemAssurance;
    part aoOptionalAssure       : SubsystemAssurance;
    part sensitivityAssure      : SubsystemAssurance;
    part angResAssure           : SubsystemAssurance;
  }

  part def ScienceOpsEnablement :> ControlService {
    attribute scienceMode : String; // e.g., "EarlyUniverse", "Exoplanets"
    port statusOut : out StatusIF;
    port tmOut     : out TelemetryIF;
  }

  //============================================================
  // Supporting parts
  //============================================================

  part def AnomalyDetector :> ControlService {
    attribute detectorVersion : String;
    port tmOut    : out TelemetryIF;
    port alarmOut : out AlarmIF;
  }

  part def FaultDiagnoser :> ControlService {
    attribute modelVersion : String;
    port tmOut : out TelemetryIF;
  }

  part def RecoveryExecutor :> ControlService {
    attribute strategy : String;
    port statusOut : out StatusIF;
    port tmOut     : out TelemetryIF;
  }

  part def MaintenanceAccessController :> ControlService {
    attribute rolePolicy : String;
    port alarmOut : out AlarmIF;
    port tmOut    : out TelemetryIF;
  }

  part def MaintenanceProcedureLibrary :> LogicalComponent {
    attribute libraryVersion : String;
  }

  part def ToolingAndHandlingInterface :> LogicalComponent {
    attribute toolingMode : String;
  }

  part def SequenceLibrary :> LogicalComponent {
    attribute name    : String;
    attribute version : String;
  }

  part def SubsystemAssurance :> ControlService {
    attribute monitoredMetric : String;
    attribute threshold       : Real;
    port tmOut : out TelemetryIF;
  }

  //============================================================
  // Top-level usage for this view (SoI slice)
  //============================================================

  part tmtOpsMaint : OpsMaintSubsystem {
    subsystemName = "TMT Operations/Maintenance";
    mode          = "Operational";
    isReady       = false;
  }

  //============================================================
  // RD-2: Requirements -> implementation mapping (satisfy)
  //============================================================

  // Science goals / stakeholder concerns (enabled by operational modes & traceability)
  satisfy MissionGoalsAndScienceDrivers by tmtOpsMaint.scienceEnablement;
  satisfy EarlyUniverseGoalMapping      by tmtOpsMaint.scienceEnablement;
  satisfy GalaxyEvolutionGoalMapping    by tmtOpsMaint.scienceEnablement;
  satisfy ExoplanetsGoalMapping         by tmtOpsMaint.scienceEnablement;
  satisfy BHgrowthGoalMapping           by tmtOpsMaint.scienceEnablement;
  satisfy StakeholderGroupConcerns      by tmtOpsMaint.logAndTrace;

  // Observing assumptions/conditions (operational gating, monitoring, and enforcement)
  satisfy ObservingAssumptionsAndConditionsDef by tmtOpsMaint.observingCondMgr;
  satisfy ObservingAssumptionsAndConditions    by tmtOpsMaint.observingCondMgr;

  // Functional/performance-related definitions (assured via performance services + fault/safe-state handling)
  satisfy FR_PhasedSegmentedApertureDef by tmtOpsMaint.performanceAssure.alignmentPhasingAssure;
  satisfy FR_PhasedSegmentedAperture    by tmtOpsMaint.performanceAssure.alignmentPhasingAssure;

  satisfy FR_PointingTrackingAndStabilityDef by tmtOpsMaint.performanceAssure.pointingTrackingAssure;
  satisfy FR_PointingTrackingAndStability    by tmtOpsMaint.performanceAssure.pointingTrackingAssure;

  satisfy FR_ObservationPhasingAndExposureControlDef by tmtOpsMaint.performanceAssure.exposureControlAssure;
  satisfy FR_ObservationPhasingAndExposureControl    by tmtOpsMaint.performanceAssure.exposureControlAssure;

  satisfy FR_AdaptiveOpticsOptionalDef by tmtOpsMaint.performanceAssure.aoOptionalAssure;
  satisfy FR_AdaptiveOpticsOptional    by tmtOpsMaint.performanceAssure.aoOptionalAssure;

  satisfy SensitivityPerformanceNFRDef       by tmtOpsMaint.performanceAssure.sensitivityAssure;
  satisfy AngularResolutionPerformanceNFRDef by tmtOpsMaint.performanceAssure.angResAssure;

  // Cross-cutting: fault response and safe state are part of achieving robust performance in operations
  satisfy FR_PointingTrackingAndStability by tmtOpsMaint.faultMgr;
  satisfy FR_PhasedSegmentedAperture      by tmtOpsMaint.safeStateCtrl;
}
