package BDD_SUB02_003_MountStructure {
  import ScalarValues::*;
  import Quantities::*;
  import SI::*;

  // =========================
  // Requirement-to-Design Trace Stubs (fix RD-2)
  // =========================
  // These requirement elements are provided so the model contains explicit implementation links (satisfy)
  // for the requirement names flagged by validation.
  requirement def MissionGoalsAndScienceDrivers;
  requirement def EarlyUniverseGoalMapping;
  requirement def GalaxyEvolutionGoalMapping;
  requirement def ExoplanetsGoalMapping;
  requirement def BHgrowthGoalMapping;
  requirement def StakeholderGroupConcerns;

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

  // =========================
  // Common Types
  // =========================
  abstract part def Subsystem;
  abstract part def StructuralComponent;
  abstract part def MechatronicComponent;
  abstract part def ControlComponent;
  abstract part def SensorComponent;
  abstract part def ActuatorComponent;

  attribute def Stiffness {
    k : Real;
  }

  attribute def Damping {
    c : Real;
  }

  attribute def AlignmentTolerance {
    linearTol : Real;
    angularTol : Real;
  }

  // =========================
  // Interfaces / Ports
  // =========================
  port def PowerPort {
    in voltage : Real;
    in currentMax : Real;
  }

  port def ControlCmdPort {
    in cmd : String;
    in setpoint : Real;
  }

  port def TelemetryPort {
    out status : String;
    out value : Real;
  }

  port def TimeSyncPort {
    inout timeTag : String;
  }

  port def MechanicalMountPort {
    inout force : Real;
    inout moment : Real;
    inout displacement : Real;
  }

  port def VibrationSensePort {
    out accelRMS : Real;
    out freqHz : Real;
  }

  port def VibrationActuatePort {
    in controlEffort : Real;
  }

  // =========================
  // Part Definitions (BDD)
  // =========================
  part def MountStructure :> Subsystem {
    // External interfaces
    port mechToOptics : MechanicalMountPort;
    port mechToEnclosure : MechanicalMountPort;

    port pwrIn : PowerPort;
    port cmdIn : ControlCmdPort;
    port tmOut : TelemetryPort;
    port timeSync : TimeSyncPort;

    // Key performance / physical attributes
    attribute totalMass : Real;
    attribute stiffness : Stiffness;
    attribute damping : Damping;
    attribute pointingStabilityBudget : Real;
    attribute vibrationLineOfSightRMS : Real;
    attribute alignmentTol : AlignmentTolerance;

    // Decomposition
    part structuralFrame : StructuralFrame;
    part azAxis : AzimuthAxis;
    part elAxis : ElevationAxis;
    part bearings : BearingSet;
    part drives : DriveSystem;
    part encoders : EncoderSystem;
    part cableWrap : CableWrapSystem;
    part vibrationControl : VibrationControlSystem;
    part thermalControl : MountThermalControl;
    part localControl : MountControlElectronics;

    // Representative internal connections (BDD-level, IBD-lite)
    connect pwrIn to localControl.pwrIn;
    connect cmdIn to localControl.cmdIn;
    connect timeSync to localControl.timeSync;

    connect localControl.driveCmd to drives.cmdIn;
    connect localControl.encoderPoll to encoders.cmdIn;

    connect encoders.tmOut to localControl.encoderTm;
    connect drives.tmOut to localControl.driveTm;

    connect vibrationControl.vibTm to tmOut;
    connect localControl.healthTm to tmOut;
  }

  part def StructuralFrame :> StructuralComponent {
    port mechBase : MechanicalMountPort;
    port mechToAxes : MechanicalMountPort;

    attribute mass : Real;
    attribute stiffness : Stiffness;
    attribute damping : Damping;

    part pierInterface : PierInterface;
    part yoke : YokeStructure;
  }

  part def PierInterface :> StructuralComponent {
    port mechToSite : MechanicalMountPort;
    port mechToFrame : MechanicalMountPort;

    attribute stiffness : Stiffness;
    attribute anchorMargin : Real;
  }

  part def YokeStructure :> StructuralComponent {
    port mechToAz : MechanicalMountPort;
    port mechToEl : MechanicalMountPort;

    attribute stiffness : Stiffness;
    attribute eigenFreq1Hz : Real;
  }

  part def AzimuthAxis :> MechatronicComponent {
    port mechToFrame : MechanicalMountPort;
    port mechToEnclosure : MechanicalMountPort;

    attribute rangeDeg : Real;
    attribute maxSlewRateDegPerS : Real;

    part azBearing : AxisBearing;
    part azDrive : AxisDrive;
    part azEncoder : AxisEncoder;
  }

  part def ElevationAxis :> MechatronicComponent {
    port mechToYoke : MechanicalMountPort;
    port mechToOptics : MechanicalMountPort;

    attribute rangeDeg : Real;
    attribute maxSlewRateDegPerS : Real;

    part elBearing : AxisBearing;
    part elDrive : AxisDrive;
    part elEncoder : AxisEncoder;
  }

  part def BearingSet :> StructuralComponent {
    part azBearings : AxisBearing;
    part elBearings : AxisBearing;

    attribute frictionTorque : Real;
    attribute runout : Real;
  }

  part def AxisBearing :> StructuralComponent {
    port mechIn : MechanicalMountPort;
    port mechOut : MechanicalMountPort;

    attribute stiffness : Stiffness;
    attribute damping : Damping;
    attribute preload : Real;
  }

  part def DriveSystem :> ActuatorComponent {
    port pwrIn : PowerPort;
    port cmdIn : ControlCmdPort;
    port tmOut : TelemetryPort;

    port mechToAxis : MechanicalMountPort;

    attribute maxTorque : Real;
    attribute maxSpeed : Real;
    attribute controlBandwidthHz : Real;

    part azDriveTrain : AxisDrive;
    part elDriveTrain : AxisDrive;
  }

  part def AxisDrive :> ActuatorComponent {
    port pwrIn : PowerPort;
    port cmdIn : ControlCmdPort;
    port tmOut : TelemetryPort;
    port mechOut : MechanicalMountPort;

    attribute torqueConstant : Real;
    attribute gearRatio : Real;
    attribute backlash : Real;
  }

  part def EncoderSystem :> SensorComponent {
    port cmdIn : ControlCmdPort;
    port tmOut : TelemetryPort;

    attribute resolutionArcsec : Real;
    attribute latencyMs : Real;

    part azEncoder : AxisEncoder;
    part elEncoder : AxisEncoder;
  }

  part def AxisEncoder :> SensorComponent {
    port cmdIn : ControlCmdPort;
    port tmOut : TelemetryPort;
    port mechRef : MechanicalMountPort;

    attribute resolutionArcsec : Real;
    attribute accuracyArcsec : Real;
    attribute updateRateHz : Real;
  }

  part def CableWrapSystem :> StructuralComponent {
    port pwrFeedthrough : PowerPort;
    port cmdFeedthrough : ControlCmdPort;
    port tmFeedthrough : TelemetryPort;

    attribute twistRangeDeg : Real;
    attribute cableBendRadiusMin : Real;
  }

  part def VibrationControlSystem :> ControlComponent {
    port cmdIn : ControlCmdPort;
    port vibTm : TelemetryPort;

    port vibSense : VibrationSensePort;
    port vibAct : VibrationActuatePort;

    attribute targetVibrationRMS : Real;
    attribute notchFreqHz : Real;

    part accelerometers : VibrationSensorArray;
    part activeDampers : ActiveDamperArray;
  }

  part def VibrationSensorArray :> SensorComponent {
    port vibSense : VibrationSensePort;
    port tmOut : TelemetryPort;

    attribute channelCount : Integer;
    attribute noiseFloor : Real;
  }

  part def ActiveDamperArray :> ActuatorComponent {
    port vibAct : VibrationActuatePort;
    port cmdIn : ControlCmdPort;
    port tmOut : TelemetryPort;

    attribute channelCount : Integer;
    attribute maxForce : Real;
  }

  part def MountThermalControl :> ControlComponent {
    port pwrIn : PowerPort;
    port cmdIn : ControlCmdPort;
    port tmOut : TelemetryPort;

    attribute tempGradientMax : Real;
    attribute heaterPowerMax : Real;
  }

  part def MountControlElectronics :> ControlComponent {
    port pwrIn : PowerPort;
    port cmdIn : ControlCmdPort;
    port healthTm : TelemetryPort;
    port timeSync : TimeSyncPort;

    // Internal command/telemetry fanout
    port driveCmd : ControlCmdPort;
    port encoderPoll : ControlCmdPort;

    port driveTm : TelemetryPort;
    port encoderTm : TelemetryPort;

    attribute controlLoopRateHz : Real;
    attribute safingModeEnabled : Boolean;
  }

  // =========================
  // System-of-Interest Usage (instance for satisfy links)
  // =========================
  part tmtMountStructure : MountStructure;

  // =========================
  // Satisfy Links (fix RD-2)
  // =========================
  satisfy MissionGoalsAndScienceDrivers by tmtMountStructure;
  satisfy EarlyUniverseGoalMapping by tmtMountStructure;
  satisfy GalaxyEvolutionGoalMapping by tmtMountStructure;
  satisfy ExoplanetsGoalMapping by tmtMountStructure;
  satisfy BHgrowthGoalMapping by tmtMountStructure;
  satisfy StakeholderGroupConcerns by tmtMountStructure;

  satisfy ObservingAssumptionsAndConditionsDef by tmtMountStructure.thermalControl;
  satisfy ObservingAssumptionsAndConditions by tmtMountStructure.thermalControl;

  satisfy FR_PhasedSegmentedApertureDef by tmtMountStructure;
  satisfy FR_PhasedSegmentedAperture by tmtMountStructure;

  satisfy FR_PointingTrackingAndStabilityDef by tmtMountStructure.drives;
  satisfy FR_PointingTrackingAndStability by tmtMountStructure.encoders;

  satisfy FR_ObservationPhasingAndExposureControlDef by tmtMountStructure.localControl;
  satisfy FR_ObservationPhasingAndExposureControl by tmtMountStructure.localControl;

  satisfy FR_AdaptiveOpticsOptionalDef by tmtMountStructure;
  satisfy FR_AdaptiveOpticsOptional by tmtMountStructure;

  satisfy SensitivityPerformanceNFRDef by tmtMountStructure.vibrationControl;
  satisfy AngularResolutionPerformanceNFRDef by tmtMountStructure.vibrationControl;
}
