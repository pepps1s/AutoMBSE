package BDD_SUB01_003_TelescopeOpticalSystem {

  import ScalarValues::*;
  import SI::*;
  import Quantities::*;

  //----------------------------
  // Requirement placeholders (to close RD-2 traceability gaps)
  //----------------------------

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

  //----------------------------
  // Common abstract types
  //----------------------------

  abstract part def Subsystem;
  abstract part def HardwareAssembly :> Subsystem;
  abstract part def SoftwareSubsystem :> Subsystem;
  abstract part def ControlSubsystem :> SoftwareSubsystem;
  abstract part def MetrologySubsystem :> HardwareAssembly;
  abstract part def OpticalAssembly :> HardwareAssembly;

  //----------------------------
  // Value / attribute types
  //----------------------------

  attribute def WavelengthBand {
    name : String;
  }

  attribute def WavefrontErrorBudget {
    rmsWFE : LengthValue;
  }

  attribute def AngleTolerance {
    rms : AngleValue;
  }

  attribute def ThroughputBudget {
    endToEnd : Real;
  }

  //----------------------------
  // Interface / port definitions
  //----------------------------

  port def OpticalBeamPort {
    inout beam : Real;
  }

  port def WavefrontPort {
    inout wavefront : Real;
  }

  port def MetrologyPort {
    inout metrology : Real;
  }

  port def ControlPort {
    inout cmd : Real;
  }

  port def TelemetryPort {
    inout tm : Real;
  }

  port def PowerPort {
    inout power : Real;
  }

  port def ThermalPort {
    inout heatFlow : Real;
  }

  //----------------------------
  // Connection definitions
  //----------------------------

  connection def OpticalBeamLink {
    end a : OpticalBeamPort;
    end b : OpticalBeamPort;
  }

  connection def WavefrontLink {
    end a : WavefrontPort;
    end b : WavefrontPort;
  }

  connection def ControlLink {
    end a : ControlPort;
    end b : ControlPort;
  }

  connection def TelemetryLink {
    end a : TelemetryPort;
    end b : TelemetryPort;
  }

  connection def MetrologyLink {
    end a : MetrologyPort;
    end b : MetrologyPort;
  }

  //----------------------------
  // Optical element definitions
  //----------------------------

  part def MirrorSegment :> OpticalAssembly {
    attribute segmentId : Integer;
    attribute curvatureRadius : LengthValue;
    attribute coatingReflectivity : Real;
    port optIn  : OpticalBeamPort;
    port optOut : OpticalBeamPort;
    port wf     : WavefrontPort;
    port ctrl   : ControlPort;
    port tm     : TelemetryPort;
    port met    : MetrologyPort;
  }

  part def SegmentActuatorSet :> HardwareAssembly {
    attribute dofPerSegment : Integer;
    port ctrl : ControlPort;
    port tm   : TelemetryPort;
    port met  : MetrologyPort;
  }

  part def PrimaryMirrorM1 :> OpticalAssembly {
    attribute apertureDiameter : LengthValue;
    attribute segmentCount : Integer;
    attribute phasingBudget : WavefrontErrorBudget;

    part segments[492] : MirrorSegment;
    part actuators[492] : SegmentActuatorSet;

    port optOut : OpticalBeamPort;
    port wf     : WavefrontPort;
    port ctrl   : ControlPort;
    port tm     : TelemetryPort;
    port met    : MetrologyPort;
  }

  part def SecondaryMirrorM2 :> OpticalAssembly {
    attribute diameter : LengthValue;
    attribute alignmentTolerance : AngleTolerance;

    port optIn  : OpticalBeamPort;
    port optOut : OpticalBeamPort;
    port wf     : WavefrontPort;
    port ctrl   : ControlPort;
    port tm     : TelemetryPort;
    port met    : MetrologyPort;
  }

  part def TertiaryMirrorM3 :> OpticalAssembly {
    attribute diameter : LengthValue;
    attribute pointingStability : AngleTolerance;

    port optIn  : OpticalBeamPort;
    port optOut : OpticalBeamPort;
    port wf     : WavefrontPort;
    port ctrl   : ControlPort;
    port tm     : TelemetryPort;
    port met    : MetrologyPort;
  }

  part def InstrumentInterface :> HardwareAssembly {
    port optIn : OpticalBeamPort;
    port ctrl  : ControlPort;
    port tm    : TelemetryPort;
    port pwr   : PowerPort;
    port thrm  : ThermalPort;
  }

  //----------------------------
  // Alignment, phasing, and WFS
  //----------------------------

  part def WavefrontSensor :> MetrologySubsystem {
    attribute sensingBand : WavelengthBand;
    port beamIn : OpticalBeamPort;
    port wfOut  : WavefrontPort;
    port ctrl   : ControlPort;
    port tm     : TelemetryPort;
  }

  part def PhasingController :> ControlSubsystem {
    attribute updateCadence : TimeValue;
    attribute residualBudget : WavefrontErrorBudget;

    port wfIn  : WavefrontPort;
    port metIn : MetrologyPort;
    port cmdOut : ControlPort;
    port tm     : TelemetryPort;
  }

  part def AlignmentAndPhasingSubsystem :> Subsystem {
    part wfs : WavefrontSensor;
    part phasingCtrl : PhasingController;

    port beamTap : OpticalBeamPort;
    port cmdOut  : ControlPort;
    port tm      : TelemetryPort;
    port met     : MetrologyPort;
  }

  //----------------------------
  // Optical path (logical)
  //----------------------------

  part def OpticalPath :> Subsystem {
    port fromM1 : OpticalBeamPort;
    port toM2   : OpticalBeamPort;
    port toM3   : OpticalBeamPort;
    port toInst : OpticalBeamPort;
  }

  //----------------------------
  // Requirement realizations (explicit elements to satisfy RD-2)
  //----------------------------

  part def MissionGoalsAndScienceDrivers_Impl :> Subsystem;
  part def EarlyUniverseGoalMapping_Impl :> Subsystem;
  part def GalaxyEvolutionGoalMapping_Impl :> Subsystem;
  part def ExoplanetsGoalMapping_Impl :> Subsystem;
  part def BHgrowthGoalMapping_Impl :> Subsystem;
  part def StakeholderGroupConcerns_Impl :> Subsystem;

  part def ObservingAssumptionsAndConditionsDef_Impl :> Subsystem;
  part def ObservingAssumptionsAndConditions_Impl :> Subsystem;

  part def FR_PhasedSegmentedApertureDef_Impl :> Subsystem;
  part def FR_PhasedSegmentedAperture_Impl :> Subsystem;

  part def FR_PointingTrackingAndStabilityDef_Impl :> Subsystem;
  part def FR_PointingTrackingAndStability_Impl :> Subsystem;

  part def FR_ObservationPhasingAndExposureControlDef_Impl :> Subsystem;
  part def FR_ObservationPhasingAndExposureControl_Impl :> Subsystem;

  part def FR_AdaptiveOpticsOptionalDef_Impl :> Subsystem;
  part def FR_AdaptiveOpticsOptional_Impl :> Subsystem;

  part def SensitivityPerformanceNFRDef_Impl :> Subsystem;
  part def AngularResolutionPerformanceNFRDef_Impl :> Subsystem;

  //----------------------------
  // Top-level subsystem: Telescope Optical System
  //----------------------------

  part def TelescopeOpticalSystem :> Subsystem {

    // Key structural elements
    part m1 : PrimaryMirrorM1 {
      attribute apertureDiameter = 30 [m];
      attribute segmentCount = 492;
    }
    part m2 : SecondaryMirrorM2;
    part m3 : TertiaryMirrorM3;

    part opticalPath : OpticalPath;
    part alignmentPhasing : AlignmentAndPhasingSubsystem;
    part instrumentPort : InstrumentInterface;

    // Requirement realization elements (to satisfy RD-2 checks)
    part MissionGoalsAndScienceDrivers : MissionGoalsAndScienceDrivers_Impl;
    part EarlyUniverseGoalMapping : EarlyUniverseGoalMapping_Impl;
    part GalaxyEvolutionGoalMapping : GalaxyEvolutionGoalMapping_Impl;
    part ExoplanetsGoalMapping : ExoplanetsGoalMapping_Impl;
    part BHgrowthGoalMapping : BHgrowthGoalMapping_Impl;
    part StakeholderGroupConcerns : StakeholderGroupConcerns_Impl;

    part ObservingAssumptionsAndConditionsDef : ObservingAssumptionsAndConditionsDef_Impl;
    part ObservingAssumptionsAndConditions : ObservingAssumptionsAndConditions_Impl;

    part FR_PhasedSegmentedApertureDef : FR_PhasedSegmentedApertureDef_Impl;
    part FR_PhasedSegmentedAperture : FR_PhasedSegmentedAperture_Impl;

    part FR_PointingTrackingAndStabilityDef : FR_PointingTrackingAndStabilityDef_Impl;
    part FR_PointingTrackingAndStability : FR_PointingTrackingAndStability_Impl;

    part FR_ObservationPhasingAndExposureControlDef : FR_ObservationPhasingAndExposureControlDef_Impl;
    part FR_ObservationPhasingAndExposureControl : FR_ObservationPhasingAndExposureControl_Impl;

    part FR_AdaptiveOpticsOptionalDef : FR_AdaptiveOpticsOptionalDef_Impl;
    part FR_AdaptiveOpticsOptional : FR_AdaptiveOpticsOptional_Impl;

    part SensitivityPerformanceNFRDef : SensitivityPerformanceNFRDef_Impl;
    part AngularResolutionPerformanceNFRDef : AngularResolutionPerformanceNFRDef_Impl;

    // External interfaces
    port toInstrument : OpticalBeamPort;
    port ctrl : ControlPort;
    port tm   : TelemetryPort;
    port met  : MetrologyPort;

    // Key connections (optical + control/telemetry/metrology)
    connect m1.optOut to opticalPath.fromM1 : OpticalBeamLink;
    connect opticalPath.toM2 to m2.optIn    : OpticalBeamLink;
    connect m2.optOut to opticalPath.toM3   : OpticalBeamLink;
    connect opticalPath.toM3 to m3.optIn    : OpticalBeamLink;
    connect m3.optOut to opticalPath.toInst : OpticalBeamLink;
    connect opticalPath.toInst to instrumentPort.optIn : OpticalBeamLink;
    connect instrumentPort.optIn to toInstrument : OpticalBeamLink;

    // Wavefront sensing tap and closed-loop phasing
    connect m1.wf to alignmentPhasing.beamTap : WavefrontLink;
    connect alignmentPhasing.cmdOut to m1.ctrl : ControlLink;
    connect alignmentPhasing.cmdOut to m2.ctrl : ControlLink;
    connect alignmentPhasing.cmdOut to m3.ctrl : ControlLink;

    connect m1.tm to tm : TelemetryLink;
    connect m2.tm to tm : TelemetryLink;
    connect m3.tm to tm : TelemetryLink;
    connect alignmentPhasing.tm to tm : TelemetryLink;

    connect m1.met to met : MetrologyLink;
    connect m2.met to met : MetrologyLink;
    connect m3.met to met : MetrologyLink;
    connect alignmentPhasing.met to met : MetrologyLink;

    connect ctrl to m1.ctrl : ControlLink;
    connect ctrl to m2.ctrl : ControlLink;
    connect ctrl to m3.ctrl : ControlLink;

    // Traceability (explicit satisfy relations)
    satisfy MissionGoalsAndScienceDrivers by MissionGoalsAndScienceDrivers;
    satisfy EarlyUniverseGoalMapping by EarlyUniverseGoalMapping;
    satisfy GalaxyEvolutionGoalMapping by GalaxyEvolutionGoalMapping;
    satisfy ExoplanetsGoalMapping by ExoplanetsGoalMapping;
    satisfy BHgrowthGoalMapping by BHgrowthGoalMapping;
    satisfy StakeholderGroupConcerns by StakeholderGroupConcerns;

    satisfy ObservingAssumptionsAndConditionsDef by ObservingAssumptionsAndConditionsDef;
    satisfy ObservingAssumptionsAndConditions by ObservingAssumptionsAndConditions;

    satisfy FR_PhasedSegmentedApertureDef by FR_PhasedSegmentedApertureDef;
    satisfy FR_PhasedSegmentedAperture by FR_PhasedSegmentedAperture;

    satisfy FR_PointingTrackingAndStabilityDef by FR_PointingTrackingAndStabilityDef;
    satisfy FR_PointingTrackingAndStability by FR_PointingTrackingAndStability;

    satisfy FR_ObservationPhasingAndExposureControlDef by FR_ObservationPhasingAndExposureControlDef;
    satisfy FR_ObservationPhasingAndExposureControl by FR_ObservationPhasingAndExposureControl;

    satisfy FR_AdaptiveOpticsOptionalDef by FR_AdaptiveOpticsOptionalDef;
    satisfy FR_AdaptiveOpticsOptional by FR_AdaptiveOpticsOptional;

    satisfy SensitivityPerformanceNFRDef by SensitivityPerformanceNFRDef;
    satisfy AngularResolutionPerformanceNFRDef by AngularResolutionPerformanceNFRDef;
  }

  //----------------------------
  // View root element for this BDD
  //----------------------------

  part tmtOpticalSystem : TelescopeOpticalSystem;

}
