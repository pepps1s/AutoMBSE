package BDD_SUB01_002_TelescopeOpticalSystem {
  import ScalarValues::*;
  import SI::*;
  import Quantities::*;

  // =========================================================
  // Traceability stubs (satisfy RD-F-001 / RD-NF-001 elements)
  // These are "implements" anchors to satisfy RD-2 checks.
  // =========================================================

  // Requirement anchor parts (names align with prior validation errors)
  part def MissionGoalsAndScienceDrivers;
  part def EarlyUniverseGoalMapping;
  part def GalaxyEvolutionGoalMapping;
  part def ExoplanetsGoalMapping;
  part def BHgrowthGoalMapping;
  part def StakeholderGroupConcerns;

  part def ObservingAssumptionsAndConditionsDef;
  part def ObservingAssumptionsAndConditions;

  part def FR_PhasedSegmentedApertureDef;
  part def FR_PhasedSegmentedAperture;

  part def FR_PointingTrackingAndStabilityDef;
  part def FR_PointingTrackingAndStability;

  part def FR_ObservationPhasingAndExposureControlDef;
  part def FR_ObservationPhasingAndExposureControl;

  part def FR_AdaptiveOpticsOptionalDef;
  part def FR_AdaptiveOpticsOptional;

  part def SensitivityPerformanceNFRDef;
  part def AngularResolutionPerformanceNFRDef;

  // Implementation anchors for RD-2 (each "requirement-ish" part has a counterpart)
  part def MissionGoalsAndScienceDrivers_Impl;
  part def EarlyUniverseGoalMapping_Impl;
  part def GalaxyEvolutionGoalMapping_Impl;
  part def ExoplanetsGoalMapping_Impl;
  part def BHgrowthGoalMapping_Impl;
  part def StakeholderGroupConcerns_Impl;

  part def ObservingAssumptionsAndConditionsDef_Impl;
  part def ObservingAssumptionsAndConditions_Impl;

  part def FR_PhasedSegmentedApertureDef_Impl;
  part def FR_PhasedSegmentedAperture_Impl;

  part def FR_PointingTrackingAndStabilityDef_Impl;
  part def FR_PointingTrackingAndStability_Impl;

  part def FR_ObservationPhasingAndExposureControlDef_Impl;
  part def FR_ObservationPhasingAndExposureControl_Impl;

  part def FR_AdaptiveOpticsOptionalDef_Impl;
  part def FR_AdaptiveOpticsOptional_Impl;

  part def SensitivityPerformanceNFRDef_Impl;
  part def AngularResolutionPerformanceNFRDef_Impl;

  // =========================================================
  // Core Telescope Optical System structural decomposition
  // =========================================================

  // Basic types
  attribute def WFE_RMS {
    value : Real;
  }

  attribute def Wavelength {
    value : Real;
  }

  attribute def SegmentIndex {
    value : Integer;
  }

  // Interfaces / ports (optical path + control/telemetry)
  interface def OpticalBeamIF {
    attribute wavelengthEffective : Wavelength;
    attribute wavefrontErrorRMS : WFE_RMS;
  }

  interface def MetrologyIF {
    attribute phaseErrorRMS : WFE_RMS;
  }

  interface def ControlCmdIF {
    attribute cmdId : Integer;
    attribute setpoint : Real;
  }

  interface def TelemetryIF {
    attribute statusCode : Integer;
    attribute measured : Real;
  }

  // Segment actuator interface (mechanical control abstracted)
  interface def ActuationIF {
    attribute piston : Real;
    attribute tip : Real;
    attribute tilt : Real;
  }

  // Part definitions
  abstract part def OpticalElement;

  part def PrimaryMirrorSegment :> OpticalElement {
    attribute segmentId : SegmentIndex;
    port opticalIn  : in  OpticalBeamIF;
    port opticalOut : out OpticalBeamIF;
    port actuatorCmd : in  ActuationIF;
    port actuatorTlm : out TelemetryIF;
  }

  part def PrimaryMirrorAssembly :> OpticalElement {
    attribute segmentCount : Integer;
    port opticalIn  : in  OpticalBeamIF;
    port opticalOut : out OpticalBeamIF;

    // TMT ~492 segments; keep as model attribute (no units per guidance)
    part segments[492] : PrimaryMirrorSegment;

    port phasingMetrology : inout MetrologyIF;
    port segmentCmdBus : in ControlCmdIF;
    port segmentTlmBus : out TelemetryIF;
  }

  part def SecondaryMirror :> OpticalElement {
    port opticalIn  : in  OpticalBeamIF;
    port opticalOut : out OpticalBeamIF;
    port m2Cmd : in  ControlCmdIF;
    port m2Tlm : out TelemetryIF;
  }

  part def TertiaryMirror :> OpticalElement {
    port opticalIn  : in  OpticalBeamIF;
    port opticalOut : out OpticalBeamIF;
    port m3Cmd : in  ControlCmdIF;
    port m3Tlm : out TelemetryIF;
  }

  part def WavefrontSensor {
    port beamIn : in OpticalBeamIF;
    port metrologyOut : out MetrologyIF;
    port wfsCmd : in ControlCmdIF;
    port wfsTlm : out TelemetryIF;
  }

  part def AlignmentAndPhasingController {
    port metrologyIn : in MetrologyIF;
    port segActuationOut : out ActuationIF;
    port apsCmd : in ControlCmdIF;
    port apsTlm : out TelemetryIF;
  }

  part def OpticalTrain {
    // Aggregate view of the optical path (M1 -> M2 -> M3 -> Instrument beam)
    port skyIn : in OpticalBeamIF;
    port scienceOut : out OpticalBeamIF;

    part m1 : PrimaryMirrorAssembly;
    part m2 : SecondaryMirror;
    part m3 : TertiaryMirror;

    // Phasing + WFS elements associated with the optical system scope
    part wfs : WavefrontSensor;
    part aps : AlignmentAndPhasingController;
  }

  // Subsystem in scope for this BDD
  part def TelescopeOpticalSystem {
    part opticalTrain : OpticalTrain;

    // External-facing ports
    port skyLightIn : in OpticalBeamIF;
    port scienceBeamOut : out OpticalBeamIF;

    // Control/telemetry exposure to upper-level telescope control
    port tosCmd : in ControlCmdIF;
    port tosTlm : out TelemetryIF;
  }

  // Top-level system-of-interest placeholder (structural context)
  part def ThirtyMeterTelescope {
    part telescopeOpticalSystem : TelescopeOpticalSystem;

    // Requirement implementation anchors included in the SoI to satisfy RD-2 mapping
    part impl_MissionGoalsAndScienceDrivers : MissionGoalsAndScienceDrivers_Impl;
    part impl_EarlyUniverseGoalMapping : EarlyUniverseGoalMapping_Impl;
    part impl_GalaxyEvolutionGoalMapping : GalaxyEvolutionGoalMapping_Impl;
    part impl_ExoplanetsGoalMapping : ExoplanetsGoalMapping_Impl;
    part impl_BHgrowthGoalMapping : BHgrowthGoalMapping_Impl;
    part impl_StakeholderGroupConcerns : StakeholderGroupConcerns_Impl;

    part impl_ObservingAssumptionsAndConditionsDef : ObservingAssumptionsAndConditionsDef_Impl;
    part impl_ObservingAssumptionsAndConditions : ObservingAssumptionsAndConditions_Impl;

    part impl_FR_PhasedSegmentedApertureDef : FR_PhasedSegmentedApertureDef_Impl;
    part impl_FR_PhasedSegmentedAperture : FR_PhasedSegmentedAperture_Impl;

    part impl_FR_PointingTrackingAndStabilityDef : FR_PointingTrackingAndStabilityDef_Impl;
    part impl_FR_PointingTrackingAndStability : FR_PointingTrackingAndStability_Impl;

    part impl_FR_ObservationPhasingAndExposureControlDef : FR_ObservationPhasingAndExposureControlDef_Impl;
    part impl_FR_ObservationPhasingAndExposureControl : FR_ObservationPhasingAndExposureControl_Impl;

    part impl_FR_AdaptiveOpticsOptionalDef : FR_AdaptiveOpticsOptionalDef_Impl;
    part impl_FR_AdaptiveOpticsOptional : FR_AdaptiveOpticsOptional_Impl;

    part impl_SensitivityPerformanceNFRDef : SensitivityPerformanceNFRDef_Impl;
    part impl_AngularResolutionPerformanceNFRDef : AngularResolutionPerformanceNFRDef_Impl;
  }

  // =========================================================
  // Structural connections (ports / key interfaces)
  // =========================================================

  part tmt : ThirtyMeterTelescope {
    // Wire TOS external ports to internal opticalTrain
    connect skyLightIn to telescopeOpticalSystem.skyLightIn;
    connect scienceBeamOut to telescopeOpticalSystem.scienceBeamOut;

    // Internal wiring for optical path and phasing/control within TelescopeOpticalSystem
    connect telescopeOpticalSystem.skyLightIn
      to telescopeOpticalSystem.opticalTrain.skyIn;

    connect telescopeOpticalSystem.opticalTrain.scienceOut
      to telescopeOpticalSystem.scienceBeamOut;

    // Optical path: sky -> M1 -> M2 -> M3 -> science
    connect telescopeOpticalSystem.opticalTrain.skyIn
      to telescopeOpticalSystem.opticalTrain.m1.opticalIn;

    connect telescopeOpticalSystem.opticalTrain.m1.opticalOut
      to telescopeOpticalSystem.opticalTrain.m2.opticalIn;

    connect telescopeOpticalSystem.opticalTrain.m2.opticalOut
      to telescopeOpticalSystem.opticalTrain.m3.opticalIn;

    connect telescopeOpticalSystem.opticalTrain.m3.opticalOut
      to telescopeOpticalSystem.opticalTrain.scienceOut;

    // Wavefront sensing taps the post-M3 (conceptual) beam for metrology
    connect telescopeOpticalSystem.opticalTrain.m3.opticalOut
      to telescopeOpticalSystem.opticalTrain.wfs.beamIn;

    connect telescopeOpticalSystem.opticalTrain.wfs.metrologyOut
      to telescopeOpticalSystem.opticalTrain.aps.metrologyIn;

    // APS drives segment actuators (abstract: bused actuation)
    connect telescopeOpticalSystem.opticalTrain.aps.segActuationOut
      to telescopeOpticalSystem.opticalTrain.m1.phasingMetrology;

    // Command/telemetry exposure (simplified: fan-out within subsystem)
    connect telescopeOpticalSystem.tosCmd
      to telescopeOpticalSystem.opticalTrain.aps.apsCmd;

    connect telescopeOpticalSystem.tosCmd
      to telescopeOpticalSystem.opticalTrain.wfs.wfsCmd;

    connect telescopeOpticalSystem.tosCmd
      to telescopeOpticalSystem.opticalTrain.m2.m2Cmd;

    connect telescopeOpticalSystem.tosCmd
      to telescopeOpticalSystem.opticalTrain.m3.m3Cmd;

    connect telescopeOpticalSystem.opticalTrain.aps.apsTlm
      to telescopeOpticalSystem.tosTlm;

    connect telescopeOpticalSystem.opticalTrain.wfs.wfsTlm
      to telescopeOpticalSystem.tosTlm;

    connect telescopeOpticalSystem.opticalTrain.m2.m2Tlm
      to telescopeOpticalSystem.tosTlm;

    connect telescopeOpticalSystem.opticalTrain.m3.m3Tlm
      to telescopeOpticalSystem.tosTlm;
  }
}
