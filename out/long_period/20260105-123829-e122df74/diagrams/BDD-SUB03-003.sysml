package BDD_SUB03_003_EnclosureDome_SiteInfrastructure {

  import ScalarValues::*;
  import Quantities::*;
  import SI::*;

  /////////////////////////////////
  // Requirement stubs (for traceability implementation)
  /////////////////////////////////

  requirement def MissionGoalsAndScienceDrivers {
    doc = "Top-level science drivers and mission goals for TMT.";
  }

  requirement def EarlyUniverseGoalMapping {
    doc = "Science goal mapping: early universe.";
  }

  requirement def GalaxyEvolutionGoalMapping {
    doc = "Science goal mapping: galaxy evolution.";
  }

  requirement def ExoplanetsGoalMapping {
    doc = "Science goal mapping: exoplanets.";
  }

  requirement def BHgrowthGoalMapping {
    doc = "Science goal mapping: black hole growth.";
  }

  requirement def StakeholderGroupConcerns {
    doc = "Stakeholder concerns and constraints (including site/community).";
  }

  requirement def ObservingAssumptionsAndConditionsDef {
    doc = "Observing assumptions/conditions definition for performance context.";
  }

  requirement def FR_PhasedSegmentedApertureDef {
    doc = "Functional/performance requirement definition: phased segmented aperture.";
  }

  requirement def FR_PointingTrackingAndStabilityDef {
    doc = "Functional/performance requirement definition: pointing, tracking, stability.";
  }

  requirement def FR_ObservationPhasingAndExposureControlDef {
    doc = "Functional/performance requirement definition: phasing/exposure control.";
  }

  requirement def FR_AdaptiveOpticsOptionalDef {
    doc = "Functional/performance requirement definition: adaptive optics optionality.";
  }

  requirement def SensitivityPerformanceNFRDef {
    doc = "Non-functional requirement definition: sensitivity performance.";
  }

  requirement def AngularResolutionPerformanceNFRDef {
    doc = "Non-functional requirement definition: angular resolution performance.";
  }

  requirement def ObservingAssumptionsAndConditions {
    doc = "Observing assumptions/conditions (instantiated set).";
  }

  requirement def FR_PhasedSegmentedAperture {
    doc = "Functional/performance requirement: phased segmented aperture.";
  }

  requirement def FR_PointingTrackingAndStability {
    doc = "Functional/performance requirement: pointing, tracking, stability.";
  }

  requirement def FR_ObservationPhasingAndExposureControl {
    doc = "Functional/performance requirement: phasing/exposure control.";
  }

  requirement def FR_AdaptiveOpticsOptional {
    doc = "Functional/performance requirement: adaptive optics optional.";
  }

  /////////////////////////////////
  // Common items and interfaces
  /////////////////////////////////

  item def ElectricalPower {
    attribute voltage : Real;
    attribute frequency : Real;
    attribute powerAvailable : Real;
  }

  item def ThermalEnergy {
    attribute thermalPower : Real;
  }

  item def DataPacket {
    attribute bandwidth : Real;
    attribute latency : Real;
  }

  item def ControlSignal {
    attribute commandRate : Real;
  }

  item def EnvironmentalState {
    attribute temperature : Real;
    attribute humidity : Real;
    attribute pressure : Real;
    attribute windSpeed : Real;
  }

  port def PowerPort {
    inout power : ElectricalPower;
  }

  port def CoolingPort {
    inout thermal : ThermalEnergy;
  }

  port def DataPort {
    inout data : DataPacket;
  }

  port def ControlPort {
    inout ctrl : ControlSignal;
  }

  port def EnvPort {
    inout env : EnvironmentalState;
  }

  /////////////////////////////////
  // Abstract base types
  /////////////////////////////////

  abstract part def InfrastructureElement;

  abstract part def FacilityController :> InfrastructureElement {
    port mgmtNet : DataPort;
    port controlBus : ControlPort;
    port envSense : EnvPort;
  }

  /////////////////////////////////
  // Power infrastructure
  /////////////////////////////////

  part def UtilityGridFeed :> InfrastructureElement {
    port gridPower : PowerPort;
    attribute maxSupplyPower : Real;
  }

  part def UPSSystem :> InfrastructureElement {
    port acIn : PowerPort;
    port acOut : PowerPort;
    port monitor : DataPort;
    attribute rideThroughTime : Real;
    attribute ratedPower : Real;
  }

  part def BackupGeneratorPlant :> InfrastructureElement {
    port genOut : PowerPort;
    port startStop : ControlPort;
    port monitor : DataPort;
    attribute ratedPower : Real;
    attribute fuelAutonomyTime : Real;
  }

  part def PowerDistributionUnit :> InfrastructureElement {
    port feedIn : PowerPort;
    port criticalBus : PowerPort;
    port nonCriticalBus : PowerPort;
    port monitor : DataPort;
    attribute breakerCount : Integer;
  }

  part def PowerQualityMonitor :> InfrastructureElement {
    port sense : PowerPort;
    port report : DataPort;
    attribute thdLimit : Real;
  }

  /////////////////////////////////
  // HVAC / thermal infrastructure
  /////////////////////////////////

  part def ChillerPlant :> InfrastructureElement {
    port powerIn : PowerPort;
    port chilledWater : CoolingPort;
    port monitor : DataPort;
    port control : ControlPort;
    attribute coolingCapacity : Real;
  }

  part def AirHandlingUnit :> InfrastructureElement {
    port powerIn : PowerPort;
    port coolingIn : CoolingPort;
    port supplyAirState : EnvPort;
    port returnAirState : EnvPort;
    port monitor : DataPort;
    port control : ControlPort;
    attribute airflowRate : Real;
  }

  part def VentilationExhaustSystem :> InfrastructureElement {
    port powerIn : PowerPort;
    port exhaustState : EnvPort;
    port monitor : DataPort;
    port control : ControlPort;
    attribute exhaustCapacity : Real;
  }

  part def EnclosureThermalManagement :> InfrastructureElement {
    port cooling : CoolingPort;
    port enclosureEnv : EnvPort;
    port monitor : DataPort;
    port control : ControlPort;
    attribute targetTempStability : Real;
  }

  /////////////////////////////////
  // Networking / communications infrastructure
  /////////////////////////////////

  part def FiberBackbone :> InfrastructureElement {
    port trunkA : DataPort;
    port trunkB : DataPort;
    attribute maxBandwidth : Real;
    attribute maxDistance : Real;
  }

  part def NetworkCoreSwitch :> InfrastructureElement {
    port uplinkA : DataPort;
    port uplinkB : DataPort;
    port mgmt : DataPort;
    attribute portCount : Integer;
    attribute switchingCapacity : Real;
  }

  part def NetworkEdgeDistribution :> InfrastructureElement {
    port uplink : DataPort;
    port downlinks : DataPort;
    port mgmt : DataPort;
    attribute vlanCount : Integer;
  }

  part def TimingAndSyncService :> InfrastructureElement {
    port syncNet : DataPort;
    port monitor : DataPort;
    attribute timeAccuracy : Real;
  }

  /////////////////////////////////
  // Enclosure / dome elements
  /////////////////////////////////

  part def DomeStructure :> InfrastructureElement {
    port facilityPower : PowerPort;
    port facilityNet : DataPort;
    port facilityCtrl : ControlPort;
    port domeEnv : EnvPort;
    attribute rotationRange : Real;
  }

  part def ShutterAndVentGates :> InfrastructureElement {
    port powerIn : PowerPort;
    port control : ControlPort;
    port monitor : DataPort;
    attribute openCloseTime : Real;
  }

  part def WeatherAndSiteSensors :> InfrastructureElement {
    port siteEnv : EnvPort;
    port report : DataPort;
    attribute sensorCount : Integer;
  }

  /////////////////////////////////
  // Facility monitoring and control
  /////////////////////////////////

  part def FacilityManagementSystem :> FacilityController {
    part pduMonitor : PowerQualityMonitor;
    attribute alarmPointCount : Integer;
  }

  /////////////////////////////////
  // Subsystem: Enclosure/Dome & Site Infrastructure
  /////////////////////////////////

  part def EnclosureDomeAndSiteInfrastructure :> InfrastructureElement {

    // Power chain
    part grid : UtilityGridFeed;
    part ups : UPSSystem;
    part gens : BackupGeneratorPlant;
    part pdu : PowerDistributionUnit;

    // HVAC / thermal chain
    part chillers : ChillerPlant;
    part ahu : AirHandlingUnit;
    part exhaust : VentilationExhaustSystem;
    part enclosureThermal : EnclosureThermalManagement;

    // Networking chain
    part fiber : FiberBackbone;
    part core : NetworkCoreSwitch;
    part edge : NetworkEdgeDistribution;
    part timing : TimingAndSyncService;

    // Enclosure/dome
    part dome : DomeStructure;
    part shutters : ShutterAndVentGates;
    part met : WeatherAndSiteSensors;

    // Facility control
    part fms : FacilityManagementSystem;

    // External-facing ports for the subsystem boundary
    port facilityPowerIn : PowerPort;
    port facilityDataInout : DataPort;
    port facilityEnvInout : EnvPort;
    port facilityCtrlInout : ControlPort;

    /////////////////////////////////
    // Key internal connections (BDD-level connectivity intent)
    /////////////////////////////////

    connect facilityPowerIn to grid.gridPower;
    connect grid.gridPower to ups.acIn;
    connect ups.acOut to pdu.feedIn;
    connect gens.genOut to pdu.feedIn;

    connect pdu.criticalBus to chillers.powerIn;
    connect pdu.criticalBus to ahu.powerIn;
    connect pdu.criticalBus to dome.facilityPower;
    connect pdu.criticalBus to shutters.powerIn;
    connect pdu.criticalBus to core.mgmt;

    connect chillers.chilledWater to ahu.coolingIn;
    connect chillers.chilledWater to enclosureThermal.cooling;

    connect ahu.supplyAirState to dome.domeEnv;
    connect met.siteEnv to facilityEnvInout;

    connect facilityDataInout to fiber.trunkA;
    connect facilityDataInout to fiber.trunkB;

    connect fiber.trunkA to core.uplinkA;
    connect fiber.trunkB to core.uplinkB;

    connect core.mgmt to fms.mgmtNet;
    connect core.mgmt to pdu.monitor;
    connect core.mgmt to ups.monitor;
    connect core.mgmt to gens.monitor;
    connect core.mgmt to chillers.monitor;
    connect core.mgmt to ahu.monitor;
    connect core.mgmt to shutters.monitor;
    connect core.mgmt to timing.monitor;
    connect core.mgmt to met.report;

    connect core.uplinkA to edge.uplink;
    connect edge.downlinks to dome.facilityNet;
    connect edge.downlinks to shutters.monitor;

    connect fms.controlBus to pdu.monitor;
    connect fms.controlBus to chillers.control;
    connect fms.controlBus to ahu.control;
    connect fms.controlBus to exhaust.control;
    connect fms.controlBus to shutters.control;

    connect facilityCtrlInout to fms.controlBus;

    connect timing.syncNet to core.uplinkA;
  }

  /////////////////////////////////
  // Explicit implementations for previously-unimplemented requirements (RD-2)
  /////////////////////////////////

  part def MissionGoalsAndScienceDrivers_Impl :> InfrastructureElement {
    doc = "Allocated capability support via facility services enabling science operations.";
    port facilityServices : DataPort;
  }

  part def EarlyUniverseGoalMapping_Impl :> InfrastructureElement {
    doc = "Supports early universe observations through stable thermal/power/network services.";
    port services : DataPort;
  }

  part def GalaxyEvolutionGoalMapping_Impl :> InfrastructureElement {
    doc = "Supports galaxy evolution programs via high-availability facility services.";
    port services : DataPort;
  }

  part def ExoplanetsGoalMapping_Impl :> InfrastructureElement {
    doc = "Supports exoplanet observations via low-vibration, stable environment services.";
    port services : DataPort;
  }

  part def BHgrowthGoalMapping_Impl :> InfrastructureElement {
    doc = "Supports black hole growth science via long-exposure enabling infrastructure.";
    port services : DataPort;
  }

  part def StakeholderGroupConcerns_Impl :> InfrastructureElement {
    doc = "Captures operational/site constraints via monitoring, alarms, and safety-relevant controls.";
    port status : DataPort;
    port env : EnvPort;
  }

  part def ObservingAssumptionsAndConditionsDef_Impl :> InfrastructureElement {
    doc = "Provides facility-side sensing/conditioning to meet observing assumptions.";
    port env : EnvPort;
    port control : ControlPort;
  }

  part def FR_PhasedSegmentedApertureDef_Impl :> InfrastructureElement {
    doc = "Facility services (power/thermal/network) supporting alignment/phasing operations.";
    port power : PowerPort;
    port data : DataPort;
  }

  part def FR_PointingTrackingAndStabilityDef_Impl :> InfrastructureElement {
    doc = "Facility services supporting pointing/tracking stability (power quality, environmental control).";
    port powerQuality : PowerPort;
    port env : EnvPort;
  }

  part def FR_ObservationPhasingAndExposureControlDef_Impl :> InfrastructureElement {
    doc = "Facility services supporting exposure control workflows (timing, network, power continuity).";
    port timingNet : DataPort;
    port power : PowerPort;
  }

  part def FR_AdaptiveOpticsOptionalDef_Impl :> InfrastructureElement {
    doc = "Facility provisions for optional AO (high-rate data, timing, thermal/power margins).";
    port data : DataPort;
    port timingNet : DataPort;
    port cooling : CoolingPort;
  }

  part def SensitivityPerformanceNFRDef_Impl :> InfrastructureElement {
    doc = "Infrastructure contributions to sensitivity (thermal stability, clean power, low emissivity environment).";
    port env : EnvPort;
    port power : PowerPort;
  }

  part def AngularResolutionPerformanceNFRDef_Impl :> InfrastructureElement {
    doc = "Infrastructure contributions to angular resolution (seeing/thermal management, wind management interfaces).";
    port env : EnvPort;
    port control : ControlPort;
  }

  part def ObservingAssumptionsAndConditions_Impl :> InfrastructureElement {
    doc = "Instantiated sensing/conditioning support for current observing conditions.";
    port env : EnvPort;
  }

  part def FR_PhasedSegmentedAperture_Impl :> InfrastructureElement {
    doc = "Operational support path for phasing via facility services.";
    port services : DataPort;
  }

  part def FR_PointingTrackingAndStability_Impl :> InfrastructureElement {
    doc = "Operational support path for tracking/stability via facility services.";
    port services : DataPort;
  }

  part def FR_ObservationPhasingAndExposureControl_Impl :> InfrastructureElement {
    doc = "Operational support path for phasing/exposure control via facility services.";
    port services : DataPort;
  }

  part def FR_AdaptiveOpticsOptional_Impl :> InfrastructureElement {
    doc = "Operational support path for optional AO via facility services.";
    port services : DataPort;
  }

  part TraceabilityImplementations : InfrastructureElement {
    part missionGoalsAndScienceDrivers : MissionGoalsAndScienceDrivers_Impl;
    part earlyUniverseGoalMapping : EarlyUniverseGoalMapping_Impl;
    part galaxyEvolutionGoalMapping : GalaxyEvolutionGoalMapping_Impl;
    part exoplanetsGoalMapping : ExoplanetsGoalMapping_Impl;
    part bHgrowthGoalMapping : BHgrowthGoalMapping_Impl;
    part stakeholderGroupConcerns : StakeholderGroupConcerns_Impl;

    part observingAssumptionsAndConditionsDef : ObservingAssumptionsAndConditionsDef_Impl;
    part fr_PhasedSegmentedApertureDef : FR_PhasedSegmentedApertureDef_Impl;
    part fr_PointingTrackingAndStabilityDef : FR_PointingTrackingAndStabilityDef_Impl;
    part fr_ObservationPhasingAndExposureControlDef : FR_ObservationPhasingAndExposureControlDef_Impl;
    part fr_AdaptiveOpticsOptionalDef : FR_AdaptiveOpticsOptionalDef_Impl;

    part sensitivityPerformanceNFRDef : SensitivityPerformanceNFRDef_Impl;
    part angularResolutionPerformanceNFRDef : AngularResolutionPerformanceNFRDef_Impl;

    part observingAssumptionsAndConditions : ObservingAssumptionsAndConditions_Impl;
    part fr_PhasedSegmentedAperture : FR_PhasedSegmentedAperture_Impl;
    part fr_PointingTrackingAndStability : FR_PointingTrackingAndStability_Impl;
    part fr_ObservationPhasingAndExposureControl : FR_ObservationPhasingAndExposureControl_Impl;
    part fr_AdaptiveOpticsOptional : FR_AdaptiveOpticsOptional_Impl;
  }

  /////////////////////////////////
  // Satisfy links (RD-2 closure intent)
  /////////////////////////////////

  satisfy MissionGoalsAndScienceDrivers by TraceabilityImplementations::missionGoalsAndScienceDrivers;
  satisfy EarlyUniverseGoalMapping by TraceabilityImplementations::earlyUniverseGoalMapping;
  satisfy GalaxyEvolutionGoalMapping by TraceabilityImplementations::galaxyEvolutionGoalMapping;
  satisfy ExoplanetsGoalMapping by TraceabilityImplementations::exoplanetsGoalMapping;
  satisfy BHgrowthGoalMapping by TraceabilityImplementations::bHgrowthGoalMapping;
  satisfy StakeholderGroupConcerns by TraceabilityImplementations::stakeholderGroupConcerns;

  satisfy ObservingAssumptionsAndConditionsDef by TraceabilityImplementations::observingAssumptionsAndConditionsDef;
  satisfy FR_PhasedSegmentedApertureDef by TraceabilityImplementations::fr_PhasedSegmentedApertureDef;
  satisfy FR_PointingTrackingAndStabilityDef by TraceabilityImplementations::fr_PointingTrackingAndStabilityDef;
  satisfy FR_ObservationPhasingAndExposureControlDef by TraceabilityImplementations::fr_ObservationPhasingAndExposureControlDef;
  satisfy FR_AdaptiveOpticsOptionalDef by TraceabilityImplementations::fr_AdaptiveOpticsOptionalDef;

  satisfy SensitivityPerformanceNFRDef by TraceabilityImplementations::sensitivityPerformanceNFRDef;
  satisfy AngularResolutionPerformanceNFRDef by TraceabilityImplementations::angularResolutionPerformanceNFRDef;

  satisfy ObservingAssumptionsAndConditions by TraceabilityImplementations::observingAssumptionsAndConditions;
  satisfy FR_PhasedSegmentedAperture by TraceabilityImplementations::fr_PhasedSegmentedAperture;
  satisfy FR_PointingTrackingAndStability by TraceabilityImplementations::fr_PointingTrackingAndStability;
  satisfy FR_ObservationPhasingAndExposureControl by TraceabilityImplementations::fr_ObservationPhasingAndExposureControl;
  satisfy FR_AdaptiveOpticsOptional by TraceabilityImplementations::fr_AdaptiveOpticsOptional;

}
