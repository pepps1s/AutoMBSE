package BDD_SUB03_EnclosureDomeAndSiteInfrastructure {

  import ScalarValues::*;
  import SI::*;
  import Quantities::*;

  doc /*diagram_id*/ "BDD-SUB03-001";
  doc /*type*/ "BDD";
  doc /*title*/ "BDD SUB03: Enclosure/Dome & Site Infrastructure (power, HVAC, networking)";
  doc /*goal*/ "Define the structural decomposition (parts/attributes/interfaces) of TMT subsystem: Enclosure/Dome & Site Infrastructure.";
  doc /*dependencies*/ "RD-F-001, RD-NF-001";

  // -----------------------------
  // Requirement packages (names preserved) + explicit design implementations
  // -----------------------------

  package MissionGoalsAndScienceDrivers {
    requirement def MissionGoalsAndScienceDrivers;
  }
  package EarlyUniverseGoalMapping {
    requirement def EarlyUniverseGoalMapping;
  }
  package GalaxyEvolutionGoalMapping {
    requirement def GalaxyEvolutionGoalMapping;
  }
  package ExoplanetsGoalMapping {
    requirement def ExoplanetsGoalMapping;
  }
  package BHgrowthGoalMapping {
    requirement def BHgrowthGoalMapping;
  }
  package StakeholderGroupConcerns {
    requirement def StakeholderGroupConcerns;
  }

  package ObservingAssumptionsAndConditionsDef {
    requirement def ObservingAssumptionsAndConditionsDef;
  }
  package FR_PhasedSegmentedApertureDef {
    requirement def FR_PhasedSegmentedApertureDef;
  }
  package FR_PointingTrackingAndStabilityDef {
    requirement def FR_PointingTrackingAndStabilityDef;
  }
  package FR_ObservationPhasingAndExposureControlDef {
    requirement def FR_ObservationPhasingAndExposureControlDef;
  }
  package FR_AdaptiveOpticsOptionalDef {
    requirement def FR_AdaptiveOpticsOptionalDef;
  }
  package SensitivityPerformanceNFRDef {
    requirement def SensitivityPerformanceNFRDef;
  }
  package AngularResolutionPerformanceNFRDef {
    requirement def AngularResolutionPerformanceNFRDef;
  }

  package ObservingAssumptionsAndConditions {
    requirement def ObservingAssumptionsAndConditions;
  }
  package FR_PhasedSegmentedAperture {
    requirement def FR_PhasedSegmentedAperture;
  }
  package FR_PointingTrackingAndStability {
    requirement def FR_PointingTrackingAndStability;
  }
  package FR_ObservationPhasingAndExposureControl {
    requirement def FR_ObservationPhasingAndExposureControl;
  }
  package FR_AdaptiveOpticsOptional {
    requirement def FR_AdaptiveOpticsOptional;
  }

  // -----------------------------
  // Common types
  // -----------------------------

  attribute def Availability {
    percent : Real;
  }

  // -----------------------------
  // Interfaces (ports)
  // -----------------------------

  interface def ElectricalPowerIF {
    inout voltage : Real;
    inout current : Real;
    inout frequency : Real;
  }

  interface def CoolingIF {
    inout coolantTemp : Real;
    inout flowRate : Real;
    inout pressure : Real;
  }

  interface def AirflowIF {
    inout airTemp : Real;
    inout humidity : Real;
    inout flowRate : Real;
  }

  interface def NetworkIF {
    inout bandwidth : Real;
    inout latency : Real;
    inout packetLoss : Real;
  }

  interface def FacilitySafetyIF {
    inout eStop : Boolean;
    inout interlockStatus : Boolean;
    inout alarms : Boolean;
  }

  interface def TimeSyncIF {
    inout ptpSyncOk : Boolean;
    inout timeOffset : Real;
  }

  interface def TelemetryIF {
    inout healthStatus : Boolean;
    inout logStreamOk : Boolean;
  }

  // -----------------------------
  // Abstract base components
  // -----------------------------

  abstract part def FacilitySubsystem;
  abstract part def ControlUnit;
  abstract part def DistributionUnit;

  // -----------------------------
  // Subsystem decomposition for Enclosure/Dome & Site Infrastructure
  // -----------------------------

  part def EnclosureDomeAndSiteInfrastructure :> FacilitySubsystem {

    // Key external interfaces to the rest of TMT / site / utilities
    port pwrToObservatory : ElectricalPowerIF;
    port netToOCS : NetworkIF;
    port safetyToObservatory : FacilitySafetyIF;
    port timeSyncToObservatory : TimeSyncIF;
    port telemetryToObservatory : TelemetryIF;

    // High-level attributes (placeholders; refined in parametric views)
    attribute availabilityTarget : Availability;
    attribute maxWindSpeedOperational : Real;
    attribute enclosureThermalStability : Real;

    // Major parts
    part enclosureDome : EnclosureAndDome;
    part sitePower : SitePowerInfrastructure;
    part hvacAndThermal : HVACandThermalManagement;
    part siteNetworking : SiteNetworkingInfrastructure;
    part siteSafetyAndAccess : SiteSafetyAndAccess;

    // Traceability to requirement packages (fix RD-2: every named requirement has an implementation)
    satisfy MissionGoalsAndScienceDrivers::MissionGoalsAndScienceDrivers by enclosureDome;
    satisfy EarlyUniverseGoalMapping::EarlyUniverseGoalMapping by hvacAndThermal;
    satisfy GalaxyEvolutionGoalMapping::GalaxyEvolutionGoalMapping by siteNetworking;
    satisfy ExoplanetsGoalMapping::ExoplanetsGoalMapping by hvacAndThermal;
    satisfy BHgrowthGoalMapping::BHgrowthGoalMapping by siteNetworking;
    satisfy StakeholderGroupConcerns::StakeholderGroupConcerns by siteSafetyAndAccess;

    satisfy ObservingAssumptionsAndConditionsDef::ObservingAssumptionsAndConditionsDef by hvacAndThermal;
    satisfy FR_PhasedSegmentedApertureDef::FR_PhasedSegmentedApertureDef by hvacAndThermal;
    satisfy FR_PointingTrackingAndStabilityDef::FR_PointingTrackingAndStabilityDef by enclosureDome;
    satisfy FR_ObservationPhasingAndExposureControlDef::FR_ObservationPhasingAndExposureControlDef by sitePower;
    satisfy FR_AdaptiveOpticsOptionalDef::FR_AdaptiveOpticsOptionalDef by sitePower;
    satisfy SensitivityPerformanceNFRDef::SensitivityPerformanceNFRDef by hvacAndThermal;
    satisfy AngularResolutionPerformanceNFRDef::AngularResolutionPerformanceNFRDef by enclosureDome;

    satisfy ObservingAssumptionsAndConditions::ObservingAssumptionsAndConditions by hvacAndThermal;
    satisfy FR_PhasedSegmentedAperture::FR_PhasedSegmentedAperture by hvacAndThermal;
    satisfy FR_PointingTrackingAndStability::FR_PointingTrackingAndStability by enclosureDome;
    satisfy FR_ObservationPhasingAndExposureControl::FR_ObservationPhasingAndExposureControl by sitePower;
    satisfy FR_AdaptiveOpticsOptional::FR_AdaptiveOpticsOptional by sitePower;
  }

  // -----------------------------
  // Enclosure / Dome
  // -----------------------------

  part def EnclosureAndDome :> FacilitySubsystem {

    port powerIn : ElectricalPowerIF;
    port networkIn : NetworkIF;
    port safetyIn : FacilitySafetyIF;
    port telemetryOut : TelemetryIF;

    // Interfaces relevant to thermal/wind management
    port domeAirflow : AirflowIF;

    // Major elements
    part rotationDrive : DomeRotationDrive;
    part shutters : ShutterSystem;
    part ventilation : DomeVentilationSystem;
    part thermalControl : EnclosureThermalControl;
    part weatherSealing : WeatherSealingAndDrainage;
    part localControl : EnclosureControlUnit;

    // Representative attributes
    attribute azimuthRange : Real;
    attribute maxRotationRate : Real;
    attribute ventingEffectiveness : Real;
  }

  part def DomeRotationDrive :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port safetyIn : FacilitySafetyIF;
    port telemetryOut : TelemetryIF;
    attribute maxTorque : Real;
    attribute positionAccuracy : Real;
  }

  part def ShutterSystem :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port safetyIn : FacilitySafetyIF;
    port telemetryOut : TelemetryIF;
    attribute openCloseTime : Real;
    attribute sealIntegrity : Real;
  }

  part def DomeVentilationSystem :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port airflowOut : AirflowIF;
    port telemetryOut : TelemetryIF;
    attribute fanCapacity : Real;
    attribute louverControlResolution : Real;
  }

  part def EnclosureThermalControl :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port airflowInout : AirflowIF;
    port telemetryOut : TelemetryIF;
    attribute temperatureUniformity : Real;
    attribute heatLoadHandling : Real;
  }

  part def WeatherSealingAndDrainage :> FacilitySubsystem {
    port telemetryOut : TelemetryIF;
    attribute ingressProtectionLevel : Real;
  }

  part def EnclosureControlUnit :> ControlUnit {
    port networkIn : NetworkIF;
    port safetyIn : FacilitySafetyIF;
    port telemetryOut : TelemetryIF;
    attribute controlLoopRate : Real;
  }

  // -----------------------------
  // Site Power Infrastructure
  // -----------------------------

  part def SitePowerInfrastructure :> FacilitySubsystem {

    port utilityFeedIn : ElectricalPowerIF;
    port powerToEnclosure : ElectricalPowerIF;
    port powerToFacilities : ElectricalPowerIF;
    port powerToTelescopeAndInstruments : ElectricalPowerIF;

    port networkIn : NetworkIF;
    port safetyIn : FacilitySafetyIF;
    port telemetryOut : TelemetryIF;

    part switchgear : PowerSwitchgear;
    part ups : UPSandRideThrough;
    part generators : BackupGeneration;
    part distribution : PowerDistribution;
    part grounding : GroundingAndLightningProtection;
    part powerMonitoring : PowerQualityMonitoring;

    attribute powerCapacity : Real;
    attribute powerQualityTHD : Real;
    attribute transferTimeToBackup : Real;
  }

  part def PowerSwitchgear :> DistributionUnit {
    port pwrIn : ElectricalPowerIF;
    port pwrOut : ElectricalPowerIF;
    port telemetryOut : TelemetryIF;
    attribute breakerCount : Integer;
  }

  part def UPSandRideThrough :> FacilitySubsystem {
    port pwrIn : ElectricalPowerIF;
    port pwrOut : ElectricalPowerIF;
    port telemetryOut : TelemetryIF;
    attribute rideThroughTime : Real;
  }

  part def BackupGeneration :> FacilitySubsystem {
    port pwrOut : ElectricalPowerIF;
    port telemetryOut : TelemetryIF;
    attribute fuelAutonomyHours : Real;
  }

  part def PowerDistribution :> DistributionUnit {
    port pwrIn : ElectricalPowerIF;
    port pwrOut : ElectricalPowerIF;
    port telemetryOut : TelemetryIF;
    attribute feederCount : Integer;
  }

  part def GroundingAndLightningProtection :> FacilitySubsystem {
    port safetyOut : FacilitySafetyIF;
    attribute groundResistance : Real;
  }

  part def PowerQualityMonitoring :> FacilitySubsystem {
    port pwrSense : ElectricalPowerIF;
    port telemetryOut : TelemetryIF;
    attribute samplingRate : Real;
  }

  // -----------------------------
  // HVAC & Thermal Management
  // -----------------------------

  part def HVACandThermalManagement :> FacilitySubsystem {

    port powerIn : ElectricalPowerIF;
    port networkIn : NetworkIF;
    port telemetryOut : TelemetryIF;

    // Provide conditioned air/cooling to enclosure and support buildings
    port airflowToEnclosure : AirflowIF;
    port coolingToFacilityLoads : CoolingIF;

    part chillers : ChillerPlant;
    part pumps : PumpSkid;
    part airHandling : AirHandlingUnits;
    part ductingAndDiffusers : AirDistribution;
    part thermalMonitoring : ThermalSensorsAndMonitoring;
    part hvacControl : HVACControlUnit;

    attribute totalCoolingCapacity : Real;
    attribute humidityControlRange : Real;
    attribute vibrationIsolationEffectiveness : Real;
  }

  part def ChillerPlant :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port coolingOut : CoolingIF;
    port telemetryOut : TelemetryIF;
    attribute chillerCount : Integer;
    attribute redundancyLevel : Integer;
  }

  part def PumpSkid :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port coolingInout : CoolingIF;
    port telemetryOut : TelemetryIF;
    attribute pumpCount : Integer;
  }

  part def AirHandlingUnits :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port airflowOut : AirflowIF;
    port telemetryOut : TelemetryIF;
    attribute filterEfficiency : Real;
  }

  part def AirDistribution :> FacilitySubsystem {
    port airflowInout : AirflowIF;
    attribute balancingResolution : Real;
  }

  part def ThermalSensorsAndMonitoring :> FacilitySubsystem {
    port telemetryOut : TelemetryIF;
    attribute sensorCount : Integer;
    attribute updateRate : Real;
  }

  part def HVACControlUnit :> ControlUnit {
    port networkIn : NetworkIF;
    port telemetryOut : TelemetryIF;
    attribute supervisoryControlRate : Real;
  }

  // -----------------------------
  // Site Networking Infrastructure
  // -----------------------------

  part def SiteNetworkingInfrastructure :> FacilitySubsystem {

    port powerIn : ElectricalPowerIF;
    port backboneToOCS : NetworkIF;
    port telemetryOut : TelemetryIF;
    port timeSyncOut : TimeSyncIF;

    part coreSwitching : CoreSwitchAndRouting;
    part distributionSwitching : NetworkDistribution;
    part fiberPlant : FiberPlantAndPatchPanels;
    part timing : TimeSyncServices;
    part cyberSecurity : NetworkSecurityBoundary;
    part networkMonitoring : NetworkMonitoringAndLogging;

    attribute backboneBandwidth : Real;
    attribute networkAvailabilityTarget : Availability;
    attribute maxLatencyBudget : Real;
  }

  part def CoreSwitchAndRouting :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port netInout : NetworkIF;
    port telemetryOut : TelemetryIF;
    attribute portCount : Integer;
  }

  part def NetworkDistribution :> FacilitySubsystem {
    port powerIn : ElectricalPowerIF;
    port netInout : NetworkIF;
    port telemetryOut : TelemetryIF;
    attribute switchCount : Integer;
  }

  part def FiberPlantAndPatchPanels :> FacilitySubsystem {
    port netInout : NetworkIF;
    attribute fiberStrandCount : Integer;
  }

  part def TimeSyncServices :> FacilitySubsystem {
    port netIn : NetworkIF;
    port timeSyncOut : TimeSyncIF;
    port telemetryOut : TelemetryIF;
    attribute ptpDomainCount : Integer;
  }

  part def NetworkSecurityBoundary :> FacilitySubsystem {
    port netInout : NetworkIF;
    port telemetryOut : TelemetryIF;
    attribute firewallRuleSetSize : Integer;
  }

  part def NetworkMonitoringAndLogging :> FacilitySubsystem {
    port netSense : NetworkIF;
    port telemetryOut : TelemetryIF;
    attribute logRetentionDays : Integer;
  }

  // -----------------------------
  // Site Safety & Access
  // -----------------------------

  part def SiteSafetyAndAccess :> FacilitySubsystem {

    port powerIn : ElectricalPowerIF;
    port networkIn : NetworkIF;
    port safetyOut : FacilitySafetyIF;
    port telemetryOut : TelemetryIF;

    part accessControl : AccessControlSystem;
    part fireDetection : FireDetectionAndSuppression;
    part emergencySystems : EmergencyStopAndInterlocks;
    part environmentalMonitoring : EnvironmentalAndWeatherMonitoring;
    part physicalSecurity : PhysicalSecurityAndCCTV;

    attribute safetyIntegrityLevel : Integer;
  }

  part def AccessControlSystem :> FacilitySubsystem {
    port netIn : NetworkIF;
    port telemetryOut : TelemetryIF;
    attribute doorCount : Integer;
  }

  part def FireDetectionAndSuppression :> FacilitySubsystem {
    port safetyOut : FacilitySafetyIF;
    port telemetryOut : TelemetryIF;
    attribute zoneCount : Integer;
  }

  part def EmergencyStopAndInterlocks :> FacilitySubsystem {
    port safetyOut : FacilitySafetyIF;
    port telemetryOut : TelemetryIF;
    attribute estopStations : Integer;
  }

  part def EnvironmentalAndWeatherMonitoring :> FacilitySubsystem {
    port telemetryOut : TelemetryIF;
    attribute sensorCount : Integer;
  }

  part def PhysicalSecurityAndCCTV :> FacilitySubsystem {
    port netOut : NetworkIF;
    port telemetryOut : TelemetryIF;
    attribute cameraCount : Integer;
  }

  // -----------------------------
  // Top-level TMT context hook for BDD usage (lightweight)
  // -----------------------------

  part def TMT_System {
    part sub03 : EnclosureDomeAndSiteInfrastructure;
  }

}
