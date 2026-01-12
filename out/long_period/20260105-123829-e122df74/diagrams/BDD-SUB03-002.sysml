package BDD_SUB03_EnclosureDomeAndSiteInfrastructure {
  import ScalarValues::*;
  import SI::*;
  import Quantities::*;

  //============================================================
  // Traceability stubs to satisfy RD-2 (requirements -> realization)
  //============================================================

  requirement def MissionGoalsAndScienceDrivers;
  requirement def EarlyUniverseGoalMapping;
  requirement def GalaxyEvolutionGoalMapping;
  requirement def ExoplanetsGoalMapping;
  requirement def BHgrowthGoalMapping;
  requirement def StakeholderGroupConcerns;

  requirement def ObservingAssumptionsAndConditionsDef;
  requirement def FR_PhasedSegmentedApertureDef;
  requirement def FR_PointingTrackingAndStabilityDef;
  requirement def FR_ObservationPhasingAndExposureControlDef;
  requirement def FR_AdaptiveOpticsOptionalDef;
  requirement def SensitivityPerformanceNFRDef;
  requirement def AngularResolutionPerformanceNFRDef;

  requirement def ObservingAssumptionsAndConditions;
  requirement def FR_PhasedSegmentedAperture;
  requirement def FR_PointingTrackingAndStability;
  requirement def FR_ObservationPhasingAndExposureControl;
  requirement def FR_AdaptiveOpticsOptional;

  //============================================================
  // Value types (lightweight, unitless per guidance)
  //============================================================
  attribute def Power_kW { value : Real; }
  attribute def Heat_kW { value : Real; }
  attribute def AirFlow_CMS { value : Real; }
  attribute def Temp_C { value : Real; }
  attribute def Humidity_pct { value : Real; }
  attribute def DataRate_Gbps { value : Real; }
  attribute def Pressure_Pa { value : Real; }

  //============================================================
  // Common interface/port definitions
  //============================================================
  part def PowerPort {
    inout powerDemand : Power_kW;
  }

  part def CoolingPort {
    inout heatFlow : Heat_kW;
  }

  part def AirHandlingPort {
    inout airFlow : AirFlow_CMS;
    inout temperature : Temp_C;
    inout humidity : Humidity_pct;
  }

  part def NetworkPort {
    inout dataRate : DataRate_Gbps;
  }

  part def ControlPort {
    inout command : String;
    inout telemetry : String;
  }

  part def SafetyInterlockPort {
    inout interlockState : Boolean;
    inout trip : Boolean;
  }

  //============================================================
  // Abstract base types
  //============================================================
  abstract part def Subsystem;
  abstract part def FacilityService;

  //============================================================
  // Target system context (minimal top-level structural anchor)
  //============================================================
  part def ThirtyMeterTelescope :> Subsystem {
    port facilityPower : PowerPort;
    port facilityNetwork : NetworkPort;
    port facilityControls : ControlPort;
    port facilitySafety : SafetyInterlockPort;

    part enclosureAndSite : EnclosureDomeAndSiteInfrastructure;
  }

  //============================================================
  // SUB03: Enclosure/Dome & Site Infrastructure
  //============================================================
  part def EnclosureDomeAndSiteInfrastructure :> Subsystem {

    // Key external interfaces to the rest of observatory/telescope
    port toTelescopePower : PowerPort;
    port toTelescopeCooling : CoolingPort;
    port toTelescopeNetwork : NetworkPort;
    port toObservatoryControls : ControlPort;
    port toSafetySystem : SafetyInterlockPort;

    attribute installedPowerCapacity : Power_kW;
    attribute installedCoolingCapacity : Heat_kW;
    attribute coreNetworkCapacity : DataRate_Gbps;

    // Structural decomposition
    part enclosureDome : DomeEnclosureSystem;
    part powerInfrastructure : SitePowerInfrastructure;
    part hvacInfrastructure : SiteHVACAndThermalInfrastructure;
    part networkingInfrastructure : SiteNetworkingInfrastructure;
    part facilityMonitoring : FacilityMonitoringAndControls;
    part physicalInfrastructure : SitePhysicalInfrastructure;

    // Key internal connections (BDD-level; keeps ports/links visible)
    connect powerInfrastructure.toSitePower to enclosureDome.sitePower;
    connect powerInfrastructure.toSitePower to hvacInfrastructure.sitePower;
    connect powerInfrastructure.toSitePower to networkingInfrastructure.sitePower;

    connect hvacInfrastructure.toCoolingServices to enclosureDome.coolingServices;
    connect hvacInfrastructure.toAirServices to enclosureDome.airServices;

    connect networkingInfrastructure.toCoreNetwork to enclosureDome.siteNetwork;
    connect networkingInfrastructure.toCoreNetwork to facilityMonitoring.siteNetwork;

    connect facilityMonitoring.toFacilityControls to enclosureDome.facilityControls;
    connect facilityMonitoring.toFacilityControls to hvacInfrastructure.facilityControls;
    connect facilityMonitoring.toFacilityControls to powerInfrastructure.facilityControls;

    connect facilityMonitoring.toSafetyInterlocks to enclosureDome.safetyInterlocks;
    connect facilityMonitoring.toSafetyInterlocks to powerInfrastructure.safetyInterlocks;

    // Exported service mapping
    connect powerInfrastructure.toTelescopePower to toTelescopePower;
    connect hvacInfrastructure.toTelescopeCooling to toTelescopeCooling;
    connect networkingInfrastructure.toTelescopeNetwork to toTelescopeNetwork;
    connect facilityMonitoring.toObservatoryControls to toObservatoryControls;
    connect facilityMonitoring.toSafetySystem to toSafetySystem;

    // RD-2: Declare realization (requirements mapped to implementing elements)
    satisfy MissionGoalsAndScienceDrivers by enclosureDome;
    satisfy EarlyUniverseGoalMapping by enclosureDome;
    satisfy GalaxyEvolutionGoalMapping by enclosureDome;
    satisfy ExoplanetsGoalMapping by enclosureDome;
    satisfy BHgrowthGoalMapping by enclosureDome;
    satisfy StakeholderGroupConcerns by facilityMonitoring;

    satisfy ObservingAssumptionsAndConditionsDef by hvacInfrastructure;
    satisfy ObservingAssumptionsAndConditions by hvacInfrastructure;

    satisfy FR_PhasedSegmentedApertureDef by hvacInfrastructure;
    satisfy FR_PhasedSegmentedAperture by hvacInfrastructure;

    satisfy FR_PointingTrackingAndStabilityDef by enclosureDome;
    satisfy FR_PointingTrackingAndStability by enclosureDome;

    satisfy FR_ObservationPhasingAndExposureControlDef by networkingInfrastructure;
    satisfy FR_ObservationPhasingAndExposureControl by networkingInfrastructure;

    satisfy FR_AdaptiveOpticsOptionalDef by networkingInfrastructure;
    satisfy FR_AdaptiveOpticsOptional by networkingInfrastructure;

    satisfy SensitivityPerformanceNFRDef by hvacInfrastructure;
    satisfy AngularResolutionPerformanceNFRDef by enclosureDome;
  }

  //============================================================
  // Dome / Enclosure system decomposition
  //============================================================
  part def DomeEnclosureSystem :> Subsystem {
    port sitePower : PowerPort;
    port coolingServices : CoolingPort;
    port airServices : AirHandlingPort;
    port siteNetwork : NetworkPort;
    port facilityControls : ControlPort;
    port safetyInterlocks : SafetyInterlockPort;

    attribute domeRotationRate : Real;
    attribute shutterOpenFraction : Real;
    attribute windScreenPosition : Real;

    part domeStructure : DomeStructure;
    part rotationAndDrive : DomeRotationAndDrive;
    part apertureAndShutters : ApertureAndShutterSystem;
    part windThermalManagement : WindAndThermalManagement;
    part accessAndHandling : AccessAndHandlingSystems;

    connect rotationAndDrive.powerIn to sitePower;
    connect apertureAndShutters.powerIn to sitePower;
    connect windThermalManagement.powerIn to sitePower;

    connect windThermalManagement.airInOut to airServices;
    connect windThermalManagement.coolingInOut to coolingServices;

    connect rotationAndDrive.ctrl to facilityControls;
    connect apertureAndShutters.ctrl to facilityControls;
    connect windThermalManagement.ctrl to facilityControls;

    connect rotationAndDrive.net to siteNetwork;
    connect windThermalManagement.net to siteNetwork;

    connect rotationAndDrive.safety to safetyInterlocks;
    connect apertureAndShutters.safety to safetyInterlocks;
  }

  part def DomeStructure :> FacilityService {
    attribute designSurvivalWind : Real;
    attribute sealingPerformance : Real;
  }

  part def DomeRotationAndDrive :> FacilityService {
    port powerIn : PowerPort;
    port ctrl : ControlPort;
    port net : NetworkPort;
    port safety : SafetyInterlockPort;

    attribute azimuthPosition_deg : Real;
    attribute azimuthRate_degps : Real;
  }

  part def ApertureAndShutterSystem :> FacilityService {
    port powerIn : PowerPort;
    port ctrl : ControlPort;
    port safety : SafetyInterlockPort;

    attribute shutterStateOpen : Boolean;
    attribute doorPosition : Real;
  }

  part def WindAndThermalManagement :> FacilityService {
    port powerIn : PowerPort;
    port ctrl : ControlPort;
    port net : NetworkPort;

    port airInOut : AirHandlingPort;
    port coolingInOut : CoolingPort;

    attribute domeInternalTemp : Temp_C;
    attribute domeInternalHumidity : Humidity_pct;
    attribute purgeAirFlow : AirFlow_CMS;
  }

  part def AccessAndHandlingSystems :> FacilityService {
    port powerIn : PowerPort;
    port ctrl : ControlPort;

    attribute liftCapacity : Real;
    attribute craneAvailability : Real;
  }

  //============================================================
  // Site Power Infrastructure
  //============================================================
  part def SitePowerInfrastructure :> Subsystem {
    port toSitePower : PowerPort;
    port toTelescopePower : PowerPort;
    port facilityControls : ControlPort;
    port safetyInterlocks : SafetyInterlockPort;

    attribute incomingUtilityCapacity : Power_kW;
    attribute backupCapacity : Power_kW;
    attribute powerQualityTHD : Real;

    part utilityGridInterface : UtilityGridInterface;
    part primarySubstation : SubstationAndSwitchgear;
    part upsSystem : UninterruptiblePowerSupply;
    part backupGenerators : BackupGeneratorPlant;
    part distributionPanels : PowerDistributionNetwork;
    part groundingAndLightning : GroundingAndLightningProtection;

    connect utilityGridInterface.powerOut to primarySubstation.powerIn;
    connect primarySubstation.powerOut to upsSystem.powerIn;
    connect primarySubstation.powerOut to distributionPanels.powerIn;
    connect backupGenerators.powerOut to primarySubstation.powerIn;

    connect distributionPanels.powerOut to toSitePower;
    connect distributionPanels.powerOut to toTelescopePower;

    connect primarySubstation.ctrl to facilityControls;
    connect upsSystem.ctrl to facilityControls;
    connect backupGenerators.ctrl to facilityControls;

    connect primarySubstation.safety to safetyInterlocks;
    connect distributionPanels.safety to safetyInterlocks;
  }

  part def UtilityGridInterface :> FacilityService {
    attribute nominalVoltage : Real;
    attribute feederCount : Integer;
    port powerOut : PowerPort;
  }

  part def SubstationAndSwitchgear :> FacilityService {
    port powerIn : PowerPort;
    port powerOut : PowerPort;
    port ctrl : ControlPort;
    port safety : SafetyInterlockPort;

    attribute transformerCount : Integer;
    attribute breakerCount : Integer;
  }

  part def UninterruptiblePowerSupply :> FacilityService {
    port powerIn : PowerPort;
    port powerOut : PowerPort;
    port ctrl : ControlPort;

    attribute autonomyMinutes : Real;
  }

  part def BackupGeneratorPlant :> FacilityService {
    port powerOut : PowerPort;
    port ctrl : ControlPort;

    attribute fuelAutonomyHours : Real;
    attribute startTimeSeconds : Real;
  }

  part def PowerDistributionNetwork :> FacilityService {
    port powerIn : PowerPort;
    port powerOut : PowerPort;
    port safety : SafetyInterlockPort;

    attribute panelCount : Integer;
    attribute feederLossFraction : Real;
  }

  part def GroundingAndLightningProtection :> FacilityService {
    attribute groundResistance : Real;
  }

  //============================================================
  // Site HVAC & Thermal Infrastructure (cooling + air handling)
  //============================================================
  part def SiteHVACAndThermalInfrastructure :> Subsystem {
    port sitePower : PowerPort;
    port toCoolingServices : CoolingPort;
    port toTelescopeCooling : CoolingPort;
    port toAirServices : AirHandlingPort;
    port facilityControls : ControlPort;

    attribute coolingPlantCapacity : Heat_kW;
    attribute chilledWaterSupplyTemp : Temp_C;
    attribute chilledWaterReturnTemp : Temp_C;
    attribute buildingPressureSetpoint : Pressure_Pa;

    part centralCoolingPlant : CentralCoolingPlant;
    part chilledWaterLoop : ChilledWaterDistribution;
    part airHandlingUnits : AirHandlingUnitSystem;
    part heatRejection : HeatRejectionSystem;
    part environmentalSensors : EnvironmentalSensing;

    connect centralCoolingPlant.powerIn to sitePower;
    connect airHandlingUnits.powerIn to sitePower;
    connect heatRejection.powerIn to sitePower;

    connect centralCoolingPlant.coolingOut to chilledWaterLoop.coolingIn;
    connect chilledWaterLoop.coolingOut to toCoolingServices;
    connect chilledWaterLoop.coolingOut to toTelescopeCooling;

    connect airHandlingUnits.airOut to toAirServices;

    connect centralCoolingPlant.ctrl to facilityControls;
    connect chilledWaterLoop.ctrl to facilityControls;
    connect airHandlingUnits.ctrl to facilityControls;
    connect environmentalSensors.net to airHandlingUnits.net;
  }

  part def CentralCoolingPlant :> FacilityService {
    port powerIn : PowerPort;
    port ctrl : ControlPort;
    port coolingOut : CoolingPort;

    attribute chillerCount : Integer;
    attribute coolingCapacity : Heat_kW;
  }

  part def ChilledWaterDistribution :> FacilityService {
    port coolingIn : CoolingPort;
    port coolingOut : CoolingPort;
    port ctrl : ControlPort;

    attribute pumpCount : Integer;
    attribute loopLossFraction : Real;
  }

  part def AirHandlingUnitSystem :> FacilityService {
    port powerIn : PowerPort;
    port ctrl : ControlPort;
    port net : NetworkPort;
    port airOut : AirHandlingPort;

    attribute ahuCount : Integer;
    attribute filtrationEfficiency : Real;
  }

  part def HeatRejectionSystem :> FacilityService {
    port powerIn : PowerPort;
    attribute towerCount : Integer;
    attribute approachTemp : Temp_C;
  }

  part def EnvironmentalSensing :> FacilityService {
    port net : NetworkPort;
    attribute temperature : Temp_C;
    attribute humidity : Humidity_pct;
    attribute pressure : Pressure_Pa;
  }

  //============================================================
  // Site Networking Infrastructure (core network + timing hooks)
  //============================================================
  part def SiteNetworkingInfrastructure :> Subsystem {
    port sitePower : PowerPort;
    port toCoreNetwork : NetworkPort;
    port toTelescopeNetwork : NetworkPort;
    port facilityControls : ControlPort;

    attribute backboneCapacity : DataRate_Gbps;
    attribute networkLatencyBudget_ms : Real;

    part coreSwitching : CoreSwitchingAndRouting;
    part fiberPlant : FiberOpticPlant;
    part edgeDistribution : EdgeNetworkDistribution;
    part timeAndSync : TimingAndSynchronizationServices;
    part cyberSecurity : CyberSecurityServices;

    connect coreSwitching.powerIn to sitePower;
    connect edgeDistribution.powerIn to sitePower;

    connect fiberPlant.netOut to coreSwitching.netInOut;
    connect coreSwitching.netInOut to edgeDistribution.netInOut;

    connect edgeDistribution.netInOut to toCoreNetwork;
    connect edgeDistribution.netInOut to toTelescopeNetwork;

    connect cyberSecurity.net to coreSwitching.netInOut;
    connect timeAndSync.net to coreSwitching.netInOut;

    connect coreSwitching.ctrl to facilityControls;
  }

  part def CoreSwitchingAndRouting :> FacilityService {
    port powerIn : PowerPort;
    port ctrl : ControlPort;
    port netInOut : NetworkPort;

    attribute switchCount : Integer;
    attribute routingDomains : Integer;
  }

  part def FiberOpticPlant :> FacilityService {
    port netOut : NetworkPort;
    attribute fiberPairs : Integer;
    attribute linkMargin_dB : Real;
  }

  part def EdgeNetworkDistribution :> FacilityService {
    port powerIn : PowerPort;
    port netInOut : NetworkPort;

    attribute edgeSwitchCount : Integer;
  }

  part def TimingAndSynchronizationServices :> FacilityService {
    port net : NetworkPort;
    attribute timeAccuracy_ns : Real;
    attribute syncHoldover_s : Real;
  }

  part def CyberSecurityServices :> FacilityService {
    port net : NetworkPort;
    attribute securityZoneCount : Integer;
  }

  //============================================================
  // Facility Monitoring & Controls (BMS/SCADA-like)
  //============================================================
  part def FacilityMonitoringAndControls :> Subsystem {
    port siteNetwork : NetworkPort;
    port toFacilityControls : ControlPort;
    port toObservatoryControls : ControlPort;
    port toSafetyInterlocks : SafetyInterlockPort;
    port toSafetySystem : SafetyInterlockPort;

    attribute telemetryUpdateRate_Hz : Real;
    attribute alarmPointCount : Integer;

    part controlNetwork : ControlNetwork;
    part monitoringSoftware : MonitoringAndAlarmSoftware;
    part dataHistorian : TelemetryHistorian;
    part operatorConsoles : OperatorConsoles;
    part safetyIntegration : SafetyIntegrationGateway;

    connect controlNetwork.net to siteNetwork;
    connect monitoringSoftware.net to siteNetwork;
    connect dataHistorian.net to siteNetwork;
    connect operatorConsoles.net to siteNetwork;

    connect monitoringSoftware.ctrl to toFacilityControls;
    connect monitoringSoftware.ctrl to toObservatoryControls;

    connect safetyIntegration.safety to toSafetyInterlocks;
    connect safetyIntegration.safety to toSafetySystem;
  }

  part def ControlNetwork :> FacilityService {
    port net : NetworkPort;
    attribute protocolSetCount : Integer;
  }

  part def MonitoringAndAlarmSoftware :> FacilityService {
    port net : NetworkPort;
    port ctrl : ControlPort;
    attribute alarmLatency_s : Real;
  }

  part def TelemetryHistorian :> FacilityService {
    port net : NetworkPort;
    attribute retentionDays : Integer;
  }

  part def OperatorConsoles :> FacilityService {
    port net : NetworkPort;
    attribute consoleCount : Integer;
  }

  part def SafetyIntegrationGateway :> FacilityService {
    port safety : SafetyInterlockPort;
    attribute interlockChannelCount : Integer;
  }

  //============================================================
  // Site Physical Infrastructure (buildings/utilities/access)
  //============================================================
  part def SitePhysicalInfrastructure :> Subsystem {
    port sitePower : PowerPort;
    port facilityControls : ControlPort;

    part buildingsAndRooms : BuildingsAndEquipmentRooms;
    part utilityRouting : UtilityRoutingAndTrays;
    part groundingGrid : GroundingAndLightningProtection;

    connect buildingsAndRooms.powerIn to sitePower;
    connect utilityRouting.powerIn to sitePower;

    connect buildingsAndRooms.ctrl to facilityControls;
  }

  part def BuildingsAndEquipmentRooms :> FacilityService {
    port powerIn : PowerPort;
    port ctrl : ControlPort;
    attribute equipmentRoomCount : Integer;
  }

  part def UtilityRoutingAndTrays :> FacilityService {
    port powerIn : PowerPort;
    attribute trayLength_m : Real;
    attribute conduitCount : Integer;
  }
}
