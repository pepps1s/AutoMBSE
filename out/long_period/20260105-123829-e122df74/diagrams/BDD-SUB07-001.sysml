package BDD_SUB07_ObservatoryControlSystem {
    /*
      diagram_id: BDD-SUB07-001
      type: BDD
      title: BDD SUB07: Observatory Control System (sequencer, config, telemetry, alarms)
      goal: Define structural decomposition (parts/attributes/interfaces) of TMT for subsystem: Observatory Control System.
      dependencies: RD-F-001, RD-NF-001
    */

    import ScalarValues::*;
    import SI::*;
    import Quantities::*;

    // =========================================================
    // Requirement placeholders (to ensure RD-2 traceability)
    // NOTE: These mirror requirement-names flagged by validation.
    // =========================================================

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

    // =========================================================
    // Common value types
    // =========================================================

    attribute def UUID {
        value : String;
    }

    attribute def SeverityLevel {
        value : Integer;
    }

    attribute def Timestamp {
        value : Real;
    }

    attribute def KeyValue {
        key : String;
        value : String;
    }

    // =========================================================
    // Interfaces / Port Types
    // =========================================================

    interface def CommandIF {
        in commandName : String;
        in parameters[0..*] : KeyValue;
        out ack : Boolean;
    }

    interface def EventIF {
        inout eventTopic : String;
        inout payload : String;
    }

    interface def TelemetryIF {
        out streamId : String;
        out timestamp : Timestamp;
        out payload : String;
    }

    interface def ConfigIF {
        in profileName : String;
        in kv[0..*] : KeyValue;
        out applied : Boolean;
    }

    interface def AlarmIF {
        out alarmId : UUID;
        out severity : SeverityLevel;
        out message : String;
        out timestamp : Timestamp;
        in ack : Boolean;
    }

    interface def TimeSyncIF {
        inout taiSeconds : Real;
        inout utcOffset : Real;
    }

    // =========================================================
    // Abstract logical components
    // =========================================================

    abstract part def OCS_Component {
        attribute componentId : UUID;
        attribute version : String;
    }

    abstract part def SubsystemAdapter :> OCS_Component {
        port cmd : CommandIF;
        port evt : EventIF;
        port tm : TelemetryIF;
        port cfg : ConfigIF;
    }

    // =========================================================
    // Key OCS components (sequencer, config, telemetry, alarms)
    // =========================================================

    part def ObservationSequencer :> OCS_Component {
        attribute state : String;
        port cmd : CommandIF;
        port evt : EventIF;
        port cfg : ConfigIF;
        port tm : TelemetryIF;
        port alarm : AlarmIF;
    }

    part def ConfigurationManager :> OCS_Component {
        attribute activeProfile : String;
        port cfg : ConfigIF;
        port evt : EventIF;
        port alarm : AlarmIF;
    }

    part def TelemetryService :> OCS_Component {
        attribute sampleRateHz : Real;
        port tmOut : TelemetryIF;
        port evt : EventIF;
        port alarm : AlarmIF;
    }

    part def AlarmManager :> OCS_Component {
        attribute activeAlarmCount : Integer;
        port alarmOut : AlarmIF;
        port evt : EventIF;
    }

    part def OperatorConsole :> OCS_Component {
        port cmdOut : CommandIF;
        port tmIn : TelemetryIF;
        port alarmIn : AlarmIF;
        port cfgInOut : ConfigIF;
        port evt : EventIF;
    }

    part def TimeSyncService :> OCS_Component {
        port time : TimeSyncIF;
        port evt : EventIF;
        port alarm : AlarmIF;
    }

    part def EventBus :> OCS_Component {
        port evt[1..*] : EventIF;
    }

    part def AuditLogService :> OCS_Component {
        attribute retentionDays : Integer;
        port evt : EventIF;
        port alarm : AlarmIF;
    }

    part def ScienceGoalTraceService :> OCS_Component {
        // maps science drivers to observation templates/constraints
        port cfg : ConfigIF;
        port evt : EventIF;
    }

    part def ObservingAssumptionsService :> OCS_Component {
        // owns observing conditions assumptions and distributes derived constraints
        port cfg : ConfigIF;
        port evt : EventIF;
        port tm : TelemetryIF;
    }

    part def PhasingAndAlignmentCoordinator :> OCS_Component {
        // coordinates segment phasing/alignment sequences
        port cmd : CommandIF;
        port evt : EventIF;
        port tm : TelemetryIF;
        port alarm : AlarmIF;
    }

    part def PointingTrackingCoordinator :> OCS_Component {
        // coordinates pointing/tracking modes and stability monitoring hooks
        port cmd : CommandIF;
        port evt : EventIF;
        port tm : TelemetryIF;
        port alarm : AlarmIF;
    }

    part def ExposureControlCoordinator :> OCS_Component {
        // exposure loop timing/phasing and calibration cadence control
        port cmd : CommandIF;
        port evt : EventIF;
        port tm : TelemetryIF;
        port cfg : ConfigIF;
        port alarm : AlarmIF;
    }

    part def AdaptiveOpticsModeCoordinator :> OCS_Component {
        // optional AO enablement, configuration, and loop state integration
        attribute aoEnabled : Boolean;
        port cmd : CommandIF;
        port cfg : ConfigIF;
        port tm : TelemetryIF;
        port evt : EventIF;
        port alarm : AlarmIF;
    }

    part def PerformanceMonitor :> OCS_Component {
        // monitors sensitivity/angular resolution proxies via telemetry/metadata
        port tmIn : TelemetryIF;
        port evt : EventIF;
        port alarm : AlarmIF;
    }

    // =========================================================
    // External subsystem adapters (placeholders for connections)
    // =========================================================

    part def TelescopeControlAdapter :> SubsystemAdapter { }
    part def InstrumentsAdapter :> SubsystemAdapter { }
    part def DataSystemAdapter :> SubsystemAdapter { }
    part def SafetyInterlocksAdapter :> SubsystemAdapter { }
    part def AlignmentPhasingAdapter :> SubsystemAdapter { }
    part def AOAdapter :> SubsystemAdapter { }

    // =========================================================
    // Top-level part: Observatory Control System (SUB07)
    // =========================================================

    part def ObservatoryControlSystem :> OCS_Component {
        attribute facilityName : String;
        attribute site : String;

        part bus : EventBus;
        part timeSync : TimeSyncService;
        part sequencer : ObservationSequencer;
        part configMgr : ConfigurationManager;
        part telemetry : TelemetryService;
        part alarms : AlarmManager;
        part console : OperatorConsole;
        part auditLog : AuditLogService;

        // Requirement-driven coordinators/services
        part scienceTrace : ScienceGoalTraceService;
        part assumptions : ObservingAssumptionsService;
        part phasingCoord : PhasingAndAlignmentCoordinator;
        part pointingCoord : PointingTrackingCoordinator;
        part exposureCoord : ExposureControlCoordinator;
        part aoCoord : AdaptiveOpticsModeCoordinator;
        part perfMon : PerformanceMonitor;

        // External adapters
        part telescope : TelescopeControlAdapter;
        part instruments : InstrumentsAdapter;
        part data : DataSystemAdapter;
        part safety : SafetyInterlocksAdapter;
        part aps : AlignmentPhasingAdapter;
        part ao : AOAdapter;

        // -------------------------
        // Key connections (IBD-like)
        // -------------------------

        // Console interactions
        connect console.cmdOut to sequencer.cmd;
        connect console.cfgInOut to configMgr.cfg;
        connect telemetry.tmOut to console.tmIn;
        connect alarms.alarmOut to console.alarmIn;
        connect console.evt to bus.evt[1];

        // Sequencer orchestration
        connect sequencer.cfg to configMgr.cfg;
        connect sequencer.tm to telemetry.tmOut;
        connect sequencer.alarm to alarms.alarmOut;
        connect sequencer.evt to bus.evt[2];

        // Event & logging backbone
        connect configMgr.evt to bus.evt[3];
        connect telemetry.evt to bus.evt[4];
        connect alarms.evt to bus.evt[5];
        connect timeSync.evt to bus.evt[6];
        connect auditLog.evt to bus.evt[7];

        // External subsystem command/telemetry pathways (simplified)
        connect sequencer.cmd to telescope.cmd;
        connect sequencer.cmd to instruments.cmd;
        connect sequencer.cmd to data.cmd;
        connect sequencer.cmd to safety.cmd;

        connect telescope.tm to telemetry.tmOut;
        connect instruments.tm to telemetry.tmOut;
        connect data.tm to telemetry.tmOut;
        connect safety.tm to telemetry.tmOut;

        connect telescope.evt to bus.evt[8];
        connect instruments.evt to bus.evt[9];
        connect data.evt to bus.evt[10];
        connect safety.evt to bus.evt[11];

        // Coordinators to their adapters
        connect phasingCoord.cmd to aps.cmd;
        connect phasingCoord.tm to telemetry.tmOut;
        connect phasingCoord.alarm to alarms.alarmOut;
        connect phasingCoord.evt to bus.evt[12];

        connect pointingCoord.cmd to telescope.cmd;
        connect pointingCoord.tm to telemetry.tmOut;
        connect pointingCoord.alarm to alarms.alarmOut;
        connect pointingCoord.evt to bus.evt[13];

        connect exposureCoord.cmd to instruments.cmd;
        connect exposureCoord.cfg to configMgr.cfg;
        connect exposureCoord.tm to telemetry.tmOut;
        connect exposureCoord.alarm to alarms.alarmOut;
        connect exposureCoord.evt to bus.evt[14];

        connect aoCoord.cmd to ao.cmd;
        connect aoCoord.cfg to configMgr.cfg;
        connect aoCoord.tm to telemetry.tmOut;
        connect aoCoord.alarm to alarms.alarmOut;
        connect aoCoord.evt to bus.evt[15];

        // Science/assumptions/performance monitoring integration
        connect scienceTrace.cfg to configMgr.cfg;
        connect scienceTrace.evt to bus.evt[16];

        connect assumptions.cfg to configMgr.cfg;
        connect assumptions.tm to telemetry.tmOut;
        connect assumptions.evt to bus.evt[17];

        connect perfMon.tmIn to telemetry.tmOut;
        connect perfMon.evt to bus.evt[18];
        connect perfMon.alarm to alarms.alarmOut;

        // Alarm/log integration
        connect alarms.alarmOut to auditLog.alarm;
        connect alarms.evt to auditLog.evt;
    }

    // =========================================================
    // RD-2: Requirement-to-Implementation mappings (satisfy)
    // =========================================================

    // Science driver / goal mapping
    satisfy MissionGoalsAndScienceDrivers::MissionGoalsAndScienceDrivers by ObservatoryControlSystem::scienceTrace;
    satisfy EarlyUniverseGoalMapping::EarlyUniverseGoalMapping by ObservatoryControlSystem::scienceTrace;
    satisfy GalaxyEvolutionGoalMapping::GalaxyEvolutionGoalMapping by ObservatoryControlSystem::scienceTrace;
    satisfy ExoplanetsGoalMapping::ExoplanetsGoalMapping by ObservatoryControlSystem::scienceTrace;
    satisfy BHgrowthGoalMapping::BHgrowthGoalMapping by ObservatoryControlSystem::scienceTrace;

    // Stakeholders & constraints handling (ops + alarm/console exposure)
    satisfy StakeholderGroupConcerns::StakeholderGroupConcerns by ObservatoryControlSystem::alarms;
    satisfy StakeholderGroupConcerns::StakeholderGroupConcerns by ObservatoryControlSystem::console;

    // Observing assumptions & conditions management
    satisfy ObservingAssumptionsAndConditionsDef::ObservingAssumptionsAndConditionsDef by ObservatoryControlSystem::assumptions;
    satisfy ObservingAssumptionsAndConditions::ObservingAssumptionsAndConditions by ObservatoryControlSystem::assumptions;

    // Functional requirements: phasing / pointing / exposure control / AO optional
    satisfy FR_PhasedSegmentedApertureDef::FR_PhasedSegmentedApertureDef by ObservatoryControlSystem::phasingCoord;
    satisfy FR_PhasedSegmentedAperture::FR_PhasedSegmentedAperture by ObservatoryControlSystem::phasingCoord;

    satisfy FR_PointingTrackingAndStabilityDef::FR_PointingTrackingAndStabilityDef by ObservatoryControlSystem::pointingCoord;
    satisfy FR_PointingTrackingAndStability::FR_PointingTrackingAndStability by ObservatoryControlSystem::pointingCoord;

    satisfy FR_ObservationPhasingAndExposureControlDef::FR_ObservationPhasingAndExposureControlDef by ObservatoryControlSystem::exposureCoord;
    satisfy FR_ObservationPhasingAndExposureControl::FR_ObservationPhasingAndExposureControl by ObservatoryControlSystem::exposureCoord;

    satisfy FR_AdaptiveOpticsOptionalDef::FR_AdaptiveOpticsOptionalDef by ObservatoryControlSystem::aoCoord;
    satisfy FR_AdaptiveOpticsOptional::FR_AdaptiveOpticsOptional by ObservatoryControlSystem::aoCoord;

    // Performance NFR hooks (telemetry + monitoring + alarms)
    satisfy SensitivityPerformanceNFRDef::SensitivityPerformanceNFRDef by ObservatoryControlSystem::perfMon;
    satisfy AngularResolutionPerformanceNFRDef::AngularResolutionPerformanceNFRDef by ObservatoryControlSystem::perfMon;

    // =========================================================
    // System-of-interest part usage (ties SUB07 into TMT context)
    // =========================================================

    part TMT :> OCS_Component {
        part ocs : ObservatoryControlSystem;
    }
}
