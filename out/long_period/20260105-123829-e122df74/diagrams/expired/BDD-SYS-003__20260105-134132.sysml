package TMT_BDD_SYS_003 {
    import ScalarValues::*;
    import SI::*;
    import Quantities::*;

    //============================================================
    // BDD-SYS-003: TMT System Decomposition (Level 2)
    //============================================================

    // Base abstractions
    abstract part def TMTComponent;
    abstract part def MechanicalSubsystem :> TMTComponent;
    abstract part def OpticalSubsystem :> TMTComponent;
    abstract part def ElectricalSubsystem :> TMTComponent;
    abstract part def SoftwareSubsystem :> TMTComponent;
    abstract part def FacilitySubsystem :> TMTComponent;
    abstract part def SafetySubsystem :> TMTComponent;
    abstract part def OperationsSubsystem :> TMTComponent;

    // Top-level system
    part def ThirtyMeterTelescope :> TMTComponent {
        // Primary architectural components (Level 2)
        part optics : TelescopeOptics;
        part mount : MountAndStructure;
        part enclosure : EnclosureAndSiteInfrastructure;

        part aps : AlignmentAndPhasingSystem;
        part ao : AdaptiveOpticsSubsystem;

        part ocs : ObservatoryControlSystem;
        part data : DataSystem;

        part safety : SafetyAndCompliance;
        part om : OperationsAndMaintenance;
    }

    //============================================================
    // Optics
    //============================================================
    part def TelescopeOptics :> OpticalSubsystem {
        part m1 : PrimaryMirrorAssembly;
        part m2 : SecondaryMirrorAssembly;
        part m3 : TertiaryMirrorAssembly;
        part beamTransfer : OpticalPathAndBeamTransfer;
        part opticsSupport : OpticsSupportAndMetrology;
    }

    part def PrimaryMirrorAssembly :> OpticalSubsystem {
        part segmentedPrimary : SegmentedPrimaryMirror;
        part segmentSupport : SegmentSupportStructure;
        part segmentActuation : SegmentActuationSystem;
        part mirrorThermal : MirrorThermalControl;
    }

    part def SegmentedPrimaryMirror :> OpticalSubsystem {
        part segment[492] : MirrorSegment;
        part segmentEdgeSensors : SegmentEdgeSensing;
    }

    part def MirrorSegment :> OpticalSubsystem {
        part surface : OpticalSurface;
        part mountKinematic : SegmentKinematicMount;
    }

    part def SecondaryMirrorAssembly :> OpticalSubsystem {
        part secondaryMirror : SecondaryMirror;
        part secondaryPositioning : SecondaryPositioningSystem;
    }

    part def TertiaryMirrorAssembly :> OpticalSubsystem {
        part tertiaryMirror : TertiaryMirror;
        part tertiarySteering : TertiarySteeringSystem;
    }

    part def OpticalPathAndBeamTransfer :> OpticalSubsystem {
        part foldOptics : FoldOptics;
        part instrumentFeed : InstrumentFeedOptics;
    }

    part def OpticsSupportAndMetrology :> OpticalSubsystem {
        part metrology : OpticalMetrology;
        part alignmentReferences : AlignmentReferences;
    }

    //============================================================
    // Mount / Structure
    //============================================================
    part def MountAndStructure :> MechanicalSubsystem {
        part altAzMount : AltAzMount;
        part drives : DriveSystem;
        part bearings : BearingSystem;
        part encoders : EncoderSystem;
        part structuralFrame : StructuralFrame;
        part vibrationControl : VibrationControlSystem;
        part structuralThermal : StructuralThermalControl;
    }

    part def AltAzMount :> MechanicalSubsystem {
        part altitudeAxis : AltitudeAxisAssembly;
        part azimuthAxis : AzimuthAxisAssembly;
    }

    part def DriveSystem :> MechanicalSubsystem {
        part motors : MotorAssemblies;
        part gearboxes : GearboxAssemblies;
        part controlElectronics : DriveControlElectronics;
    }

    part def BearingSystem :> MechanicalSubsystem {
        part altitudeBearings : AltitudeBearings;
        part azimuthBearings : AzimuthBearings;
    }

    part def EncoderSystem :> MechanicalSubsystem {
        part altitudeEncoders : AltitudeEncoders;
        part azimuthEncoders : AzimuthEncoders;
    }

    //============================================================
    // Enclosure / Dome & Site Infrastructure
    //============================================================
    part def EnclosureAndSiteInfrastructure :> FacilitySubsystem {
        part dome : DomeAndShutters;
        part rotation : EnclosureRotationSystem;
        part windThermalMgmt : WindAndThermalManagement;

        part sitePower : PowerInfrastructure;
        part hvac : HVACAndCooling;
        part networking : SiteNetworking;
        part accessHandling : AccessAndHandlingSystems;
        part facilitiesMonitoring : FacilitiesMonitoring;
    }

    part def DomeAndShutters :> FacilitySubsystem {
        part enclosureShell : EnclosureShell;
        part shutters : ShutterSystem;
        part vents : VentilationSystem;
    }

    part def PowerInfrastructure :> FacilitySubsystem {
        part primaryPower : PrimaryPowerDistribution;
        part backupPower : BackupPowerAndUPS;
        part grounding : GroundingAndLightningProtection;
    }

    part def HVACAndCooling :> FacilitySubsystem {
        part airHandling : AirHandlingUnits;
        part coolantLoops : CoolantLoopSystem;
        part thermalSensors : FacilityThermalSensing;
    }

    //============================================================
    // APS (Alignment & Phasing)
    //============================================================
    part def AlignmentAndPhasingSystem :> OpticalSubsystem {
        part wavefrontSensing : WavefrontSensingSubsystem;
        part phasingControl : PhasingControlSubsystem;
        part segmentControl : SegmentControlSubsystem;
        part apsMetrology : APSMetrologySubsystem;
    }

    part def WavefrontSensingSubsystem :> OpticalSubsystem {
        part shWfs : ShackHartmannWFS;
        part acquisitionOptics : WFSAcquisitionOptics;
        part wfsCalibration : WFSCalibrationUnit;
    }

    part def PhasingControlSubsystem :> OpticalSubsystem {
        part phasingAlgorithms : PhasingAlgorithms;
        part phasingController : PhasingController;
    }

    part def SegmentControlSubsystem :> ElectricalSubsystem {
        part segmentActuators : SegmentActuatorNetwork;
        part actuatorDrivers : ActuatorDriverElectronics;
        part segmentSensors : SegmentSensorNetwork;
    }

    part def APSMetrologySubsystem :> OpticalSubsystem {
        part referenceSources : MetrologyReferenceSources;
        part measurementDevices : MetrologyMeasurementDevices;
        part alignmentModels : AlignmentModels;
    }

    //============================================================
    // AO (Adaptive Optics)
    //============================================================
    part def AdaptiveOpticsSubsystem :> OpticalSubsystem {
        part wfs : AOWavefrontSensing;
        part deformableOptics : DeformableOpticsSubsystem;
        part rtc : AORealTimeController;
        part guideStar : GuideStarSubsystem;
        part aoCalibration : AOCalibrationSubsystem;
    }

    part def AOWavefrontSensing :> OpticalSubsystem {
        part ngsWfs : NaturalGuideStarWFS;
        part lgsWfs : LaserGuideStarWFS;
    }

    part def DeformableOpticsSubsystem :> OpticalSubsystem {
        part deformableMirror : DeformableMirror;
        part tipTiltMirror : TipTiltMirror;
    }

    part def AORealTimeController :> SoftwareSubsystem {
        part computePlatform : RTCComputePlatform;
        part controlLoops : AOControlLoops;
        part telemetry : AOTelemetryPublisher;
    }

    part def GuideStarSubsystem :> OpticalSubsystem {
        part ngsSelection : NGSSelection;
        part lgsFacility : LaserGuideStarFacility;
    }

    part def AOCalibrationSubsystem :> OpticalSubsystem {
        part interactionMatrix : InteractionMatrixManagement;
        part calibrationSources : CalibrationSources;
    }

    //============================================================
    // OCS (Observatory Control System)
    //============================================================
    part def ObservatoryControlSystem :> SoftwareSubsystem {
        part sequencer : ObservationSequencer;
        part schedulerHooks : SchedulingInterface;
        part configuration : ConfigurationManagement;
        part telemetry : TelemetryAndLogging;
        part alarms : AlarmAndEventManagement;
        part operatorUI : OperatorInterface;
        part timeSync : TimeSynchronization;
    }

    part def TelemetryAndLogging :> SoftwareSubsystem {
        part telemetryBus : TelemetryBus;
        part loggingService : LoggingService;
        part healthMonitoring : HealthMonitoring;
    }

    //============================================================
    // Data System
    //============================================================
    part def DataSystem :> SoftwareSubsystem {
        part acquisition : DataAcquisition;
        part metadata : MetadataManagement;
        part quicklook : QuicklookQA;
        part pipeline : SciencePipeline;
        part archive : DataArchive;
        part distribution : DataDistributionInterfaces;
    }

    part def DataAcquisition :> SoftwareSubsystem {
        part detectorReadout : DetectorReadoutIntegration;
        part ingest : DataIngestService;
    }

    part def DataArchive :> SoftwareSubsystem {
        part storage : ArchiveStorage;
        part catalog : ArchiveCatalog;
        part accessAPI : ArchiveAccessAPI;
    }

    //============================================================
    // Safety
    //============================================================
    part def SafetyAndCompliance :> SafetySubsystem {
        part interlocks : InterlockSystem;
        part emergencyStop : EmergencyStopSystem;
        part personnelSafety : PersonnelSafetySubsystem;
        part operationalConstraints : OperationalConstraintsManagement;
        part laserSafety : LaserSafetySubsystem;
        part incidentMgmt : IncidentManagement;
    }

    part def InterlockSystem :> SafetySubsystem {
        part doorGates : DoorAndGateInterlocks;
        part motionInterlocks : MotionInterlocks;
        part highVoltageInterlocks : ElectricalSafetyInterlocks;
    }

    //============================================================
    // Operations & Maintenance (O&M)
    //============================================================
    part def OperationsAndMaintenance :> OperationsSubsystem {
        part startupShutdown : StartupShutdownProcedures;
        part maintenanceModes : MaintenanceModes;
        part faultHandling : FaultDetectionAndRecovery;
        part sparesLogistics : SparesAndLogistics;
        part calibrationOps : CalibrationOperations;
        part verificationSupport : VerificationAndCommissioningSupport;
    }

    part def FaultDetectionAndRecovery :> OperationsSubsystem {
        part anomalyDetection : AnomalyDetection;
        part safeStateMgmt : SafeStateManagement;
        part diagnostics : DiagnosticsSupport;
        part recoveryWorkflows : RecoveryWorkflows;
    }

    //============================================================
    // Leaf placeholders (minimal typing anchors)
    //============================================================
    part def OpticalSurface :> OpticalSubsystem;
    part def SegmentKinematicMount :> MechanicalSubsystem;
    part def SegmentSupportStructure :> MechanicalSubsystem;
    part def SegmentActuationSystem :> ElectricalSubsystem;
    part def MirrorThermalControl :> FacilitySubsystem;

    part def SecondaryMirror :> OpticalSubsystem;
    part def SecondaryPositioningSystem :> MechanicalSubsystem;

    part def TertiaryMirror :> OpticalSubsystem;
    part def TertiarySteeringSystem :> MechanicalSubsystem;

    part def FoldOptics :> OpticalSubsystem;
    part def InstrumentFeedOptics :> OpticalSubsystem;

    part def OpticalMetrology :> OpticalSubsystem;
    part def AlignmentReferences :> OpticalSubsystem;

    part def StructuralFrame :> MechanicalSubsystem;
    part def VibrationControlSystem :> MechanicalSubsystem;
    part def StructuralThermalControl :> FacilitySubsystem;

    part def AltitudeAxisAssembly :> MechanicalSubsystem;
    part def AzimuthAxisAssembly :> MechanicalSubsystem;

    part def MotorAssemblies :> MechanicalSubsystem;
    part def GearboxAssemblies :> MechanicalSubsystem;
    part def DriveControlElectronics :> ElectricalSubsystem;

    part def AltitudeBearings :> MechanicalSubsystem;
    part def AzimuthBearings :> MechanicalSubsystem;

    part def AltitudeEncoders :> ElectricalSubsystem;
    part def AzimuthEncoders :> ElectricalSubsystem;

    part def EnclosureRotationSystem :> FacilitySubsystem;
    part def WindAndThermalManagement :> FacilitySubsystem;
    part def AccessAndHandlingSystems :> FacilitySubsystem;
    part def FacilitiesMonitoring :> FacilitySubsystem;

    part def EnclosureShell :> FacilitySubsystem;
    part def ShutterSystem :> FacilitySubsystem;
    part def VentilationSystem :> FacilitySubsystem;

    part def PrimaryPowerDistribution :> FacilitySubsystem;
    part def BackupPowerAndUPS :> FacilitySubsystem;
    part def GroundingAndLightningProtection :> FacilitySubsystem;

    part def AirHandlingUnits :> FacilitySubsystem;
    part def CoolantLoopSystem :> FacilitySubsystem;
    part def FacilityThermalSensing :> FacilitySubsystem;

    part def ShackHartmannWFS :> OpticalSubsystem;
    part def WFSAcquisitionOptics :> OpticalSubsystem;
    part def WFSCalibrationUnit :> OpticalSubsystem;

    part def PhasingAlgorithms :> SoftwareSubsystem;
    part def PhasingController :> SoftwareSubsystem;

    part def SegmentActuatorNetwork :> ElectricalSubsystem;
    part def ActuatorDriverElectronics :> ElectricalSubsystem;
    part def SegmentSensorNetwork :> ElectricalSubsystem;

    part def MetrologyReferenceSources :> OpticalSubsystem;
    part def MetrologyMeasurementDevices :> OpticalSubsystem;
    part def AlignmentModels :> SoftwareSubsystem;

    part def NaturalGuideStarWFS :> OpticalSubsystem;
    part def LaserGuideStarWFS :> OpticalSubsystem;

    part def DeformableMirror :> OpticalSubsystem;
    part def TipTiltMirror :> OpticalSubsystem;

    part def RTCComputePlatform :> SoftwareSubsystem;
    part def AOControlLoops :> SoftwareSubsystem;
    part def AOTelemetryPublisher :> SoftwareSubsystem;

    part def NGSSelection :> OpticalSubsystem;
    part def LaserGuideStarFacility :> FacilitySubsystem;

    part def InteractionMatrixManagement :> SoftwareSubsystem;
    part def CalibrationSources :> OpticalSubsystem;

    part def ObservationSequencer :> SoftwareSubsystem;
    part def SchedulingInterface :> SoftwareSubsystem;
    part def ConfigurationManagement :> SoftwareSubsystem;
    part def AlarmAndEventManagement :> SoftwareSubsystem;
    part def OperatorInterface :> SoftwareSubsystem;
    part def TimeSynchronization :> SoftwareSubsystem;

    part def TelemetryBus :> SoftwareSubsystem;
    part def LoggingService :> SoftwareSubsystem;
    part def HealthMonitoring :> SoftwareSubsystem;

    part def MetadataManagement :> SoftwareSubsystem;
    part def QuicklookQA :> SoftwareSubsystem;
    part def SciencePipeline :> SoftwareSubsystem;
    part def DataDistributionInterfaces :> SoftwareSubsystem;

    part def DetectorReadoutIntegration :> SoftwareSubsystem;
    part def DataIngestService :> SoftwareSubsystem;

    part def ArchiveStorage :> SoftwareSubsystem;
    part def ArchiveCatalog :> SoftwareSubsystem;
    part def ArchiveAccessAPI :> SoftwareSubsystem;

    part def EmergencyStopSystem :> SafetySubsystem;
    part def PersonnelSafetySubsystem :> SafetySubsystem;
    part def OperationalConstraintsManagement :> SafetySubsystem;
    part def LaserSafetySubsystem :> SafetySubsystem;
    part def IncidentManagement :> SafetySubsystem;

    part def DoorAndGateInterlocks :> SafetySubsystem;
    part def MotionInterlocks :> SafetySubsystem;
    part def ElectricalSafetyInterlocks :> SafetySubsystem;

    part def StartupShutdownProcedures :> OperationsSubsystem;
    part def MaintenanceModes :> OperationsSubsystem;
    part def SparesAndLogistics :> OperationsSubsystem;
    part def CalibrationOperations :> OperationsSubsystem;
    part def VerificationAndCommissioningSupport :> OperationsSubsystem;

    part def AnomalyDetection :> OperationsSubsystem;
    part def SafeStateManagement :> OperationsSubsystem;
    part def DiagnosticsSupport :> OperationsSubsystem;
    part def RecoveryWorkflows :> OperationsSubsystem;
}
