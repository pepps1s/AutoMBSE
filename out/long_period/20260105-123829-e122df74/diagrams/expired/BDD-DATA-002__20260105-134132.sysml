package BDD-INS-DATA-002 {
    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    part def Telescope {
        part opticalSystem : TelescopeOpticalSystem;
        part structure : TelescopeStructure;
        part enclosure : EnclosureInfrastructure;
        part alignmentSystem : AlignmentSystem;
        part adaptiveOptics : AdaptiveOpticsSystem;
        part scienceInstruments : ScienceInstruments;
        part observatoryControl : ObservatoryControl;
        part dataSystem : DataSystem;
        part safetySystem : SafetySystem;
        part maintenanceSystem : MaintenanceSystem;
    }

    part def TelescopeOpticalSystem {
        part primaryMirror : Mirror;
        part secondaryMirror : Mirror;
        part tertiaryMirror : Mirror;
    }

    part def TelescopeStructure {
        part mount : MountSystem;
        part vibrationControl : VibrationControl;
    }

    part def EnclosureInfrastructure {
        part domeRotation : DomeSystem;
        part environmentalControl : EnvironmentalSystem;
    }

    part def AlignmentSystem {
        part wavefrontSensor : WavefrontSensor;
        part segmentActuators : SegmentActuators;
    }

    part def AdaptiveOpticsSystem {
        part realTimeController : AOController;
        part wavefrontSensors : AOWavefrontSensors;
        part deformableMirrors : DeformableMirror;
    }

    part def ScienceInstruments {
        part instrumentPorts : InstrumentPort[4];  % Example: 4 ports available %
        part instrumentControl : InstrumentControl;
        part calibrationUnits : CalibrationUnit;
    }

    part def ObservatoryControl {
        part sequencer : Sequencer;
        part scheduler : Scheduler;
        part telemetry : TelemetrySystem;
        part healthMonitoring : HealthMonitoring;
    }

    part def DataSystem {
        part acquisition : AcquisitionNode;
        part metadataService : MetadataService;
        part pipeline : PipelineCompute;
        part archive : ArchiveInterface;
        part distributionGateway : DistributionGateway;
    }

    part def SafetySystem {
        part interlocks : SafetyInterlocks;
        part emergencyStop : EmergencyStop;
    }

    part def MaintenanceSystem {
        part faultHandling : FaultHandler;
        part subsystemCheck : SubsystemCheck;
        part maintenanceMode : MaintenanceMode;
    }
    
    part def MountSystem {
        part azimuthDrive : Drive;
        part altitudeDrive : Drive;
        part encoders : Encoder;
    }

    part def Mirror {
        attribute size : LengthValue;
        attribute coating : MaterialType;
    }

    part def Drive {
        attribute maxSpeed : VelocityValue;
        attribute powerConsumption : PowerValue;
    }

    part def Encoder {
        attribute resolution : ResolutionValue;
    }

    part def Sequencer {
        attribute sequenceType : SequenceType;
        attribute startTime : TimeValue;
    }

    part def Scheduler {
        attribute scheduleTime : TimeValue;
        attribute taskPriority : Integer;
    }

    part def TelemetrySystem {
        attribute dataRate : DataRateValue;
        attribute updateFrequency : FrequencyValue;
    }

    part def HealthMonitoring {
        attribute systemHealth : SystemHealthStatus;
        attribute failureMode : FailureMode;
    }

    part def AcquisitionNode {
        attribute sensorType : SensorType;
        attribute dataRate : DataRateValue;
    }

    part def MetadataService {
        attribute metadataFormat : MetadataFormat;
        attribute metadataVolume : VolumeValue;
    }

    part def PipelineCompute {
        attribute processingPower : PowerValue;
        attribute processingTime : TimeValue;
    }

    part def ArchiveInterface {
        attribute storageCapacity : StorageCapacityValue;
        attribute redundancyLevel : Integer;
    }

    part def DistributionGateway {
        attribute bandwidth : BandwidthValue;
        attribute distributionLatency : TimeValue;
    }

    part def SafetyInterlocks {
        attribute status : InterlockStatus;
        attribute failureDetection : FailureDetection;
    }

    part def EmergencyStop {
        attribute triggerCondition : TriggerCondition;
        attribute responseAction : ResponseAction;
    }

    part def FaultHandler {
        attribute errorCode : Integer;
        attribute recoveryAction : RecoveryAction;
    }

    part def SubsystemCheck {
        attribute componentStatus : ComponentStatus;
        attribute checkFrequency : FrequencyValue;
    }

    part def MaintenanceMode {
        attribute modeType : ModeType;
        attribute modeDuration : TimeValue;
    }

    abstract part def InstrumentPort;
    abstract part def InstrumentControl;
    abstract part def CalibrationUnit;
    
    abstract part def AOController;
    abstract part def AOWavefrontSensors;
    abstract part def DeformableMirror;
    abstract part def WavefrontSensor;
    abstract part def SegmentActuators;
    
    abstract part def SafetyInterlocks;
    abstract part def EmergencyStop;
}
