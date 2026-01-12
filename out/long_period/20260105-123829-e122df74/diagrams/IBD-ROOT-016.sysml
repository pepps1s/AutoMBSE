package IBDROOT016_ConfigDistribution {
    package ConfigurationPropagation {
        // Part Definitions for Subsystems
        part def ObservatoryControlSystem {
            port configInPort: ConfigInputPort;
            port configOutPort: ConfigOutputPort;
            port telemetryInPort: TelemetryInputPort;
            port telemetryOutPort: TelemetryOutputPort;
            port safetyInhibitPort: SafetyInhibitPort;
            attribute configManager: ConfigurationManager;
        }

        part def Subsystem1 {
            port configInPort: ConfigInputPort;
            port metadataOutPort: MetadataOutputPort;
            port telemetryInPort: TelemetryInputPort;
            attribute subsystemConfig: SubsystemConfiguration;
        }

        part def Subsystem2 {
            port configInPort: ConfigInputPort;
            port metadataOutPort: MetadataOutputPort;
            port telemetryInPort: TelemetryInputPort;
            attribute subsystemConfig: SubsystemConfiguration;
        }

        part def Subsystem3 {
            port configInPort: ConfigInputPort;
            port metadataOutPort: MetadataOutputPort;
            port telemetryInPort: TelemetryInputPort;
            attribute subsystemConfig: SubsystemConfiguration;
        }

        part def DataSystemProvenance {
            port metadataInPort: MetadataInputPort;
            port telemetryOutPort: TelemetryOutputPort;
            attribute provenanceMetadata: ProvenanceMetadata;
        }

        // Port Definitions
        port def ConfigInputPort {
            in item configData: ConfigurationData;
        }

        port def ConfigOutputPort {
            out item configData: ConfigurationData;
        }

        port def TelemetryInputPort {
            in item telemetryData: TelemetryData;
        }

        port def TelemetryOutputPort {
            out item telemetryData: TelemetryData;
        }

        port def SafetyInhibitPort {
            in item safetyInhibitSignal: SafetySignal;
        }

        port def MetadataOutputPort {
            out item metadataRecord: MetadataRecord;
        }

        port def MetadataInputPort {
            in item metadataRecord: MetadataRecord;
        }

        // Attribute Definitions
        attribute def ConfigurationManager {
            attribute configVersion: String;
            attribute configTimestamp: TimeStamp;
        }

        attribute def SubsystemConfiguration {
            attribute configID: String;
            attribute subsystemStatus: String;
        }

        // Provenance/metadata includes configuration context (RD-F-007 intent)
        attribute def ProvenanceMetadata {
            attribute metadataID: String;
            attribute dataIntegrity: Boolean;
            attribute timestamp: TimeStamp;
            attribute configurationContext: ConfigurationData;
        }

        // Data Definitions
        item def ConfigurationData {
            attribute configSource: String;
            attribute configValues: String;
            attribute configVersion: String;
            attribute configTimestamp: TimeStamp;
        }

        item def TelemetryData {
            attribute dataType: String;
            attribute dataValue: String;
        }

        item def MetadataRecord {
            attribute recordID: String;
            attribute timestamp: TimeStamp;
            attribute sourceSystem: String;
            attribute metadataType: String;
            attribute configurationContext: ConfigurationData;
        }

        item def SafetySignal {
            attribute signalStatus: Boolean;
            attribute signalType: String;
        }

        item def TimeStamp {
            attribute time: String;
        }

        // IBD Context: instantiate BDD types so their attributes exist in instance context (BBD-1)
        part def IBD_ROOT_016_ConfigDistributionAndProvenanceCapture {
            part ocs : ObservatoryControlSystem;
            part subsystem1 : Subsystem1;
            part subsystem2 : Subsystem2;
            part subsystem3 : Subsystem3;
            part dataSystemProvenance : DataSystemProvenance;

            // Configuration Distribution Connectors (separate from command/telemetry lines)
            connector configDistribution {
                end1 port configOutPort of ocs;
                end2 port configInPort of subsystem1;
            }

            connector configDistribution2 {
                end1 port configOutPort of ocs;
                end2 port configInPort of subsystem2;
            }

            connector configDistribution3 {
                end1 port configOutPort of ocs;
                end2 port configInPort of subsystem3;
            }

            // Metadata Capture for Provenance
            connector metadataCapture {
                end1 port metadataOutPort of subsystem1;
                end2 port metadataInPort of dataSystemProvenance;
            }

            connector metadataCapture2 {
                end1 port metadataOutPort of subsystem2;
                end2 port metadataInPort of dataSystemProvenance;
            }

            connector metadataCapture3 {
                end1 port metadataOutPort of subsystem3;
                end2 port metadataInPort of dataSystemProvenance;
            }

            // Telemetry Distribution (non-config line)
            connector telemetryDistribution {
                end1 port telemetryOutPort of dataSystemProvenance;
                end2 port telemetryInPort of ocs;
            }
        }
    }
}
