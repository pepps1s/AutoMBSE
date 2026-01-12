package IBD_InterfaceModel {

    package InstrumentsDataSystem {

        // Port Definitions for Instrument and Data System Interfaces
        port def RawDataFlow {
            out rawData: RawFrame;
            in ackRetry: Acknowledgment;
        }

        port def CalibrationDataFlow {
            out calibrationData: CalibrationFrame;
            in ackRetry: Acknowledgment;
        }

        port def MetadataFlow {
            out metadata: MetadataRecord;
            in ackRetry: Acknowledgment;
        }

        // Part Definitions for Instrument and Data System
        part def Instrument {
            port out rawDataPort: RawDataFlow;
            port out calibrationDataPort: CalibrationDataFlow;
            port out metadataPort: MetadataFlow;
        }

        part def DataSystem {
            port in rawDataPort: RawDataFlow;
            port in calibrationDataPort: CalibrationDataFlow;
            port in metadataPort: MetadataFlow;
        }

        // Interface Definition for Acknowledgment and Retry Behavior
        part def Acknowledgment {
            attribute ackStatus: Boolean;
            attribute retryCount: Integer;
        }

        // Data Structures Definitions
        part def RawFrame {
            attribute dataContent: String;
        }

        part def CalibrationFrame {
            attribute dataContent: String;
            attribute calibrationType: String;
        }

        part def MetadataRecord {
            attribute timestamp: String;
            attribute configurationContext: String;
        }

        // Interface Behavior for Raw Data, Calibration Data, and Metadata
        behavior RawDataFlowBehavior {
            // Acknowledgment and Retry Logic for Raw Data Flow
            state MachineState {
                state awaitingData: {
                    transition on rawData to awaitingAck;
                }
                state awaitingAck: {
                    transition on ackRetry.ackStatus == true to awaitingData;
                    transition on ackRetry.retryCount >= 3 to failed;
                }
                state failed: {
                    // Logic for failed acknowledgment/retry
                }
            }
        }

        behavior CalibrationDataFlowBehavior {
            // Acknowledgment and Retry Logic for Calibration Data Flow
            state MachineState {
                state awaitingData: {
                    transition on calibrationData to awaitingAck;
                }
                state awaitingAck: {
                    transition on ackRetry.ackStatus == true to awaitingData;
                    transition on ackRetry.retryCount >= 3 to failed;
                }
                state failed: {
                    // Logic for failed acknowledgment/retry
                }
            }
        }

        behavior MetadataFlowBehavior {
            // Acknowledgment and Retry Logic for Metadata Flow
            state MachineState {
                state awaitingData: {
                    transition on metadata to awaitingAck;
                }
                state awaitingAck: {
                    transition on ackRetry.ackStatus == true to awaitingData;
                    transition on ackRetry.retryCount >= 3 to failed;
                }
                state failed: {
                    // Logic for failed acknowledgment/retry
                }
            }
        }
    }

    // Contract Between Instrument and Data System (Interface View)
    block InstrumentDataSystemContract {
        part instrument: Instrument;
        part dataSystem: DataSystem;

        // Connectors for Data Flows
        connector rawDataConnector: instrument.rawDataPort -> dataSystem.rawDataPort;
        connector calibrationDataConnector: instrument.calibrationDataPort -> dataSystem.calibrationDataPort;
        connector metadataConnector: instrument.metadataPort -> dataSystem.metadataPort;
    }
}
