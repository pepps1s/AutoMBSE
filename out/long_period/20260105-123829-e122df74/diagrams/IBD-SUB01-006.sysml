package IBD_SUB01_006 {

    package TelescopeOpticalSystem {
        // Part Definitions
        part def SegmentActuator {
            port actuatorCommandPort: ActuatorCommand;
            port actuatorStatusPort: ActuatorStatus;
            attribute actuatorType: String;
        }

        part def MetrologySensor {
            port metrologyDataPort: MetrologyData;
            port metrologySyncPort: TimeSync;
            attribute sensorType: String;
        }

        part def LocalController {
            port controlCommandPort: ControlCommand;
            port controlStatusPort: ControlStatus;
            attribute controllerType: String;
        }

        // Port Definitions
        port def ActuatorCommand {
            out command: ActuationCommand;
        }

        port def ActuatorStatus {
            in status: ActuatorState;
        }

        port def MetrologyData {
            in data: WavefrontData;
        }

        port def TimeSync {
            in syncSignal: TimeSyncSignal;
            out syncSignal: TimeSyncSignal;
        }

        port def ControlCommand {
            out command: ControlSignal;
        }

        port def ControlStatus {
            in status: ControllerState;
        }

        // Flow Definitions
        part def ActuationCommand;
        part def ActuatorState;
        part def WavefrontData;
        part def TimeSyncSignal;
        part def ControlSignal;
        part def ControllerState;

        // Attribute Definitions
        attribute def ActuatorType;
        attribute def SensorType;
        attribute def ControllerType;
    }

    // Defining the Internal Block Diagram (IBD) Structure
    internalBlockDiagram IBD_SUB01_006 {
        part SegmentActuator1: SegmentActuator;
        part MetrologySensor1: MetrologySensor;
        part LocalController1: LocalController;

        // Connecting parts with ports
        connector SegmentActuator1.actuatorCommandPort -> LocalController1.controlCommandPort;
        connector SegmentActuator1.actuatorStatusPort -> LocalController1.controlStatusPort;
        connector MetrologySensor1.metrologyDataPort -> LocalController1.controlCommandPort;
        connector MetrologySensor1.metrologySyncPort -> LocalController1.controlStatusPort;
        connector LocalController1.controlCommandPort -> SegmentActuator1.actuatorCommandPort;

        // Defining Timing Sync for Metrology and Actuation
        connector MetrologySensor1.metrologySyncPort -> TimeSyncSignal;
        connector LocalController1.controlStatusPort -> TimeSyncSignal;

        // Timing Relationships (Metrology Sampling / Actuation Cadence)
        part def TimeSyncSignal {
            in timeSync: TimeSyncSignal;
            out timeSync: TimeSyncSignal;
        }
    }
}
