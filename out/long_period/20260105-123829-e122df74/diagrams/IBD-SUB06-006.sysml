package IBD_SUB06_006_InstrumentIsolationAndInterlockWiring {
    package Instrumentation {
        part def Instrument {
            port isolationControlPort: IsolationControl;
            port shutdownStatusPort: ShutdownStatus;
            port safetyInterlockPort: SafetyInterlock;
        }

        part def SafetySubsystem {
            port isolationControlPort: IsolationControl;
            port shutdownStatusPort: ShutdownStatus;
            port safetyInterlockPort: SafetyInterlock;
        }

        part def OpsSubsystem {
            port shutdownControlPort: ShutdownControl;
            port maintenanceStatusPort: MaintenanceStatus;
            port safetyInterlockPort: SafetyInterlock;
        }

        part def IsolationControl {
            out status: StatusFeedback;
        }

        part def ShutdownStatus {
            in status: ShutdownFeedback;
        }

        part def SafetyInterlock {
            in signal: SafetySignal;
            out status: SafetyStatus;
        }

        part def ShutdownControl {
            out command: ShutdownCommand;
        }

        part def MaintenanceStatus {
            in status: MaintenanceFeedback;
        }

        part def SafetySignal;
        part def StatusFeedback;
        part def ShutdownFeedback;
        part def SafetyStatus;
        part def ShutdownCommand;
        part def MaintenanceFeedback;

        // Interfaces for Wiring and Integration
        port def IsolationFeedbackPort {
            out status: StatusFeedback;
            in command: ShutdownCommand;
        }

        port def SafetyInterlockPort {
            in signal: SafetySignal;
            out status: SafetyStatus;
        }

        part Instrument1 : Instrument {
            port isolationControlPort : IsolationControl;
            port shutdownStatusPort : ShutdownStatus;
            port safetyInterlockPort : SafetyInterlock;
        }

        part SafetySubsystem1 : SafetySubsystem {
            port isolationControlPort : IsolationControl;
            port shutdownStatusPort : ShutdownStatus;
            port safetyInterlockPort : SafetyInterlock;
        }

        part OpsSubsystem1 : OpsSubsystem {
            port shutdownControlPort : ShutdownControl;
            port maintenanceStatusPort : MaintenanceStatus;
            port safetyInterlockPort : SafetyInterlock;
        }

        // Connections for Isolation, Shutdown, and Interlock Feedback
        connection Instrument1.isolationControlPort -> SafetySubsystem1.isolationControlPort;
        connection SafetySubsystem1.shutdownStatusPort -> Instrument1.shutdownStatusPort;
        connection SafetySubsystem1.safetyInterlockPort -> Instrument1.safetyInterlockPort;
        connection Instrument1.safetyInterlockPort -> OpsSubsystem1.safetyInterlockPort;
        connection SafetySubsystem1.safetyInterlockPort -> OpsSubsystem1.safetyInterlockPort;
        connection OpsSubsystem1.shutdownControlPort -> SafetySubsystem1.shutdownStatusPort;
        connection OpsSubsystem1.maintenanceStatusPort -> SafetySubsystem1.shutdownStatusPort;
        connection SafetySubsystem1.safetyInterlockPort -> SafetySubsystem1.safetyInterlockPort;
    }
}
