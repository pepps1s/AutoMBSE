package IBD_SUB10_004_OperationsMaintenanceStateMachine {

    // Define parts
    part def ModeCommandSource {
        port in ModeCommand: ModeCommandPort;
    }

    part def ReadinessAggregator {
        port in ReadinessStatus: ReadinessPort;
        port out AggregatedStatus: AggregatedStatusPort;
    }

    part def ModeStatusOutputs {
        port out ModeStatus: ModeStatusPort;
    }

    // Mode and State identifiers based on RD-OPS-001
    part def ModeIdentifier {
        attribute modeName: ModeName;
        attribute modeState: ModeState;
    }

    part def TelemetrySystem {
        port out TelemetryOut: TelemetryPort;
    }

    part def AlarmSystem {
        port out AlarmOut: AlarmPort;
    }

    // Define ports
    port def ModeCommandPort {
        in ModeCommand: ModeCommand;
    }

    port def ReadinessPort {
        in ReadinessStatus: ReadinessStatus;
    }

    port def AggregatedStatusPort {
        out AggregatedStatus: AggregatedStatus;
    }

    port def ModeStatusPort {
        out ModeStatus: ModeStatus;
    }

    port def TelemetryPort {
        out TelemetrySignal: TelemetrySignal;
    }

    port def AlarmPort {
        out AlarmSignal: AlarmSignal;
    }

    // Define data types (Mode, Readiness, Telemetry, etc.)
    part def ModeCommand;
    part def ReadinessStatus;
    part def AggregatedStatus;
    part def ModeStatus;
    part def TelemetrySignal;
    part def AlarmSignal;

    // Instantiating system parts and connections
    part ModeCommandSourceInstance: ModeCommandSource {
        port in ModeCommandPort: ModeCommand;
    }

    part ReadinessAggregatorInstance: ReadinessAggregator {
        port in ReadinessPort: ReadinessStatus;
        port out AggregatedStatusPort: AggregatedStatus;
    }

    part ModeStatusOutputsInstance: ModeStatusOutputs {
        port out ModeStatusPort: ModeStatus;
    }

    part TelemetrySystemInstance: TelemetrySystem {
        port out TelemetryOut: TelemetrySignal;
    }

    part AlarmSystemInstance: AlarmSystem {
        port out AlarmOut: AlarmSignal;
    }

    // Define internal wiring for operations/maintenance control
    connector ModeCommandToReadiness {
        ModeCommandPort from ModeCommandSourceInstance;
        ReadinessPort to ReadinessAggregatorInstance;
    }

    connector ReadinessToStatusAggregation {
        ReadinessPort from ReadinessAggregatorInstance;
        AggregatedStatusPort to ModeStatusOutputsInstance;
    }

    connector ModeStatusToTelemetry {
        ModeStatusPort from ModeStatusOutputsInstance;
        TelemetryPort to TelemetrySystemInstance;
    }

    connector ModeStatusToAlarm {
        ModeStatusPort from ModeStatusOutputsInstance;
        AlarmPort to AlarmSystemInstance;
    }

}
