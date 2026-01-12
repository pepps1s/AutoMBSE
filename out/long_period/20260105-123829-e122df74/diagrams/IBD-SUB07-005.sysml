package OCS_Telemetry_Aggregation_Alarm_Correlation {

    package InternalWiring {

        // Part Definitions
        part def TelemetryAggregator {
            port telemetryInputPort: TelemetrySignal;
            port telemetryOutputPort: TelemetrySignal;
            attribute correlationID: String;
        }

        part def AlarmCorrelator {
            port alarmInputPort: AlarmSignal;
            port alarmOutputPort: AlarmSignal;
            attribute traceCorrelationId: String;
        }

        part def OperatorDisplay {
            port displayOutputPort: TelemetrySignal;
        }

        part def LogOutput {
            port logOutputPort: AlarmSignal;
        }

        // Part Instances
        part TelemetryAggregatorInst : TelemetryAggregator {
            correlationID = "TelemetryCorrelation";
        }

        part AlarmCorrelatorInst : AlarmCorrelator {
            traceCorrelationId = "TraceCorrelation";
        }

        // Port Definitions
        port def TelemetrySignal {
            in telemetryIn: TelemetrySignal;
            out telemetryOut: TelemetrySignal;
        }

        port def AlarmSignal {
            in alarmIn: AlarmSignal;
            out alarmOut: AlarmSignal;
        }

        // Flow Definitions
        part def TelemetrySignal;
        part def AlarmSignal;
        part def AlarmLevel;

        // System Connections
        connection def TelemetryFlow {
            source: SafetySubsystem.telemetryOutPort;
            destination: TelemetryAggregatorInst.telemetryInputPort;
        }

        connection def AlarmFlow {
            source: SafetySubsystem.alarmOutPort;
            destination: AlarmCorrelatorInst.alarmInputPort;
        }

        connection def DisplayFlow {
            source: AlarmCorrelatorInst.alarmOutputPort;
            destination: OperatorDisplay.displayOutputPort;
        }

        connection def LogFlow {
            source: AlarmCorrelatorInst.alarmOutputPort;
            destination: LogOutput.logOutputPort;
        }

    }

}
