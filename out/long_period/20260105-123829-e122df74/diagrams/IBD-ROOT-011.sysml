package UnifiedTelemetryBackbone {

    package TelemetryFlow {
        
        // Part Definitions
        part def Subsystem {
            port telemetryOutPort: TelemetryOut;
            port telemetryInPort: TelemetryIn;
            attribute telemetryCadence: TelemetryCadence;
        }

        part def OCS {
            port telemetryInPort: TelemetryIn;
            attribute telemetryCadence: TelemetryCadence;
            port alarmOutPort: AlarmOut;
        }

        part def DataSystem {
            port telemetryInPort: TelemetryIn;
            attribute telemetryCadence: TelemetryCadence;
            port alarmOutPort: AlarmOut;
        }

        part def AlarmSystem {
            port alarmInPort: AlarmIn;
            attribute alarmLevel: AlarmLevel;
        }

        // Port Definitions
        port def TelemetryOut {
            out telemetrySignal: TelemetrySignal;
        }

        port def TelemetryIn {
            in telemetrySignal: TelemetrySignal;
        }

        port def AlarmOut {
            out alarmSignal: AlarmSignal;
        }

        port def AlarmIn {
            in alarmSignal: AlarmSignal;
        }

        // Flow Definitions
        part def TelemetrySignal;
        part def AlarmSignal;

        // Attribute Definitions
        attribute def TelemetryCadence;
        attribute def AlarmLevel;

    }

    // Root System Integration
    part TelemetryBackbone {
        part telescopeOpticalSystem: Subsystem;
        part mountStructure: Subsystem;
        part enclosureSiteInfrastructure: Subsystem;
        part alignmentPhasingSystem: Subsystem;
        part adaptiveOpticsSubsystem: Subsystem;
        part instruments: Subsystem;
        part ocsSystem: OCS;
        part dataSystem: DataSystem;
        part alarmSystem: AlarmSystem;

        // Connecting parts to the telemetry flow
        telescopeOpticalSystem.telemetryOutPort --> ocsSystem.telemetryInPort;
        mountStructure.telemetryOutPort --> ocsSystem.telemetryInPort;
        enclosureSiteInfrastructure.telemetryOutPort --> ocsSystem.telemetryInPort;
        alignmentPhasingSystem.telemetryOutPort --> ocsSystem.telemetryInPort;
        adaptiveOpticsSubsystem.telemetryOutPort --> ocsSystem.telemetryInPort;
        instruments.telemetryOutPort --> dataSystem.telemetryInPort;

        ocsSystem.alarmOutPort --> alarmSystem.alarmInPort;
        dataSystem.alarmOutPort --> alarmSystem.alarmInPort;

        // Telemetry cadence definition for receivers
        ocsSystem.telemetryCadence = 10;
        dataSystem.telemetryCadence = 10;
    }

}
