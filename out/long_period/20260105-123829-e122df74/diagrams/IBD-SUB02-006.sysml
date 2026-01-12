package IBD_SUB02_VibrationSensingActuation {

    package InternalWiring {

        part def VibrationSensor {
            port VibrationSensePort: VibrationSensorPort;
            attribute sensorSensitivity: SensitivityLevel;
        }

        part def VibrationActuator {
            port VibrationActuatePort: VibrationActuatorPort;
            attribute actuatorType: ActuatorType;
        }

        part def VibrationControlSystem {
            port telemetryOutPort: TelemetryPort;
            port telemetryInPort: TelemetryPort;
            attribute telemetryFrequency: Frequency;
        }

        // Port Definitions
        port def VibrationSensorPort {
            out VibrationSignal: VibrationMetrics;
        }

        port def VibrationActuatorPort {
            in VibrationCommand: VibrationSignal;
            out VibrationResponse: VibrationSignal;
        }

        port def TelemetryPort {
            in VibrationData: VibrationMetrics;
            out ControlSignal: ControlCommand;
        }

        // Flow Definitions
        part def VibrationSignal;
        part def VibrationMetrics;
        part def ControlCommand;

        // Attribute Definitions
        attribute def SensitivityLevel;
        attribute def ActuatorType;
        attribute def Frequency;

        // System-level connections
        part def VibrationSensingActuation {
            part vibrationSensor: VibrationSensor;
            part vibrationActuator: VibrationActuator;
            part vibrationControl: VibrationControlSystem;
            port controlPort: TelemetryPort;
            port telemetryPort: TelemetryPort;

            // Connectors
            connector def SensorToControl {
                end1: VibrationSensor.VibrationSensePort -> VibrationControlSystem.telemetryInPort;
                end2: VibrationControlSystem.telemetryOutPort -> VibrationActuator.VibrationActuatePort;
            }
        }

    }

    // Dependency connections to other subsystems
    dependency BDD_SUB02_003_MountStructure;
    dependency RD_NF_002_Performance_NFR_Pointing_Tracking_Stability_Jitter;
    dependency IBD_ROOT_011_TelescopeOpticalSystem;

}
