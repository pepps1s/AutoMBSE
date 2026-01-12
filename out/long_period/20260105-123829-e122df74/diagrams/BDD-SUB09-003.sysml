package BDD_SUB09_Safety {
    
    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    % Define parts and subsystems for the Safety subsystem %
    part def SafetyInterlocks {
        description = "Handles safety interlocks to prevent unsafe operations.";
        part emergencyStop :> SafetyComponents::EmergencyStop;
        part operationalConstraints :> SafetyComponents::OperationalConstraints;
        part interlockPorts[2] : Port {
            direction = inout;
            type = Boolean;
            description = "Ports for safety interlock signals.";
        }
    }
    
    part def EmergencyStop {
        description = "Manages the emergency stop functionality.";
        part button :> SafetyComponents::Button {
            description = "Physical emergency stop button.";
        }
        part signal :> SafetyComponents::Signal {
            direction = out;
            description = "Signal output for emergency stop action.";
        }
    }
    
    part def OperationalConstraints {
        description = "Defines and enforces operational constraints for the system.";
        part temperatureLimit :> SafetyComponents::TemperatureLimit {
            value = 40 [C];
            description = "Maximum allowable temperature.";
        }
        part vibrationLimit :> SafetyComponents::VibrationLimit {
            value = 0.1 [m/s^2];
            description = "Maximum allowable vibration threshold.";
        }
    }
    
    % Define the interfaces/ports for the Safety subsystem %
    interface def SafetyInterface {
        port emergencyStopPort : Boolean {
            direction = in;
            description = "Emergency stop signal input.";
        }
        port operationalLimitPort : Boolean {
            direction = out;
            description = "Operational constraints alert output.";
        }
    }
    
    % Define connections between parts and interfaces %
    part TMT_SafetySystem {
        part safetySubsystem : SafetyInterlocks;
        part emergencyStopSubsystem : EmergencyStop;
        part operationalConstraintsSubsystem : OperationalConstraints;
        interface safetyInterface :> SafetyInterface;
    }

    % Define dependencies between parts %
    dependency safetyDependency {
        from = TMT_SafetySystem;
        to = RD_F_001;
        description = "Safety subsystem depends on RD_F-001 for defining safety functions.";
    }

    dependency safetyOperationalConstraint {
        from = OperationalConstraints;
        to = RD_NF_001;
        description = "Operational constraints from RD-NF-001 to ensure safety compliance.";
    }
}
