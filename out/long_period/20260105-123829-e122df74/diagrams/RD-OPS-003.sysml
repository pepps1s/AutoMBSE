package FaultResponseAndSafeStateRequirements {

    import ScalarValues::*;
    import Quantities::*;

    requirement def FaultResponseRequirement {
        doc /* This requirement defines the operational requirements for fault detection, annunciation, safe-state entry, recovery, and logging across fault classes (motion, thermal, power, network, sensor failure) for the Thirty Meter Telescope (TMT). */

        attribute faultClass : String;  // Types of faults: motion, thermal, power, network, sensor
        attribute detectionMethod : String;  // How faults are detected: sensor, threshold crossing, etc.
        attribute annunciationMethod : String;  // How faults are announced: alarms, visual indicators, etc.
        attribute safeStateAction : String;  // Action to be taken when entering a safe state
        attribute recoveryMethod : String;  // Method of recovery from faults
        attribute loggingRequirements : String;  // Minimum telemetry and logging capture requirements during faults
        attribute operatorNotification : String;  // Required operator notifications during fault events

        require constraint{ 
            faultClass in ["motion", "thermal", "power", "network", "sensor"];
            detectionMethod != "" and annunciationMethod != "";
            safeStateAction != "" and recoveryMethod != "";
            loggingRequirements != "";
        }
    }

    requirement def MotionFaultResponseReq {
        doc /* Requirements for fault detection, annunciation, safe-state actions, recovery, and logging for motion-related faults. */
        
        attribute detectionMethod : String;
        attribute annunciationMethod : String;
        attribute safeStateAction : String;
        attribute recoveryMethod : String;
        attribute loggingRequirements : String;
        attribute operatorNotification : String;

        require constraint{
            detectionMethod == "motion sensor";
            annunciationMethod == "visual and audio alarms";
            safeStateAction == "halt motion and secure telescope";
            recoveryMethod == "manual reset or automated recovery";
            loggingRequirements == "motion sensor data, system logs during fault";
            operatorNotification == "alert via system interface and alarms";
        }
    }

    requirement def ThermalFaultResponseReq {
        doc /* Requirements for fault detection, annunciation, safe-state actions, recovery, and logging for thermal-related faults. */
        
        attribute detectionMethod : String;
        attribute annunciationMethod : String;
        attribute safeStateAction : String;
        attribute recoveryMethod : String;
        attribute loggingRequirements : String;
        attribute operatorNotification : String;

        require constraint{
            detectionMethod == "temperature sensor";
            annunciationMethod == "audible alarms";
            safeStateAction == "shutdown thermal subsystem to prevent damage";
            recoveryMethod == "automated cooldown or manual intervention";
            loggingRequirements == "temperature readings, thermal subsystem logs";
            operatorNotification == "alert via system interface and alarms";
        }
    }

    requirement def PowerFaultResponseReq {
        doc /* Requirements for fault detection, annunciation, safe-state actions, recovery, and logging for power-related faults. */
        
        attribute detectionMethod : String;
        attribute annunciationMethod : String;
        attribute safeStateAction : String;
        attribute recoveryMethod : String;
        attribute loggingRequirements : String;
        attribute operatorNotification : String;

        require constraint{
            detectionMethod == "power failure sensor";
            annunciationMethod == "audible and visual alarms";
            safeStateAction == "switch to backup power or shut down safely";
            recoveryMethod == "automatic switch to backup or manual recovery";
            loggingRequirements == "power system logs, battery status, energy consumption";
            operatorNotification == "alert via system interface and alarms";
        }
    }

    requirement def NetworkFaultResponseReq {
        doc /* Requirements for fault detection, annunciation, safe-state actions, recovery, and logging for network-related faults. */
        
        attribute detectionMethod : String;
        attribute annunciationMethod : String;
        attribute safeStateAction : String;
        attribute recoveryMethod : String;
        attribute loggingRequirements : String;
        attribute operatorNotification : String;

        require constraint{
            detectionMethod == "network status monitor";
            annunciationMethod == "network failure indicator";
            safeStateAction == "enter local control mode, isolate fault";
            recoveryMethod == "automated network reconnection or manual intervention";
            loggingRequirements == "network status logs, error logs";
            operatorNotification == "alert via system interface and alarms";
        }
    }

    requirement def SensorFailureFaultResponseReq {
        doc /* Requirements for fault detection, annunciation, safe-state actions, recovery, and logging for sensor-related faults. */
        
        attribute detectionMethod : String;
        attribute annunciationMethod : String;
        attribute safeStateAction : String;
        attribute recoveryMethod : String;
        attribute loggingRequirements : String;
        attribute operatorNotification : String;

        require constraint{
            detectionMethod == "sensor failure detection";
            annunciationMethod == "sensor fault alarms";
            safeStateAction == "switch to backup sensor or deactivate faulty sensor";
            recoveryMethod == "sensor replacement or recalibration";
            loggingRequirements == "sensor error logs, status reports";
            operatorNotification == "alert via system interface and alarms";
        }
    }

    requirement def SafeStateObjective {
        doc /* The system must transition to a safe state in response to faults and notify the operator accordingly. */
        
        attribute safeStateAction : String;
        attribute recoveryMethod : String;
        attribute operatorNotification : String;

        require constraint{
            safeStateAction != "" and recoveryMethod != "";
            operatorNotification != "";
        }
    }

    requirement def LoggingTelemetryCapture {
        doc /* Minimum logging and telemetry capture requirements during faults. */
        
        attribute faultClass : String;
        attribute loggingRequirements : String;

        require constraint{
            faultClass in ["motion", "thermal", "power", "network", "sensor"];
            loggingRequirements != "";
        }
    }
}
