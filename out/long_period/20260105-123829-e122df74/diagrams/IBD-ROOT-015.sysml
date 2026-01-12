package IBD_ROOT_015 {
    package SafetyInhibitDistribution {
        // Part Definitions
        part def SafetySubsystem {
            port inhibitOut: InhibitPort;
            port estopLoop: EStopPort;
            attribute tripActive: Boolean;
            attribute estopSys: EStopSystem;
            attribute sensors: SensorArray;
        }

        part def MotionCapableSubsystem {
            port inhibitIn: InhibitPort;
            port resetAuth: ResetAuthorizationPort;
            attribute motionControl: MotionControlSystem;
            attribute tripCondition: Boolean;
        }

        // Port Definitions
        port def InhibitPort {
            out inhibitSignal: InhibitSignal;
        }

        port def EStopPort {
            in estopSignal: EStopSignal;
        }

        port def ResetAuthorizationPort {
            in resetSignal: ResetSignal;
        }

        // Flow Definitions
        part def InhibitSignal;
        part def EStopSignal;
        part def ResetSignal;

        part def EStopSystem;
        part def MotionControlSystem;
        part def SensorArray;

        // Interaction Flow Example
        part def SafetyInhibitFlow {
            part safetySubsystem: SafetySubsystem;
            part motionSubsystem: MotionCapableSubsystem;
            
            // Connection between SafetySubsystem and MotionCapableSubsystem
            safetySubsystem.inhibitOut -> motionSubsystem.inhibitIn;
            safetySubsystem.estopLoop -> motionSubsystem.resetAuth; // Reset authorization

            // Including an example of latched trip condition and reset authorization path
            safetySubsystem.tripActive -> motionSubsystem.tripCondition;
            motionSubsystem.resetAuth -> safetySubsystem.estopSys;
        }
    }
}
