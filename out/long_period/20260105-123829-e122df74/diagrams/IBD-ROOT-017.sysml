package IBD_ROOT_DegradedModeSignaling {
    package TMT_DegradedModeSignals {
        
        // Part Definitions
        part def DegradedMode {
            port out degradedSignalPort: DegradedSignal;
            port in overrideSafetyPort: SafetyOverride;
            attribute modeIdentifier: ModeID;
            attribute modeStatus: ModeStatus;
            attribute affectedSubsystems: SubsystemList;
        }

        // Port Definitions
        port def DegradedSignal {
            out signalLevel: SignalStrength;
            in signalInput: SignalInput;
        }

        port def SafetyOverride {
            in overrideCommand: OverrideCommand;
            out safetyStatus: SafetyStatus;
        }

        // Attribute Definitions
        attribute def ModeID {
            type String;
            description "Identifier for the degraded mode (e.g., LimitedTracking, LimitedAlignment, PartialInstrument)";
        }

        attribute def ModeStatus {
            type String;
            description "Current status of the degraded mode (e.g., Active, Inactive)";
        }

        attribute def SubsystemList {
            type List<String>;
            description "List of subsystems affected by the degraded mode";
        }

        attribute def SignalStrength {
            type Real;
            description "Signal strength of the degraded mode signal";
        }

        attribute def SignalInput {
            type Real;
            description "Input signal associated with the degraded mode";
        }

        attribute def OverrideCommand {
            type String;
            description "Command issued to override safety restrictions";
        }

        attribute def SafetyStatus {
            type String;
            description "Status of safety override (e.g., Engaged, Disengaged)";
        }

        // Mode Definitions
        part def LimitedTrackingMode: DegradedMode {
            modeIdentifier: "LimitedTracking";
            modeStatus: "Inactive";
            affectedSubsystems: ["Mount", "APS"];
        }

        part def LimitedAlignmentMode: DegradedMode {
            modeIdentifier: "LimitedAlignment";
            modeStatus: "Inactive";
            affectedSubsystems: ["Mount", "APS", "Instruments"];
        }

        part def PartialInstrumentMode: DegradedMode {
            modeIdentifier: "PartialInstrument";
            modeStatus: "Inactive";
            affectedSubsystems: ["Instruments"];
        }

        // Safety Override Path
        part def SafetyOverridePath {
            port safetyIn: SafetyOverride;
            port safetyOut: SafetyStatus;
            part safetyOverrideCmd: SafetyOverride;
        }

    }

    // Dependencies
    import RD_OPS_001;
    import RD_OPS_003;
    import BDD_SUB10_001;
    import BDD_SUB09_001;
    import IBD_ROOT_011;
    import IBD_ROOT_012;
}
