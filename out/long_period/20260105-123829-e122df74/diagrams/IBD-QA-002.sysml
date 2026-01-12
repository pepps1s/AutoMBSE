package IBD_QA_CrossSubsystemPortNamingHarmonization {
    
    // Defining canonical port names
    port def CommandPort {
        out commandSignal : CommandSignal;
        in commandAcknowledgment : CommandAcknowledgment;
    }
    
    port def TelemetryPort {
        out telemetrySignal : TelemetrySignal;
        in telemetryData : TelemetryData;
    }
    
    port def TimeSyncPort {
        in timeReference : TimeReference;
        out timeSyncSignal : TimeSyncSignal;
    }
    
    port def OpticalBeamPort {
        out opticalBeam : OpticalBeam;
        in opticalFeedback : OpticalFeedback;
    }
    
    port def SafetyInhibitPort {
        out safetyInhibitSignal : SafetyInhibitSignal;
        in safetyStatus : SafetyStatus;
    }

    // Example subsystems and their ports
    part def TelescopeOpticalSystem {
        port opticalBeamPort : OpticalBeamPort;
        port timeSyncPort : TimeSyncPort;
    }
    
    part def MountStructure {
        port commandPort : CommandPort;
        port telemetryPort : TelemetryPort;
    }
    
    part def AdaptiveOpticsSubsystem {
        port opticalBeamPort : OpticalBeamPort;
        port telemetryPort : TelemetryPort;
        port safetyInhibitPort : SafetyInhibitPort;
    }
    
    part def ObservatoryControlSystem {
        port commandPort : CommandPort;
        port telemetryPort : TelemetryPort;
        port timeSyncPort : TimeSyncPort;
        port safetyInhibitPort : SafetyInhibitPort;
    }

    // Constraint ensuring canonical port naming usage
    constraint HarmonizedPortNaming {
        // All ports must use canonical names defined above
        TelescopeOpticalSystem::opticalBeamPort == OpticalBeamPort;
        MountStructure::commandPort == CommandPort;
        AdaptiveOpticsSubsystem::safetyInhibitPort == SafetyInhibitPort;
        ObservatoryControlSystem::timeSyncPort == TimeSyncPort;
    }

    // Acceptance check examples
    acceptance_check {
        // Ensure canonical port names are used across subsystems
        includesCanonicalPortNames : TelescopeOpticalSystem::opticalBeamPort == OpticalBeamPort;
        includesCanonicalPortNames : AdaptiveOpticsSubsystem::safetyInhibitPort == SafetyInhibitPort;
    }

    // Dependency references (IBD-ROOT dependencies)
    dependency IBD-ROOT-009;
    dependency IBD-ROOT-011;
    dependency IBD-ROOT-012;
    dependency IBD-ROOT-013;
}
