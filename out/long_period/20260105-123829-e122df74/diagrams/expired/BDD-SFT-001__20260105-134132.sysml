package ObservatoryControlSystem_StructuralDecomposition {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    abstract part def SoftwareSubsystem;
    abstract part def SequencingService;
    abstract part def ConfigurationService;
    abstract part def TelemetryService;
    abstract part def AlarmsService;
    abstract part def OperatorUIService;
    abstract part def AuthorizationService;
    
    part def ObservatoryControlSystem {
        part sequencing : SequencingService;
        part configuration : ConfigurationService;
        part telemetry : TelemetryService;
        part alarms : AlarmsService;
        part operatorUI : OperatorUIService;
        part authorization : AuthorizationService;
    }
    
    part def SequencingService : SoftwareSubsystem {
        description = "Responsible for managing and executing observation sequences.";
    }
    
    part def ConfigurationService : SoftwareSubsystem {
        description = "Handles the configuration management for all subsystems.";
    }
    
    part def TelemetryService : SoftwareSubsystem {
        description = "Collects and transmits telemetry data for system monitoring.";
    }
    
    part def AlarmsService : SoftwareSubsystem {
        description = "Monitors the system for faults and triggers alarm conditions.";
    }
    
    part def OperatorUIService : SoftwareSubsystem {
        description = "Provides the interface for operators to control the system and monitor status.";
    }
    
    part def AuthorizationService : SoftwareSubsystem {
        description = "Manages access control and security for the system.";
    }

    dependency BDD-SYS-003;
    
    acceptance_checks {
        "Sequencing, configuration, telemetry, and alarms services are defined.";
        "Operator UI and authorization services are included.";
    }
    
}
