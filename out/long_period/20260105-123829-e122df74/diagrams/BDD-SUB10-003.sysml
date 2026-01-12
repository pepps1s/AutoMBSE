package BDD_SUB10_003_OperationsMaintenance {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Define parts for Operations and Maintenance subsystem %
    part def TelescopeOpticalSystem;
    part def TelescopeStructureMount;
    part def EnclosureDomeSiteInfrastructure;
    part def AlignmentPhasingWavefrontSensing;
    part def AdaptiveOpticsSubsystem;
    part def ScienceInstruments;
    part def ObservatoryControlSystem;
    part def DataSystem;
    part def SafetyCompliance;
    part def OperationsMaintenance;

    % Define attributes for Operations/Maintenance related parts %
    attribute def StartupShutdownMode {
        mode : String;
        description : String;
    }

    attribute def FaultHandlingMode {
        mode : String;
        description : String;
    }

    attribute def MaintenanceMode {
        mode : String;
        description : String;
    }

    % Define parts with attributes for each mode %
    part OperationsMaintenance {
        part startupShutdownMode : StartupShutdownMode {
            mode = "Startup";
            description = "System startup sequence including power-up and self-check.";
        }
        part faultHandlingMode : FaultHandlingMode {
            mode = "Fault Detection";
            description = "Handles fault detection, safe-state transitions, and recovery.";
        }
        part maintenanceMode : MaintenanceMode {
            mode = "Routine Maintenance";
            description = "Routine maintenance procedures including calibration and parts replacement.";
        }
    }

    % Define key interfaces and connections %
    interface def StartupInterface {
        direction : inout;
        type : String;
        connectionType : "power, data";
    }

    interface def FaultHandlingInterface {
        direction : inout;
        type : String;
        connectionType : "diagnostics, recovery";
    }

    interface def MaintenanceInterface {
        direction : inout;
        type : String;
        connectionType : "mechanical, electrical";
    }

    % Define port connections for operations and maintenance modes %
    part OperationsMaintenance {
        port startupPort : StartupInterface;
        port faultHandlingPort : FaultHandlingInterface;
        port maintenancePort : MaintenanceInterface;
    }

    % Define system connections %
    OperationsMaintenance::startupPort --> TelescopeOpticalSystem::startupPort;
    OperationsMaintenance::faultHandlingPort --> TelescopeStructureMount::faultPort;
    OperationsMaintenance::maintenancePort --> EnclosureDomeSiteInfrastructure::maintenancePort;

    % Define the dependency connections to other subsystems %
    dependency TelescopeOpticalSystem;
    dependency TelescopeStructureMount;
    dependency EnclosureDomeSiteInfrastructure;
    dependency AlignmentPhasingWavefrontSensing;
    dependency AdaptiveOpticsSubsystem;
    dependency ScienceInstruments;
    dependency ObservatoryControlSystem;
    dependency DataSystem;
    dependency SafetyCompliance;

}
