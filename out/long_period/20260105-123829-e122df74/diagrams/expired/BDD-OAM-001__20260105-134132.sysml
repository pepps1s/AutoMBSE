package BDD_OAM_001 {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    part def OAndMSubsystem;
    part def MaintenanceTools;
    part def SegmentHandling;
    part def CalibrationTooling;
    part def DiagnosticAccess;
    
    // Defining attributes for the system
    attribute def OAndMGoal {
        maintenanceToolsInterfaces : Boolean;
        segmentHandlingInterfaces : Boolean;
        calibrationToolingDefined : Boolean;
        diagnosticAccessDefined : Boolean;
    }
    
    // Structural elements and their relationships
    part def OAndMSubsystem {
        part maintenanceTools : MaintenanceTools;
        part segmentHandling : SegmentHandling;
        part calibrationTooling : CalibrationTooling;
        part diagnosticAccess : DiagnosticAccess;
    }
    
    part def MaintenanceTools {
        part tools : CustomToolsType {
            description = "Tools for maintenance purposes, including cleaning, adjustment, and repair.";
        }
        port input : CustomInputType {
            direction = inout;
        }
    }
    
    part def SegmentHandling {
        part handlingUnit : HandlingUnitType {
            description = "Handles the transport, replacement, and adjustment of telescope segments.";
        }
        port connection : CustomConnectionType {
            direction = out;
        }
    }
    
    part def CalibrationTooling {
        part calibrationUnit : CalibrationUnitType {
            description = "Tools for calibrating the telescope subsystems.";
        }
        port diagnosticConnection : CustomConnectionType {
            direction = inout;
        }
    }
    
    part def DiagnosticAccess {
        part diagnosticUnit : DiagnosticUnitType {
            description = "Provides access points for diagnostic tools and monitoring.";
        }
        port diagnosticPort : CustomPortType {
            direction = in;
        }
    }
    
    // System attributes and operational description
    part OAndMSubsystem {
        attribute goal : OAndMGoal {
            maintenanceToolsInterfaces = true;
            segmentHandlingInterfaces = true;
            calibrationToolingDefined = true;
            diagnosticAccessDefined = true;
        }
    }
    
    // Dependencies
    dependency BDD_SYS_003;
    
    // Acceptance checks
    check maintenanceToolsInterfaces == true;
    check segmentHandlingInterfaces == true;
    check calibrationToolingDefined == true;
    check diagnosticAccessDefined == true;
}
