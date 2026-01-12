package RD_F_002_ObservationLifecycleFunctionalDecomposition {
    import ScalarValues::*;
    import Quantities::*;
    
    requirement def StartupReq {
        doc /* The system shall initialize all subsystems, perform self-checks, and confirm readiness. */
        
        attribute startupTime : Real;
        attribute subsystemStatus : Boolean;
        
        require constraint {
            startupTime <= 15.0
            subsystemStatus == true
        }
    }
    
    requirement def SlewReq {
        doc /* The system shall slew to the target position with an accuracy of 0.1 arcseconds. */
        
        attribute slewTime : Real;
        attribute accuracy : Real;
        
        require constraint {
            slewTime <= 10.0
            accuracy <= 0.1
        }
    }
    
    requirement def AcquisitionReq {
        doc /* The system shall acquire the target within 30 seconds and acquire a guide star with an accuracy of 0.2 arcseconds. */
        
        attribute acquisitionTime : Real;
        attribute guideStarAccuracy : Real;
        
        require constraint {
            acquisitionTime <= 30.0
            guideStarAccuracy <= 0.2
        }
    }
    
    requirement def AlignmentPhasingReq {
        doc /* The system shall align and phase the segmented mirror to within 0.05 arcseconds. */
        
        attribute alignmentTime : Real;
        attribute alignmentAccuracy : Real;
        
        require constraint {
            alignmentTime <= 60.0
            alignmentAccuracy <= 0.05
        }
    }
    
    requirement def ExposureLoopReq {
        doc /* The system shall execute the exposure loop with a cadence of 2 seconds, capturing image data with the specified quality. */
        
        attribute exposureTime : Real;
        attribute exposureQuality : String;
        
        require constraint {
            exposureTime == 2.0
            exposureQuality == "High"
        }
    }
    
    requirement def CalibrationReq {
        doc /* The system shall perform calibration routines including flat field, dark field, and arc, with the results meeting the required standards. */
        
        attribute calibrationType : String;
        attribute calibrationTime : Real;
        
        require constraint {
            calibrationType in {"Flat", "Dark", "Arc"}
            calibrationTime <= 30.0
        }
    }
    
    requirement def ShutdownReq {
        doc /* The system shall safely shutdown, ensuring all subsystems are powered down and no faults are present. */
        
        attribute shutdownTime : Real;
        attribute shutdownStatus : Boolean;
        
        require constraint {
            shutdownTime <= 10.0
            shutdownStatus == true
        }
    }

    requirement def AlternatePathWeatherAbortReq {
        doc /* In case of inclement weather, the system shall abort the observation and transition to a safe state. */
        
        attribute weatherCondition : String;
        attribute abortStatus : Boolean;
        
        require constraint {
            weatherCondition == "Inclement"
            abortStatus == true
        }
    }

    requirement def AlternatePathFaultAbortReq {
        doc /* In case of a system fault, the system shall abort the observation and transition to a safe state. */
        
        attribute faultCondition : String;
        attribute abortStatus : Boolean;
        
        require constraint {
            faultCondition == "SystemFault"
            abortStatus == true
        }
    }

    requirement def AlternatePathInstrumentUnavailableReq {
        doc /* If an instrument becomes unavailable, the system shall abort the observation and notify the operator. */
        
        attribute instrumentStatus : Boolean;
        attribute abortStatus : Boolean;
        
        require constraint {
            instrumentStatus == false
            abortStatus == true
        }
    }

    requirement def FunctionalRequirementLinkReq {
        doc /* Each lifecycle step shall be mapped to at least one functional requirement placeholder. */
        
        attribute lifecycleStep : String;
        attribute functionalRequirement : String;
        
        require constraint {
            lifecycleStep in {"Startup", "Slew", "Acquisition", "Alignment", "Exposure", "Calibration", "Shutdown"}
            functionalRequirement != ""
        }
    }
}
