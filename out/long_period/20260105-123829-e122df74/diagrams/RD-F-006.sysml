package ObservatoryControlSequencingRequirements {

    import ScalarValues::*;
    import Quantities::*;

    requirement def OCSFunctionalRequirements {
        doc /* Define functional requirements for sequencing, configuration, telemetry, alarms, and operator interactions. */

        attribute sequencingControl : Boolean;
        attribute configurationControl : Boolean;
        attribute telemetrySystem : Boolean;
        attribute alarmsSystem : Boolean;
        attribute operatorInteraction : Boolean;

        // Requirements for sequencing, state management, and automation
        requirement sequencingReq {
            doc /* The system shall provide sequencing control for observations. */
            attribute sequencingTime : Real;
            attribute sequencingAccuracy : Real;
            require constraint{ sequencingTime <= 10.0 }  // Example constraint: sequencing time should be ≤ 10 seconds
        }

        requirement stateManagementReq {
            doc /* The system shall manage states between observation tasks. */
            attribute stateChangeTime : Real;
            attribute stateTransitionAccuracy : Real;
            require constraint{ stateChangeTime <= 5.0 }  // Example constraint: state change time should be ≤ 5 seconds
        }

        requirement commandAuthorizationReq {
            doc /* The system shall authorize commands before execution to ensure safety and accuracy. */
            attribute authorizationDelay : Real;
            require constraint{ authorizationDelay <= 1.0 }  // Command authorization delay ≤ 1 second
        }

        // Fault detection and safe-state transitions
        requirement faultDetectionReq {
            doc /* The system shall detect faults and transition to safe-state automatically. */
            attribute faultDetectionTime : Real;
            attribute safeStateTransitionTime : Real;
            require constraint{ faultDetectionTime <= 1.0 }  // Fault detection time ≤ 1 second
            require constraint{ safeStateTransitionTime <= 3.0 }  // Safe-state transition time ≤ 3 seconds
        }

        // Telemetry and logging for post-run analysis
        requirement telemetryLoggingReq {
            doc /* The system shall log telemetry data for post-run analysis. */
            attribute telemetryDataFrequency : Real;
            attribute telemetryDataSize : Real;
            require constraint{ telemetryDataFrequency >= 0.5 }  // Telemetry frequency ≥ 0.5 Hz
            require constraint{ telemetryDataSize >= 50.0 }  // Telemetry data size ≥ 50 MB
        }

        // Acceptance check: At least 15 functional requirements for various OCS tasks
        requirement functionalCoverageCheck {
            doc /* The system shall define at least 15 functional requirements covering sequencing, state management, automation, fault detection, and telemetry. */
        }

    }
}
