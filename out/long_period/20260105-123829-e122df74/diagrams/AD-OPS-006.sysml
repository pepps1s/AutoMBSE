package AD_OPS_WeatherAbortAndControlledPause {

    // Define items for environmental conditions and operational states
    item def EnvironmentStatus;
    item def CurrentOpsState;
    
    // Define actions and flow
    action weatherAbortAndPause {
        in item environment : EnvironmentStatus;
        in item opsState : CurrentOpsState;
        
        // Behavior starts here
        first start;
        
        // Check environmental conditions: wind and temperature
        then action checkWindSpeed {
            in item wind : EnvironmentStatus;
            // Guard for wind speed threshold
            if wind.speed > 15 { // Threshold for wind speed
                then action abortExposure; // Abort exposure if wind is too strong
            }
            else {
                then done;
            }
        }

        then action checkTemperature {
            in item temperature : EnvironmentStatus;
            // Guard for temperature gradient
            if temperature.gradient > 5 { // Threshold for temperature gradient
                then action pauseExposure; // Pause exposure if temperature gradient exceeds limit
            }
            else {
                then done;
            }
        }

        // Action steps for safe termination
        then action stopExposureSafely {
            in item opsState : CurrentOpsState;
            // Ensure exposure is safely stopped
            then action parkShutters;
            then action closeShutters;
            then action logEvent;
            then action finalizeExposureRecord;
        }

        then action safeParkMode {
            // Transition to a safe park mode to ensure no further exposure
            then done;
        }
        
        // Action flow ends here
        then done;
    }

    // Define specific actions for stopping exposure and protecting optics
    action abortExposure {
        // Abort exposure and take necessary steps
        then done;
    }

    action pauseExposure {
        // Pause exposure and protect the system
        then done;
    }

    action parkShutters {
        // Park the shutters to protect optics/instruments
        then done;
    }

    action closeShutters {
        // Close shutters or equivalent protection mechanism
        then done;
    }

    action logEvent {
        // Log event for weather abort or pause
        then done;
    }

    action finalizeExposureRecord {
        // Ensure data and metadata are complete before termination
        then done;
    }
}
