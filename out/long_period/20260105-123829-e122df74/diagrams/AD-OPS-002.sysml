package behavior {

    // Define the input item for target coordinates and time reference
    item def TargetCoordinates;
    item def TimeReference;

    // Define the main action for slew-to-target behavior
    action SlewToTarget {
        in item targetCoordinates : TargetCoordinates; // Target coordinates input
        in item timeReference : TimeReference;         // Time reference input

        first start;
        then action 'ApplyPointingModel';              // Apply pointing model action
        then action 'CommandMountMotion';              // Command the mount motion action
        then action 'MonitorEncoders';                 // Monitor the encoders
        then action 'SettleVerifyResidualError';       // Settle and verify residual error
        then action 'WindThermalConstraintCheck';      // Check for wind/thermal constraint violations

        // Guard for wind/thermal constraint violation, branching to pause/abort
        guard condition windThermalConstraintViolated : RD-NF-004;
        then action 'PauseOrAbort';                    // Pause or abort the operation based on constraints

        then action 'MonitorAdjust';                   // Feedback loop: monitor and adjust
        then action 'SlewComplete';                    // Mark the slew as complete

        then done;                                     // End of the action sequence
    }

    // Define specific actions and their dependencies
    action def 'ApplyPointingModel';
    action def 'CommandMountMotion';
    action def 'MonitorEncoders';
    action def 'SettleVerifyResidualError';
    action def 'WindThermalConstraintCheck';
    action def 'PauseOrAbort';
    action def 'MonitorAdjust';
    action def 'SlewComplete';

}
