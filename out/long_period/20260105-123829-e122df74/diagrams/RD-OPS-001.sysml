package RD_OPS_001_OperationalModesStatesCatalog {

    import ScalarValues::*;
    import Quantities::*;
    
    requirement def StartupModeReq {
        doc /* The system shall enter the startup mode where all critical subsystems are powered up and self-checked. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "TelescopeOpticalSystem" and "ObservatoryControlSystem";
            subsystemsDisabled includes "AdaptiveOptics" and "Safety";
        }
        
        entry criteria /* Subsystems power up, telemetry is active, initial calibration starts. */
        exit criteria /* All self-checks pass, subsystems reach operational readiness. */
        allowed transitions /* Startup → Standby, Startup → Maintenance */;
    }

    requirement def StandbyModeReq {
        doc /* The system shall enter standby mode, with all major subsystems in a low-power state, ready for observation. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "ObservatoryControlSystem";
            subsystemsDisabled includes "AdaptiveOptics" and "TelescopeOpticalSystem";
        }
        
        entry criteria /* Minimal telemetry is active, health checks pass. */
        exit criteria /* Transition to Observation or Calibration mode initiated. */
        allowed transitions /* Standby → Observing, Standby → Calibration */;
    }

    requirement def ObservingModeReq {
        doc /* The system shall enter observing mode, enabling full operational capability for data acquisition. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "TelescopeOpticalSystem" and "AdaptiveOptics" and "DataSystem";
            subsystemsDisabled includes "Maintenance" and "SafePark";
        }
        
        entry criteria /* Full subsystem functionality, ready for target acquisition. */
        exit criteria /* Transition initiated due to target completion or failure. */
        allowed transitions /* Observing → Standby, Observing → Maintenance, Observing → SafePark */;
    }

    requirement def CalibrationModeReq {
        doc /* The system shall enter calibration mode to perform necessary calibration of instruments and subsystems. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "TelescopeOpticalSystem" and "DataSystem";
            subsystemsDisabled includes "AdaptiveOptics" and "Safety";
        }
        
        entry criteria /* Calibration routines and target sources activated. */
        exit criteria /* Calibration completed, transition to Observation or Maintenance mode. */
        allowed transitions /* Calibration → Observing, Calibration → Maintenance */;
    }

    requirement def MaintenanceModeReq {
        doc /* The system shall enter maintenance mode, disabling non-essential subsystems and enabling diagnostic routines. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "ObservatoryControlSystem" and "DataSystem";
            subsystemsDisabled includes "TelescopeOpticalSystem" and "AdaptiveOptics";
        }
        
        entry criteria /* System in non-critical state, ready for subsystem checks and repairs. */
        exit criteria /* Maintenance completed, all operational systems checked and verified. */
        allowed transitions /* Maintenance → Standby, Maintenance → SafePark */;
    }

    requirement def SafeParkModeReq {
        doc /* The system shall enter safe park mode, ensuring all subsystems are powered down and safely secured. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "Safety";
            subsystemsDisabled includes "TelescopeOpticalSystem" and "ObservatoryControlSystem";
        }
        
        entry criteria /* System shutdown initiated, critical safety protocols active. */
        exit criteria /* Safe state maintained until power-up is required. */
        allowed transitions /* SafePark → Startup */;
    }

    requirement def DegradedModePartialInstrumentCapabilityReq {
        doc /* The system shall enter a degraded mode where only critical instruments are available for limited observation. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "TelescopeOpticalSystem";
            subsystemsDisabled includes "AdaptiveOptics" and "DataSystem";
        }
        
        entry criteria /* Instrument limitations detected, degraded functionality in place. */
        exit criteria /* Full capability restored or transition to Maintenance mode. */
        allowed transitions /* Degraded → Standby, Degraded → Maintenance */;
    }

    requirement def DegradedModeLimitedTrackingReq {
        doc /* The system shall enter a degraded mode with limited tracking capabilities, suitable for coarse observations. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "ObservatoryControlSystem";
            subsystemsDisabled includes "AdaptiveOptics" and "DataSystem";
        }
        
        entry criteria /* Tracking system degraded, coarse tracking enabled. */
        exit criteria /* Tracking restored to full capability. */
        allowed transitions /* Degraded → Standby, Degraded → Maintenance */;
    }

    requirement def DegradedModeLimitedAlignmentReq {
        doc /* The system shall enter a degraded mode with limited alignment, suitable for low-accuracy observations. */
        
        attribute subsystemsEnabled : Set<Subsystem>;
        attribute subsystemsDisabled : Set<Subsystem>;
        
        require constraint {
            subsystemsEnabled includes "TelescopeOpticalSystem";
            subsystemsDisabled includes "AdaptiveOptics";
        }
        
        entry criteria /* Alignment errors detected, coarse alignment established. */
        exit criteria /* Alignment restored to full precision. */
        allowed transitions /* Degraded → Standby, Degraded → Maintenance */;
    }

}
