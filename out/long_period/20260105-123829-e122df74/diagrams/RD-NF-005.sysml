package RD_NF_005_Interface_NFR_Timing_Latency_Synchronization {
    
    import ScalarValues::*;
    import Quantities::*;
    
    requirement def TimingSynchronizationReqDef {
        doc /* This requirement defines the critical timing, latency, and synchronization non-functional requirements for the telescope control system. It ensures the synchronization of clocks, timestamps, and command/telemetry rates, essential for the integrity and accuracy of the system. */
        
        attribute clockSyncAccuracy : Real;
        attribute timestampAccuracy : Real;
        attribute commandLatency : Real;
        attribute telemetryRate : Real;
        
        require constraint {
            clockSyncAccuracy <= 1.0;  // Maximum allowable clock synchronization error in milliseconds
            timestampAccuracy <= 0.5;  // Maximum allowable timestamp error in milliseconds
            commandLatency <= 100.0;  // Maximum command latency in milliseconds
            telemetryRate >= 5.0;     // Minimum telemetry rate in Hz
        }
    }

    requirement def LoopClosureTimingReqDef {
        doc /* Defines the timing constraints required for loop closure in adaptive optics and phasing systems, including phasing accuracy and guiding latency. */
        
        attribute loopClosurePhasing : Real;
        attribute loopClosureGuiding : Real;
        
        require constraint {
            loopClosurePhasing <= 0.2;  // Maximum phasing error in micrometers
            loopClosureGuiding <= 50.0;  // Maximum guiding latency in milliseconds
        }
    }

    requirement def VerificationApproachReqDef {
        doc /* Defines the verification approach for assessing the system’s time correlation, latency measurements, and synchronization accuracy. This will be achieved through time correlation tests and end-to-end latency measurements. */
        
        attribute timeCorrelationTestEnabled : Boolean;
        attribute latencyMeasurementEnabled : Boolean;
        
        require constraint {
            timeCorrelationTestEnabled == true;  // Time correlation test must be enabled for validation
            latencyMeasurementEnabled == true;   // End-to-end latency measurement must be enabled
        }
    }

    requirement def TimingLatencySyncReqDef {
        doc /* This requirement specifies that the system must ensure real-time synchronization and data integrity for control loops and telemetry systems. */
        
        attribute realTimeSyncEnabled : Boolean;
        attribute dataIntegrityMaintained : Boolean;
        
        require constraint {
            realTimeSyncEnabled == true;   // Real-time synchronization must be enabled for control loops
            dataIntegrityMaintained == true; // Data integrity must be ensured at all times
        }
    }

    requirement def SyncDependencyReqDef {
        doc /* States the dependencies between timing and latency systems, referencing RD-F-006 and RD-F-007 for further system-specific details. */
        
        attribute dependentSystems : String;
        
        require constraint {
            dependentSystems == "RD-F-006, RD-F-007";  // Reference to required system dependencies for full implementation
        }
    }

}
