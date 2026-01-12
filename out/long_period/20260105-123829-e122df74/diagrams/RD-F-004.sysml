package OpticalAlignmentAndSegmentPhasingRequirements {

    import ScalarValues::*;
    import Quantities::*;
    
    requirement def OpticalAlignmentReq {
        doc /* The optical alignment of the segmented primary mirror 
             * shall be maintained within the specified tolerances during
             * all operational modes. */
        
        attribute alignmentTolerance : Real;
        attribute alignmentStability : Real;
        
        require constraint { alignmentStability >= 0.99 }
    }
    
    requirement def SegmentPhasingReq {
        doc /* The segment phasing system shall ensure that all segments 
             * are correctly phased to within the system's performance 
             * specifications during all phases of operation. */
        
        attribute phasingAccuracy : Real;
        attribute segmentCount : Integer;
        
        require constraint { phasingAccuracy <= 0.1 }
    }
    
    requirement def MetrologyReq {
        doc /* The metrology system shall provide real-time position and
             * alignment data for each segment, with a minimum update rate 
             * of 1Hz. */
        
        attribute updateRate : Real;
        attribute segmentPositionAccuracy : Real;
        
        require constraint { updateRate >= 1.0 }
    }

    requirement def SegmentActuationReq {
        doc /* The segment actuators shall provide precise control for 
             * segment positioning, maintaining segment positions with 
             * a maximum deviation of 0.1mm. */
        
        attribute actuatorPrecision : Real;
        attribute actuatorResponseTime : Real;
        
        require constraint { actuatorPrecision <= 0.1 }
    }
    
    requirement def PhasingLoopBehaviorReq {
        doc /* The phasing loop shall maintain segment phasing within 
             * defined performance limits during the alignment, continuous 
             * maintenance, and recovery modes. */
        
        attribute loopStability : Real;
        attribute recoveryTime : Real;
        
        require constraint { loopStability >= 0.98 }
        require constraint { recoveryTime <= 60.0 }
    }
    
    requirement def InitialAlignmentReq {
        doc /* The initial alignment mode shall achieve optical alignment
             * and segment phasing within 30 minutes after system startup. */
        
        attribute alignmentTime : Real;
        
        require constraint { alignmentTime <= 30.0 }
    }
    
    requirement def ContinuousMaintenanceReq {
        doc /* The continuous maintenance mode shall ensure that alignment
             * and phasing are continuously monitored and adjusted in real-time. */
        
        attribute monitoringFrequency : Real;
        
        require constraint { monitoringFrequency >= 0.1 }
    }
    
    requirement def RecoveryAfterDisturbanceReq {
        doc /* After a disturbance, the system shall restore optical alignment 
             * and segment phasing to within tolerances within 5 minutes. */
        
        attribute disturbanceRecoveryTime : Real;
        
        require constraint { disturbanceRecoveryTime <= 5.0 }
    }

    // Define dependencies with other diagrams
    dependency RD_F_001 {
        referencedPackage "RD_F_001" {
            // Define specific parts or actions if needed
        }
    }
    
    dependency RD_CTX_001 {
        referencedPackage "RD_CTX_001" {
            // Define specific parts or actions if needed
        }
    }
    
}
