package PD_ARM_001_ReliabilityAllocationBudget {

    // Model for Reliability Allocation Budget (MTBF/Failure Rate by Subsystem)
    
    import TMT_SystemModel::*;
    
    part reliabilityBudget :> TMT_SystemModel {
        
        // Define subsystems involved in the reliability allocation
        part mountSubsystem :> TMT_Subsystem {
            attribute MTBF : Real;
            attribute failureRate : Real;
            attribute repairTime : Real;
        }
        
        part opticsControlSubsystem :> TMT_Subsystem {
            attribute MTBF : Real;
            attribute failureRate : Real;
            attribute repairTime : Real;
        }
        
        part enclosureSubsystem :> TMT_Subsystem {
            attribute MTBF : Real;
            attribute failureRate : Real;
            attribute repairTime : Real;
        }
        
        part ocsSubsystem :> TMT_Subsystem {
            attribute MTBF : Real;
            attribute failureRate : Real;
            attribute repairTime : Real;
        }
        
        part dataSubsystem :> TMT_Subsystem {
            attribute MTBF : Real;
            attribute failureRate : Real;
            attribute repairTime : Real;
        }
        
        part safetySubsystem :> TMT_Subsystem {
            attribute MTBF : Real;
            attribute failureRate : Real;
            attribute repairTime : Real;
        }
        
        // Link dependencies
        dependency PD_AVL_001 :> TMT_Dependency {
            reference ReliabilityTarget;
        }
        
        // Define system-level allocation
        part systemReliabilityTarget :> TMT_System {
            attribute overallMTBF : Real;
            attribute overallFailureRate : Real;
            attribute overallRepairTime : Real;
        }
        
        // Sanity constraint to ensure no negative MTBF values
        constraint positiveMTBFConstraint {
            // Ensure MTBF cannot be negative or zero
            (mountSubsystem.MTBF > 0) and 
            (opticsControlSubsystem.MTBF > 0) and 
            (enclosureSubsystem.MTBF > 0) and 
            (ocsSubsystem.MTBF > 0) and 
            (dataSubsystem.MTBF > 0) and 
            (safetySubsystem.MTBF > 0)
        }

        // Roll-up logic to aggregate subsystem MTBF values to system-level target
        allocation {
            systemReliabilityTarget.overallMTBF = 
                (mountSubsystem.MTBF + opticsControlSubsystem.MTBF + enclosureSubsystem.MTBF + 
                 ocsSubsystem.MTBF + dataSubsystem.MTBF + safetySubsystem.MTBF) / 6;
        }
    }
}
