package PD_MNT_001 {
    
    import TMT_System_Architecture::*;
    import TMT_Operations_Maintenance::*;
    
    part MaintainabilityServiceBudget {
        
        goal "Budget maintenance durations and frequencies for key maintainability drivers that impact availability and ops planning";
        dependencies PD_AVL_001, RD_NF_003, RD_F_002, BDD_SUB10_001, BDD_SUB06_001, BDD_SUB04_001;
        
        part ServiceDurationCadence {
            attribute duration: Time;
            attribute cadence: Time;
            attribute crewSize: Integer;
        }

        part MTTRRelatedValues {
            attribute meanTimeToRepair: Time;
            attribute repairCrewSize: Integer;
            attribute failureRate: Real;
        }
        
        part MaintenanceActivities {
            part activity1 : Activity {
                name "Segment Handling";
                frequency 12;  // In months
                duration 4;  // In hours
            }
            part activity2 : Activity {
                name "Actuator Calibration";
                frequency 6;  // In months
                duration 8;  // In hours
            }
            part activity3 : Activity {
                name "Instrument Swap";
                frequency 24;  // In months
                duration 12;  // In hours
            }
            part activity4 : Activity {
                name "Mirror Cleaning";
                frequency 36;  // In months
                duration 3;  // In hours
            }
            part activity5 : Activity {
                name "System Diagnostics";
                frequency 3;  // In months
                duration 2;  // In hours
            }
            part activity6 : Activity {
                name "Thermal Control Calibration";
                frequency 24;  // In months
                duration 5;  // In hours
            }
            part activity7 : Activity {
                name "Power System Inspection";
                frequency 18;  // In months
                duration 6;  // In hours
            }
            part activity8 : Activity {
                name "Cooling System Maintenance";
                frequency 12;  // In months
                duration 7;  // In hours
            }
        }
        
        part MaintenanceTimeLoss {
            constraint "Planned maintenance time loss per observing period";
            // Roll-up constraint computation for planned downtime
            function computeTimeLoss: Real {
                return sum(activity.duration * activity.frequency for activity in MaintenanceActivities);
            }
        }
        
        part MaintenanceModeConstraint {
            constraint "Maintenance tasks only in allowed ops modes";
            operationMode: Mode;
            allowedModes { "standby", "maintenance" };
            rule "If activity scheduled in operational mode, it must be in standby or maintenance";
        }

        part InstrumentSwap {
            name "Instrument Swap Task";
            dependency BDD_SUB06_001;
            constraint "Ensure the inclusion of instrument swap with correct subsystem references";
            referenceSubsystems { "InstrumentSubSystem" };
        }

        part SegmentHandlingTask {
            name "Segment Handling Task";
            dependency BDD_SUB04_001;
            constraint "Segment-related task references specific subsystems";
            referenceSubsystems { "OpticalSystem" };
        }

    }

}
