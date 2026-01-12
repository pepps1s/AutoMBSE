package PD_PWR_001 {

    import TMT_Resource_Architecture::*;
    
    part ElectricalPowerBudget :> PowerBudget {
        part redefines steadyStatePower : PowerAllocation {
            powerType = kW;
            value = 1000;  // Example steady-state power allocation
        }
        part redefines peakPower : PowerAllocation {
            powerType = kVA;
            value = 1200;  // Example peak power allocation
        }
        part redefines backupPower : PowerBackup {
            value = 1500;  // Example backup power
            backupDuration = 10;  // Backup duration in hours
        }
    }

    part PowerAllocation {
        attribute powerType : String;  // kW or kVA
        attribute value : Float;       // Power value in kW or kVA
    }

    part PowerBackup {
        attribute value : Float;       // Backup power value
        attribute backupDuration : Float;  // Duration in hours
    }
    
    part PowerSubsystems :> PowerSystem {
        part telescopeOpticalSystem : PowerSubsystem {
            powerType = kW;
            value = 200;
        }
        part telescopeMountStructure : PowerSubsystem {
            powerType = kW;
            value = 300;
        }
        part enclosureAndSiteInfrastructure : PowerSubsystem {
            powerType = kW;
            value = 150;
        }
        part adaptiveOpticsSubsystem : PowerSubsystem {
            powerType = kW;
            value = 100;
        }
        part instrumentsSubsystem : PowerSubsystem {
            powerType = kW;
            value = 250;
        }
        part controlSystem : PowerSubsystem {
            powerType = kW;
            value = 120;
        }
        part safetySystem : PowerSubsystem {
            powerType = kW;
            value = 80;
        }
    }

    part PowerSystem {
        attribute powerType : String;  // kW
        attribute value : Float;       // Power value
    }

    constraint SteadyStatePowerMargin {
        steadyStatePower.value <= (totalPowerAllocation.value * 1.05);  // Allowing for a 5% margin
    }

    constraint PeakPowerMargin {
        peakPower.value <= (totalPowerAllocation.value * 1.1);  // Allowing for a 10% margin during peak
    }

    constraint BackupPowerSupport {
        backupPower.value >= totalPowerAllocation.value * 0.2;  // Backup power should cover at least 20% of total power
    }
    
    constraint SafetyCriticalLoads {
        safetySystem.value >= 50;  // Ensuring safety-critical load is supported
    }
}
