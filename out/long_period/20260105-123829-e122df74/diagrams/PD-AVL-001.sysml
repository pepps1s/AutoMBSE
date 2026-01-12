package PD_AVL_001_AvailabilityBudget {

    import RD_NF_003_ArmRequirements;
    import RD_F_002_ObservationLifecycleFunctionalDecomposition;
    import RD_OPS_001_OperationalModesStatesCatalog;
    import BDD_SUB10_001_OperationsMaintenance;

    // Define parameters for availability, downtime hours, and loss categories
    parameter AvailabilityTarget : Real;
    parameter DowntimeHours : Real;
    parameter LossCategories : List<String>;

    // Define types for the different categories of downtime (planned vs unplanned)
    enum DowntimeCategory {
        Weather,
        Faults,
        Maintenance,
        CalibrationOverhead,
        StartupShutdown
    }

    // Create a list of loss categories and their impact on availability
    lossCategory Weather : DowntimeCategory {
        duration : Real;
        impact : Real;
    }
    lossCategory Faults : DowntimeCategory {
        duration : Real;
        impact : Real;
    }
    lossCategory Maintenance : DowntimeCategory {
        duration : Real;
        impact : Real;
    }
    lossCategory CalibrationOverhead : DowntimeCategory {
        duration : Real;
        impact : Real;
    }
    lossCategory StartupShutdown : DowntimeCategory {
        duration : Real;
        impact : Real;
    }

    // Include planned and unplanned downtime separation
    plannedDowntime : List<LossCategory> {
        includes Weather, Faults, Maintenance
    }
    unplannedDowntime : List<LossCategory> {
        includes CalibrationOverhead, StartupShutdown
    }

    // Define the roll-up constraint for achieving the required availability target
    constraint AchievedAvailabilityConstraint {
        AchievedAvailability >= AvailabilityTarget
    }

    // Reference calibration cadence/time from RD-F-002
    parameter CalibrationCadenceTime : Real from RD_F_002_ObservationLifecycleFunctionalDecomposition;

    // Define check for achieved availability against target from RD-NF-003
    check AchievedAvailabilityCheck {
        AchievedAvailability >= AvailabilityTarget
    }

    // Dependencies on other requirements
    dependency RD_NF_003;
    dependency RD_F_002;
    dependency RD_OPS_001;
    dependency BDD_SUB10_001;

}
