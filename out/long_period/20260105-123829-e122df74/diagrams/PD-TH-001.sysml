package PD_TH_001_ObservatoryThermalStabilityBudget {
    
    import RD_NF_004_EnvironmentalConstraints::*;
    import PD_OPT_001_OpticalWavefrontErrorBudget::*;
    import BDD_SUB03_EnclosureAndSiteInfrastructure::*;
    import BDD_SUB01_TelescopeOpticalSystem::*;
    
    part temperature : ThermalParameter {
        attribute units = "Kelvin";
        attribute minValue = 0;
        attribute maxValue = 1000;
    }
    
    part gradient : ThermalParameter {
        attribute units = "K/m";
        attribute minValue = 0;
        attribute maxValue = 10;
    }
    
    part timeConstant : ThermalParameter {
        attribute units = "hours";
        attribute minValue = 1;
        attribute maxValue = 24;
    }
    
    part mirrorTemperature : temperature {
        :>> value = 20;
    }
    
    part enclosureTemperature : temperature {
        :>> value = 25;
    }
    
    part instrumentBayTemperature : temperature {
        :>> value = 22;
    }
    
    part hvacExchangeTemperature : temperature {
        :>> value = 18;
    }
    
    part nightTimeCooldown : timeConstant {
        :>> value = 12;
    }
    
    part temperatureGradient : gradient {
        :>> value = 1.5;
    }
    
    part allowableDelta : gradient {
        :>> value = 0.5;
    }
    
    part operationalThresholds : OperationalConstraints {
        attribute pauseThreshold = 40;
        attribute abortThreshold = 50;
    }
    
    part thermalBudgetAllocation : BudgetConstraint {
        :>> targetSystem = "TMT";
        :>> componentName = "Optics + Enclosure + HVAC";
    }
    
    constraint thermalGradientWFE {
        (temperatureGradient > 0.5) -> (WFEImpact = "High");
        (temperatureGradient <= 0.5) -> (WFEImpact = "Low");
    }
    
    constraint complianceCheck {
        thermalBudgetAllocation :> operationalThresholds;
        operationalThresholds.pauseThreshold <= allowableDelta;
    }
    
    package dependencies {
        import PD_OPT_001_OpticalWavefrontErrorBudget;
        import RD_NF_004_EnvironmentalConstraints;
        import BDD_SUB03_EnclosureAndSiteInfrastructure;
        import BDD_SUB01_TelescopeOpticalSystem;
    }
}
