package EnvironmentalNFRs_TMT {
    import ScalarValues::*;
    import Quantities::*;
    
    requirement def ThermalWindDustSiteConditions {
        doc /* This requirement defines environment-driven non-functional requirements (NFRs) for the Thirty Meter Telescope (TMT) concerning thermal, wind, dust, and site conditions, and their operational constraints. */
        
        attribute temperatureGradient : Real; // Placeholder for temperature gradient limits
        attribute windSpeedLimit : Real; // Placeholder for wind speed constraints
        attribute cleanlinessClass : Integer; // Placeholder for cleanliness class (impacting instrument performance)
        
        require constraint {
            temperatureGradient <= 30.0 // Measurable constraint for temperature gradient in degrees Celsius
            windSpeedLimit <= 30.0 // Measurable wind speed limit in meters per second
            cleanlinessClass >= 2 // Ensure cleanliness class of 2 or better
        }
    }
    
    requirement def OperationalConstraintsForPausing {
        doc /* This requirement defines operational constraints for when observation must be paused or aborted due to environmental conditions. */
        
        attribute temperatureThreshold : Real; // Placeholder for temperature threshold
        attribute windSpeedThreshold : Real; // Placeholder for wind speed threshold
        attribute dustVisibilityLimit : Real; // Placeholder for dust visibility limits
        
        require constraint {
            temperatureThreshold <= 40.0 // Temperature at which observation must pause in Celsius
            windSpeedThreshold <= 25.0 // Wind speed threshold for pausing observation in meters per second
            dustVisibilityLimit <= 0.5 // Dust visibility limit where observation must be aborted
        }
    }
    
    requirement def EnvironmentalConstraintsImpactOnFunctions {
        doc /* This requirement maps each environmental constraint to the impacted functions such as tracking, dome, phasing, and instrument stability. */
        
        attribute impactedFunction : String; // Placeholder for impacted function (e.g., tracking, dome, phasing)
        attribute environmentalCondition : String; // Placeholder for environmental condition (e.g., temperature, wind)
        
        require constraint {
            (impactedFunction == "tracking" && environmentalCondition == "wind") // Wind impacting tracking function
            (impactedFunction == "dome" && environmentalCondition == "temperature") // Temperature impacting dome function
            (impactedFunction == "phasing" && environmentalCondition == "wind") // Wind affecting phasing
            (impactedFunction == "instrument stability" && environmentalCondition == "dust") // Dust affecting instrument stability
        }
    }
}
