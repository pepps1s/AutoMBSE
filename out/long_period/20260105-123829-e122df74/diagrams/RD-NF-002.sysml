package RD_NF_002_Performance_NFR_Pointing_Tracking_Stability_Jitter {

    import ScalarValues::*;
    import Quantities::*;
    
    requirement def PointingTrackingStabilityJitter {
        doc /* The pointing and tracking system shall maintain stability and minimize jitter for long exposure imaging. Measurable criteria should include frequency band, RMS, settling time, and jitter tolerance. Environmental factors such as wind and thermal effects must be accounted for as assumptions or influencing factors. */

        attribute stabilityError : Real;         // Maximum allowable pointing error in arcseconds
        attribute jitterRms : Real;             // Root mean square jitter in arcseconds
        attribute settlingTime : Real;          // Time to settle after slew in seconds
        attribute driftRate : Real;             // Drift rate in arcseconds per minute
        
        // Measurable criteria with placeholders
        attribute frequencyBand : Real;        // Frequency band in Hz for tracking stability
        attribute rmsTolerance : Real;         // RMS tolerance in arcseconds
        attribute settlingTolerance : Real;    // Settling time tolerance in seconds

        // Environmental contributors
        attribute windImpact : Real;           // Wind speed effect on stability in m/s
        attribute thermalEffect : Real;        // Thermal drift effect on stability in degrees Celsius
        
        require constraint {
            stabilityError <= 0.1  // Pointing error must be within 0.1 arcseconds
            jitterRms <= 0.05      // RMS jitter must be less than 0.05 arcseconds
            settlingTime <= 10     // Settling time should be less than 10 seconds
            driftRate <= 0.02      // Drift rate should not exceed 0.02 arcseconds per minute
        }
    }

    requirement def EnvironmentalImpactAssumption {
        doc /* Assumptions on environmental factors (wind, thermal) that influence system stability and jitter. These factors must be accounted for in the design and analysis of the pointing and tracking system. */
        
        attribute windSpeedImpact : Real;      // Effect of wind on stability (m/s)
        attribute thermalVariation : Real;     // Effect of temperature variation on stability (°C)
        
        require constraint {
            windSpeedImpact <= 5  // Wind speed effect should not exceed 5 m/s
            thermalVariation <= 2 // Temperature variation impact should not exceed 2°C
        }
    }
    
    requirement def ConsistentTerminology {
        doc /* Ensure terminology consistency across all documents and views. Terms like stability, accuracy, and repeatability must be used correctly and consistently across all system models. */
        
        attribute termStability : Boolean;      // Consistent use of the term "stability"
        attribute termAccuracy : Boolean;       // Consistent use of the term "accuracy"
        attribute termRepeatability : Boolean;  // Consistent use of the term "repeatability"
        
        require constraint {
            termStability == true   // Ensure "stability" is used consistently
            termAccuracy == true    // Ensure "accuracy" is used consistently
            termRepeatability == true  // Ensure "repeatability" is used consistently
        }
    }
}
