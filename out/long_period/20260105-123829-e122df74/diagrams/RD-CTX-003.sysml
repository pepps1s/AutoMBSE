package RD_CTX_003_MissionGoalsAndScienceDrivers {
    import ScalarValues::*;
    import Quantities::*;

    // Requirement for defining mission goals and science drivers
    requirement def MissionGoalsAndScienceDrivers {
        doc /* Define mission goals and science drivers as top-level sources for functional and performance requirements. */
        
        // Define the mission/science goals
        attribute earlyUniverse : Boolean;
        attribute galaxyEvolution : Boolean;
        attribute exoplanets : Boolean;
        attribute BHgrowth : Boolean;
        attribute highAngularResolution : Boolean;
        attribute longExposures : Boolean;
        
        // Define capabilities to map to each goal
        attribute collectLight : Boolean;
        attribute highAngularResolutionCapability : Boolean;
        attribute longExposureCapability : Boolean;

        // Define constraints to ensure mapping is correct
        require constraint {
            earlyUniverse == collectLight
            galaxyEvolution == highAngularResolution
            exoplanets == longExposures
            BHgrowth == collectLight
            highAngularResolution == highAngularResolutionCapability
            longExposures == longExposureCapability
        }

        // Define a consistent goal/capability taxonomy
        doc /* The taxonomy for goals and capabilities must be reused across the RD traceability. */
        
        // Map each goal to capability
        requirement def EarlyUniverseGoalMapping {
            doc /* The goal 'earlyUniverse' is mapped to capability 'collectLight'. */
            attribute mappedToCapability : collectLight;
        }

        requirement def GalaxyEvolutionGoalMapping {
            doc /* The goal 'galaxyEvolution' is mapped to capability 'highAngularResolution'. */
            attribute mappedToCapability : highAngularResolutionCapability;
        }

        requirement def ExoplanetsGoalMapping {
            doc /* The goal 'exoplanets' is mapped to capability 'longExposures'. */
            attribute mappedToCapability : longExposureCapability;
        }

        requirement def BHgrowthGoalMapping {
            doc /* The goal 'BHgrowth' is mapped to capability 'collectLight'. */
            attribute mappedToCapability : collectLight;
        }
    }
}
