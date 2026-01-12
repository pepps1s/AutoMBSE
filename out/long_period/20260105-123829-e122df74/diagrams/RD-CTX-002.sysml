package StakeholdersAndConcernsCatalog {

    import ScalarValues::*;
    import Quantities::*;
    
    requirement def StakeholderGroupConcerns {
        doc /* Capture stakeholder groups and their primary concerns to drive requirements completeness and prioritization. */

        attribute scienceUsersConcerns : Set<Concern>;
        attribute operatorsConcerns : Set<Concern>;
        attribute maintainersConcerns : Set<Concern>;
        attribute safetyConcerns : Set<Concern>;
        attribute dataUsersConcerns : Set<Concern>;
        attribute siteConcerns : Set<Concern>;
        attribute partnersConcerns : Set<Concern>;

        require constraint{ 
            // Ensure each stakeholder has 3+ concerns mapped to relevant requirement themes
            scienceUsersConcerns->size() >= 3;
            operatorsConcerns->size() >= 3;
            maintainersConcerns->size() >= 3;
            safetyConcerns->size() >= 3;
            dataUsersConcerns->size() >= 3;
            siteConcerns->size() >= 3;
            partnersConcerns->size() >= 3;
        }
        
        // Concerns should map to requirement themes
        requirementMapping concernMapping {
            scienceUsersConcerns->exists( concern | concern.theme == "performance" || concern.theme == "safety" || concern.theme == "availability" || concern.theme == "maintainability" || concern.theme == "dataQuality");
            operatorsConcerns->exists( concern | concern.theme == "performance" || concern.theme == "safety" || concern.theme == "availability" || concern.theme == "maintainability" || concern.theme == "dataQuality");
            maintainersConcerns->exists( concern | concern.theme == "performance" || concern.theme == "safety" || concern.theme == "availability" || concern.theme == "maintainability" || concern.theme == "dataQuality");
            safetyConcerns->exists( concern | concern.theme == "performance" || concern.theme == "safety" || concern.theme == "availability" || concern.theme == "maintainability" || concern.theme == "dataQuality");
            dataUsersConcerns->exists( concern | concern.theme == "performance" || concern.theme == "safety" || concern.theme == "availability" || concern.theme == "maintainability" || concern.theme == "dataQuality");
            siteConcerns->exists( concern | concern.theme == "performance" || concern.theme == "safety" || concern.theme == "availability" || concern.theme == "maintainability" || concern.theme == "dataQuality");
            partnersConcerns->exists( concern | concern.theme == "performance" || concern.theme == "safety" || concern.theme == "availability" || concern.theme == "maintainability" || concern.theme == "dataQuality");
        }
        
        // Traceability to requirements
        traceability requirementTraceability {
            // Requirements are traced based on concern title alignment
            scienceUsersConcerns->exists( concern | concern.traceableToRequirement->exists(req | req.title matches "performance" || req.title matches "safety" || req.title matches "availability" || req.title matches "maintainability" || req.title matches "dataQuality"));
            operatorsConcerns->exists( concern | concern.traceableToRequirement->exists(req | req.title matches "performance" || req.title matches "safety" || req.title matches "availability" || req.title matches "maintainability" || req.title matches "dataQuality"));
            maintainersConcerns->exists( concern | concern.traceableToRequirement->exists(req | req.title matches "performance" || req.title matches "safety" || req.title matches "availability" || req.title matches "maintainability" || req.title matches "dataQuality"));
            safetyConcerns->exists( concern | concern.traceableToRequirement->exists(req | req.title matches "performance" || req.title matches "safety" || req.title matches "availability" || req.title matches "maintainability" || req.title matches "dataQuality"));
            dataUsersConcerns->exists( concern | concern.traceableToRequirement->exists(req | req.title matches "performance" || req.title matches "safety" || req.title matches "availability" || req.title matches "maintainability" || req.title matches "dataQuality"));
            siteConcerns->exists( concern | concern.traceableToRequirement->exists(req | req.title matches "performance" || req.title matches "safety" || req.title matches "availability" || req.title matches "maintainability" || req.title matches "dataQuality"));
            partnersConcerns->exists( concern | concern.traceableToRequirement->exists(req | req.title matches "performance" || req.title matches "safety" || req.title matches "availability" || req.title matches "maintainability" || req.title matches "dataQuality"));
        }
    }

    class Concern {
        doc /* Defines a concern raised by a stakeholder group, including the theme and traceability to requirements */
        
        attribute title : String;
        attribute theme : String; // e.g., performance, safety, availability, maintainability, data quality
        attribute traceableToRequirement : Set<Requirement>;
    }
}
