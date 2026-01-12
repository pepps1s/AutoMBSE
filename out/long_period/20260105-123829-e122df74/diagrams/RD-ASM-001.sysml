package RD-ASM-001 {
    import ScalarValues::*;
    import Quantities::*;
    
    requirement def Assumptions {
        doc /* Document assumptions related to the Thirty Meter Telescope (TMT) project, including environmental and operational assumptions. */
        attribute assumptionDescription : String;
        attribute status : String;  // Assumed, Confirmed, Open
        attribute impactArea : String; // Ops, Safety, Performance, Cost, Schedule
        
        require constraint {
            assumptionDescription != ""
        }
    }

    requirement def Dependencies {
        doc /* Document external dependencies related to TMT, including system integration and external components. */
        attribute dependencyDescription : String;
        attribute status : String;  // Assumed, Confirmed, Open
        attribute impactArea : String; // Ops, Safety, Performance, Cost, Schedule
        
        require constraint {
            dependencyDescription != ""
        }
    }

    requirement def Constraints {
        doc /* Document programmatic and physical constraints, including site constraints and their effects on operational requirements. */
        attribute constraintDescription : String;
        attribute status : String;  // Assumed, Confirmed, Open
        attribute impactArea : String; // Ops, Safety, Performance, Cost, Schedule
        
        require constraint {
            constraintDescription != ""
        }
    }

    // Creating the Assumptions, Dependencies, and Constraints Register (RD-ASM-001)
    requirement def RDASMRegister {
        doc /* Assumptions, Dependencies, and Constraints Register for TMT, documenting assumptions, external dependencies, and programmatic/physical constraints. */
        attribute assumptionsList : List<Assumptions>;
        attribute dependenciesList : List<Dependencies>;
        attribute constraintsList : List<Constraints>;
        attribute siteRelatedConstraints : List<Constraints>;  // Site-related constraints
        
        require constraint {
            assumptionsList.size() >= 15
            constraintsList.size() >= 10
        }
        
        // Acceptance Check: Mark each item with status and impact area
        require constraint {
            assumptionsList.each { assumption ->
                assumption.status != ""
                assumption.impactArea != ""
            }
            constraintsList.each { constraint ->
                constraint.status != ""
                constraint.impactArea != ""
            }
        }
        
        // Acceptance Check: Ensure site-related constraints are included
        require constraint {
            siteRelatedConstraints.size() > 0
        }
        
        // Acceptance Check: Document rationale for each item
        require constraint {
            assumptionsList.each { assumption ->
                assumption.assumptionDescription != ""
            }
            constraintsList.each { constraint ->
                constraint.constraintDescription != ""
            }
        }
    }
}
