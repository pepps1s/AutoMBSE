package TraceabilityBackbone_RDT_002 {
    import ScalarValues::*;
    import Quantities::*;
    
    requirement def RD_RequirementToVerificationTrace {
        doc /* This requirement defines the traceability from the set of requirements 
             * to the verification methods and evidence placeholders for the Thirty Meter Telescope (TMT). 
             * It ensures at least 40 requirements trace to one verification method and evidence placeholder, 
             * and identifies high-risk requirements needing multiple verification methods (e.g., performance + safety). */
        
        attribute requirementID : String;
        attribute verificationMethod : String;
        attribute evidencePlaceholder : String;
        attribute traceLinkConsistency : Boolean;
        attribute highRiskFlag : Boolean;
        
        require constraint { 
            requirementID != '' && verificationMethod != '' && evidencePlaceholder != '';
            traceLinkConsistency == true; 
            highRiskFlag == true implies verificationMethod != 'single';
        }
    }
    
    requirement def TraceabilityVerificationConsistency {
        doc /* This requirement ensures that trace links are consistent and non-duplicative. 
             * It enforces that each requirement is traced uniquely to a verification method and evidence placeholder. */
        
        attribute requirementID : String;
        attribute verificationMethod : String;
        attribute evidencePlaceholder : String;
        
        require constraint {
            traceLinkConsistency == true;
            requirementID != '' && verificationMethod != '' && evidencePlaceholder != '';
        }
    }

    requirement def HighRiskRequirementIdentification {
        doc /* This requirement identifies high-risk requirements which need multiple verification methods, 
             * such as those related to performance and safety, ensuring they are flagged for extra attention. */
        
        attribute requirementID : String;
        attribute riskLevel : String;
        attribute verificationMethodsCount : Integer;
        
        require constraint {
            riskLevel == 'high' implies verificationMethodsCount > 1;
        }
    }
    
    // Traceability Links for Specific Requirements
    package RD_VER_001 {
        import ScalarValues::*;
        import Quantities::*;
        
        requirement def VerificationMethodAndEvidence {
            doc /* This requirement defines the verification methods and associated evidence placeholders for traceability. 
                 * It links specific verification strategies to the corresponding evidence required for validation. */
            
            attribute verificationMethod : String;
            attribute evidenceType : String;
            attribute associatedRequirement : String;
        
            require constraint {
                verificationMethod != '' && evidenceType != '';
                associatedRequirement != '';
            }
        }
    }
    
    package RD_TR_001 {
        import ScalarValues::*;
        import Quantities::*;
        
        requirement def TraceableRequirement {
            doc /* This requirement defines the trace from the set of requirements to verification methods and evidence placeholders. */
        
            attribute requirementID : String;
            attribute verificationMethod : String;
            attribute evidencePlaceholder : String;
            attribute requirementType : String;
            attribute verificationType : String;
        
            require constraint {
                requirementID != '' && verificationMethod != '' && evidencePlaceholder != '';
                requirementType == 'functional' || requirementType == 'non-functional';
                verificationType == 'test' || verificationType == 'inspection' || verificationType == 'analysis';
            }
        }
    }
}
