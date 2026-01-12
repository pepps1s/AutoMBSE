package VerificationStrategyAndEvidence {
    import ScalarValues::*;
    import Quantities::*;
    
    requirement def VerificationStrategyReqDef {
        doc /* Define the verification approach for the TMT system, including methods such as analysis, test, inspection, and demonstration. 
             * For each key requirement group, establish evidence expectations such as test reports, logs, calibration records, and analysis models. */
        
        attribute verificationMethod : String;
        attribute evidenceType : String;
        attribute safetyConstraints : Boolean;
        attribute dataIntegrityConstraints : Boolean;
        
        require constraint{ 
            (verificationMethod == "Analysis" or verificationMethod == "Test" or verificationMethod == "Inspection" or verificationMethod == "Demonstration") and 
            (evidenceType == "Test Reports" or evidenceType == "Logs" or evidenceType == "Calibration Records" or evidenceType == "Analysis Models")
        }
    }
    
    requirement def VerificationEvidenceReqDef {
        doc /* For each verification method, define a list of evidence artifacts, ensuring safety and data integrity requirements are met. */
        
        attribute testReports : Boolean;
        attribute logs : Boolean;
        attribute calibrationRecords : Boolean;
        attribute analysisModels : Boolean;
        
        require constraint{ 
            testReports or logs or calibrationRecords or analysisModels
        }
    }

    requirement def SafetyAndIntegrityConstraintsReqDef {
        doc /* Define the safety and data integrity verification constraints, including safety measures and data integrity checks. */
        
        attribute safetyVerification : Boolean;
        attribute dataIntegrityVerification : Boolean;
        
        require constraint{ 
            safetyVerification and dataIntegrityVerification
        }
    }
}
