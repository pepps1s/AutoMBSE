package SafetyAndInterlocksSubsystem {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    part def InterlockController;
    part def EStopChain;
    part def HazardDetectionSensors;
    part def AccessControl;

    part SafetySubsystem {
        part interlockController : InterlockController;
        part eStopChain : EStopChain;
        part hazardDetectionSensors : HazardDetectionSensors;
        part accessControl : AccessControl;
    }

    attribute def SafetyGoal {
        goal : String;
        description : String;
    }

    part SafetySystemGoal : SafetyGoal {
        goal = "Define safety subsystem structure: interlock controller, E-stop chain, hazard detection sensors, access control.";
        description = "This goal aims to establish the fundamental components of the safety subsystem for the Thirty Meter Telescope, ensuring personnel safety and operational integrity.";
    }

    // Acceptance checks for verification
    part SafetyAcceptanceCheck {
        part interlockControllerDefined : Boolean = true;
        part eStopChainDefined : Boolean = true;
        part hazardDetectionSensorsIncluded : Boolean = true;
        part accessControlIncluded : Boolean = true;
    }
}
