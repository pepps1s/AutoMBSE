package IBD_SUB04_PhaseLoopStabilityRecovery {

    import ScalarValues::*;
    import Quantities::*;

    part def PhasingLoopStabilityRecovery {
        attribute disturbanceDetected : Boolean;
        attribute loopState : LoopState;
        attribute recoveryTrigger : RecoveryTrigger;
        attribute recoveryTime : Duration;
        attribute loopStability : Real;

        port in disturbanceDetection : DisturbanceDetection;
        port in loopStatus : LoopStatus;
        port out recoveryAction : RecoveryAction;
        port out recoveryTimeOut : RecoveryTime;

        flow disturbanceDetection : disturbanceDetected;
        flow loopStatus : loopState;
        flow recoveryAction : recoveryTrigger;
        flow recoveryTimeOut : recoveryTime;

        part APS : AlignmentAndPhasingSystem {
            part disturbanceDetectionUnit : DisturbanceDetectionUnit;
            part loopStatusUnit : LoopStatusUnit;
            part recoveryActionUnit : RecoveryActionUnit;
            part recoveryTimeUnit : RecoveryTimeUnit;
        }
    }

    // Instantiate the BDD-defined type so its attributes are instantiated in this IBD context
    part plsRecovery : PhasingLoopStabilityRecovery;

    part def DisturbanceDetection {
        port out disturbanceDetected : Boolean;
    }

    part def LoopStatus {
        port out loopState : LoopState;
    }

    part def RecoveryAction {
        port in recoveryTrigger : RecoveryTrigger;
        port out recoveryAction : RecoveryAction;
    }

    part def RecoveryTime {
        port in recoveryTime : Duration;
    }

    part def AlignmentAndPhasingSystem {
        port in disturbanceDetectionUnit : DisturbanceDetection;
        port in loopStatusUnit : LoopStatus;
        port in recoveryActionUnit : RecoveryAction;
        port in recoveryTimeUnit : RecoveryTime;
    }

    part def DisturbanceDetectionUnit {
        attribute disturbanceDetected : Boolean;
    }

    part def LoopStatusUnit {
        attribute loopState : LoopState;
    }

    part def RecoveryActionUnit {
        attribute recoveryTrigger : RecoveryTrigger;
        attribute recoveryAction : RecoveryAction;
    }

    part def RecoveryTimeUnit {
        attribute recoveryTime : Duration;
    }

    enum LoopState {
        stable,
        unstable,
        recovering
    }

    enum RecoveryTrigger {
        manual,
        automatic
    }

    enum RecoveryAction {
        noAction,
        reset,
        reinitialize
    }
}
