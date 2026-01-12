package AD_OPS_FaultDetection {

    item def FaultIndication;
    item def FaultClass;
    item def SafeStateCommand;
    item def RecoveryMethod;
    item def Logging;

    action faultDetectionAction {
        in item faultIndication : FaultIndication;
        in item faultClass : FaultClass;

        first start;
        then action classifyFault;
        then action transitionToSafeState;
        then action notifyOperator;
        then action attemptRecovery;
        then action logAudit;
        then done;
    }

    action classifyFault {
        first start;
        then action 'classify fault based on faultClass';
        then done;
    }

    action transitionToSafeState {
        in item safeStateCommand : SafeStateCommand;

        first start;
        then action 'send safe state command';
        then action 'verify safe state is maintained';
        then done;
    }

    action notifyOperator {
        first start;
        then action 'send operator notification';
        then done;
    }

    action attemptRecovery {
        in item recoveryMethod : RecoveryMethod;

        first start;
        then action 'attempt recovery via selected recoveryMethod';
        then action 'check if recovery successful';
        then action 'retry if automatic recovery failed';
        then action 'escalate to maintenance if manual intervention required';
        then done;
    }

    action logAudit {
        in item logging : Logging;

        first start;
        then action 'log fault and recovery actions for audit purposes';
        then done;
    }

}
