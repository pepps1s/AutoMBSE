package IBD_OPS_006_FaultAbortTriggerWiring_v2 {

  /* Diagram metadata */
  doc /*diagram_id*/ "IBD-OPS-006";
  doc /*type*/ "IBD";
  doc /*title*/ "IBD OPS: Fault Abort Trigger Wiring (Power/Network/Motion) v2";
  doc /*goal*/ "Re-issue wiring of fault conditions leading to abort and safe-state behavior across fault classes.";
  doc /*dependencies*/ "RD-F-002, RD-OPS-003, IBD-ROOT-014, IBD-ROOT-010, IBD-ROOT-015, IBD-SUB10-005";

  package Types {
    enum def FaultClass { Power; Network; Motion; }
    enum def SeverityLevel { Info; Warning; Critical; }

    value type FaultCode : Integer;

    item def PowerFaultTrigger {
      attribute faultClass : FaultClass = FaultClass::Power;
      attribute code : FaultCode;
      attribute severity : SeverityLevel;
      attribute sourceId : String;
      attribute detail : String;
    }

    item def NetworkFaultTrigger {
      attribute faultClass : FaultClass = FaultClass::Network;
      attribute code : FaultCode;
      attribute severity : SeverityLevel;
      attribute sourceId : String;
      attribute detail : String;
    }

    item def MotionFaultTrigger {
      attribute faultClass : FaultClass = FaultClass::Motion;
      attribute code : FaultCode;
      attribute severity : SeverityLevel;
      attribute sourceId : String;
      attribute detail : String;
    }

    item def AbortDecision {
      attribute abortRequested : Boolean;
      attribute abortReasonClass : FaultClass;
      attribute abortReasonCode : FaultCode;
      attribute abortReasonDetail : String;
      attribute decidedBy : String;
    }

    item def InhibitSignal {
      attribute inhibitActive : Boolean;
      attribute reason : String;
    }

    item def LogEvent {
      attribute topic : String;
      attribute severity : SeverityLevel;
      attribute message : String;
      attribute sourceId : String;
    }
  }

  package Ports {
    import Types::*;

    port def PowerFaultSensePort { out powerFault : PowerFaultTrigger; }
    port def NetworkFaultSensePort { out networkFault : NetworkFaultTrigger; }
    port def MotionFaultSensePort { out motionFault : MotionFaultTrigger; }

    port def AbortDecisionInPort {
      in powerFault : PowerFaultTrigger;
      in networkFault : NetworkFaultTrigger;
      in motionFault : MotionFaultTrigger;
    }

    port def AbortDecisionOutPort { out abort : AbortDecision; }

    port def InhibitOutPort { out inhibit : InhibitSignal; }
    port def InhibitInPort { in inhibit : InhibitSignal; }

    port def LogOutPort { out log : LogEvent; }
    port def LogInPort { in log : LogEvent; }
  }

  package Parts {
    import Types::*;
    import Ports::*;

    part def PowerFaultSensor {
      port faultOut : PowerFaultSensePort;
      attribute sensorId : String;
    }

    part def NetworkFaultMonitor {
      port faultOut : NetworkFaultSensePort;
      attribute monitorId : String;
    }

    part def MotionController {
      port faultOut : MotionFaultSensePort;
      port inhibitIn : InhibitInPort;
      attribute controllerId : String;
    }

    part def AbortDecisionUnit {
      port faultsIn : AbortDecisionInPort;
      port abortOut : AbortDecisionOutPort;
      attribute unitId : String;
      attribute policy : String;
    }

    part def SafeStateInhibitDistributor {
      port abortIn : AbortDecisionOutPort;
      port inhibitOut : InhibitOutPort;
      attribute distributorId : String;
    }

    part def OpsLog {
      port logIn : LogInPort;
      attribute logName : String;
    }

    part def DataLog {
      port logIn : LogInPort;
      attribute logName : String;
    }

    part def AbortTelemetryLogger {
      port abortIn : AbortDecisionOutPort;
      port logOut : LogOutPort;
      attribute loggerId : String;
      attribute topicBase : String;
    }
  }

  package Structure {
    import Parts::*;
    import Ports::*;
    import Types::*;

    part def TMT_FaultAbortWiring {
      /* Fault sources (required by acceptance: no trigger without source part) */
      part powerFaultSensor : PowerFaultSensor {
        sensorId = "PWR-SENSE-01";
      }

      part networkFaultMonitor : NetworkFaultMonitor {
        monitorId = "NET-MON-01";
      }

      part motionController : MotionController {
        controllerId = "MOTION-CTRL-01";
      }

      /* Abort decision */
      part abortDecision : AbortDecisionUnit {
        unitId = "ABORT-DECIDE-01";
        policy = "Abort on any Critical fault; otherwise operator-mediated.";
      }

      /* Safe-state inhibit distribution */
      part inhibitDistributor : SafeStateInhibitDistributor {
        distributorId = "INHIBIT-DIST-01";
      }

      /* Logging / telemetry of abort reason */
      part abortLogger : AbortTelemetryLogger {
        loggerId = "ABORT-LOG-01";
        topicBase = "tmt.ops.abort";
      }

      part opsLog : OpsLog {
        logName = "OpsLog";
      }

      part dataLog : DataLog {
        logName = "DataLog";
      }

      /* Internal wiring (IBD connectors) */

      // (1) Fault triggers routed into abort decision (explicit power/network/motion signals)
      connect pwrFaultToAbort : powerFaultSensor.faultOut :: abortDecision.faultsIn
        { item powerFault = powerFault; }

      connect netFaultToAbort : networkFaultMonitor.faultOut :: abortDecision.faultsIn
        { item networkFault = networkFault; }

      connect motionFaultToAbort : motionController.faultOut :: abortDecision.faultsIn
        { item motionFault = motionFault; }

      // (2) Abort decision routed to safe-state inhibit distribution
      connect abortToInhibitDist : abortDecision.abortOut :: inhibitDistributor.abortIn;

      // (3) Safe-state inhibit distributed to motion controller inhibit input (representative consumer)
      connect inhibitToMotion : inhibitDistributor.inhibitOut :: motionController.inhibitIn;

      // (4) Abort reason logged/telemetry to ops + data logs
      connect abortToLogger : abortDecision.abortOut :: abortLogger.abortIn;

      connect loggerToOpsLog : abortLogger.logOut :: opsLog.logIn;
      connect loggerToDataLog : abortLogger.logOut :: dataLog.logIn;
    }
  }

  /* View root */
  import Structure::*;
  part ibdRoot : TMT_FaultAbortWiring;
}
