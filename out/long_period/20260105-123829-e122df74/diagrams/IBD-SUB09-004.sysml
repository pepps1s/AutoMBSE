package IBD_SUB09_004_Safety_PLC_Internal_IO_Wiring {

  // Dependencies (referenced definitions; ports/types must already exist in BDD-SUB09-001)
  import BDD_SUB09_001_Safety_TMT::*;
  import TMT_RD_F_008_Safety_Interlocks_EmergencyResponse::*;
  import TMT_RD_NF_006_SafetyCompliance_StandardsPoliciesAlignment::*;
  import IBD_ROOT_011_UnifiedTelemetryBackbone::*;

  // IBD: Safety PLC internal IO wiring (E-Stop, Interlocks, Inhibits, Alarms)
  part ibdSub09_004_SafetySubsystem : SafetySubsystem {

    // Core controller
    part safetyPLC : SafetyPLC;

    // Explicit IO parts (inputs)
    part estopLoopIO : customComponent {
      port estopLoop : EStopLoopPort;
    }

    part accessInterlockIO : customComponent {
      port accessIfc : AccessInterlockPort;
    }

    part discreteInterlockIO : customComponent {
      port ioInterlock : InterlockDiscreteIO;
    }

    // Outputs (separated inhibit vs alarm)
    part inhibitOutputs : customComponent {
      port inhibitOut : MotionInhibitPort;
    }

    part alarmAnnunciator : customComponent {
      port alarmOut : AlarmAnnunciatorPort;
    }

    // Event / logging path to audit/safety event record
    part logger : customComponent {
      port tlmIn : SafetyTelemetryPort;
    }

    part auditLog : AuditLog;
    part safetyEventRecord : SafetyEventRecord;

    // --- Internal wiring/connectors ---

    // Inputs into PLC (E-Stop and access/interlock discrete IO)
    connect estopLoopIO.estopLoop to safetyPLC.tripIn;
    connect accessInterlockIO.accessIfc to safetyPLC.tripIn;
    connect discreteInterlockIO.ioInterlock to safetyPLC.tripIn;

    // PLC command/telemetry exposure (internal tie to subsystem ports, if present in BDD)
    connect cmdIn to safetyPLC.cmdIn;
    connect safetyPLC.tlmOut to tlmOut;

    // Separate output flows: inhibit outputs vs alarm outputs
    connect safetyPLC.inhibitOut to inhibitOutputs.inhibitOut;
    connect safetyPLC.alarmOut to alarmAnnunciator.alarmOut;
    connect alarmAnnunciator.alarmOut to alarmOut;

    // Event/logging connector feeding audit/safety event record path
    connect safetyPLC.tlmOut to logger.tlmIn;
    connect logger to safetyEventRecord;
    connect safetyEventRecord to auditLog;

    // Optional: forward safety telemetry to unified telemetry backbone (root IBD dependency)
    connect tlmOut to UnifiedTelemetryBackbone::telemetryOutPort;
    connect alarmOut to UnifiedTelemetryBackbone::alarmOutPort;
  }

}
