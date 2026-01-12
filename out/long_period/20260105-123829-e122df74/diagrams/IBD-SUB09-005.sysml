package IBD_SUB09_005_SafetyEvent_AuditLog_Wiring_ComplianceEvidence_v2 {

  import ScalarValues::*;

  // =========================
  // Items (flows)
  // =========================
  item def UtcTimestamp {
    attribute epochMillis : Integer;
    attribute iso8601Utc  : String;
  }

  item def TraceCorrelationId {
    attribute value : String;
  }

  item def SafetyEvent {
    attribute eventType          : String;
    attribute severityLevel      : String;
    attribute impactedSubsystem  : String;
    attribute requiresReporting  : Boolean;
    attribute correlationId      : TraceCorrelationId;
  }

  item def AuditLogEntry {
    attribute action             : String;
    attribute actorId            : String;
    attribute changeReason       : String;
    attribute correlationId      : TraceCorrelationId;
  }

  item def SafetyEventRecord {
    attribute eventId            : String;
    attribute event              : SafetyEvent;
    attribute eventTimestampUtc  : UtcTimestamp;
    attribute traceCorrelationId : TraceCorrelationId;
  }

  item def AuditLog {
    attribute logId              : String;
    attribute entry              : AuditLogEntry;
    attribute logTimestampUtc    : UtcTimestamp;
    attribute traceCorrelationId : TraceCorrelationId;
    attribute timeSource         : String;
  }

  // =========================
  // Ports (defined here to avoid referencing undefined ports)
  // =========================
  port def TimeInPort {
    in timeUtc : UtcTimestamp;
  }

  port def TimeOutPort {
    out timeUtc : UtcTimestamp;
  }

  port def SafetyEventOutPort {
    out safetyEventRecord : SafetyEventRecord;
  }

  port def AuditLogOutPort {
    out auditLog : AuditLog;
  }

  port def SafetyEventInPort {
    in safetyEventRecord : SafetyEventRecord;
  }

  port def AuditLogInPort {
    in auditLog : AuditLog;
  }

  // Optional evidence query ports (retention/search goal support)
  port def EvidenceQueryPort {
    in query : String;
    out result : String;
  }

  // =========================
  // Parts (key elements for the IBD)
  // =========================
  part def TimeSourceService {
    attribute timeSourceId : String;
    port timeOut : TimeOutPort;
  }

  part def SafetyLogger {
    // Logger emits both audit logs and safety event records, and consumes time
    port timeIn        : TimeInPort;
    port auditLogOut   : AuditLogOutPort;
    port safetyEvtOut  : SafetyEventOutPort;

    // Local policy flags (can mirror RD-NF-006 intent)
    attribute includesActorId       : Boolean = true;
    attribute includesChangeReason  : Boolean = true;
  }

  part def AuditLogStoreService {
    // Sink/service that persists audit logs as immutable compliance evidence
    port auditLogIn   : AuditLogInPort;
    port queryPort    : EvidenceQueryPort;

    // Immutable storage/retention attributes (instantiated)
    attribute immutableStorageEnabled : Boolean = true;
    attribute tamperEvident           : Boolean = true;
    attribute retentionDays           : Integer;
    attribute searchable              : Boolean = true;
  }

  part def SafetyEventRecordStoreService {
    // Sink/service that persists safety event records as compliance evidence
    port safetyEvtIn  : SafetyEventInPort;
    port queryPort    : EvidenceQueryPort;

    // Immutable storage/retention attributes (instantiated)
    attribute immutableStorageEnabled : Boolean = true;
    attribute tamperEvident           : Boolean = true;
    attribute retentionDays           : Integer;
    attribute searchable              : Boolean = true;
  }

  // SUB09 boundary (Safety subsystem internal wiring focus)
  part def SafetySubsystem_Sub09 {
    // Internal parts
    part timeSource  : TimeSourceService;
    part logger      : SafetyLogger;
    part auditSink   : AuditLogStoreService;
    part safetySink  : SafetyEventRecordStoreService;

    // Instantiate retention policy consistent with compliance evidence v2
    // (values can be refined/derived by RD-NF-006 constraints)
    attribute auditRetentionDays  : Integer = 3650; // 10 years
    attribute eventRetentionDays  : Integer = 3650; // 10 years

    // Bind instantiated policy values into sinks
    constraint { auditSink.retentionDays == auditRetentionDays; }
    constraint { safetySink.retentionDays == eventRetentionDays; }

    // =========================
    // Connections (IBD wiring)
    // =========================
    // Timestamping path: logger consumes authoritative UTC time
    connect timeSource.timeOut to logger.timeIn;

    // Evidence flows: logger emits records/logs into immutable stores
    connect logger.auditLogOut to auditSink.auditLogIn;
    connect logger.safetyEvtOut to safetySink.safetyEvtIn;
  }

  // =========================
  // IBD instance for diagram_id: IBD-SUB09-005
  // =========================
  package IBD_SUB09_005 {
    // Title: IBD SUB09: Safety Event + Audit Log Wiring (Compliance Evidence v2)
    // Goal: Model internal connections that generate and persist audit logs and safety event records.

    part ibdSub09_005_SafetyEvidenceWiring : SafetySubsystem_Sub09 {
      // Instance-level time source identity to satisfy "time source/timestamping path" intent
      timeSource.timeSourceId = "TMT-UTC-Authoritative-Time";

      // Ensure stores are explicitly tamper-evident & immutable (instantiated flags already true in defs)
      auditSink.immutableStorageEnabled = true;
      auditSink.tamperEvident = true;

      safetySink.immutableStorageEnabled = true;
      safetySink.tamperEvident = true;
    }
  }

}
