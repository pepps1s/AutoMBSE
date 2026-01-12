package IBD_SUB05_007_AO_Safety_LGS_Interlock_Wiring {

  // Diagram: IBD-SUB05-007
  // Title: IBD SUB05: AO Safety & Laser Guide Star Interlock Wiring (If LGS Enabled)
  // Dependencies: BDD-SUB05-002, BDD-SUB09-001, RD-F-008, IBD-ROOT-015

  //--------------------------------------------
  // Item (flow) definitions
  //--------------------------------------------
  item def LgsEnableRequest {
    attribute requested : Boolean;
    attribute reason : String;
  }

  item def LgsPermission {
    attribute permitted : Boolean;
    attribute constraints : String;
  }

  item def InhibitSignal {
    attribute active : Boolean;
    attribute cause : String;
  }

  item def SafetyEvent {
    attribute eventId : String;
    attribute severity : String;
    attribute message : String;
  }

  //--------------------------------------------
  // Port definitions (named to match expected subsystem port intent)
  //--------------------------------------------
  port def SafetyCommandPort {
    in lgsEnableReq : LgsEnableRequest;
  }

  port def SafetyTelemetryPort {
    out lgsPermission : LgsPermission;
    out tripActive : Boolean;
  }

  port def InhibitPort {
    out inhibit : InhibitSignal;
  }

  port def AuditEventPort {
    in eventIn : SafetyEvent;
  }

  port def AoLaserControlPort {
    out lgsEnableReq : LgsEnableRequest;
    in lgsPermission : LgsPermission;
  }

  port def AoSafetyInhibitPort {
    in inhibitIn : InhibitSignal;
  }

  port def AoEventPort {
    out safetyEvent : SafetyEvent;
  }

  //--------------------------------------------
  // Part definitions (ensure all referenced safety/laser ports exist on AO and Safety)
  //--------------------------------------------
  part def AdaptiveOpticsSubsystem {
    // Existing-ish generic AO ports (kept minimal; focus is safety/laser wiring)
    port ocsCmdIn;
    port statusOut;
    port telemetryOut;
    port pwrIn;
    port net;

    // Safety/LGS ports used by this IBD
    port lgsIfc : AoLaserControlPort;
    port inhibitIn : AoSafetyInhibitPort;
    port eventOut : AoEventPort;

    // Internal state relevant to gating (illustrative)
    attribute aoEmissionAllowed : Boolean = false;
    attribute aoSafeMode : Boolean = true;
  }

  part def SafetySubsystem {
    // Common Safety ports
    port cmdIn : SafetyCommandPort;
    port tlmOut : SafetyTelemetryPort;
    port inhibitOut : InhibitPort;

    // Event logging / audit hookup
    port loggerOut : (out eventOut : SafetyEvent);
  }

  part def SafetyEventLogger {
    port auditIn : AuditEventPort;
  }

  //--------------------------------------------
  // IBD: internal wiring
  //--------------------------------------------
  package IBD_Structure {

    part ao : AdaptiveOpticsSubsystem;
    part safety : SafetySubsystem;
    part audit : SafetyEventLogger;

    //
    // (1) LGS enable/permission flow gated by Safety
    // AO requests LGS enable; Safety returns permission (and trip state).
    //
    connect ao.lgsIfc.lgsEnableReq -> safety.cmdIn.lgsEnableReq;
    connect safety.tlmOut.lgsPermission -> ao.lgsIfc.lgsPermission;

    //
    // (2) Inhibit path: Safety can disable AO emission or force safe mode
    // Safety inhibit drives AO inhibit input; AO should use this to force safe state.
    //
    connect safety.inhibitOut.inhibit -> ao.inhibitIn.inhibitIn;

    //
    // (3) Safety event logging connector tied to audit/event path
    // AO emits safety-relevant events; Safety forwards to audit logger.
    //
    connect ao.eventOut.safetyEvent -> safety.loggerOut.eventOut;
    connect safety.loggerOut.eventOut -> audit.auditIn.eventIn;

    //
    // (4) Explicit intent/constraints (lightweight, model-checkable hints)
    //
    constraint def LgsPermissionGatedBySafety {
      // If trip is active OR inhibit is active, permission must be false
      // and AO must be in safe mode / emission not allowed.
      // (Expressed as constraints over visible flow/state signals.)
      // NOTE: These are declarative checks; implementations map them to behaviors.
      assert (safety.tlmOut.tripActive == true) implies (safety.tlmOut.lgsPermission.permitted == false);
      assert (safety.inhibitOut.inhibit.active == true) implies (safety.tlmOut.lgsPermission.permitted == false);
      assert (safety.inhibitOut.inhibit.active == true) implies (ao.aoSafeMode == true);
      assert (safety.tlmOut.lgsPermission.permitted == false) implies (ao.aoEmissionAllowed == false);
    }

    constraint lgsSafetyGate : LgsPermissionGatedBySafety;

  }

}
