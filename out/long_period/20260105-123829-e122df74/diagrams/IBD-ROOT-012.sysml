package IBD_ROOT_012_CommandRouting_AuthorizationGates {

  import ScalarValues::*;
  import Quantities::*;

  // ===== Items / Signals =====
  item def OperatorCommand;
  item def RoutedCommand;
  item def CommandAck;
  item def AuthToken;
  item def SafetyInhibit;        // true => inhibit asserted
  item def SafetyGateStatus;     // gate state/health
  item def TimingStamp;

  // ===== Common Ports =====
  port def OperatorCommandPort {
    out item cmd : OperatorCommand;
    in  item ack : CommandAck;
  }

  port def CommandPort {
    in  item cmd : RoutedCommand;
    out item ack : CommandAck;
  }

  port def RouterIngressPort {
    in  item cmd : OperatorCommand;
    in  item token : AuthToken;
    out item ack : CommandAck;
  }

  port def RouterEgressPort {
    out item routedCmd : RoutedCommand;
    in  item ack : CommandAck;
  }

  port def SafetyInhibitPort {
    in  item inhibit : SafetyInhibit;
    out item status  : SafetyGateStatus;
  }

  // ===== Parts (Definitions) =====
  part def OperatorConsole {
    port opCmd : OperatorCommandPort;
  }

  part def OCS_CommandRouter {
    port inFromOperator : RouterIngressPort;

    port toOptics   : RouterEgressPort;
    port toMount    : RouterEgressPort;
    port toEnclosure: RouterEgressPort;
    port toAPS      : RouterEgressPort;
    port toAO       : RouterEgressPort;
    port toInstr    : RouterEgressPort;

    // RD-F-006 semantics: authorization delay on command acceptance/routing
    attribute authorizationDelay_s : Real;   // seconds (nominal authorization check time)
    attribute routingLatency_s     : Real;   // seconds (internal distribution latency)
  }

  part def SafetyGate {
    port inCmd  : RouterEgressPort;
    port outCmd : RouterEgressPort;
    port inhibitIfc : SafetyInhibitPort;

    // Fail-closed behavior
    attribute failClosed : Boolean = true;
    attribute authorized : Boolean = false;

    // RD-NF-005 / RD-F-006 style timing semantics: gating/decision time
    attribute gateDecisionTime_s : Real;     // seconds
  }

  // Subsystem endpoints (each MUST have a corresponding command port)
  part def TelescopeOpticalSystem {
    port cmdIn : CommandPort;
  }

  part def MountStructure {
    port cmdIn : CommandPort;
  }

  part def EnclosureSiteInfrastructure {
    port cmdIn : CommandPort;
  }

  part def AlignmentPhasingSystem {
    port cmdIn : CommandPort;
  }

  part def AdaptiveOpticsSubsystem {
    port cmdIn : CommandPort;
  }

  part def InstrumentsSubsystem {
    port cmdIn : CommandPort;
  }

  // Safety subsystem provides inhibit (per Safety viewpoint; abstracted here)
  part def SafetySubsystem {
    port motionInhibitOut : SafetyInhibitPort;
  }

  // ===== IBD Root: Command Routing + Authorization Gates =====
  part def TMT_CommandRouting_AuthorizationGates {

    part operator : OperatorConsole;
    part ocs      : OCS_CommandRouter;

    part safety   : SafetySubsystem;

    // Per-target safety gates (explicit gating points)
    part gateOptics    : SafetyGate;
    part gateMount     : SafetyGate;
    part gateEnclosure : SafetyGate;
    part gateAPS       : SafetyGate;
    part gateAO        : SafetyGate;
    part gateInstr     : SafetyGate;

    // Command endpoints
    part optics   : TelescopeOpticalSystem;
    part mount    : MountStructure;
    part enclosure: EnclosureSiteInfrastructure;
    part aps      : AlignmentPhasingSystem;
    part ao       : AdaptiveOpticsSubsystem;
    part instr    : InstrumentsSubsystem;

    // Instantiate representative timing attributes (RD-NF-005 / RD-F-006 semantics)
    attribute ocs_authorizationDelay_s : Real = 0.250;
    attribute ocs_routingLatency_s     : Real = 0.050;

    attribute gateDecisionTime_s       : Real = 0.020;

    // Bind instance timing to router/gates
    bind ocs.authorizationDelay_s = ocs_authorizationDelay_s;
    bind ocs.routingLatency_s     = ocs_routingLatency_s;

    bind gateOptics.gateDecisionTime_s    = gateDecisionTime_s;
    bind gateMount.gateDecisionTime_s     = gateDecisionTime_s;
    bind gateEnclosure.gateDecisionTime_s = gateDecisionTime_s;
    bind gateAPS.gateDecisionTime_s       = gateDecisionTime_s;
    bind gateAO.gateDecisionTime_s        = gateDecisionTime_s;
    bind gateInstr.gateDecisionTime_s     = gateDecisionTime_s;

    // ---- Operator -> OCS ingress ----
    connect operator.opCmd.cmd -> ocs.inFromOperator.cmd;
    // (Auth token arrives via a separate channel; modeled as a routed token input)
    // If you want a concrete source, model it as part AuthService; kept abstract here.

    // ---- OCS egress -> Safety gates (gating connectors) ----
    connect ocs.toOptics.routedCmd    -> gateOptics.inCmd.routedCmd;
    connect ocs.toMount.routedCmd     -> gateMount.inCmd.routedCmd;
    connect ocs.toEnclosure.routedCmd -> gateEnclosure.inCmd.routedCmd;
    connect ocs.toAPS.routedCmd       -> gateAPS.inCmd.routedCmd;
    connect ocs.toAO.routedCmd        -> gateAO.inCmd.routedCmd;
    connect ocs.toInstr.routedCmd     -> gateInstr.inCmd.routedCmd;

    // ---- Safety inhibit affects command enable/disable (explicit gating influence) ----
    connect safety.motionInhibitOut.inhibit -> gateOptics.inhibitIfc.inhibit;
    connect safety.motionInhibitOut.inhibit -> gateMount.inhibitIfc.inhibit;
    connect safety.motionInhibitOut.inhibit -> gateEnclosure.inhibitIfc.inhibit;
    connect safety.motionInhibitOut.inhibit -> gateAPS.inhibitIfc.inhibit;
    connect safety.motionInhibitOut.inhibit -> gateAO.inhibitIfc.inhibit;
    connect safety.motionInhibitOut.inhibit -> gateInstr.inhibitIfc.inhibit;

    // ---- Safety gates -> Subsystem command endpoints ----
    connect gateOptics.outCmd.routedCmd    -> optics.cmdIn.cmd;
    connect gateMount.outCmd.routedCmd     -> mount.cmdIn.cmd;
    connect gateEnclosure.outCmd.routedCmd -> enclosure.cmdIn.cmd;
    connect gateAPS.outCmd.routedCmd       -> aps.cmdIn.cmd;
    connect gateAO.outCmd.routedCmd        -> ao.cmdIn.cmd;
    connect gateInstr.outCmd.routedCmd     -> instr.cmdIn.cmd;

    // ---- Acks back (subsystems -> gates -> OCS -> operator) ----
    connect optics.cmdIn.ack   -> gateOptics.outCmd.ack;
    connect mount.cmdIn.ack    -> gateMount.outCmd.ack;
    connect enclosure.cmdIn.ack-> gateEnclosure.outCmd.ack;
    connect aps.cmdIn.ack      -> gateAPS.outCmd.ack;
    connect ao.cmdIn.ack       -> gateAO.outCmd.ack;
    connect instr.cmdIn.ack    -> gateInstr.outCmd.ack;

    connect gateOptics.inCmd.ack    -> ocs.toOptics.ack;
    connect gateMount.inCmd.ack     -> ocs.toMount.ack;
    connect gateEnclosure.inCmd.ack -> ocs.toEnclosure.ack;
    connect gateAPS.inCmd.ack       -> ocs.toAPS.ack;
    connect gateAO.inCmd.ack        -> ocs.toAO.ack;
    connect gateInstr.inCmd.ack     -> ocs.toInstr.ack;

    connect ocs.inFromOperator.ack -> operator.opCmd.ack;

    // ===== Fail-Closed Intent (behavioral constraint as invariant) =====
    // If inhibit is asserted and failClosed is true, gate must not authorize forwarding.
    // (Modeled as an invariant-style constraint placeholder; detailed action semantics go in an AD/SM.)
    constraint def FailClosedInvariant(g : SafetyGate, inh : SafetyInhibit) {
      // Informal invariant:
      // (g.failClosed && inh) => (g.authorized == false)
    }

    constraint failClosed_optics    : FailClosedInvariant(gateOptics,    gateOptics.inhibitIfc.inhibit);
    constraint failClosed_mount     : FailClosedInvariant(gateMount,     gateMount.inhibitIfc.inhibit);
    constraint failClosed_enclosure : FailClosedInvariant(gateEnclosure, gateEnclosure.inhibitIfc.inhibit);
    constraint failClosed_aps       : FailClosedInvariant(gateAPS,       gateAPS.inhibitIfc.inhibit);
    constraint failClosed_ao        : FailClosedInvariant(gateAO,        gateAO.inhibitIfc.inhibit);
    constraint failClosed_instr     : FailClosedInvariant(gateInstr,     gateInstr.inhibitIfc.inhibit);
  }

  // Diagram root part (for the IBD view)
  part tmtRootCommandRouting : TMT_CommandRouting_AuthorizationGates;

}
