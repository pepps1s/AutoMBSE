package IBD_SUB03_013_EnclosureMechanisms_InternalWiring {

  // Dependencies (imported for type/port consistency)
  import BDD_SUB03_003_EnclosureDome_SiteInfrastructure::*;
  import BDD_SUB09_001_Safety_TMT::*;
  import BDD_SUB10_001_OperationsMaintenance::*;
  import IBD_ROOT_012_CommandRouting_AuthorizationGates::*;
  import IBD_ROOT_015::*;
  import RD_F_008_SafetyInterlocksEmergencyResponse::*;

  //
  // Local signal/item typings (lightweight; ports themselves come from referenced BDDs/IBDs)
  //
  item def MechanismState;
  item def MechanismFault;
  item def MechanismAlarm;

  //
  // Local controller wrapper to bind existing ports (must exist in referenced BDDs)
  //
  part def EnclosureMechanismController {
    // From BDD_SUB10_001_OperationsMaintenance
    port cmdIn  : CommandPort;
    port tlmOut : TelemetryPort;
    port status : StatusPort;

    // From BDD_SUB09_001_Safety_TMT and/or IBD_ROOT_015 MotionCapableSubsystem
    port inhibitIn : SafetyInterlockPort;
    port alarmOut  : AlarmAnnunciatorPort;
  }

  //
  // Mechanism equipment (actuators/sensors) is modeled as InfrastructureElement for wiring containment.
  //
  part def EnclosureMechanismAssembly : InfrastructureElement {
    part controller : EnclosureMechanismController;

    // Infrastructure-side control/data plumbing (from BDD_SUB03_003 ports)
    port controlBus : ControlPort;
    port dataIfc    : DataPort;
    port pwr        : PowerPort;
    port envSense   : EnvPort;
  }

  //
  // Safety/Ops constraints (local, connected to inhibit flows)
  //
  constraint def MotionMustBeInhibitedOnTrip {
    in  tripActive : Boolean;
    out inhibitReq : Boolean;
    assert inhibitReq == tripActive;
  }

  //
  // IBD: Enclosure mechanisms internal wiring (rotation / shutters / wind screens)
  //
  part def IBD_SUB03_013_EnclosureMechanisms_InternalWiring {

    // Upstream command routing / authorization (from IBD_ROOT_012)
    part cmdRouter : OCS_CommandRouter;

    // Enclosure subsystem entry point (from IBD_ROOT_012: EnclosureSiteInfrastructure has cmdIn)
    part enclosure : EnclosureSiteInfrastructure;

    // Safety inhibit distribution (from IBD_ROOT_015)
    part safetyInhibit : SafetyInhibitDistribution;
    part safety        : SafetySubsystem;

    // Ops / OCS operator interface endpoints (from BDD_SUB10_001)
    part opsConsole : OperatorConsole;

    // Mechanism assemblies (SUB03)
    part rotationDrive : EnclosureMechanismAssembly;
    part shutterSystem : EnclosureMechanismAssembly;
    part windScreens   : EnclosureMechanismAssembly;

    // Constraint instances (one per mechanism)
    constraint rotTripRule : MotionMustBeInhibitedOnTrip;
    constraint shtTripRule : MotionMustBeInhibitedOnTrip;
    constraint wndTripRule : MotionMustBeInhibitedOnTrip;

    //
    // Command routing to enclosure and then to local mechanism controllers
    //
    connect cmdRouter.toEnclosure  to enclosure.cmdIn;

    // Controller command wiring (all are CommandPort instances)
    connect enclosure.cmdIn        to rotationDrive.controller.cmdIn;
    connect enclosure.cmdIn        to shutterSystem.controller.cmdIn;
    connect enclosure.cmdIn        to windScreens.controller.cmdIn;

    //
    // Safety inhibit distribution reaching mechanism controllers
    //
    connect safetyInhibit.safetySubsystem  to safety;
    connect safety.motionInhibitOut        to safetyInhibit.motionSubsystem.inhibitIn;

    // Bind each mechanism as a motion-capable subsystem target of inhibit
    connect safety.motionInhibitOut        to rotationDrive.controller.inhibitIn;
    connect safety.motionInhibitOut        to shutterSystem.controller.inhibitIn;
    connect safety.motionInhibitOut        to windScreens.controller.inhibitIn;

    //
    // Telemetry/state to OCS/Ops; alarms where relevant
    //
    connect rotationDrive.controller.tlmOut   to opsConsole.opCmd;      // ops console is the aggregation point
    connect shutterSystem.controller.tlmOut   to opsConsole.opCmd;
    connect windScreens.controller.tlmOut     to opsConsole.opCmd;

    connect rotationDrive.controller.status   to opsConsole.opCmd;
    connect shutterSystem.controller.status   to opsConsole.opCmd;
    connect windScreens.controller.status     to opsConsole.opCmd;

    // Alarms annunciated via Safety subsystem annunciator/logger path
    connect rotationDrive.controller.alarmOut to safety.annunciator.alarmOut;
    connect shutterSystem.controller.alarmOut to safety.annunciator.alarmOut;
    connect windScreens.controller.alarmOut   to safety.annunciator.alarmOut;

    //
    // Local safety/ops constraint hookups (tripActive -> inhibit requirement)
    //
    // Trip source: SafetySubsystem.tripActive (from BDD_SUB09_001)
    bind rotTripRule.tripActive = safety.tripActive;
    bind shtTripRule.tripActive = safety.tripActive;
    bind wndTripRule.tripActive = safety.tripActive;

    // Inhibit requirement mapped to each controller inhibitIn presence (boolean proxy)
    // (Model intent: when tripActive, inhibit must be asserted at each inhibitIn)
    bind rotTripRule.inhibitReq = safety.tripActive;
    bind shtTripRule.inhibitReq = safety.tripActive;
    bind wndTripRule.inhibitReq = safety.tripActive;

  }

}
