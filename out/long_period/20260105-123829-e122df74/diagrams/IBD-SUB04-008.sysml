```sysml
package IBD_SUB04_008_APS_Safety_Interlock_Consumption {

  // ========== Items (explicit signals/flows) ==========

  item def InhibitSignal {
    attribute inhibitActive : Boolean;
    attribute reason : String;
    attribute source : String;
  }

  item def SafeStateStatus {
    attribute safeStateConfirmed : Boolean;
    attribute apsReady : Boolean;
    attribute lastTrip : String;
  }

  item def MotionInhibitSignal :> InhibitSignal {
    attribute axisGroup : String;
  }

  item def AccessInterlockSignal :> InhibitSignal {
    attribute zoneId : String;
    attribute doorClosed : Boolean;
  }

  item def LaserInterlockSignal :> InhibitSignal {
    attribute lgsEnabled : Boolean;
    attribute beamPermit : Boolean;
  }

  item def ApsActuationCommand {
    attribute enableActuation : Boolean;
    attribute mode : String;
  }

  item def ApsActuationStatus {
    attribute actuationAllowed : Boolean;
    attribute inhibited : Boolean;
    attribute inhibitLatched : Boolean;
  }

  // ========== Port definitions (aligned with Safety / APS inhibit + status patterns) ==========

  port def MotionInhibitPort {
    out item motionInhibit : MotionInhibitSignal;
  }

  port def AccessInterlockPort {
    out item accessInterlock : AccessInterlockSignal;
  }

  port def OpsConstraintPort {
    out item laserInterlock : LaserInterlockSignal;
  }

  port def SafetyTelemetryPort {
    in item safeState : SafeStateStatus;
  }

  port def InhibitPort {
    in item inhibit : InhibitSignal;
  }

  port def ApsSafetyStatusPort {
    out item safeState : SafeStateStatus;
  }

  port def CommandPort {
    in item actCmd : ApsActuationCommand;
  }

  port def TelemetryPort {
    out item actStatus : ApsActuationStatus;
  }

  // ========== Part definitions ==========

  part def SafetySubsystem {
    port motionInhibitOut : MotionInhibitPort;
    port accessIfc : AccessInterlockPort;
    port opsIfc : OpsConstraintPort;
    port tlmOut : SafetyTelemetryPort; // receives confirmations from APS (status path back)
  }

  part def AlignmentAndPhasingSystem {
    port inhibitIn : InhibitPort;           // aggregated inhibit input into
