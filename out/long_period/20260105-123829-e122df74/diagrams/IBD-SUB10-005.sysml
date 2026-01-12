```sysml
package IBD_SUB10_005_FaultHandlingWiring {

  // ============================================================
  // IBD-SUB10-005
  // IBD SUB10: Fault Handling Wiring (Detect→Annunciate→Safe State→Recover)
  // ============================================================

  // ----------------------------
  // Type Library (items/attrs)
  // ----------------------------
  package Types {

    // Fault classes (acceptance: at least three; includes motion, power, network, thermal)
    item def MotionFault;
    item def PowerFault;
    item def NetworkFault;
    item def ThermalFault;

    // Derived/normalized fault event for downstream wiring
    item def FaultEvent;

    // Alarm + logging / evidence
    item def AlarmSignal;
    item def LogRecord;
    item def EvidencePointer;

    // Safety / inhibit / safe-state
    item def SafeStateRequest;
    item def InhibitSignal;

    // Recovery gating + coordination
    item def RecoveryRequest;
    item def RecoveryAuthorization;
    item def RecoveryCommand;
    item def RecoveryStatus;

    // Severity as attribute type
    attribute def SeverityLevel;
    attribute def FaultCode;

    // Simple boolean-ish attribute type
    attribute def BooleanFlag;
  }

  // ----------------------------
  // Port Definitions
  // ----------------------------
  package Ports {
    import Types::*;

    // Fault detection ports (typed by fault class)
    port def MotionFaultSensePort { out motionFault : MotionFault; }
    port def PowerFaultSensePort  { out powerFault  : PowerFault;  }
    port def NetworkFaultSensePort{ out networkFault: NetworkFault;}
    port def ThermalFaultSensePort{ out thermalFault: ThermalFault;}

    // Generic fault/event routing
    port def FaultEventOutPort { out faultEvent : FaultEvent; }
    port def FaultEventInPort  { in  faultEvent : FaultEvent; }

    // Alarm annunciation
    port def AlarmOutPort { out alarm : AlarmSignal; }
    port def AlarmInPort  { in  alarm : AlarmSignal; }

    // Logging + evidence store
    port def LogOutPort { out record : LogRecord; }
    port def LogInPort  { in  record : LogRecord; }
    port def EvidenceOutPort { out evidencePtr : EvidencePointer; }
    port def EvidenceInPort  { in  evidencePtr : EvidencePointer; }

    // Safe-state request + inhibit distribution
    port def SafeStateOutPort { out safeReq : SafeStateRequest; }
    port def SafeStateInPort  { in  safeReq : SafeStateRequest; }
    port def InhibitOutPort { out inhibit : InhibitSignal; }
    port def InhibitInPort  { in  inhibit : InhibitSignal; }

    // Recovery
    port def RecoveryReqOutPort { out recReq : RecoveryRequest; }
    port def RecoveryReqInPort  { in  recReq : RecoveryRequest; }
    port def RecoveryAuthOutPort{ out recAuth: RecoveryAuthorization; }
    port def RecoveryAuthInPort { in  recAuth: RecoveryAuthorization; }
    port def RecoveryCmdOutPort { out recCmd : RecoveryCommand; }
    port def RecoveryCmdInPort  { in  recCmd : RecoveryCommand; }
    port def RecoveryStatusOutPort { out recStatus : RecoveryStatus; }
    port def RecoveryStatusInPort  { in  recStatus : RecoveryStatus; }
  }

  // ----------------------------
  // Part Definitions (SUB10 focus + linked root subsystems)
  // ----------------------------
  package Parts {
    import Types::*;
    import Ports::*;

    // Fault normalization + routing (Detect → Annunciate/Log → Safe State/Recover)
    part def FaultSupervisor {
      // Inputs from detectors
      port motionIn  : MotionFaultSensePort;
      port powerIn   : PowerFaultSensePort;
      port netIn     : NetworkFaultSensePort;
      port thermalIn : ThermalFaultSensePort;

      // Normalized downstream fan-out
      port faultEventOut : FaultEventOutPort;

      // Annunciation + logging
      port alarmOut : AlarmOutPort;
      port logOut   : LogOutPort;

      // Safe-state trigger path
      port safeReqOut : SafeStateOutPort;

      // Recovery request path (must be gated later)
      port recReqOut : RecoveryReqOutPort;

      // Local attributes for classification/thresholding
      attribute lastFaultCode : FaultCode;
      attribute lastSeverity  : SeverityLevel;
      attribute latched       : BooleanFlag;
    }

    // Alarm system (per root telemetry/alarm backbone concept)
    part def AlarmSystem {
      port alarmIn : AlarmInPort;
    }

    // Evidence store / logger (data system evidence sink)
    part def EvidenceStore {
      port logIn : LogInPort;
      port evidenceOut : EvidenceOutPort;
    }

    // Safety subsystem & inhibit distribution (per IBD-ROOT-015 concept)
    part def SafetyInhibitDistribution {
      port safeReqIn   : SafeStateInPort;
      port inhibitOut  : InhibitOutPort;
    }

    // Motion-capable subsystem that must honor inhibits (abstracted)
    part def MotionSubsystem {
      port inhibitIn : InhibitInPort;
      port recCmdIn  : RecoveryCmdInPort;
      port recStatusOut : RecoveryStatusOutPort;
    }

    // Power subsystem (abstracted)
    part def PowerSubsystem {
      port recCmdIn  : RecoveryCmdInPort;
      port recStatusOut : RecoveryStatusOutPort;
    }

    // Network/controls subsystem (abstracted)
    part def NetworkSubsystem {
      port recCmdIn  : RecoveryCmdInPort;
      port recStatusOut : RecoveryStatusOutPort;
    }

    // Thermal control subsystem (abstracted)
    part def ThermalSubsystem {
      port recCmdIn  : RecoveryCmdInPort;
      port recStatusOut : RecoveryStatusOutPort;
    }

    // Authorization gate (per IBD-ROOT-012 concept): fail-closed, blocks unsafe recovery
    part def RecoveryAuthorizationGate {
      // Requests in; authorized token out
      port recReqIn  : RecoveryReqInPort;
      port recAuthOut: RecoveryAuthOutPort;

      attribute failClosed : BooleanFlag = true;
      attribute authorized : BooleanFlag;
    }

    // Recovery coordinator: only emits recovery commands when authorization present
    part def RecoveryCoordinator {
      port recReqIn   : RecoveryReqInPort;
      port recAuthIn  : RecoveryAuthInPort;

      // Controlled recovery outputs
      port recCmdToMotion  : RecoveryCmdOutPort;
      port recCmdToPower   : RecoveryCmdOutPort;
      port recCmdToNetwork : RecoveryCmdOutPort;
      port recCmdToThermal : RecoveryCmdOutPort;

      // Status aggregation inputs
      port statusFromMotion  : RecoveryStatusInPort;
      port statusFromPower   : RecoveryStatusInPort;
      port statusFromNetwork : RecoveryStatusInPort;
      port statusFromThermal : RecoveryStatusInPort;

      // Logging hook (optional but useful)
      port logOut : LogOutPort;
    }
  }

  // ----------------------------
  // IBD: Internal fault-response wiring
  // ----------------------------
  package IBD {
    import Types::*;
    import Ports::*;
    import Parts.*;

    part sub10FaultHandling : part def SUB10_FaultHandlingWiring {

      // Internal parts
      part faultSup   : FaultSupervisor;
      part alarmSys   : AlarmSystem;
      part evidence   : EvidenceStore;
      part safetyDist : SafetyInhibitDistribution;

      part authGate   : RecoveryAuthorizationGate;
      part recCoord   : RecoveryCoordinator;

      // Representative affected subsystems (coordination targets)
      part motionSub  : MotionSubsystem;
      part powerSub   : PowerSubsystem;
      part netSub     : NetworkSubsystem;
      part thermalSub : ThermalSubsystem;

      // --------------------------------------------
      // Annunciation path to alarms
      // --------------------------------------------
      connect faultSup.alarmOut to alarmSys.alarmIn;

      // --------------------------------------------
      // Logging path to evidence stores
      // --------------------------------------------
      connect faultSup.logOut to evidence.logIn;
      // evidence store emits pointers for downstream audit chains (kept internal here)
      // (Optional loop-back logging for recovery coordinator)
      connect recCoord.logOut to evidence.logIn;

      // --------------------------------------------
      // Safe-state trigger path to Safety inhibit distribution
      // --------------------------------------------
      connect faultSup.safeReqOut
