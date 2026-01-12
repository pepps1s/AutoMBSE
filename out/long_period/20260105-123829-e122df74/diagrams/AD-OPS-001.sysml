package AD_OPS_001_ObservatoryStartup {

  // ===== View Metadata =====
  doc /*diagram_id*/ AD-OPS-001;
  doc /*type*/ AD;
  doc /*title*/ "AD OPS: Observatory Startup (Power-Up → Ready) Activity";
  doc /*goal*/ "Nominal startup from site power availability through subsystem self-checks to operator-declared Ready for Slew, aligned to operational modes and lifecycle requirements.";
  doc /*dependencies*/ "RD-OPS-001, RD-F-002, BDD-SUB03-001, BDD-SUB07-001, BDD-SUB09-001, BDD-SUB10-001";

  // ===== Items (Inputs/Signals) =====
  item def OperatorCommand;
  item def ScheduleTrigger;
  item def TimeSyncReadiness;
  item def FacilityStatus;
  item def NetworkStatus;
  item def HVACStatus;
  item def SafetyInterlockStatus;
  item def TelemetryRecord;
  item def ModeName;
  item def ReadinessState;

  // ===== Enumerations / Named Values =====
  enum def OpsModeName {
    StartupMode;
    StandbyMode;
    SafeParkMode;
  }

  enum def ReadyStateName {
    ReadyForSlew;
    NotReady;
  }

  // ===== Action Definitions (named for reuse & mode trace) =====
  action def EnterStartupMode {
    out item mode : ModeName;
  }

  action def VerifySitePowerAndFacilities {
    in item facility : FacilityStatus;
    in item hvac : HVACStatus;
    in item net : NetworkStatus;
    out item facilitiesReady : FacilityStatus;
  }

  action def ConfirmTimeSyncReadiness {
    in item timeSync : TimeSyncReadiness;
    out item timeSyncOK : TimeSyncReadiness;
  }

  action def InitializeOCSSequencerAndLogging {
    out item logStart : TelemetryRecord;
  }

  action def AcquireBaselineTelemetrySnapshot {
    out item snapshot : TelemetryRecord;
  }

  action def CheckSafetyInterlocks {
    in item safety : SafetyInterlockStatus;
    out item interlockOK : SafetyInterlockStatus;
  }

  action def HandoffToSafeState {
    in item safety : SafetyInterlockStatus;
    out item mode : ModeName;
  }

  action def EnableSubsystemsForStartup {
    out item enabled : TelemetryRecord;
  }

  action def RunSubsystemSelfChecks {
    out item selfCheckReport : TelemetryRecord;
  }

  action def TransitionToStandbyMode {
    out item mode : ModeName;
  }

  action def OperatorDeclareReadyForSlew {
    in item cmd : OperatorCommand;
    out item ready : ReadinessState;
  }

  action def LogStartupCompletion {
    out item logDone : TelemetryRecord;
  }

  // ===== Activity (Main Behavior) =====
  action StartupObservatory {
    // Acceptance: at least 1 in item for operator command or schedule trigger
    in item operatorCmd : OperatorCommand;
    in item schedule : ScheduleTrigger;

    // Acceptance: at least 1 in item for time sync readiness
    in item timeSyncReady : TimeSyncReadiness;

    // Facilities readiness (site power/HVAC/network)
    in item facilityStatus : FacilityStatus;
    in item hvacStatus : HVACStatus;
    in item networkStatus : NetworkStatus;

    // Safety interlock input
    in item safetyStatus : SafetyInterlockStatus;

    // Internal / outputs
    out item opsMode : ModeName;
    out item readiness : ReadinessState;
    out item startupLog : TelemetryRecord;

    // ---- Start ----
    first start;

    // Mode alignment with RD-OPS-001: StartupMode
    then action enterStartup : EnterStartupMode;

    // OCS telemetry/logging action (required)
    then action initOCS : InitializeOCSSequencerAndLogging;

    // Facilities readiness action (required)
    then action verifyFacilities : VerifySitePowerAndFacilities {
      in item facility : FacilityStatus = StartupObservatory.facilityStatus;
      in item hvac : HVACStatus = StartupObservatory.hvacStatus;
      in item net : NetworkStatus = StartupObservatory.networkStatus;
    }

    // Ensure time sync readiness (required)
    then action confirmTimeSync : ConfirmTimeSyncReadiness {
      in item timeSync : TimeSyncReadiness = StartupObservatory.timeSyncReady;
    }

    // Baseline telemetry snapshot (telemetry/logging)
    then action baselineTlm : AcquireBaselineTelemetrySnapshot;

    // Safety interlock check with explicit decision/guard
    then action checkInterlocks : CheckSafetyInterlocks {
      in item safety : SafetyInterlockStatus = StartupObservatory.safetyStatus;
    }

    // Decision node (explicit guards)
    then decision SafetyInterlockDecision;

    // Guarded flows:
    // - If interlock tripped -> safe-state handoff
    // - Else continue nominal startup
    flow checkInterlocks.interlockOK to SafetyInterlockDecision;

    then action safeHandoff : HandoffToSafeState {
      in item safety : SafetyInterlockStatus = StartupObservatory.safetyStatus;
    } when (SafetyInterlockDecision, "Safety interlock tripped");

    // Mode alignment with RD-OPS-001: SafeParkMode (handoff)
    then action logSafe : InitializeOCSSequencerAndLogging
      when (SafetyInterlockDecision, "Safety interlock tripped");

    then done when (SafetyInterlockDecision, "Safety interlock tripped");

    // Nominal path (no trip)
    then action enableSubsystems : EnableSubsystemsForStartup
      when (SafetyInterlockDecision, "No safety trip");

    then action runSelfChecks : RunSubsystemSelfChecks
      when (SafetyInterlockDecision, "No safety trip");

    // Transition StartupMode -> StandbyMode (RD-OPS-001 reference)
    then action toStandby : TransitionToStandbyMode
      when (SafetyInterlockDecision, "No safety trip");

    // Operator declares readiness
    then action declareReady : OperatorDeclareReadyForSlew {
      in item cmd : OperatorCommand = StartupObservatory.operatorCmd;
    } when (SafetyInterlockDecision, "No safety trip");

    // Log completion
    then action logCompletion : LogStartupCompletion
      when (SafetyInterlockDecision, "No safety trip");

    // Clear end condition 'ReadyForSlew'
    then done when (declareReady.ready == ReadyStateName::ReadyForSlew);

  }

}
