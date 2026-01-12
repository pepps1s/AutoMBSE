package AD_SUB04_001_SegmentAlignmentAndPhasing_APS_LoopClosure {

  // Diagram metadata
  doc /*diagram_id*/ = "AD-SUB04-001";
  doc /*type*/       = "AD";
  doc /*title*/      = "AD SUB04: Segment Alignment & Phasing (APS Loop Closure) Activity";
  doc /*goal*/       = "Represent APS behavior: WFS/metrology -> pose error -> actuator commands -> converge within phasing tolerances.";
  doc /*dependencies*/ = "RD-F-004, BDD-SUB04-001, BDD-SUB01-001, RD-NF-005";

  // --- Items (interfaces / context) ---

  item def TimeStamp;
  item def TimeSync {
    attribute syncAccuracy;
    attribute timestampAccuracy;
  }

  item def SegmentId;
  item def SegmentCount {
    attribute n;
  }

  item def WfsFrame {
    attribute t : TimeStamp;
  }

  item def MetrologyInput {
    attribute t : TimeStamp;
  }

  item def WavefrontEstimate {
    attribute t : TimeStamp;
  }

  item def SegmentPoseError {
    attribute segment : SegmentId;
    attribute t : TimeStamp;
  }

  item def ActuatorCommand {
    attribute segment : SegmentId;
    attribute t : TimeStamp;
  }

  item def PhasingTolerance {
    attribute pistonTolerance;
    attribute tipTiltTolerance;
  }

  item def LoopCadence {
    attribute updateRateHz;
    attribute maxCommandLatency;
  }

  item def RecoveryTime {
    attribute seconds;
  }

  item def FaultHandoff {
    attribute reason;
  }

  item def LoopMetrics {
    attribute iteration;
    attribute elapsedTime;
    attribute maxResidualError;
    attribute commandLatency;
    attribute syncQuality;
  }

  // --- Actions (steps / behaviors) ---

  action def AcquireWavefrontAndMetrology {
    in item wfs      : WfsFrame;
    in item metrology: MetrologyInput;
    in item timeSync : TimeSync;
    first start;
    then done;
  }

  action def CheckClockSyncAndCadence {
    in item timeSync : TimeSync;
    in item cadence  : LoopCadence;
    first start;
    then done;
  }

  action def EstimateWavefront {
    in item wfs      : WfsFrame;
    in item metrology: MetrologyInput;
    first start;
    then done;
  }

  action def ComputeSegmentPoseErrors {
    in item estimate : WavefrontEstimate;
    in item segments : SegmentCount;
    first start;
    then done;
  }

  action def ComputeActuatorCommands {
    in item poseError: SegmentPoseError;
    in item cadence  : LoopCadence;
    first start;
    then done;
  }

  action def IssueActuatorCommands {
    in item cmd      : ActuatorCommand;
    in item timeSync : TimeSync;
    first start;
    then done;
  }

  action def VerifyConvergence {
    in item poseError : SegmentPoseError;
    in item tolerance : PhasingTolerance;
    first start;
    then done;
  }

  action def CheckRecoveryTimeGuard {
    in item recoveryTime : RecoveryTime;
    first start;
    then done;
  }

  action def FaultResponseHandoff {
    in item fault : FaultHandoff;
    first start;
    then done;
  }

  action def ForEachSegment {
    in item segments : SegmentCount;
    first start;
    then done;
  }

  // --- Main loop closure behavior ---

  action ClosePhasingLoop {
    // Acceptance: in items for WfsFrame/MetrologyInput and TimeSync
    in item wfsFrame     : WfsFrame;
    in item metrologyIn  : MetrologyInput;
    in item timeSync     : TimeSync;

    // Context
    in item segments     : SegmentCount;
    in item tolerance    : PhasingTolerance;
    in item cadence      : LoopCadence;
    in item recoveryTime : RecoveryTime;

    // Loop bookkeeping (conceptual item)
    item metrics : LoopMetrics;

    first start;

    // Timing/latency sensitivity: clock sync / loop cadence considerations
    then action timingGate : CheckClockSyncAndCadence {
      in item timeSync : TimeSync;
      in item cadence  : LoopCadence;
    }

    // ---- LOOP ITERATION 1 (sense -> estimate -> command -> verify) ----

    then action sense1 : AcquireWavefrontAndMetrology {
      in item wfs      : WfsFrame       = wfsFrame;
      in item metrology: MetrologyInput = metrologyIn;
      in item timeSync : TimeSync       = timeSync;
    }

    then action estimate1 : EstimateWavefront {
      in item wfs      : WfsFrame       = wfsFrame;
      in item metrology: MetrologyInput = metrologyIn;
    }

    then action errAll1 : ComputeSegmentPoseErrors {
      in item estimate : WavefrontEstimate;
      in item segments : SegmentCount = segments; // segment count context
    }

    then action perSeg1 : ForEachSegment {
      in item segments : SegmentCount = segments; // named per-segment iteration context
    }

    then action command1 : ComputeActuatorCommands {
      in item poseError: SegmentPoseError;
      in item cadence  : LoopCadence = cadence; // latency-aware command computation
    }

    then action actuate1 : IssueActuatorCommands {
      in item cmd      : ActuatorCommand;
      in item timeSync : TimeSync = timeSync; // timestamping / correlation
    }

    then action verify1 : VerifyConvergence {
      in item poseError : SegmentPoseError;
      in item tolerance : PhasingTolerance = tolerance;
    }

    // Explicit convergence check (guarded decision)
    then action convergenceCheck1;

    // If converged -> done
    // Else -> guard: cannot converge within recoveryTime -> fault handoff
    // Else -> iterate again (loop pattern via repeated then steps)

    // ---- LOOP ITERATION 2 (repeat pattern) ----

    then action recoveryGuard1 : CheckRecoveryTimeGuard {
      in item recoveryTime : RecoveryTime = recoveryTime;
    }

    then action sense2 : AcquireWavefrontAndMetrology {
      in item wfs      : WfsFrame       = wfsFrame;
      in item metrology: MetrologyInput = metrologyIn;
      in item timeSync : TimeSync       = timeSync;
    }

    then action estimate2 : EstimateWavefront {
      in item wfs      : WfsFrame       = wfsFrame;
      in item metrology: MetrologyInput = metrologyIn;
    }

    then action errAll2 : ComputeSegmentPoseErrors {
      in item estimate : WavefrontEstimate;
      in item segments : SegmentCount = segments;
    }

    then action perSeg2 : ForEachSegment {
      in item segments : SegmentCount = segments;
    }

    then action command2 : ComputeActuatorCommands {
      in item poseError: SegmentPoseError;
      in item cadence  : LoopCadence = cadence;
    }

    then action actuate2 : IssueActuatorCommands {
      in item cmd      : ActuatorCommand;
      in item timeSync : TimeSync = timeSync;
    }

    then action verify2 : VerifyConvergence {
      in item poseError : SegmentPoseError;
      in item tolerance : PhasingTolerance = tolerance;
    }

    then action convergenceCheck2;

    // ---- Fault route (cannot converge within recoveryTime) ----
    then action cannotConvergeWithinRecoveryTime : FaultResponseHandoff {
      in item fault : FaultHandoff;
    }

    then done;
  }

}
