package IBD_SUB02_005_AltAzDriveLoops {

  import ScalarValues::*;
  import Quantities::*;

  // Dependency imports (types defined upstream)
  import BDD_SUB02_003_MountStructure::*;          // PowerPort, ControlCmdPort, TelemetryPort, TimeSyncPort
  import RD_F_003_PointingTrackingStabilization_FunctionalRequirementsSet::*;
  import RD_NF_005_Interface_NFR_Timing_Latency_Synchronization::*;
  import IBD_ROOT_013::*;                          // TimeFrequencyDistributionBackbone (time sync backbone)

  // ---- Local part definitions (use ONLY port types from BDD-SUB02-003) ----

  part def MountControlProcessor {
    port cmdOut        : ControlCmdPort;
    port telemetryIn   : TelemetryPort;
    port timeSyncIn    : TimeSyncPort;
    port pwrIn         : PowerPort;
  }

  part def DriveController {
    port cmdIn         : ControlCmdPort;
    port telemetryOut  : TelemetryPort;
    port timeSyncIn    : TimeSyncPort;
    port pwrIn         : PowerPort;

    // Internal axis interfaces
    port motorDriveOut : ControlCmdPort;
    port encFbIn       : TelemetryPort;
  }

  part def Motor {
    port driveIn       : ControlCmdPort;
    port telemetryOut  : TelemetryPort;
    port pwrIn         : PowerPort;
  }

  part def Encoder {
    port telemetryOut  : TelemetryPort;
    port timeSyncIn    : TimeSyncPort;
    port pwrIn         : PowerPort;
  }

  // ---- IBD context: Alt-Az Drive Loops wiring (Motor/Drive/Encoder) ----

  part def AltAzDriveLoops {

    // Control-loop timing / update-rate (instantiate at least one)
    attribute commandLatency : Real = 1.0[ms];
    attribute encoderUpdateRateHz : Real = 2000.0[Hz];

    // Parts
    part mcp      : MountControlProcessor;
    part altDrive : DriveController;
    part azDrive  : DriveController;

    part altMotor : Motor;
    part azMotor  : Motor;

    part altEnc   : Encoder;
    part azEnc    : Encoder;

    // Time reference distribution backbone (from IBD-ROOT-013)
    part timeBackbone : TimeFrequencyDistributionBackbone;

    // ---- Command paths (MCP -> Drive Controllers) ----
    connect mcp.cmdOut to altDrive.cmdIn;
    connect mcp.cmdOut to azDrive.cmdIn;

    // ---- Drive -> Motor wiring ----
    connect altDrive.motorDriveOut to altMotor.driveIn;
    connect azDrive.motorDriveOut  to azMotor.driveIn;

    // ---- Encoder feedback wiring ----
    connect altEnc.telemetryOut to altDrive.encFbIn;
    connect azEnc.telemetryOut  to azDrive.encFbIn;

    // ---- Telemetry back to mount control processor ----
    connect altDrive.telemetryOut to mcp.telemetryIn;
    connect azDrive.telemetryOut  to mcp.telemetryIn;

    // (Optional) direct encoder telemetry visibility at MCP level as well
    connect altEnc.telemetryOut to mcp.telemetryIn;
    connect azEnc.telemetryOut  to mcp.telemetryIn;

    // ---- Time synchronization wiring (Backbone -> MCP/Drives/Encoders) ----
    connect timeBackbone.timeSyncOut to mcp.timeSyncIn;

    connect timeBackbone.timeSyncOut to altDrive.timeSyncIn;
    connect timeBackbone.timeSyncOut to azDrive.timeSyncIn;

    connect timeBackbone.timeSyncOut to altEnc.timeSyncIn;
    connect timeBackbone.timeSyncOut to azEnc.timeSyncIn;

    // ---- Power distribution (logical wiring references) ----
    connect mcp.pwrIn      to altDrive.pwrIn;
    connect mcp.pwrIn      to azDrive.pwrIn;

    connect altDrive.pwrIn to altMotor.pwrIn;
    connect azDrive.pwrIn  to azMotor.pwrIn;

    connect altDrive.pwrIn to altEnc.pwrIn;
    connect azDrive.pwrIn  to azEnc.pwrIn;
  }

  // View instance for diagram IBD-SUB02-005
  part ibd_sub02_005 : AltAzDriveLoops;

}
