package IBD_IF_INT_006_InternalIF_MountOptics_CouplingSignals {

  import ScalarValues::*;
  import Quantities::*;

  //========================
  // Value / Item Definitions
  //========================

  attribute def AlignmentTolerance {
    attribute tipTiltTolArcsec : Real;
    attribute decenterTolUm    : Real;
    attribute focusTolUm       : Real;
  }

  attribute def VibrationLevel {
    attribute vibRmsUm     : Real;
    attribute dominantHz   : Real;
  }

  attribute def TemperatureGradient {
    attribute dTdz_C_per_m : Real;
    attribute dTdt_C_per_hr: Real;
  }

  item def FlexureState {
    attribute deflectionUm : Real;
    attribute rotationArcsec : Real;
  }

  item def VibrationTelemetry {
    attribute vibRmsUm   : Real;
    attribute bandHzLow  : Real;
    attribute bandHzHigh : Real;
  }

  item def ThermalTelemetry {
    attribute dTdz_C_per_m : Real;
  }

  item def AlignmentTelemetry {
    attribute tipTiltErrArcsec : Real;
    attribute decenterErrUm    : Real;
    attribute focusErrUm       : Real;
  }

  item def VibrationCompCmd {
    attribute notchHz     : Real;
    attribute gain        : Real;
  }

  item def AlignmentCompCmd {
    attribute tipTiltBiasArcsec : Real;
    attribute focusBiasUm       : Real;
  }

  //========================
  // Port Definitions (names constrained to SUB01/SUB02 BDD)
  //========================

  // Structural/mechanical coupling interface (SUB02 port name)
  port def MechanicalMountPort {
    out flexureState : FlexureState;
  }

  // Vibration sensing interface (SUB02 port name)
  port def VibrationSensePort {
    out vibrationTelemetry : VibrationTelemetry;
  }

  // Vibration actuation interface (SUB02 port name)
  port def VibrationActuatePort {
    in  vibrationCompCmd : VibrationCompCmd;
  }

  // Thermal coupling interface (SUB01 port name)
  port def ThermalPort {
    out thermalTelemetry : ThermalTelemetry;
  }

  // Telemetry interface (SUB01/SUB02 port name)
  port def TelemetryPort {
    out alignmentTelemetry : AlignmentTelemetry;
    out vibrationTelemetry : VibrationTelemetry;
    out thermalTelemetry   : ThermalTelemetry;
  }

  // Control command interface (SUB02 port name; also compatible with SUB01 control usage)
  port def ControlCmdPort {
    out vibrationCompCmd : VibrationCompCmd;
    out alignmentCompCmd : AlignmentCompCmd;
  }

  //========================
  // Part Definitions (minimal stubs aligned to SUB01/SUB02 concepts)
  //========================

  // SUB02: Mount/Structure
  part def MountStructure {
    // Required attributes (instantiate vibration + thermal)
    attribute vibration : VibrationLevel;
    attribute thermalEffect : TemperatureGradient;

    // Ports (names constrained to the dependency BDD samples)
    port mechToOptics : MechanicalMountPort;
    port vibSenseOut  : VibrationSensePort;
    port vibActIn     : VibrationActuatePort;
    port thermalOut   : ThermalPort;
    port telOut       : TelemetryPort;
  }

  // SUB01: Optical system element receiving coupled effects
  part def OpticalAssembly {
    // Required attribute (instantiate alignmentTolerance)
    attribute alignmentTolerance : AlignmentTolerance;

    // Ports (use only dependency-listed port names)
    port mechFromMount : ~MechanicalMountPort;
    port thermalIn     : ~ThermalPort;
    port telOut        : TelemetryPort;

    // Optional: local compensation entry via control
    port ctrlIn        : ~ControlCmdPort;
  }

  // Cross-subsystem control/compensation logic (fits SUB01/SUB02 ControlComponent/ControlSubsystem intent)
  part def AlignmentCompensationController {
    port telIn  : ~TelemetryPort;
    port cmdOut : ControlCmdPort;
  }

  //========================
  // IBD: Internal IF Mount↔Optics Structural/Alignment Coupling Signals
  //========================

  part IBD_Internal_IF_Mount_Optics_StructuralAlignmentCoupling {

    part mount      : MountStructure {
      // concrete instantiations (examples)
      attribute vibration.vibRmsUm     = 0.8;
      attribute vibration.dominantHz   = 8.0;

      attribute thermalEffect.dTdz_C_per_m  = 0.3;
      attribute thermalEffect.dTdt_C_per_hr = 0.5;
    }

    part optics     : OpticalAssembly {
      // concrete instantiation (examples)
      attribute alignmentTolerance.tipTiltTolArcsec = 0.02;
      attribute alignmentTolerance.decenterTolUm    = 5.0;
      attribute alignmentTolerance.focusTolUm       = 10.0;
    }

    part controller : AlignmentCompensationController;

    //------------------------
    // Coupling: sensed effects (Mount/Structure -> Optics/Controller)
    //------------------------

    // Structural flexure coupling as a sensed/measured effect
    connect mount.mechToOptics.flexureState -> optics.mechFromMount.flexureState;

    // Vibration sensed telemetry routed to controller for estimation/compensation
    connect mount.vibSenseOut.vibrationTelemetry -> controller.telIn.vibrationTelemetry;

    // Thermal gradient / state routed to optics and controller (environmental coupling)
    connect mount.thermalOut.thermalTelemetry -> optics.thermalIn.thermalTelemetry;
    connect mount.thermalOut.thermalTelemetry -> controller.telIn.thermalTelemetry;

    // Optical alignment telemetry to controller (metrology/diagnostics)
    connect optics.telOut.alignmentTelemetry -> controller.telIn.alignmentTelemetry;

    //------------------------
    // Coupling: control / compensation commands (Controller -> Mount/Optics)
    //------------------------

    // Commanded vibration compensation to mount active vibration control
    connect controller.cmdOut.vibrationCompCmd -> mount.vibActIn.vibrationCompCmd;

    // Commanded alignment compensation bias to optics (e.g., focus/tip-tilt bias feed-forward)
    connect controller.cmdOut.alignmentCompCmd -> optics.ctrlIn.alignmentCompCmd;

    //------------------------
    // Telemetry aggregation (optional internal fan-out to TelemetryPort)
    //------------------------

    // Mount health/telemetry out (vibration + thermal) for subsystem telemetry bus use
    connect mount.vibSenseOut.vibrationTelemetry -> mount.telOut.vibrationTelemetry;
    connect mount.thermalOut.thermalTelemetry    -> mount.telOut.thermalTelemetry;

    // Optics alignment telemetry routed to its telemetry port
    connect optics.telOut.alignmentTelemetry -> optics.telOut.alignmentTelemetry;

  }

}
