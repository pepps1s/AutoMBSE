package IBD_OPS_005_WeatherAbortTriggerWiring_v2 {

  //==========================================================================
  // Imports (minimal placeholders; keep model self-contained)
  //==========================================================================
  package ScalarValues {
    attribute def Real;
    attribute def Boolean;
    attribute def String;
  }

  package Quantities {
    attribute def m_per_s;
    attribute def degC;
    attribute def m;
  }

  import ScalarValues::*;
  import Quantities::*;

  //==========================================================================
  // Types & Items (signals)
  //==========================================================================
  attribute def WindSpeed : Real;
  attribute def Temperature : Real;
  attribute def Visibility : Real;

  attribute def BooleanFlag : Boolean;

  item def WindSample {
    attribute windSpeed : WindSpeed;
  }

  item def TemperatureSample {
    attribute temperature : Temperature;
  }

  item def VisibilitySample {
    attribute visibility : Visibility; // dust/visibility equivalent
  }

  item def AbortDecision {
    attribute abortRequired : BooleanFlag;
    attribute reason : String;
  }

  item def SafeStateTrigger {
    attribute trigger : BooleanFlag;
    attribute cause : String;
  }

  item def SafeParkCommand {
    attribute parkNow : BooleanFlag;
    attribute cause : String;
  }

  //==========================================================================
  // Ports
  //==========================================================================
  port def WindOutPort {
    out item wind : WindSample;
  }

  port def TemperatureOutPort {
    out item temp : TemperatureSample;
  }

  port def VisibilityOutPort {
    out item vis : VisibilitySample;
  }

  port def AbortInputsPort {
    in item wind : WindSample;
    in item temp : TemperatureSample;
    in item vis  : VisibilitySample;
  }

  port def AbortDecisionOutPort {
    out item decision : AbortDecision;
  }

  port def SafeStateInPort {
    in item safeTrig : SafeStateTrigger;
  }

  port def SafeParkInPort {
    in item parkCmd : SafeParkCommand;
  }

  port def SafeStateOutPort {
    out item safeTrig : SafeStateTrigger;
  }

  port def SafeParkOutPort {
    out item parkCmd : SafeParkCommand;
  }

  //==========================================================================
  // Part Definitions (BDD-typed parts used in IBD)
  //==========================================================================
  part def WindSensor {
    port windOut : WindOutPort;
  }

  part def TemperatureSensor {
    port tempOut : TemperatureOutPort;
  }

  part def DustVisibilitySensor {
    port visOut : VisibilityOutPort;
  }

  part def EnvironmentalSensorSuite {
    part wind : WindSensor;
    part temp : TemperatureSensor;
    part dustVis : DustVisibilitySensor;

    // expose suite-level ports (typed) for IBD wiring convenience
    port windOut : WindOutPort;
    port tempOut : TemperatureOutPort;
    port visOut  : VisibilityOutPort;

    // internal bindings (suite ports reflect sub-sensor outputs)
    binding windOut = wind.windOut;
    binding tempOut = temp.tempOut;
    binding visOut  = dustVis.visOut;
  }

  part def WeatherAbortDecisionLogic {

    // Thresholds consistent with RD-NF-004 (instantiate at least one)
    attribute windSpeedThreshold : WindSpeed [1] = 15.0; // m/s (example operational threshold)
    attribute temperatureThreshold : Temperature [1] = 0.0; // degC (example pause/abort threshold)
    attribute dustVisibilityLimit : Visibility [1] = 1000.0; // m (example minimum visibility)

    port envIn : AbortInputsPort;
    port abortOut : AbortDecisionOutPort;
  }

  part def SafetySubsystem {
    port safeStateIn : SafeStateInPort;
    port safeStateOut : SafeStateOutPort;
  }

  part def OpsMaintenanceSubsystem {
    port safeParkIn : SafeParkInPort;
    port safeParkOut : SafeParkOutPort;
  }

  //==========================================================================
  // IBD: Weather Abort Trigger Wiring (Env Sensors→Decision→Safe Park v2)
  //==========================================================================
  part def TMT_Operations_Abort_Wiring_Context {
    // Parts (explicitly typed)
    part envSensors : EnvironmentalSensorSuite;
    part abortLogic : WeatherAbortDecisionLogic;
    part safety : SafetySubsystem;
    part opsMaint : OpsMaintenanceSubsystem;

    // Derived trigger translation (kept as explicit ports/flows via items)
    // Decision fan-out: to Safety safe-state and Ops/Maint safe-park
    // (No "abort trigger port" exists without corresponding typed parts: envSensors + abortLogic present)

    // Connect sensors to abort logic
    connect envSensors.windOut to abortLogic.envIn.wind;
    connect envSensors.tempOut to abortLogic.envIn.temp;
    connect envSensors.visOut  to abortLogic.envIn.vis;

    // Connect abort decision to safety and ops/maint actions via subsystem output ports
    // Map AbortDecision -> triggers/commands (represented as flow items through ports)
    // For IBD purposes, treat abortOut.decision as source that informs triggers/commands.
    connect abortLogic.abortOut.decision to safety.safeStateIn.safeTrig;
    connect abortLogic.abortOut.decision to opsMaint.safeParkIn.parkCmd;
  }

  //==========================================================================
  // View Instance (diagram anchor)
  //==========================================================================
  part IBD_OPS_005 : TMT_Operations_Abort_Wiring_Context;

}
