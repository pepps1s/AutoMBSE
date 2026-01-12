package TMT_SUB03_FacilityMonitoringTelemetryRouting {

  import ScalarValues::*;
  import Quantities::*;

  // =========================
  // IBD-SUB03-014
  // IBD SUB03: Facility Monitoring Telemetry Routing (Env/Power/Network to OCS+Data)
  // Goal: Detail facility telemetry routing for environment, power quality, and network health
  //       to both OCS/Ops and Data System.
  // Dependencies: BDD-SUB03-003, IBD-ROOT-011, IBD-SUB08-005
  // =========================

  // -------- Items (signals / records) --------
  abstract item def TelemetrySignal;
  item def EnvTelemetry : TelemetrySignal;
  item def PowerQualityTelemetry : TelemetrySignal;
  item def NetworkHealthTelemetry : TelemetrySignal;

  item def TelemetryEnvelope : TelemetrySignal {
    // simple provenance-friendly wrapper
    attribute sourceId : String;
    attribute schemaVersion : String;
    attribute timestampUtc : String;
  }

  // -------- Attribute / Quantity Types (minimal stubs) --------
  attribute def Percentage;
  attribute def Hertz;

  // -------- Ports --------
  port def TelemetryOut {
    out telemetry : TelemetryEnvelope;
  }

  port def TelemetryIn {
    in telemetry : TelemetryEnvelope;
  }

  // Optional: specialized sinks (distinct consumers)
  port def OpsTelemetryIn {
    in telemetry : TelemetryEnvelope;
  }

  port def ProvenanceTelemetryIn {
    in telemetry : TelemetryEnvelope;
  }

  // -------- Source Parts (telemetry producers) --------
  part def EnvSenseArray {
    // acceptance: includes envSense element as telemetry source
    port telemetryOut : TelemetryOut;
    attribute monitoringFrequency : Hertz = 1.0[Hz];  // RD-IF-003 style: monitoringFrequency / cadence
  }

  part def ControlBusMonitor {
    // acceptance: includes controlBus element as telemetry source
    port telemetryOut : TelemetryOut;
    attribute uptimePercentage : Percentage = 99.90[%]; // RD-IF-003 style: uptimePercentage
  }

  part def MgmtNetworkMonitor {
    // acceptance: includes mgmtNet element as telemetry source
    port telemetryOut : TelemetryOut;
    attribute monitoringFrequency : Hertz = 0.2[Hz];
  }

  // -------- Routing / Fan-out --------
  part def TelemetryRouter {
    port envIn  : TelemetryIn;
    port pwrIn  : TelemetryIn;
    port netIn  : TelemetryIn;

    // Separate consumers (no undefined consumer termination)
    port opsOutEnv  : TelemetryOut;
    port opsOutPwr  : TelemetryOut;
    port opsOutNet  : TelemetryOut;

    port provOutEnv : TelemetryOut;
    port provOutPwr : TelemetryOut;
    port provOutNet : TelemetryOut;
  }

  // -------- Consumers (distinct sinks) --------
  part def OpsTelemetrySink {
    // OCS/Ops consumer
    port telemetryIn : OpsTelemetryIn;
  }

  part def DataProvenanceSink {
    // Data System consumer
    port telemetryIn : ProvenanceTelemetryIn;
  }

  // -------- IBD Context: SUB03 Facility Monitoring Telemetry Routing --------
  part def FacilityMonitoringTelemetryRouting_SUB03 {

    // Producers (sources)
    part envSense   : EnvSenseArray;
    part controlBus : ControlBusMonitor;
    part mgmtNet    : MgmtNetworkMonitor;

    // Router / backbone element
    part telemetryRouter : TelemetryRouter;

    // Separate consumers (sinks)
    part ocsOpsTelemetrySink : OpsTelemetrySink;
    part dataProvenanceSink  : DataProvenanceSink;

    // --- Connections: Sources -> Router ---
    connect EnvSense_To_Router
      from envSense.telemetryOut.telemetry
      to   telemetryRouter.envIn.telemetry;

    connect ControlBus_To_Router
      from controlBus.telemetryOut.telemetry
      to   telemetryRouter.pwrIn.telemetry;

    connect MgmtNet_To_Router
      from mgmtNet.telemetryOut.telemetry
      to   telemetryRouter.netIn.telemetry;

    // --- Connections: Router -> Ops (OCS/Ops telemetry sink) ---
    connect RouterEnv_To_Ops
      from telemetryRouter.opsOutEnv.telemetry
      to   ocsOpsTelemetrySink.telemetryIn.telemetry;

    connect RouterPwr_To_Ops
      from telemetryRouter.opsOutPwr.telemetry
      to   ocsOpsTelemetrySink.telemetryIn.telemetry;

    connect RouterNet_To_Ops
      from telemetryRouter.opsOutNet.telemetry
      to   ocsOpsTelemetrySink.telemetryIn.telemetry;

    // --- Connections: Router -> Data (data provenance sink) ---
    connect RouterEnv_To_Data
      from telemetryRouter.provOutEnv.telemetry
      to   dataProvenanceSink.telemetryIn.telemetry;

    connect RouterPwr_To_Data
      from telemetryRouter.provOutPwr.telemetry
      to   dataProvenanceSink.telemetryIn.telemetry;

    connect RouterNet_To_Data
      from telemetryRouter.provOutNet.telemetry
      to   dataProvenanceSink.telemetryIn.telemetry;
  }

  // Package-level instance for the IBD view (optional but commonly used)
  part IBD_SUB03_014 : FacilityMonitoringTelemetryRouting_SUB03;

}
