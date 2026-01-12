package IBD_SUB05_006_AO_Telemetry_Performance_Metrics_Wiring {

  //
  // Diagram metadata
  //
  metadata def DiagramInfo {
    attribute diagramId : String;
    attribute diagramType : String;
    attribute title : String;
    attribute goal : String;
    attribute dependencies : String;
  }

  metadata diagram : DiagramInfo {
    diagramId = "IBD-SUB05-006";
    diagramType = "IBD";
    title = "IBD SUB05: AO Telemetry & Performance Metrics Wiring (Residual WFE/Strehl)";
    goal = "Model AO performance telemetry generation and routing to OCS/Ops and Data System for provenance.";
    dependencies = "BDD-SUB05-002, BDD-SUB08-002, RD-NF-001, RD-F-007, IBD-ROOT-011";
  }

  //
  // Basic scalar/type stubs (kept minimal for standalone parsing)
  //
  attribute def Real;
  attribute def String;
  attribute def Timestamp;

  //
  // Telemetry / provenance item types
  //
  item def AOPerformanceMetrics {
    // Required metrics
    attribute residualWFE_nm : Real;
    attribute strehlRatio : Real;
    attribute aoLoopRate_Hz : Real;

    // Provenance-relevant context (for Data System capture)
    attribute timestamp : Timestamp;
    attribute configurationId : String;
    attribute schemaVersion : String;
  }

  item def MetadataRecord {
    attribute timestamp : Timestamp;
    attribute configurationId : String;
    attribute schemaVersion : String;
    attribute unitsDeclared : String;
    attribute framesDeclared : String;
    attribute sourceSubsystem : String;
    attribute signalName : String;
  }

  //
  // Ports (aligned with names referenced in dependency summaries)
  //
  port def TelemetryOutPort {
    out item telemetryOut : AOPerformanceMetrics;
  }

  port def TelemetryInPort {
    in item telemetryIn : AOPerformanceMetrics;
  }

  port def MetadataOutPort {
    out item metadataOut : MetadataRecord;
  }

  port def MetadataInPort {
    in item metadataIn : MetadataRecord;
  }

  //
  // Parts (sources/sinks and backbone consistent with IBD-ROOT-011 concepts)
  //
  part def AOController {
    // From BDD-SUB05-002 sample: TelemetryOutPort / telemetryOut
    port telemetryOut : TelemetryOutPort;

    // Internal state (also appears in BDD-SUB05-002 sample attrs)
    attribute residualWFE_nm : Real;
    attribute strehlRatio : Real;
    attribute aoLoopRate_Hz : Real;
  }

  part def OCSOps {
    // From IBD-ROOT-011 sample: OCS has telemetryInPort
    port telemetryInPort : TelemetryInPort;
  }

  part def TelemetryBackbone {
    // Unified telemetry routing backbone (IBD-ROOT-011 concept)
    port telemetryInPort : TelemetryInPort;
    port telemetryOutPort_OCS : TelemetryOutPort;
    port telemetryOutPort_Data : TelemetryOutPort;
  }

  //
  // Data System provenance capture path (RD-F-007 intent)
  //
  part def ProvenanceCaptureService {
    port telemetryInPort : TelemetryInPort;
    port metadataOutPort : MetadataOutPort;

    // Minimal indicative attributes for provenance policy hooks
    attribute supportsChecksumOrDigest : String;
    attribute schemaVersionRecorded : String;
    attribute unitsAndFramesDeclared : String;
    attribute recordsTimeReferenceInfo : String;
  }

  part def MetadataStore {
    port metadataInPort : MetadataInPort;
  }

  part def DataSystem {
    // From IBD-ROOT-011 sample: DataSystem has telemetryInPort
    port telemetryInPort : TelemetryInPort;

    // Explicit metadata/provenance capture path linkage
    part provenanceCapture : ProvenanceCaptureService;
    part metadataStore : MetadataStore;

    connect telemetryInPort.telemetryIn to provenanceCapture.telemetryInPort.telemetryIn;
    connect provenanceCapture.metadataOutPort.metadataOut to metadataStore.metadataInPort.metadataIn;
  }

  //
  // IBD: AO telemetry generation and routing to OCS/Ops + Data System
  //
  part IBD_SUB05_006 : Subsystem {
    // Key participants
    part aoController : AOController;
    part telemetryBackbone : TelemetryBackbone;
    part ocsOps : OCSOps;
    part dataSystem : DataSystem;

    // Source -> backbone
    connect aoController.telemetryOut.telemetryOut to telemetryBackbone.telemetryInPort.telemetryIn;

    // Backbone -> sinks (OCS/Ops and Data System)
    connect telemetryBackbone.telemetryOutPort_OCS.telemetryOut to ocsOps.telemetryInPort.telemetryIn;
    connect telemetryBackbone.telemetryOutPort_Data.telemetryOut to dataSystem.telemetryInPort.telemetryIn;
  }

  //
  // Minimal stubs to keep the IBD self-contained
  //
  abstract part def Subsystem;

}
