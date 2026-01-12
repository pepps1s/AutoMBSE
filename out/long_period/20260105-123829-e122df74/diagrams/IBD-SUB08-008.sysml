package IBD_SUB08_008_QuickLook_QA_Wiring {

  //========================================================
  // Diagram Metadata
  //========================================================
  metadata def DiagramInfo {
    attribute diagram_id : String;
    attribute type : String;
    attribute title : String;
    attribute goal : String;
  }

  metadata diagram : DiagramInfo {
    diagram_id = "IBD-SUB08-008";
    type = "IBD";
    title = "IBD SUB08: Quick-Look QA Wiring (Near-Real-Time QA to Sequencer Decision)";
    goal = "Model quick-look QA generation and feedback wiring to observation sequencing decisions before the next exposure.";
  }

  // Shallow dependency anchors (trace hooks)
  comment about diagram
    "dependencies: RD-F-007, RD-F-002, IBD-ROOT-011, IBD-OPS-004";

  //========================================================
  // Basic Types / Items
  //========================================================
  attribute def Duration;

  item def RawScienceFrame;
  item def CalibratedFrame;
  item def ObservationMetadata;

  item def QAFlags;
  item def QAMetrics;

  item def ExposureTrigger;           // end-of-exposure / readout complete trigger
  item def NextExposureDecision;      // decision gate for next exposure

  item def ProvenanceRecord;          // provenance linkage for QA products
  item def LogEntry;

  //========================================================
  // Ports
  //========================================================
  port def RawDataStreamIn {
    in raw : RawScienceFrame;
  }

  port def CalDataStreamIn {
    in cal : CalibratedFrame;
  }

  port def MetadataStreamIn {
    in meta : ObservationMetadata;
  }

  port def ExposureTimingIn {
    in exposureDone : ExposureTrigger;
  }

  port def QAOut {
    out flags : QAFlags;
    out metrics : QAMetrics;
  }

  port def DecisionGateIn {
    in qaFlags : QAFlags;
    in qaMetrics : QAMetrics;
  }

  port def DecisionGateOut {
    out decision : NextExposureDecision;
  }

  port def ProvenanceOut {
    out prov : ProvenanceRecord;
    out log  : LogEntry;
  }

  port def ProvenanceIn {
    in prov : ProvenanceRecord;
    in log  : LogEntry;
  }

  //========================================================
  // Part Definitions
  //========================================================
  part def DataAcquisitionContext {
    // Produces raw/calibrated/metadata streams for near-real-time consumers
    port rawOut  : ~RawDataStreamIn;     // out raw
    port calOut  : ~CalDataStreamIn;     // out cal
    port metaOut : ~MetadataStreamIn;    // out meta
    port timingOut : ~ExposureTimingIn;  // out exposureDone
  }

  part def QuickLookQAGenerator {
    // Consumes raw/cal/metadata and produces QA flags/metrics
    port rawIn  : RawDataStreamIn;
    port calIn  : CalDataStreamIn;
    port metaIn : MetadataStreamIn;

    port timingIn : ExposureTimingIn;

    port qaOut : QAOut;
    port provenanceOut : ProvenanceOut;

    // Timing attributes for constraint evaluation
    attribute qaLatency : Duration;              // time from exposureDone -> QA available
  }

  part def OCS_OpsDecisionPoint {
    // Represents OCS / Ops decision gate before next exposure
    port qaIn : DecisionGateIn;
    port decisionOut : DecisionGateOut;

    // Decision timing
    attribute decisionLeadTime : Duration;       // time from exposureDone -> latest decision moment
  }

  part def DataSystemContext {
    // Logs/stores QA outputs and provenance
    port provenanceIn : ProvenanceIn;
  }

  //========================================================
  // Constraint: QA available before next exposure decision
  //========================================================
  constraint def QAAvailableBeforeNextExposureDecision {
    attribute qaLatency : Duration;
    attribute decisionLeadTime : Duration;

    // Required: QA is available prior to next exposure decision point.
    // (Interpretation: QA generation completes no later than decision gate deadline)
    assert qaLatency <= decisionLeadTime;
  }

  //========================================================
  // IBD: Internal Wiring (Near-real-time QA -> Sequencer Decision)
  //========================================================
  part def TMT_QuickLook_QA_Wiring_IBD {
    // Context / participating elements
    part daq : DataAcquisitionContext;
    part qa  : QuickLookQAGenerator;
    part ocs : OCS_OpsDecisionPoint;
    part ds  : DataSystemContext;

    // Instantiate the timing requirement constraint
    constraint qaBeforeDecision : QAAvailableBeforeNextExposureDecision {
      qaLatency = qa.qaLatency;
      decisionLeadTime = ocs.decisionLeadTime;
    }

    //======================================================
    // Data streams into QA generator
    //======================================================
    connect daq.rawOut::raw  to qa.rawIn::raw;
    connect daq.calOut::cal  to qa.calIn::cal;
    connect daq.metaOut::meta to qa.metaIn::meta;

    // Exposure timing trigger into QA generator (start/anchor for latency)
    connect daq.timingOut::exposureDone to qa.timingIn::exposureDone;

    //======================================================
    // QA outputs to OCS/Ops decision gate (before next exposure)
    //======================================================
    connect qa.qaOut::flags   to ocs.qaIn::qaFlags;
    connect qa.qaOut::metrics to ocs.qaIn::qaMetrics;

    // Decision output (for completeness of the decision point in this IBD)
    // (Not wired further in this view; represents handoff to sequencer/exposure loop)
    // ocs.decisionOut::decision

    //======================================================
    // QA outputs logged/stored in Data System context (provenance linkage)
    //======================================================
    connect qa.provenanceOut::prov to ds.provenanceIn::prov;
    connect qa.provenanceOut::log  to ds.provenanceIn::log;
  }

}
