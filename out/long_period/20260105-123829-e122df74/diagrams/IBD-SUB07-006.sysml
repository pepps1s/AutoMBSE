package IBD_SUB07_006_OCS_ConfigManager_InternalWiring {

  import ScalarValues::*;

  // ========== Items (flows) ==========
  item def ConfigSelectionCriteria;
  item def ConfigSnapshot;
  item def ConfigDistribution;
  item def ProvenanceRecord;
  item def TimeSyncSignal;

  // ========== Ports ==========
  port def ConfigSelectionSourcePort {
    out selectionCriteria : ConfigSelectionCriteria;
    out selectedConfig    : ConfigSnapshot;
  }

  port def ConfigInPort {
    in config : ConfigSnapshot;
  }

  port def ConfigOutPort {
    out config : ConfigDistribution;
  }

  port def ConfigSinkPort {
    in config : ConfigDistribution;
  }

  port def ProvenanceOutPort {
    out record : ProvenanceRecord;
  }

  port def ProvenanceInPort {
    in record : ProvenanceRecord;
  }

  port def TimeSyncInPort {
    in timeSync : TimeSyncSignal;
  }

  port def TimeSyncOutPort {
    out timeSync : TimeSyncSignal;
  }

  // ========== Parts ==========
  part def ConfigSelectionSource {
    port outSel : ConfigSelectionSourcePort;
  }

  part def ConfigDistributorFanout {
    port inCfg  : ConfigInPort;
    port outCfg : ConfigOutPort;
  }

  part def ConfigSinkEndpoint {
    port cfgSink : ConfigSinkPort;
  }

  part def ProvenanceRecorder {
    // Metadata completeness (acceptance check)
    attribute schemaVersionRecorded   : String;
    attribute unitsAndFramesDeclared  : Boolean;

    // Timestamping linkage (acceptance check)
    attribute configTimestampRecorded : String;

    port inCfg      : ConfigInPort;
    port timeSyncIn : TimeSyncInPort;
    port outProv    : ProvenanceOutPort;
  }

  part def TimeSyncInterface {
    port timeSyncOut : TimeSyncOutPort;
  }

  // External sink representing Data System provenance ingestion (dependency context: IBD-ROOT-016)
  part def DataSystemProvenanceSink {
    port provIn : ProvenanceInPort;
  }

  // ========== IBD: OCS Configuration Manager Internal Wiring ==========
  part def OCSConfigurationManagerInternalWiring {

    // Internal functions: Select → Distribute → Record
    part selector    : ConfigSelectionSource;
    part distributor : ConfigDistributorFanout;
    part recorder    : ProvenanceRecorder;

    // Time sync backbone linkage (dependency context: IBD-ROOT-013)
    part timeSync    : TimeSyncInterface;

    // Fanout targets (all have config sink ports — acceptance check)
    part ocsCoreSink         : ConfigSinkEndpoint;
    part instrumentsSink     : ConfigSinkEndpoint;
    part adaptiveOpticsSink  : ConfigSinkEndpoint;
    part enclosureSiteSink   : ConfigSinkEndpoint;
    part telescopeOpticsSink : ConfigSinkEndpoint;
    part dataSystemSink      : ConfigSinkEndpoint;

    // Provenance output target
    part dataProvenance      : DataSystemProvenanceSink;

    // ----- Internal wiring -----

    // Selection output feeds distributor input (selected config becomes distributable config)
    connect selector.outSel.selectedConfig to distributor.inCfg.config;

    // Distribution fanout to subsystem config sinks (no target without a config sink port)
    connect distributor.outCfg.config to ocsCoreSink.cfgSink.config;
    connect distributor.outCfg.config to instrumentsSink.cfgSink.config;
    connect distributor.outCfg.config to adaptiveOpticsSink.cfgSink.config;
    connect distributor.outCfg.config to enclosureSiteSink.cfgSink.config;
    connect distributor.outCfg.config to telescopeOpticsSink.cfgSink.config;
    connect distributor.outCfg.config to dataSystemSink.cfgSink.config;

    // Recorder captures the same selected configuration for provenance/reproducibility
    connect selector.outSel.selectedConfig to recorder.inCfg.config;

    // Time synchronization for configuration timestamping
    connect timeSync.timeSyncOut.timeSync to recorder.timeSyncIn.timeSync;

    // Provenance record forwarded to Data System provenance sink
    connect recorder.outProv.record to dataProvenance.provIn.record;

    // Recorder metadata initialization (completeness attributes)
    recorder.schemaVersionRecorded  = "SysMLv2-OCS-CFG-1.0";
    recorder.unitsAndFramesDeclared = true;
  }

  // ========== IBD-level instantiation to satisfy BDD attribute instantiation rule ==========
  part ocsConfigMgrInternalWiring : OCSConfigurationManagerInternalWiring {
    schemaVersionRecorded  = "SysMLv2-OCS-CFG-1.0";
    unitsAndFramesDeclared = true;
  }

}
