package BDD_SUB08_003_DataSystem {
  import ScalarValues::*;
  import SI::*;

  // ============
  // Interfaces
  // ============

  interface def DataStreamIF {
    inout rawBytes : String;
    inout frameCount : Integer;
  }

  interface def MetadataIF {
    inout metadataJson : String;
  }

  interface def TelemetryIF {
    out telemetryJson : String;
  }

  interface def ControlIF {
    in commandJson : String;
    out ackJson : String;
  }

  interface def DataProductIF {
    out productUri : String;
    out productType : String;
  }

  interface def ArchiveAccessIF {
    in queryJson : String;
    out resultJson : String;
  }

  interface def AlertIF {
    out alertJson : String;
  }

  // ==============
  // Common Types
  // ==============

  attribute def DataRate {
    value : Real;
  }

  attribute def StorageCapacity {
    value : Real;
  }

  attribute def Latency {
    value : Real;
  }

  // ==========================
  // Abstract Component Classes
  // ==========================

  abstract part def SubsystemComponent;

  abstract part def ServiceComponent :> SubsystemComponent;

  // ======================
  // Data System Components
  // ======================

  part def DataSystem :> SubsystemComponent {
    // External-facing ports
    port instrumentDataIn : DataStreamIF;
    port ocsControl : ControlIF;
    port ocsTelemetry : TelemetryIF;
    port dataDeliveryOut : DataProductIF;
    port archiveAccess : ArchiveAccessIF;
    port alertsOut : AlertIF;

    // Key attributes (no units per modeling guideline)
    attribute peakIngestRate : DataRate;
    attribute archiveCapacity : StorageCapacity;
    attribute quicklookLatency : Latency;

    // Structural decomposition
    part acquisition : DataAcquisition;
    part metadata : MetadataManagement;
    part pipeline : ProcessingPipeline;
    part archive : ArchiveAndDistribution;
    part catalog : CatalogAndIndex;
    part qa : QualityAssurance;
    part security : SecurityAndAccessControl;
    part orchestration : WorkflowOrchestration;

    // Internal integration (conceptual connections)
    connect instrumentDataIn to acquisition.rawIn;
    connect acquisition.rawOut to metadata.rawIn;
    connect metadata.enrichedOut to pipeline.dataIn;
    connect pipeline.productOut to archive.productIn;
    connect archive.deliveryOut to dataDeliveryOut;

    connect ocsControl to orchestration.controlIn;
    connect orchestration.telemetryOut to ocsTelemetry;

    connect archiveAccess to archive.accessIn;

    connect qa.alertOut to alertsOut;
    connect security.alertOut to alertsOut;
  }

  part def DataAcquisition :> ServiceComponent {
    port rawIn : DataStreamIF;
    port rawOut : DataStreamIF;
    port controlIn : ControlIF;
    port telemetryOut : TelemetryIF;

    attribute bufferDepthFrames : Integer;
    attribute losslessCaptureEnabled : Boolean;
  }

  part def MetadataManagement :> ServiceComponent {
    port rawIn : DataStreamIF;
    port metadataIn : MetadataIF;
    port enrichedOut : DataStreamIF;
    port metadataOut : MetadataIF;
    port telemetryOut : TelemetryIF;

    attribute schemaVersion : String;
    attribute provenanceEnabled : Boolean;
  }

  part def ProcessingPipeline :> ServiceComponent {
    port dataIn : DataStreamIF;
    port metadataIn : MetadataIF;
    port productOut : DataProductIF;
    port telemetryOut : TelemetryIF;
    port alertOut : AlertIF;

    attribute supportsRealtimeQuicklook : Boolean;
    attribute maxConcurrentJobs : Integer;
  }

  part def ArchiveAndDistribution :> ServiceComponent {
    port productIn : DataProductIF;
    port accessIn : ArchiveAccessIF;
    port deliveryOut : DataProductIF;
    port telemetryOut : TelemetryIF;

    attribute retentionPolicy : String;
    attribute replicationEnabled : Boolean;
  }

  part def CatalogAndIndex :> ServiceComponent {
    port metadataIn : MetadataIF;
    port accessIn : ArchiveAccessIF;
    port telemetryOut : TelemetryIF;

    attribute indexType : String;
  }

  part def QualityAssurance :> ServiceComponent {
    port dataIn : DataStreamIF;
    port productIn : DataProductIF;
    port alertOut : AlertIF;
    port telemetryOut : TelemetryIF;

    attribute qaRulesetVersion : String;
  }

  part def SecurityAndAccessControl :> ServiceComponent {
    port controlIn : ControlIF;
    port accessIn : ArchiveAccessIF;
    port alertOut : AlertIF;
    port telemetryOut : TelemetryIF;

    attribute authnMode : String;
    attribute authzPolicy : String;
  }

  part def WorkflowOrchestration :> ServiceComponent {
    port controlIn : ControlIF;
    port telemetryOut : TelemetryIF;
    port alertOut : AlertIF;

    attribute schedulerType : String;
  }

  // ==========================
  // RD-2 Fix: Provide explicit
  // implementations for prior
  // requirement packages/names
  // ==========================

  // Mission goals & mappings (implemented as data products + metadata/pipeline services)
  part def MissionGoalsAndScienceDrivers :> ServiceComponent {
    port productOut : DataProductIF;
    port metadataOut : MetadataIF;
  }

  part def EarlyUniverseGoalMapping :> ServiceComponent {
    port productOut : DataProductIF;
    port metadataOut : MetadataIF;
  }

  part def GalaxyEvolutionGoalMapping :> ServiceComponent {
    port productOut : DataProductIF;
    port metadataOut : MetadataIF;
  }

  part def ExoplanetsGoalMapping :> ServiceComponent {
    port productOut : DataProductIF;
    port metadataOut : MetadataIF;
  }

  part def BHgrowthGoalMapping :> ServiceComponent {
    port productOut : DataProductIF;
    port metadataOut : MetadataIF;
  }

  part def StakeholderGroupConcerns :> ServiceComponent {
    port alertOut : AlertIF;
    port telemetryOut : TelemetryIF;
  }

  // Observing assumptions & conditions: realized as metadata schema + provenance
  part def ObservingAssumptionsAndConditionsDef :> ServiceComponent {
    port metadataOut : MetadataIF;
  }

  part def ObservingAssumptionsAndConditions :> ServiceComponent {
    port metadataOut : MetadataIF;
  }

  // Functional requirement realizations (traceable service elements)
  part def FR_PhasedSegmentedApertureDef :> ServiceComponent {
    port metadataOut : MetadataIF;
    port telemetryOut : TelemetryIF;
  }

  part def FR_PhasedSegmentedAperture :> ServiceComponent {
    port metadataOut : MetadataIF;
    port telemetryOut : TelemetryIF;
  }

  part def FR_PointingTrackingAndStabilityDef :> ServiceComponent {
    port telemetryOut : TelemetryIF;
  }

  part def FR_PointingTrackingAndStability :> ServiceComponent {
    port telemetryOut : TelemetryIF;
  }

  part def FR_ObservationPhasingAndExposureControlDef :> ServiceComponent {
    port controlIn : ControlIF;
    port metadataOut : MetadataIF;
  }

  part def FR_ObservationPhasingAndExposureControl :> ServiceComponent {
    port controlIn : ControlIF;
    port metadataOut : MetadataIF;
  }

  part def FR_AdaptiveOpticsOptionalDef :> ServiceComponent {
    port telemetryOut : TelemetryIF;
    port metadataOut : MetadataIF;
  }

  part def FR_AdaptiveOpticsOptional :> ServiceComponent {
    port telemetryOut : TelemetryIF;
    port metadataOut : MetadataIF;
  }

  // Non-functional performance requirements: realized as QA + pipeline + archive constraints/services
  part def SensitivityPerformanceNFRDef :> ServiceComponent {
    port productOut : DataProductIF;
    port telemetryOut : TelemetryIF;
  }

  part def AngularResolutionPerformanceNFRDef :> ServiceComponent {
    port productOut : DataProductIF;
    port telemetryOut : TelemetryIF;
  }

  // ==========================
  // Integrate RD-2 fix elements
  // into DataSystem structure
  // ==========================

  part def DataSystemWithTraceability :> DataSystem {
    part scienceDrivers : MissionGoalsAndScienceDrivers;
    part mapEarlyUniverse : EarlyUniverseGoalMapping;
    part mapGalaxyEvolution : GalaxyEvolutionGoalMapping;
    part mapExoplanets : ExoplanetsGoalMapping;
    part mapBHgrowth : BHgrowthGoalMapping;
    part stakeholderConcerns : StakeholderGroupConcerns;

    part obsCondDef : ObservingAssumptionsAndConditionsDef;
    part obsCond : ObservingAssumptionsAndConditions;

    part frPhasingDef : FR_PhasedSegmentedApertureDef;
    part frPhasing : FR_PhasedSegmentedAperture;

    part frPointingDef : FR_PointingTrackingAndStabilityDef;
    part frPointing : FR_PointingTrackingAndStability;

    part frExposureDef : FR_ObservationPhasingAndExposureControlDef;
    part frExposure : FR_ObservationPhasingAndExposureControl;

    part frAODef : FR_AdaptiveOpticsOptionalDef;
    part frAO : FR_AdaptiveOpticsOptional;

    part nfrSensitivity : SensitivityPerformanceNFRDef;
    part nfrAngRes : AngularResolutionPerformanceNFRDef;

    // Minimal connections to demonstrate implementation linkage through metadata/products/telemetry
    connect metadata.metadataOut to obsCondDef.metadataOut;
    connect metadata.metadataOut to obsCond.metadataOut;

    connect pipeline.productOut to scienceDrivers.productOut;
    connect pipeline.productOut to mapEarlyUniverse.productOut;
    connect pipeline.productOut to mapGalaxyEvolution.productOut;
    connect pipeline.productOut to mapExoplanets.productOut;
    connect pipeline.productOut to mapBHgrowth.productOut;

    connect ocsTelemetry to frPointing.telemetryOut;
    connect ocsTelemetry to frPhasing.telemetryOut;

    connect ocsControl to frExposure.controlIn;

    connect pipeline.productOut to nfrSensitivity.productOut;
    connect pipeline.productOut to nfrAngRes.productOut;

    connect qa.alertOut to stakeholderConcerns.alertOut;
    connect security.alertOut to stakeholderConcerns.alertOut;
  }
}
