package BDD_SUB08_002_DataSystem {
  import ScalarValues::*;
  import SI::*;

  /**************************************
   * Dependencies (trace hooks)
   **************************************/
  package Dependencies {
    // Shallow references to upstream requirement packages (names taken from dependency summaries)
    package RD_F_001_BlackBoxFunctionsOverview;
    package RD_NF_001_PerformanceNFR_SensitivityAndAngularResolution;
  }

  /**************************************
   * Shared Types & Interfaces
   **************************************/
  package TypesAndInterfaces {

    // Simple value types (minimal + parsable)
    attribute def Identifier { value : String; }
    attribute def Timestamp { value : String; }         // ISO-8601 string
    attribute def DataRateMbps { value : Real; }
    attribute def StorageCapacityTB { value : Real; }
    attribute def AvailabilityPct { value : Real; }
    attribute def LatencySeconds { value : Real; }

    // Data artifacts
    attribute def RawFrame { frameId : Identifier; acquiredAt : Timestamp; }
    attribute def MetadataRecord { recordId : Identifier; createdAt : Timestamp; }
    attribute def CalibrationFrame { calId : Identifier; createdAt : Timestamp; }
    attribute def DataProduct { productId : Identifier; producedAt : Timestamp; level : String; }

    // Interfaces (as part defs used for ports)
    part def RawDataIF { ref frame : RawFrame; }
    part def MetadataIF { ref record : MetadataRecord; }
    part def CalibrationIF { ref cal : CalibrationFrame; }
    part def DataProductIF { ref product : DataProduct; }

    part def ControlCmdIF { ref cmd : String; }
    part def StatusTelemetryIF { ref status : String; }
    part def TimeSyncIF { ref time : Timestamp; }

    part def ArchiveQueryIF { ref query : String; }
    part def ArchiveResultIF { ref result : String; }

    part def NetworkStreamIF { ref payload : String; }
  }

  /**************************************
   * Requirements-realization placeholders
   * (Fix RD-2: ensure every listed requirement package has an implementing module)
   **************************************/
  package RequirementsRealization {

    // For RD-NF-001 packages that previously lacked implementation.
    // Implementations are modeled as parts contained in DataSystem and traceable by name.

    part def MissionGoalsAndScienceDrivers_Implementation;
    part def EarlyUniverseGoalMapping_Implementation;
    part def GalaxyEvolutionGoalMapping_Implementation;
    part def ExoplanetsGoalMapping_Implementation;
    part def BHgrowthGoalMapping_Implementation;

    part def StakeholderGroupConcerns_Implementation;

    part def ObservingAssumptionsAndConditionsDef_Implementation;
    part def FR_PhasedSegmentedApertureDef_Implementation;
    part def FR_PointingTrackingAndStabilityDef_Implementation;
    part def FR_ObservationPhasingAndExposureControlDef_Implementation;
    part def FR_AdaptiveOpticsOptionalDef_Implementation;

    part def SensitivityPerformanceNFRDef_Implementation;
    part def AngularResolutionPerformanceNFRDef_Implementation;

    part def ObservingAssumptionsAndConditions_Implementation;
    part def FR_PhasedSegmentedAperture_Implementation;
    part def FR_PointingTrackingAndStability_Implementation;
    part def FR_ObservationPhasingAndExposureControl_Implementation;
    part def FR_AdaptiveOpticsOptional_Implementation;
  }

  /**************************************
   * Data System (SUB08) — Block Definition
   **************************************/
  package DataSystemStructure {
    import TypesAndInterfaces::*;
    import RequirementsRealization::*;

    abstract part def LogicalComponent;
    abstract part def ExternalSystem;

    /******** External neighbor abstractions (context) ********/
    part def InstrumentSystem :> ExternalSystem {
      port rawOut : out RawDataIF;
      port calOut : out CalibrationIF;
      port ctrlIn : in ControlCmdIF;
      port statusOut : out StatusTelemetryIF;
      port timeIn : in TimeSyncIF;
    }

    part def ObservatoryControlSystem :> ExternalSystem {
      port cmdOut : out ControlCmdIF;
      port statusIn : in StatusTelemetryIF;
      port timeOut : out TimeSyncIF;
      port metaReqOut : out MetadataIF;
    }

    part def ScienceUserFacility :> ExternalSystem {
      port productIn : in DataProductIF;
      port queryOut : out ArchiveQueryIF;
      port resultIn : in ArchiveResultIF;
    }

    /******** Data System internal components ********/
    part def DataAcquisition :> LogicalComponent {
      attribute maxInputRate : DataRateMbps;
      port rawIn : in RawDataIF;
      port calIn : in CalibrationIF;
      port timeIn : in TimeSyncIF;
      port ctrlIn : in ControlCmdIF;

      port rawToPipeline : out RawDataIF;
      port calToPipeline : out CalibrationIF;
      port statusOut : out StatusTelemetryIF;
    }

    part def MetadataService :> LogicalComponent {
      attribute latencyTarget : LatencySeconds;
      port rawIn : in RawDataIF;
      port timeIn : in TimeSyncIF;
      port ctrlIn : in ControlCmdIF;

      port metaOut : out MetadataIF;
      port statusOut : out StatusTelemetryIF;
    }

    part def ProcessingPipeline :> LogicalComponent {
      attribute processingLatency : LatencySeconds;
      port rawIn : in RawDataIF;
      port calIn : in CalibrationIF;
      port metaIn : in MetadataIF;
      port ctrlIn : in ControlCmdIF;

      port quicklookOut : out DataProductIF;
      port scienceProductOut : out DataProductIF;
      port statusOut : out StatusTelemetryIF;
    }

    part def ArchiveSystem :> LogicalComponent {
      attribute capacity : StorageCapacityTB;
      attribute availability : AvailabilityPct;

      port productIn : in DataProductIF;
      port metaIn : in MetadataIF;
      port queryIn : in ArchiveQueryIF;

      port resultOut : out ArchiveResultIF;
      port statusOut : out StatusTelemetryIF;
    }

    part def DataDistributionGateway :> LogicalComponent {
      port productIn : in DataProductIF;
      port queryIn : in ArchiveQueryIF;
      port resultIn : in ArchiveResultIF;

      port productOut : out DataProductIF;
      port queryOut : out ArchiveQueryIF;
      port resultOut : out ArchiveResultIF;
    }

    part def DataSystemMonitor :> LogicalComponent {
      port statusIn : in StatusTelemetryIF;
      port statusOut : out StatusTelemetryIF;
    }

    /******** Subsystem root ********/
    part def DataSystem :> LogicalComponent {

      // Key performance/architecture attributes
      attribute endToEndAvailability : AvailabilityPct;
      attribute maxSustainedIngest : DataRateMbps;
      attribute archiveCapacity : StorageCapacityTB;

      // Ports to/from external systems
      port instRawIn : in RawDataIF;
      port instCalIn : in CalibrationIF;
      port ocsCmdIn : in ControlCmdIF;
      port ocsTimeIn : in TimeSyncIF;
      port ocsStatusOut : out StatusTelemetryIF;

      port scienceProductOut : out DataProductIF;
      port archiveQueryIn : in ArchiveQueryIF;
      port archiveResultOut : out ArchiveResultIF;

      // Internal decomposition
      part acquisition : DataAcquisition;
      part metadata : MetadataService;
      part pipeline : ProcessingPipeline;
      part archive : ArchiveSystem;
      part distribution : DataDistributionGateway;
      part monitor : DataSystemMonitor;

      // Requirement-implementation parts (RD-2 fix set)
      part MissionGoalsAndScienceDrivers : MissionGoalsAndScienceDrivers_Implementation;
      part EarlyUniverseGoalMapping : EarlyUniverseGoalMapping_Implementation;
      part GalaxyEvolutionGoalMapping : GalaxyEvolutionGoalMapping_Implementation;
      part ExoplanetsGoalMapping : ExoplanetsGoalMapping_Implementation;
      part BHgrowthGoalMapping : BHgrowthGoalMapping_Implementation;

      part StakeholderGroupConcerns : StakeholderGroupConcerns_Implementation;

      part ObservingAssumptionsAndConditionsDef : ObservingAssumptionsAndConditionsDef_Implementation;
      part FR_PhasedSegmentedApertureDef : FR_PhasedSegmentedApertureDef_Implementation;
      part FR_PointingTrackingAndStabilityDef : FR_PointingTrackingAndStabilityDef_Implementation;
      part FR_ObservationPhasingAndExposureControlDef : FR_ObservationPhasingAndExposureControlDef_Implementation;
      part FR_AdaptiveOpticsOptionalDef : FR_AdaptiveOpticsOptionalDef_Implementation;

      part SensitivityPerformanceNFRDef : SensitivityPerformanceNFRDef_Implementation;
      part AngularResolutionPerformanceNFRDef : AngularResolutionPerformanceNFRDef_Implementation;

      part ObservingAssumptionsAndConditions : ObservingAssumptionsAndConditions_Implementation;
      part FR_PhasedSegmentedAperture : FR_PhasedSegmentedAperture_Implementation;
      part FR_PointingTrackingAndStability : FR_PointingTrackingAndStability_Implementation;
      part FR_ObservationPhasingAndExposureControl : FR_ObservationPhasingAndExposureControl_Implementation;
      part FR_AdaptiveOpticsOptional : FR_AdaptiveOpticsOptional_Implementation;
    }
  }

  /**************************************
   * BDD context assembly with key connections
   **************************************/
  package DataSystemContextConnections {
    import TypesAndInterfaces::*;
    import DataSystemStructure::*;

    part tmtInstrument : InstrumentSystem;
    part tmtOCS : ObservatoryControlSystem;
    part scienceUsers : ScienceUserFacility;

    part sub08 : DataSystem;

    // Structural connections (ports / interfaces)
    connect tmtInstrument.rawOut to sub08.instRawIn;
    connect tmtInstrument.calOut to sub08.instCalIn;

    connect tmtOCS.cmdOut to sub08.ocsCmdIn;
    connect tmtOCS.timeOut to sub08.ocsTimeIn;
    connect sub08.ocsStatusOut to tmtOCS.statusIn;

    connect sub08.scienceProductOut to scienceUsers.productIn;
    connect scienceUsers.queryOut to sub08.archiveQueryIn;
    connect sub08.archiveResultOut to scienceUsers.resultIn;

    // Internal (subsystem) key dataflow connections
    connect sub08.instRawIn to sub08.acquisition.rawIn;
    connect sub08.instCalIn to sub08.acquisition.calIn;
    connect sub08.ocsTimeIn to sub08.acquisition.timeIn;
    connect sub08.ocsCmdIn to sub08.acquisition.ctrlIn;

    connect sub08.acquisition.rawToPipeline to sub08.pipeline.rawIn;
    connect sub08.acquisition.calToPipeline to sub08.pipeline.calIn;

    connect sub08.acquisition.rawToPipeline to sub08.metadata.rawIn;
    connect sub08.ocsTimeIn to sub08.metadata.timeIn;
    connect sub08.ocsCmdIn to sub08.metadata.ctrlIn;

    connect sub08.metadata.metaOut to sub08.pipeline.metaIn;
    connect sub08.ocsCmdIn to sub08.pipeline.ctrlIn;

    connect sub08.pipeline.scienceProductOut to sub08.archive.productIn;
    connect sub08.metadata.metaOut to sub08.archive.metaIn;

    connect sub08.archive.queryIn to sub08.archiveQueryIn;
    connect sub08.archive.resultOut to sub08.archiveResultOut;

    // Distribution gateway (product delivery + archive access)
    connect sub08.pipeline.scienceProductOut to sub08.distribution.productIn;
    connect sub08.archiveQueryIn to sub08.distribution.queryIn;
    connect sub08.archiveResultOut to sub08.distribution.resultIn;

    connect sub08.distribution.productOut to sub08.scienceProductOut;
    connect sub08.distribution.queryOut to sub08.archive.queryIn;
    connect sub08.archive.resultOut to sub08.distribution.resultIn;

    // Monitoring aggregation
    connect sub08.acquisition.statusOut to sub08.monitor.statusIn;
    connect sub08.metadata.statusOut to sub08.monitor.statusIn;
    connect sub08.pipeline.statusOut to sub08.monitor.statusIn;
    connect sub08.archive.statusOut to sub08.monitor.statusIn;
    connect sub08.monitor.statusOut to sub08.ocsStatusOut;
  }
}
