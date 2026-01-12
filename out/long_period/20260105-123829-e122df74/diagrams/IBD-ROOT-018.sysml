package IBDROOT018 {
    // Port and Part Definitions for Exposure Synchronization System

    part def ExposureTriggerSource {
        port out exposureTrigger : TriggerSignal;
        attribute triggerLatency : Real;
    }

    part def InstrumentReadoutSink {
        port in instrumentData : DataProduct;
        attribute readoutLatency : Real;
    }

    part def MetadataTimestampSink {
        port in metadata : MetadataRecord;
        attribute timestampAccuracy : Real;
    }

    part def TimeSync {
        port in timeSyncIn : TimeSyncSignal;
        port out timeSyncOut : TimeSyncSignal;
    }

    part def DataSync {
        port in dataSyncIn : DataSyncSignal;
        port out dataSyncOut : DataSyncSignal;
    }

    part def ExposureSequenceCoordinator {
        port in exposureTrigger : TriggerSignal;
        port out dataProductOut : DataProduct;
        port out metadataOut : MetadataRecord;
        port out timeSyncOut : TimeSyncSignal;
        attribute syncCadence : Real;
        attribute latencyConstraint : Real;
    }

    part def TimeSyncProcessor {
        port in timeSyncIn : TimeSyncSignal;
        port out timeSyncOut : TimeSyncSignal;
        attribute timeSyncAccuracy : Real;
    }

    // Exposing the interconnections and timing constraints
    block ExposureSynchronizationSystem {
        part exposureTriggerSource : ExposureTriggerSource;
        part instrumentReadoutSink : InstrumentReadoutSink;
        part metadataTimestampSink : MetadataTimestampSink;
        part exposureSequenceCoordinator : ExposureSequenceCoordinator;
        part timeSyncProcessor : TimeSyncProcessor;

        // Connections between parts
        exposureTriggerSource.exposureTrigger -> exposureSequenceCoordinator.exposureTrigger;
        exposureSequenceCoordinator.dataProductOut -> instrumentReadoutSink.instrumentData;
        exposureSequenceCoordinator.metadataOut -> metadataTimestampSink.metadata;
        exposureSequenceCoordinator.timeSyncOut -> timeSyncProcessor.timeSyncIn;
        timeSyncProcessor.timeSyncOut -> exposureSequenceCoordinator.timeSyncOut;

        // Timing and synchronization constraints
        exposureSequenceCoordinator.syncCadence = 5.0; // Example: 5 seconds sync cadence
        exposureSequenceCoordinator.latencyConstraint = 0.1; // Example: 100 ms latency tolerance
        timeSyncProcessor.timeSyncAccuracy = 0.01; // Example: 10 ms accuracy
    }

    // TimeSyncSignal and DataSyncSignal definitions for intercommunication
    part def TimeSyncSignal {
        attribute timeStamp : Time;
    }

    part def DataSyncSignal {
        attribute dataReady : Boolean;
    }

    // Attributes for handling the system's time and synchronization requirements
    attribute def Time {
        unit : "seconds";
    }

    attribute def TriggerSignal {
        unit : "boolean";
    }

    attribute def DataProduct {
        unit : "bytes";
    }

    attribute def MetadataRecord {
        unit : "metadata";
    }

    attribute def MetadataCompleteness {
        unit : "boolean";
    }
}
