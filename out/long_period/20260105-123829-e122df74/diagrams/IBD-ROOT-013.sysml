package IBD_ROOT_013 {
    // IBD ROOT: Time & Frequency Distribution Backbone (Re-issue)

    // Flow items
    item def TimeSignal;
    item def TimeReferenceSignal;

    // Port types (time ports)
    port def TimeSyncInPort { in item time : TimeSignal; }
    port def TimeSyncOutPort { out item time : TimeSignal; }
    port def TimeReferenceIntakePort { in item ref : TimeReferenceSignal; }
    port def TimeReferenceFanoutPort { out item ref : TimeReferenceSignal; }

    // Root backbone definition
    part def TimeFrequencyDistributionBackbone {
        port timeRefIn  : TimeReferenceIntakePort;
        port timeRefOut : TimeReferenceFanoutPort;
        port timeSyncIn : TimeSyncInPort;
        port timeSyncOut: TimeSyncOutPort;
    }

    // Distribution processing elements (types reused from BDD; attributes instantiated, not re-defined)
    part def TimeSyncProcessor {
        port timeIn  : TimeSyncInPort;
        port timeOut : TimeSyncOutPort;

        // RD-NF-005-consistent attributes (instance-valued in IBD context)
        attribute clockSyncAccuracy : ClockSyncAccuracy;
        attribute timestampAccuracy : TimestampAccuracy;
    }

    part def TimeSyncComponent {
        port timeIn  : TimeSyncInPort;
        port timeOut : TimeSyncOutPort;

        // Attribute required by cross-view rule
        attribute clockSyncAccuracy : ClockSyncAccuracy;
        attribute timestampAccuracy : TimestampAccuracy;
    }

    // Root instance with intake, internal node(s), and fanout to consumers
    part ibd_root_013 : TimeFrequencyDistributionBackbone {
        // Time/Frequency reference intake + internal distribution nodes
        part tsProc : TimeSyncProcessor;
        part distNode : TimeSyncComponent;

        // Consumers (types/ports expected to exist in dependent BDDs/IBDs)
        part aps     : AlignmentPhasingSystem;
        part ao      : AdaptiveOpticsSubsystem;
        part dataSys : DataSystem;

        // Internal backbone chaining
        connect timeSyncIn to tsProc.timeIn;
        connect tsProc.timeOut to distNode.timeIn;
        connect distNode.timeOut to timeSyncOut;

        // Fanout to timeSync ports on consumers (terminate on existing time ports)
        connect distNode.timeOut to aps.timing;
        connect distNode.timeOut to ao.timeIn;
        connect distNode.timeOut to dataSys.timeIn;
    }
}
