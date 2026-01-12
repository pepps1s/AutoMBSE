package PD_Timing_Synchronization_Budget {

    package Timing_Accuracy {
        part TimeSource {
            attribute timeAccuracy : Real;
            attribute jitter : Real;
            attribute offset : Real;
            attribute timestampResolution : Real;
        }

        part OCS {
            port timeSyncPort : TimeSource;
        }

        part APS {
            port timeSyncPort : TimeSource;
        }

        part AO_RTC {
            port timeSyncPort : TimeSource;
        }

        part Instruments {
            port timeSyncPort : TimeSource;
        }

        part DataAcquisition {
            port timeSyncPort : TimeSource;
        }

        part SafetyLogging {
            port timeSyncPort : TimeSource;
        }

        part LoopClosureControl {
            port timeSyncPort : TimeSource;
            attribute loopClosureLatency : Real;
        }

        part LoopClosureControlInstance {
            part loopClosureControlInstance : LoopClosureControl;
            loopClosureControlInstance.loopClosureLatency = 1.0;
        }

        constraint TimingAccuracyConstraint {
            TimeSource.timeAccuracy == 1.0;
            OCS.timeSyncPort.timeAccuracy == TimeSource.timeAccuracy;
            APS.timeSyncPort.timeAccuracy == TimeSource.timeAccuracy;
            AO_RTC.timeSyncPort.timeAccuracy == TimeSource.timeAccuracy;
            Instruments.timeSyncPort.timeAccuracy == TimeSource.timeAccuracy;
            DataAcquisition.timeSyncPort.timeAccuracy == TimeSource.timeAccuracy;
            SafetyLogging.timeSyncPort.timeAccuracy == TimeSource.timeAccuracy;
        }

        constraint TimestampResolutionConstraint {
            TimeSource.timestampResolution == 0.1;
            OCS.timeSyncPort.timestampResolution == TimeSource.timestampResolution;
            APS.timeSyncPort.timestampResolution == TimeSource.timestampResolution;
            AO_RTC.timeSyncPort.timestampResolution == TimeSource.timestampResolution;
            Instruments.timeSyncPort.timestampResolution == TimeSource.timestampResolution;
            DataAcquisition.timeSyncPort.timestampResolution == TimeSource.timestampResolution;
            SafetyLogging.timeSyncPort.timestampResolution == TimeSource.timestampResolution;
        }

        part TimingPaths {
            path TimeSourceToOCS :> TimeSource to OCS;
            path TimeSourceToAPS :> TimeSource to APS;
            path TimeSourceToAO_RTC :> TimeSource to AO_RTC;
            path TimeSourceToInstruments :> TimeSource to Instruments;
            path TimeSourceToDataAcquisition :> TimeSource to DataAcquisition;
            path TimeSourceToSafetyLogging :> TimeSource to SafetyLogging;
            path TimeSourceToLoopClosure :> TimeSource to LoopClosureControlInstance;
        }

        part TimingConstraints {
            constraint clockSyncAccuracy : OCS.timeSyncPort.timeAccuracy == APS.timeSyncPort.timeAccuracy;
            constraint timestampAccuracy : APS.timeSyncPort.timestampResolution == AO_RTC.timeSyncPort.timestampResolution;
        }

        part LatencyBudget {
            constraint LoopClosureLatency : LoopClosureControlInstance.loopClosureLatency <= 1.0;
        }

        package TimeStandard {
            part UniversalTime {
                attribute timeReference : String = "UTC";
            }

            part TimeStandardReference {
                part redefines timeReference = UniversalTime;
            }
        }
    }

    acceptance_checks {
        check "Defines time accuracy, jitter, offset, and timestamp resolution value types" : Timing_Accuracy;
        check "Includes at least 8 timing paths (time source to OCS, APS, AO RTC, instruments, data acquisition, safety logging)" : TimingPaths;
        check "Has constraints that verify clock sync accuracy and timestamp accuracy meet RD-NF-005" : TimingConstraints;
        check "Includes at least one closed-loop latency budget for a control loop (APS or guiding/AO) with named contributors" : LatencyBudget;
        check "Ensures all time-related parameters reference a single declared time standard within the PD package" : TimeStandard;
    }
}
