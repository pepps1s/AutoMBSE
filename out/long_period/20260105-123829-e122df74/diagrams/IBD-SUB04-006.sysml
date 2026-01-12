package IBD_SUB04_006_APS_Internal_Dataflow {
    import ScalarValues::*;

    % ===== Basic Attribute / Item Types (stubs for typing) ===== %
    attribute def TimeStamp;
    attribute def EstimateData;
    attribute def CommandData;
    attribute def TimeSync;

    % ===== Dependency-Reused Timing Port Names (IBD-ROOT-013) ===== %
    port def TimeSyncInPort {
        in item timeIn : TimeSync;
        attribute clockSyncAccuracy : Real;
        attribute timestampAccuracy : Real;
    }

    port def TimeSyncOutPort {
        out item timeOut : TimeSync;
        attribute clockSyncAccuracy : Real;
        attribute timestampAccuracy : Real;
    }

    % ===== Payload Part Definitions (from BDD-SUB04-001; attributes must exist on instance context) ===== %
    part def WfsFrame {
        attribute timestamp : TimeStamp;
    }

    part def WavefrontEstimate {
        attribute estimateData : EstimateData;
    }

    part def ActuatorCommand {
        attribute commandData : CommandData;
    }

    % ===== Data Ports (typed + directional) ===== %
    port def WfsFrameOutPort {
        out item frame : WfsFrame;
    }
    port def WfsFrameInPort {
        in item frame : WfsFrame;
    }

    port def WavefrontEstimateOutPort {
        out item estimate : WavefrontEstimate;
    }
    port def WavefrontEstimateInPort {
        in item estimate : WavefrontEstimate;
    }

    port def ActuatorCommandOutPort {
        out item cmd : ActuatorCommand;
    }
    port def ActuatorCommandInPort {
        in item cmd : ActuatorCommand;
    }

    % ===== Internal APS Parts (WFS → Estimator → Controller → Cmd Distribution) ===== %
    part def WfsAcquisition {
        port wfsFrameOut : WfsFrameOutPort;
        port timeSyncIn : TimeSyncInPort;
    }

    part def WavefrontEstimator {
        port wfsFrameIn : WfsFrameInPort;
        port wavefrontOut : WavefrontEstimateOutPort;
        port timeSyncIn : TimeSyncInPort;
    }

    part def SegmentController {
        % RD-F-004 intent: instantiate attributes on the controlling element
        attribute updateRate : Real;
        attribute segmentPositionAccuracy : Real;

        port wavefrontIn : WavefrontEstimateInPort;
        port cmdOut : ActuatorCommandOutPort;
        port timeSyncIn : TimeSyncInPort;
    }

    part def ActuatorCommandDistributor {
        port cmdIn : ActuatorCommandInPort;
        port cmdOut : ActuatorCommandOutPort;
        port timeSyncIn : TimeSyncInPort;
    }

    part def SegmentActuatorArray {
        port cmdIn : ActuatorCommandInPort;
        port timeSyncIn : TimeSyncInPort;
    }

    % ===== Timing Backbone (reuse names from IBD-ROOT-013) ===== %
    part def TimeFrequencyDistributionBackbone {
        port timeRefIn : TimeSyncInPort;
        port timeRefOut : TimeSyncOutPort;
    }

    part def TimeSyncProcessor {
        attribute clockSyncAccuracy : Real;
        attribute timestampAccuracy : Real;

        port timeIn : TimeSyncInPort;
        port timeOut : TimeSyncOutPort;
    }

    % ===== IBD Top-Level (contains instances required by BBD-1) ===== %
    part def APS_Internal_Dataflow_IBD {
        % Payload instances (BBD-1: ensure attributes are instantiated in an instance context)
        part wfsFrame : WfsFrame;
        part wfEstimate : WavefrontEstimate;
        part actCmd : ActuatorCommand;

        % Functional chain parts (>= 3 internal parts for flows)
        part wfsAcq : WfsAcquisition;
        part estimator : WavefrontEstimator;
        part controller : SegmentController;
        part cmdDist : ActuatorCommandDistributor;
        part actuators : SegmentActuatorArray;

        % Timing parts (explicit timeSync)
        part distNode : TimeFrequencyDistributionBackbone;
        part tsProc : TimeSyncProcessor;

        % ===== Data Connections (typed + directionally consistent) =====
        connect wfsAcq.wfsFrameOut::frame to estimator.wfsFrameIn::frame;
        connect estimator.wavefrontOut::estimate to controller.wavefrontIn::estimate;
        connect controller.cmdOut::cmd to cmdDist.cmdIn::cmd;
        connect cmdDist.cmdOut::cmd to actuators.cmdIn::cmd;

        % ===== Explicit Timing / Cadence / Timestamping Connections =====
        connect distNode.timeRefOut::timeOut to tsProc.timeIn::timeIn;
        connect tsProc.timeOut::timeOut to wfsAcq.timeSyncIn::timeIn;
        connect tsProc.timeOut::timeOut to estimator.timeSyncIn::timeIn;
        connect tsProc.timeOut::timeOut to controller.timeSyncIn::timeIn;
        connect tsProc.timeOut::timeOut to cmdDist.timeSyncIn::timeIn;
        connect tsProc.timeOut::timeOut to actuators.timeSyncIn::timeIn;
    }

    % Instantiate the view root for this IBD
    part ibd_sub04_006 : APS_Internal_Dataflow_IBD;
}
