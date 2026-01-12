package IBD_SUB08_RawCalMetadataInternalPipelines {

    part def RawCapture {
        port rawOutPort: RawFrameOut;
    }

    part def CalibrationCapture {
        port calOutPort: CalibrationFrameOut;
    }

    part def MetadataService {
        port metadataOutPort: MetadataRecordOut;
    }

    part def StagingBuffer {
        port bufferInPort: RawFrameIn;
        port bufferOutPort: RawFrameOut;
    }

    part def PipelineCompute {
        port computeInPort: RawFrameIn;
        port computeOutPort: RawFrameOut;
    }

    // Flow Definitions
    part def RawFrame;
    part def CalibrationFrame;
    part def MetadataRecord;

    // Interface Definitions
    port def RawFrameOut {
        out RawFrame rawData;
    }

    port def CalibrationFrameOut {
        out CalibrationFrame calData;
    }

    port def MetadataRecordOut {
        out MetadataRecord metadata;
    }

    port def RawFrameIn {
        in RawFrame rawData;
    }

    port def CalibrationFrameIn {
        in CalibrationFrame calData;
    }

    port def MetadataRecordIn {
        in MetadataRecord metadata;
    }

    // Internal Connectors for Data Flow
    connector RawToStaging {
        source RawCapture.rawOutPort;
        target StagingBuffer.bufferInPort;
    }

    connector CalToStaging {
        source CalibrationCapture.calOutPort;
        target StagingBuffer.bufferInPort;
    }

    connector StagingToCompute {
        source StagingBuffer.bufferOutPort;
        target PipelineCompute.computeInPort;
    }

    connector MetadataToCompute {
        source MetadataService.metadataOutPort;
        target PipelineCompute.computeInPort;
    }

    connector ComputeToStaging {
        source PipelineCompute.computeOutPort;
        target StagingBuffer.bufferOutPort;
    }

}
