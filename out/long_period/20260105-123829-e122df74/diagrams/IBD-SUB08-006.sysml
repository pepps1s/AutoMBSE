package IBD_SUB08_006_DataIntegrityPatternApplication {
  // IBD-SUB08-006
  // Type: IBD
  // Title: IBD SUB08: Data Integrity Pattern Application (Ack/Retry/Checksum on Interfaces)
  // Goal: Apply a consistent wiring pattern for ack/retry/checksum across critical internal data interfaces.
  // Dependencies: RD-F-007, BDD-SUB08-002, IBD-SUB08-005

  package TypesAndInterfaces {

    // Payload items (simplified from Data System concepts)
    item def RawFrame;
    item def StagedFrame;
    item def PipelineInputFrame;

    // Control / integrity items
    item def AckSignal;
    item def RetryRequest;

    // Integrity mechanism: checksum/digest represented as an attribute type
    attribute def Checksum32;
    attribute def SeqNumber;

    // Generic "data-with-integrity" envelope pattern
    item def DataEnvelope {
      attribute seq : SeqNumber;
      attribute checksum : Checksum32;
    }

    // Specializations used on links
    item def RawFrameEnvelope : DataEnvelope {
      item payload : RawFrame;
    }

    item def StagedFrameEnvelope : DataEnvelope {
      item payload : StagedFrame;
    }

    item def PipelineInputEnvelope : DataEnvelope {
      item payload : PipelineInputFrame;
    }

    // Port type: Payload flows one direction; ack/retry flow opposite
    port def DataIntegrityLink<P> {
      // Payload direction is decided by the part's port usage (out/in)
      in item  ackIn   : AckSignal;
      out item ackOut  : AckSignal;

      in item  retryIn : RetryRequest;
      out item retryOut: RetryRequest;

      // Integrity attribute carried/associated on the interface
      attribute linkChecksum : Checksum32;

      // Payload item (generic)
      inout item payload : P;
    }
  }

  package Structure {

    import TypesAndInterfaces::*;

    // Internal components participating in the pattern demonstration
    part def RawCaptureUnit {
      port txToStaging : DataIntegrityLink<RawFrameEnvelope>;
    }

    part def StagingBufferUnit {
      // Receives raw->staging, then transmits staging->pipeline
      port rxFromCapture : DataIntegrityLink<RawFrameEnvelope>;
      port txToPipeline  : DataIntegrityLink<PipelineInputEnvelope>;
    }

    part def PipelineComputeUnit {
      port rxFromStaging : DataIntegrityLink<PipelineInputEnvelope>;
    }

    // IBD subject for SUB08 internal data integrity wiring
    part def IBD_SUB08_DataIntegrityPatternSubject {
      part capture : RawCaptureUnit;
      part staging : StagingBufferUnit;
      part pipeline: PipelineComputeUnit;

      // Link 1: capture → staging (payload forward), ack/retry back
      // Payload direction: capture.txToStaging.payload (out) -> staging.rxFromCapture.payload (in)
      // Ack/retry direction: staging.rxFromCapture.ackOut/retryOut -> capture.txToStaging.ackIn/retryIn
      connect CaptureToStagingPayload
        connect capture.txToStaging.payload to staging.rxFromCapture.payload;

      connect CaptureToStagingAck
        connect staging.rxFromCapture.ackOut to capture.txToStaging.ackIn;

      connect CaptureToStagingRetry
        connect staging.rxFromCapture.retryOut to capture.txToStaging.retryIn;

      // Link 2: staging → pipeline (payload forward), ack/retry back
      connect StagingToPipelinePayload
        connect staging.txToPipeline.payload to pipeline.rxFromStaging.payload;

      connect StagingToPipelineAck
        connect pipeline.rxFromStaging.ackOut to staging.txToPipeline.ackIn;

      connect StagingToPipelineRetry
        connect pipeline.rxFromStaging.retryOut to staging.txToPipeline.retryIn;

      // Make checksum presence explicit on interfaces (integrity mechanism)
      // These bindings show that each interface carries/uses a checksum attribute.
      bind CaptureStagingChecksum
        bind capture.txToStaging.linkChecksum = staging.rxFromCapture.linkChecksum;

      bind StagingPipelineChecksum
        bind staging.txToPipeline.linkChecksum = pipeline.rxFromStaging.linkChecksum;
    }
  }
}
