package AD_SUB08_001_DataCapture_Integrity_Staging {

  // === Item / Value Types ===
  item def RawDataUnit {
    attribute frameId : String;
    attribute sourceId : String;
    attribute byteCount : Integer;
    attribute expectedByteCount : Integer;
  }

  item def ObservingContext {
    attribute observationId : String;
    attribute instrumentId : String;
    attribute site : String;
    attribute timeReference : String;
  }

  item def ConfigurationContext {
    attribute configId : String;
    attribute schemaVersion : String;
    attribute buildId : String;
  }

  item def Digest {
    attribute algorithm : String;
    attribute value : String;
  }

  item def IntegrityResult {
    attribute ok : Boolean;
    attribute failureReason : String;
    attribute detectedMissingOrTruncated : Boolean;
  }

  item def ProvenanceRecord {
    attribute provenanceId : String;
    attribute observationId : String;
    attribute configId : String;
    attribute ingestTimestamp : String;
  }

  item def MetadataRecord {
    attribute metadataId : String;
    attribute schemaVersion : String;
    attribute frameId : String;
    attribute observationId : String;
  }

  item def PipelineStagingHandle {
    attribute stagingId : String;
    attribute locationUri : String;
  }

  item def DataProductHandle {
    attribute productId : String;
    attribute deliveryToken : String;
    attribute locationUri : String;
    attribute status : String;
  }

  // === Reusable Action Definitions ===
  action def computeDigest {
    in item raw : RawDataUnit;
    out item digest : Digest;

    first start;
    then action 'compute checksum/digest';
    then done;
  }

  action def detectMissingOrTruncatedData {
    in item raw : RawDataUnit;
    out item missingOrTruncated : Boolean;

    first start;
    then action 'detect missing blocks / truncated frame';
    then done;
  }

  action def evaluateIntegrity {
    in item raw : RawDataUnit;
    in item digest : Digest;
    in item missingOrTruncated : Boolean;
    out item result : IntegrityResult;

    first start;
    then action 'evaluate digest and completeness';
    then done;
  }

  action def registerProvenance {
    in item obs : ObservingContext;
    in item cfg : ConfigurationContext;
    in item raw : RawDataUnit;
    out item prov : ProvenanceRecord;

    first start;
    then action 'register provenance (source, timestamps, context linkage)';
    then done;
  }

  action def persistMetadataSchemaVersion {
    in item cfg : ConfigurationContext;
    in item prov : ProvenanceRecord;
    in item raw : RawDataUnit;
    out item meta : MetadataRecord;

    first start;
    then action 'persist metadata with schema version';
    then done;
  }

  action def stageForPipeline {
    in item raw : RawDataUnit;
    in item meta : MetadataRecord;
    in item prov : ProvenanceRecord;
    out item staging : PipelineStagingHandle;

    first start;
    then action 'stage raw + metadata for pipeline ingestion';
    then done;
  }

  action def produceDeliveryToken {
    in item staging : PipelineStagingHandle;
    in item meta : MetadataRecord;
    out item product : DataProductHandle;

    first start;
    then action 'create delivery-ready artifact handle/token';
    then done;
  }

  action def sendAcknowledgement {
    in item raw : RawDataUnit;
    in item ok : Boolean;

    first start;
    then action 'send acknowledgement (ACK/NACK) per data flow interface';
    then done;
  }

  action def requestRetryOrRetransmit {
    in item raw : RawDataUnit;
    in item reason : String;

    first start;
    then action 'request retry / retransmit per interface expectations';
    then done;
  }

  // === Activity (AD) ===
  action AD_SUB08_001_DataCapture_IntegrityCheck_DeliveryStaging {
    // Inputs required by acceptance checks
    in item rawIn : RawDataUnit;
    in item obsIn : ObservingContext;
    in item cfgIn : ConfigurationContext;

    // Output required by acceptance checks
    out item dataProduct : DataProductHandle;

    // Internal flow artifacts
    item digest : Digest;
    item missingOrTruncated : Boolean;
    item integrity : IntegrityResult;
    item prov : ProvenanceRecord;
    item meta : MetadataRecord;
    item staging : PipelineStagingHandle;

    first start;

    // 1) Integrity check (checksum/digest + missing/truncated detection)
    then action digestStep : computeDigest { in raw = rawIn; }
    then action missingStep : detectMissingOrTruncatedData { in raw = rawIn; }
    then action integrityStep : evaluateIntegrity {
      in raw = rawIn;
      in digest = digestStep.digest;
      in missingOrTruncated = missingStep.missingOrTruncated;
    }

    // 2) Branch on integrity failure (explicit)
    then action integrityDecision if integrityStep.result.ok == true {
      // PASS path
      then action ackPass : sendAcknowledgement { in raw = rawIn; in ok = true; }

      // 3) Register provenance
      then action provStep : registerProvenance { in obs = obsIn; in cfg = cfgIn; in raw = rawIn; }

      // 4) Persist metadata schema version
      then action metaStep : persistMetadataSchemaVersion { in cfg = cfgIn; in prov = provStep.prov; in raw = rawIn; }

      // 5) Stage for pipeline
      then action stageStep : stageForPipeline { in raw = rawIn; in meta = metaStep.meta; in prov = provStep.prov; }

      // 6) Produce delivery-ready token/handle
      then action deliverStep : produceDeliveryToken { in staging = stageStep.staging; in meta = metaStep.meta; }

      then action 'set output handle' {
        // bind output
      }
      then done;

    } else {
      // FAIL path (retry + acknowledgement behavior)
      then action nackFail : sendAcknowledgement { in raw = rawIn; in ok = false; }
      then action retryReq : requestRetryOrRetransmit {
        in raw = rawIn;
        in reason = integrityStep.result.failureReason;
      }

      // Optional local retry loop hook (named, explicit)
      then action retryDecision if false {
        then action 'placeholder: local retry attempt' { }
        then done;
      } else {
        // Terminal failure outcome as a handle (delivery token indicates failure)
        then action failToken : 'produce failure DataProductHandle' {
          // semantic intent: return a delivery token that indicates failure for downstream handling
        }
        then done;
      }
    }

    // Output binding (nominally from deliverStep.product, or failure token)
    // (Modeling tools may resolve binding via feature chains; kept minimal here.)
  }

}
