package RD_F_007_DataAcquisition_Metadata_DataProduct_FunctionalRequirementsSet {
    import ScalarValues::*;
    import Quantities::*;

    doc /* RD-F-007 — Data Acquisition, Metadata & Data Product Functional Requirements Set
         * Goal: Define functional requirements for raw data capture, metadata integrity,
         *       quick-look QA, pipeline interfaces, and archive delivery.
         * Dependencies: RD-F-002, RD-CTX-001
         * Target System: Thirty Meter Telescope (TMT, 三十米望远镜)
         */

    // ---- Trace anchors to upstream views (black-box trace points) ----
    requirement def RD_F_007_DependencyAnchorDef {
        doc /* This definition provides explicit anchors for traceability to upstream
             * requirement/context sets without committing to implementation choices. */
        attribute dependsOn_RD_F_002 : Boolean;
        attribute dependsOn_RD_CTX_001 : Boolean;
        require constraint { dependsOn_RD_F_002 == true and dependsOn_RD_CTX_001 == true }
    }

    requirement DepAnchors : RD_F_007_DependencyAnchorDef {
        dependsOn_RD_F_002 = true;
        dependsOn_RD_CTX_001 = true;
    }

    // ---- Common interface and provenance vocabulary (black-box, tech-agnostic) ----
    requirement def DataFlowInterfaceDef {
        doc /* Abstract interface for data transfer between subsystems.
             * This is black-box: it constrains observable behaviors, not storage technology. */
        attribute supportsPull : Boolean;
        attribute supportsPush : Boolean;
        attribute supportsAcknowledgement : Boolean;
        attribute supportsRetry : Boolean;
        require constraint { (supportsPull == true) or (supportsPush == true) }
    }

    requirement def ProvenanceContextDef {
        doc /* Provenance context required to enable traceable lineage of raw data and products:
             * configuration + environmental + calibration context. */
        attribute includesConfigurationContext : Boolean;
        attribute includesEnvironmentalContext : Boolean;
        attribute includesCalibrationContext : Boolean;
        require constraint {
            includesConfigurationContext == true and
            includesEnvironmentalContext == true and
            includesCalibrationContext == true
        }
    }

    requirement def IntegrityCheckDef {
        doc /* Integrity checks required to detect corruption and ensure end-to-end fidelity. */
        attribute supportsChecksumOrDigest : Boolean;
        attribute supportsEndToEndVerification : Boolean;
        attribute detectsMissingOrTruncatedData : Boolean;
        require constraint {
            supportsChecksumOrDigest == true and
            supportsEndToEndVerification == true and
            detectsMissingOrTruncatedData == true
        }
    }

    requirement def MetadataCompletenessDef {
        doc /* Metadata completeness required for scientific usability, traceability,
             * and downstream processing. */
        attribute requiredFieldsPresent : Boolean;
        attribute schemaVersionRecorded : Boolean;
        attribute unitsAndFramesDeclared : Boolean;
        require constraint {
            requiredFieldsPresent == true and
            schemaVersionRecorded == true and
            unitsAndFramesDeclared == true
        }
    }

    requirement def TimestampingDef {
        doc /* Timestamping requirements for exposures/readouts/events.
             * Focus is on ordering, correlation, and traceability (not time source technology). */
        attribute timestampsRecorded : Boolean;
        attribute supportsMonotonicOrdering : Boolean;
        attribute recordsTimeReferenceInfo : Boolean;
        require constraint {
            timestampsRecorded == true and
            supportsMonotonicOrdering == true and
            recordsTimeReferenceInfo == true
        }
    }

    requirement def QuickLookQADef {
        doc /* Quick-look QA requirements to provide timely feedback for observing decisions,
             * without specifying algorithmic implementation. */
        attribute qaGenerated : Boolean;
        attribute qaAvailableBeforeNextExposureDecision : Boolean;
        attribute qaIncludesBasicQualityFlags : Boolean;
        require constraint {
            qaGenerated == true and
            qaAvailableBeforeNextExposureDecision == true and
            qaIncludesBasicQualityFlags == true
        }
    }

    // ---- Functional Requirements (at least 12) ----

    requirement def RawDataCaptureReqDef {
        doc /* The Data System shall capture raw detector readouts for each commanded exposure
             * and preserve them as identifiable raw data units suitable for downstream processing. */
        attribute captureEnabled : Boolean;
        attribute rawUnitIdentifiable : Boolean;
        require constraint { captureEnabled == true and rawUnitIdentifiable == true }
    }

    requirement def ExposureTimestampingReqDef {
        doc /* The Data System shall record timestamps for exposure start/end and readout events
             * sufficient to correlate with observatory telemetry and configuration states. */
        attribute timestampsRecorded : Boolean;
        attribute correlatableWithTelemetry : Boolean;
        require constraint { timestampsRecorded == true and correlatableWithTelemetry == true }
    }

    requirement def MetadataCompletenessReqDef {
        doc /* The Data System shall capture metadata required to interpret raw data,
             * including instrument configuration, observing parameters, and coordinate frames. */
        attribute requiredFieldsPresent : Boolean;
        attribute schemaVersionRecorded : Boolean;
        require constraint { requiredFieldsPresent == true and schemaVersionRecorded == true }
    }

    requirement def MetadataIntegrityReqDef {
        doc /* The Data System shall ensure metadata integrity by detecting missing, malformed,
             * or inconsistent metadata prior to downstream handoff. */
        attribute detectsMissingMetadata : Boolean;
        attribute detectsMalformedMetadata : Boolean;
        attribute detectsInconsistentMetadata : Boolean;
        require constraint {
            detectsMissingMetadata == true and
            detectsMalformedMetadata == true and
            detectsInconsistentMetadata == true
        }
    }

    requirement def RawDataIntegrityReqDef {
        doc /* The Data System shall perform integrity verification on raw data units
             * to detect corruption or truncation during capture and transfer. */
        attribute supportsChecksumOrDigest : Boolean;
        attribute detectsMissingOrTruncatedData : Boolean;
        require constraint { supportsChecksumOrDigest == true and detectsMissingOrTruncatedData == true }
    }

    requirement def EndToEndTransferVerificationReqDef {
        doc /* The Data System shall support end-to-end verification for transfers
             * from acquisition to downstream consumers, ensuring bitwise fidelity. */
        attribute supportsEndToEndVerification : Boolean;
        attribute supportsAcknowledgement : Boolean;
        require constraint { supportsEndToEndVerification == true and supportsAcknowledgement == true }
    }

    requirement def ProvenanceTraceabilityReqDef {
        doc /* The Data System shall record provenance sufficient to trace each raw data unit
             * and derived product back to configuration state, environmental conditions,
             * and calibration context used during acquisition. */
        attribute includesConfigurationContext : Boolean;
        attribute includesEnvironmentalContext : Boolean;
        attribute includesCalibrationContext : Boolean;
        require constraint {
            includesConfigurationContext == true and
            includesEnvironmentalContext == true and
            includesCalibrationContext == true
        }
    }

    requirement def ConfigurationContextCaptureReqDef {
        doc /* The Data System shall capture configuration context identifiers and values
             * (e.g., instrument mode, detector settings, AO state, observing template identifiers)
             * needed to reproduce processing and interpretation. */
        attribute configContextCaptured : Boolean;
        attribute configContextTraceableToSource : Boolean;
        require constraint { configContextCaptured == true and configContextTraceableToSource == true }
    }

    requirement def EnvironmentalContextCaptureReqDef {
        doc /* The Data System shall capture environmental context associated with acquisition
             * (e.g., site conditions and relevant telescope environment telemetry) sufficient for
             * data quality assessment and later analysis. */
        attribute envContextCaptured : Boolean;
        attribute envContextTimeAligned : Boolean;
        require constraint { envContextCaptured == true and envContextTimeAligned == true }
    }

    requirement def CalibrationContextCaptureReqDef {
        doc /* The Data System shall associate calibration context with raw data and products,
             * including references to calibration exposures and applicable calibration states. */
        attribute calibContextCaptured : Boolean;
        attribute calibReferencesResolvable : Boolean;
        require constraint { calibContextCaptured == true and calibReferencesResolvable == true }
    }

    requirement def QuickLookQAReqDef {
        doc /* The Data System shall generate quick-look QA outputs for each exposure
             * to support operational decisions (continue/adjust/abort) without delay. */
        attribute qaGenerated : Boolean;
        attribute qaAvailableBeforeNextExposureDecision : Boolean;
        require constraint { qaGenerated == true and qaAvailableBeforeNextExposureDecision == true }
    }

    requirement def QuickLookQAContentReqDef {
        doc /* Quick-look QA shall include basic quality indicators (e.g., saturation/invalid pixels,
             * background level indicators, and completeness flags) suitable for operator review. */
        attribute qaIncludesBasicQualityFlags : Boolean;
        attribute qaIncludesCompletenessIndicators : Boolean;
        require constraint { qaIncludesBasicQualityFlags == true and qaIncludesCompletenessIndicators == true }
    }

    requirement def PipelineHandoffInterfaceReqDef {
        doc /* The Data System shall provide a black-box data delivery interface to downstream
             * processing pipelines, enabling transfer of raw data units and associated metadata. */
        attribute supportsPull : Boolean;
        attribute supportsPush : Boolean;
        attribute supportsRetry : Boolean;
        require constraint { (supportsPull == true or supportsPush == true) and supportsRetry == true }
    }

    requirement def PipelineInterfaceContractReqDef {
        doc /* The pipeline handoff interface shall provide an explicit contract for:
             * (1) data unit identification, (2) metadata schema/version, and (3) delivery confirmation,
             * without committing to specific storage technologies or transports. */
        attribute rawUnitIdentifiable : Boolean;
        attribute schemaVersionRecorded : Boolean;
        attribute supportsAcknowledgement : Boolean;
        require constraint { rawUnitIdentifiable == true and schemaVersionRecorded == true and supportsAcknowledgement == true }
    }

    requirement def ArchiveDeliveryInterfaceReqDef {
        doc /* The Data System shall provide a black-box delivery interface for archiving and distribution
             * of raw data and defined data products, including integrity verification and provenance. */
        attribute supportsChecksumOrDigest : Boolean;
        attribute supportsAcknowledgement : Boolean;
        attribute includesConfigurationContext : Boolean;
        attribute includesEnvironmentalContext : Boolean;
        attribute includesCalibrationContext : Boolean;
        require constraint {
            supportsChecksumOrDigest == true and
            supportsAcknowledgement == true and
            includesConfigurationContext == true and
            includesEnvironmentalContext == true and
            includesCalibrationContext == true
        }
    }

    requirement def DeliveryFailureHandlingReqDef {
        doc /* The Data System shall detect and handle delivery failures by supporting retries
             * and preserving sufficient diagnostics to enable recovery and audit. */
        attribute supportsRetry : Boolean;
        attribute failureDiagnosable : Boolean;
        attribute preservesAuditTrail : Boolean;
        require constraint { supportsRetry == true and failureDiagnosable == true and preservesAuditTrail == true }
    }

    // ---- Requirement instances (RD view contents) ----

    requirement RD_F_007_RawDataCapture : RawDataCaptureReqDef {
        captureEnabled = true;
        rawUnitIdentifiable = true;
    }

    requirement RD_F_007_Timestamping : ExposureTimestampingReqDef {
        timestampsRecorded = true;
        correlatableWithTelemetry = true;
    }

    requirement RD_F_007_MetadataCompleteness : MetadataCompletenessReqDef {
        requiredFieldsPresent = true;
        schemaVersionRecorded = true;
    }

    requirement RD_F_007_MetadataIntegrity : MetadataIntegrityReqDef {
        detectsMissingMetadata = true;
        detectsMalformedMetadata = true;
        detectsInconsistentMetadata = true;
    }

    requirement RD_F_007_RawDataIntegrity : RawDataIntegrityReqDef {
        supportsChecksumOrDigest = true;
        detectsMissingOrTruncatedData = true;
    }

    requirement RD_F_007_EndToEndTransferVerification : EndToEndTransferVerificationReqDef {
        supportsEndToEndVerification = true;
        supportsAcknowledgement = true;
    }

    requirement RD_F_007_ProvenanceTraceability : ProvenanceTraceabilityReqDef {
        includesConfigurationContext = true;
        includesEnvironmentalContext = true;
        includesCalibrationContext = true;
    }

    requirement RD_F_007_ConfigContextCapture : ConfigurationContextCaptureReqDef {
        configContextCaptured = true;
        configContextTraceableToSource = true;
    }

    requirement RD_F_007_EnvContextCapture : EnvironmentalContextCaptureReqDef {
        envContextCaptured = true;
        envContextTimeAligned = true;
    }

    requirement RD_F_007_CalibrationContextCapture : CalibrationContextCaptureReqDef {
        calibContextCaptured = true;
        calibReferencesResolvable = true;
    }

    requirement RD_F_007_QuickLookQA : QuickLookQAReqDef {
        qaGenerated = true;
        qaAvailableBeforeNextExposureDecision = true;
    }

    requirement RD_F_007_QuickLookQAContent : QuickLookQAContentReqDef {
        qaIncludesBasicQualityFlags = true;
        qaIncludesCompletenessIndicators = true;
    }

    requirement RD_F_007_PipelineHandoffInterface : PipelineHandoffInterfaceReqDef {
        supportsPull = true;
        supportsPush = true;
        supportsRetry = true;
    }

    requirement RD_F_007_PipelineInterfaceContract : PipelineInterfaceContractReqDef {
        rawUnitIdentifiable = true;
        schemaVersionRecorded = true;
        supportsAcknowledgement = true;
    }

    requirement RD_F_007_ArchiveDeliveryInterface : ArchiveDeliveryInterfaceReqDef {
        supportsChecksumOrDigest = true;
        supportsAcknowledgement = true;
        includesConfigurationContext = true;
        includesEnvironmentalContext = true;
        includesCalibrationContext = true;
    }

    requirement RD_F_007_DeliveryFailureHandling : DeliveryFailureHandlingReqDef {
        supportsRetry = true;
        failureDiagnosable = true;
        preservesAuditTrail = true;
    }

    // ---- Cross-cutting constraint bundles (optional reusable defs) ----

    requirement RD_F_007_Interface_Primitives : DataFlowInterfaceDef {
        supportsPull = true;
        supportsPush = true;
        supportsAcknowledgement = true;
        supportsRetry = true;
    }

    requirement RD_F_007_RequiredProvenance : ProvenanceContextDef {
        includesConfigurationContext = true;
        includesEnvironmentalContext = true;
        includesCalibrationContext = true;
    }

    requirement RD_F_007_RequiredIntegrityChecks : IntegrityCheckDef {
        supportsChecksumOrDigest = true;
        supportsEndToEndVerification = true;
        detectsMissingOrTruncatedData = true;
    }

    requirement RD_F_007_RequiredMetadataCompleteness : MetadataCompletenessDef {
        requiredFieldsPresent = true;
        schemaVersionRecorded = true;
        unitsAndFramesDeclared = true;
    }

    requirement RD_F_007_RequiredTimestamping : TimestampingDef {
        timestampsRecorded = true;
        supportsMonotonicOrdering = true;
        recordsTimeReferenceInfo = true;
    }

    requirement RD_F_007_RequiredQuickLookQA : QuickLookQADef {
        qaGenerated = true;
        qaAvailableBeforeNextExposureDecision = true;
        qaIncludesBasicQualityFlags = true;
    }
}
