package TMT_RD_NF_006_SafetyCompliance_StandardsPoliciesAlignment {

    import ScalarValues::*;
    import Quantities::*;

    package RD_NF_006_View {
        doc /* RD-NF-006 — Safety & Compliance NFR: Standards/Policies Alignment
             * Goal: Capture compliance-oriented NFRs (safety procedures, operational constraints, auditability)
             * relevant to site and partners for the Thirty Meter Telescope (TMT, 三十米望远镜).
             *
             * Dependencies: RD-F-008, RD-ASM-001
             *
             * Notes:
             * - This view avoids hardcoding local regulations; placeholders are used where jurisdiction-specific
             *   rules may apply.
             * - Explicit auditability + traceability requirements are included for safety events.
             */

        package Dependencies {
            requirement RD_F_008_SafetyInterlocksEmergencyResponse_Dependency {
                doc /* Shallow dependency placeholder: RD-F-008 provides safety interlock / emergency response
                     * functional requirements and hazard mitigation scaffolding.
                     */
                attribute dependencyId : String;
                attribute status : String;
                attribute impactArea : String;

                require constraint { dependencyId != "" }
            }

            requirement RD_ASM_001_AssumptionsRegister_Dependency {
                doc /* Shallow dependency placeholder: RD-ASM-001 contains assumptions/constraints/dependencies
                     * register impacting compliance interpretations.
                     */
                attribute dependencyId : String;
                attribute status : String;
                attribute impactArea : String;

                require constraint { dependencyId != "" }
            }
        }

        package CommonComplianceModel {
            enum def ComplianceStatusEnum {
                literal Draft;
                literal Proposed;
                literal Approved;
                literal Verified;
            }

            enum def EvidenceTypeEnum {
                literal TrainingRecord;
                literal AccessLog;
                literal AuditLog;
                literal Certificate;
                literal TestReport;
                literal ProcedureRecord;
                literal IncidentReport;
                literal ChangeRecord;
            }

            part def ComplianceArtifact {
                attribute artifactId : String;
                attribute evidenceType : EvidenceTypeEnum;
                attribute ownerOrg : String;
                attribute retentionDays : Integer;
                attribute tamperEvident : Boolean;
            }

            part def SafetyEventRecord {
                attribute eventId : String;
                attribute eventType : String;              // e.g., interlockTrip, emergencyStop, nearMiss (placeholder taxonomy)
                attribute eventTimestampUtc : String;      // ISO-8601 UTC timestamp string to avoid locale-specific date formats
                attribute severityLevel : Integer;         // 1..5 (placeholder)
                attribute impactedSubsystem : String;      // free text or ref-id placeholder
                attribute traceCorrelationId : String;     // links event -> logs -> telemetry -> analysis (traceability backbone)
                attribute requiresReporting : Boolean;     // jurisdiction-specific rules handled via assumptions
            }

            part def AuditLog {
                attribute logId : String;
                attribute timeSource : String;             // e.g., NTP/PTP/observatory time service (placeholder)
                attribute immutableStorageEnabled : Boolean;
                attribute retentionDays : Integer;
                attribute searchable : Boolean;
                attribute includesActorId : Boolean;
                attribute includesChangeReason : Boolean;
            }

            part def AccessControlPolicy {
                attribute policyId : String;
                attribute roleBasedAccess : Boolean;
                attribute multiFactorAuthRequired : Boolean;
                attribute leastPrivilege : Boolean;
                attribute periodicReviewDays : Integer;
            }

            part def TrainingProgram {
                attribute programId : String;
                attribute requiredForRole : Boolean;
                attribute refresherIntervalDays : Integer;
                attribute tracksCompletion : Boolean;
            }

            part def CertificationHook {
                attribute certificationSchemeName : String; // placeholder: do not hardcode
                attribute certificateId : String;
                attribute expiresAfterDays : Integer;
                attribute coversSubsystem : String;         // placeholder mapping key
            }
        }

        package ComplianceNFRs {

            // --- 1) Training ---
            requirement def ComplianceTrainingReqDef {
                doc /* Personnel performing safety-critical operations shall complete role-appropriate training,
                     * with recorded completion evidence and periodic refreshers.
                     */
                attribute trainingRequired : Boolean;
                attribute refresherIntervalDays : Integer;
                attribute evidenceRetentionDays : Integer;

                require constraint { trainingRequired == true }
            }

            requirement NFR_01_Training_RequiredAndTracked : ComplianceTrainingReqDef {
                doc /* All personnel roles that interact with safety interlocks, emergency stop, high-voltage,
                     * laser (if applicable), cryogenics (if applicable), and confined spaces (if applicable)
                     * shall have a defined training path with tracked completion evidence.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute applicableRolesDefined : Boolean;

                require constraint { applicableRolesDefined == true }
            }

            // --- 2) Physical access control ---
            requirement def PhysicalAccessControlReqDef {
                doc /* Physical access to restricted areas shall be controlled, logged, and periodically reviewed. */
                attribute accessControlEnabled : Boolean;
                attribute periodicReviewDays : Integer;
                attribute accessLogRetentionDays : Integer;
                attribute visitorEscortRequired : Boolean;

                require constraint { accessControlEnabled == true }
            }

            requirement NFR_02_PhysicalAccess_ControlAndLogging : PhysicalAccessControlReqDef {
                doc /* Restricted site areas (e.g., equipment rooms, enclosure machinery zones, HV rooms)
                     * shall enforce access control with audit logs and review cadence.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute restrictedZonesEnumerated : Boolean;

                require constraint { restrictedZonesEnumerated == true }
            }

            // --- 3) Logical access control (systems/software) ---
            requirement def LogicalAccessControlReqDef {
                doc /* Logical access to safety-relevant control systems shall enforce least privilege and
                     * strong authentication with auditable identity attribution.
                     */
                attribute roleBasedAccess : Boolean;
                attribute multiFactorAuthRequired : Boolean;
                attribute leastPrivilege : Boolean;
                attribute accountReviewDays : Integer;

                require constraint { roleBasedAccess == true }
            }

            requirement NFR_03_LogicalAccess_RBAC_MFA_Review : LogicalAccessControlReqDef {
                doc /* Observatory control software, safety PLC/interlock interfaces, and log systems
                     * shall implement RBAC, MFA (where feasible), and periodic account reviews.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute breakGlassProcedureDefined : Boolean;

                require constraint { breakGlassProcedureDefined == true }
            }

            // --- 4) Audit logs (immutability + retention) ---
            requirement def AuditLoggingReqDef {
                doc /* Safety-relevant actions and system state changes shall be logged with integrity protections
                     * and defined retention.
                     */
                attribute immutableStorageEnabled : Boolean;
                attribute retentionDays : Integer;
                attribute timeSynchronized : Boolean;
                attribute includesActorId : Boolean;

                require constraint { retentionDays > 0 }
            }

            requirement NFR_04_AuditLogs_Immutable_TimeSync_Retention : AuditLoggingReqDef {
                doc /* Interlock trips, emergency stop actions, overrides, configuration changes, and
                     * privileged access events shall produce immutable, time-synchronized audit logs.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute coversSafetyEvents : Boolean;

                require constraint { coversSafetyEvents == true }
            }

            // --- 5) Traceability for safety events (end-to-end correlation) ---
            requirement def SafetyEventTraceabilityReqDef {
                doc /* Safety events shall be traceable across telemetry, command history, audit logs,
                     * incident records, and corrective actions using correlation identifiers.
                     */
                attribute correlationIdRequired : Boolean;
                attribute linksToIncidentRecord : Boolean;
                attribute linksToCorrectiveAction : Boolean;

                require constraint { correlationIdRequired == true }
            }

            requirement NFR_05_SafetyEvent_Traceability_CorrelationBackbone : SafetyEventTraceabilityReqDef {
                doc /* Each safety event shall include a correlation identifier enabling reconstruction of:
                     * trigger -> detection -> interlock action -> operator acknowledgement -> recovery steps.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute reconstructionSupported : Boolean;

                require constraint { reconstructionSupported == true }
            }

            // --- 6) Auditability of safety procedures (procedural compliance) ---
            requirement def SafetyProcedureAuditabilityReqDef {
                doc /* Safety procedures shall be version-controlled, acknowledged by operators, and auditable. */
                attribute proceduresVersionControlled : Boolean;
                attribute acknowledgementRecorded : Boolean;
                attribute changeApprovalRequired : Boolean;

                require constraint { proceduresVersionControlled == true }
            }

            requirement NFR_06_SafetyProcedures_Versioned_Acknowledged_Auditable : SafetyProcedureAuditabilityReqDef {
                doc /* Startup/shutdown, maintenance modes, interlock override procedures, and emergency response
                     * procedures shall be auditable with recorded acknowledgements.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute coversEmergencyResponse : Boolean;

                require constraint { acknowledgementRecorded == true }
            }

            // --- 7) Configuration management & change control ---
            requirement def ChangeControlReqDef {
                doc /* Safety-critical configuration changes shall be controlled, reviewed, and fully auditable. */
                attribute changeRequestRequired : Boolean;
                attribute reviewApprovalRequired : Boolean;
                attribute rollbackAvailable : Boolean;
                attribute changeLogRetentionDays : Integer;

                require constraint { changeRequestRequired == true }
            }

            requirement NFR_07_ChangeControl_SafetyCritical_Config : ChangeControlReqDef {
                doc /* Changes to interlock logic, emergency stop behavior, hazard thresholds, and
                     * access policies shall follow controlled change management with rollback capability.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute includesReasonForChange : Boolean;

                require constraint { includesReasonForChange == true }
            }

            // --- 8) Safety certification hooks (partner/site compliance) ---
            requirement def SafetyCertificationHookReqDef {
                doc /* The system shall provide hooks to attach and verify compliance certificates or attestations
                     * required by site/operator/partners (placeholders for specific schemes).
                     */
                attribute certificationRequired : Boolean;
                attribute certificateTracked : Boolean;
                attribute expiryMonitored : Boolean;

                require constraint { certificateTracked == true }
            }

            requirement NFR_08_CertificationHooks_Attach_Verify_Expiry : SafetyCertificationHookReqDef {
                doc /* Provide data model elements to reference safety certificates/attestations for
                     * safety-relevant subsystems and operational processes without hardcoding local regulations.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute supportsMultipleSchemes : Boolean;

                require constraint { supportsMultipleSchemes == true }
            }

            // --- 9) Incident reporting readiness (jurisdiction-agnostic) ---
            requirement def IncidentReportingReadinessReqDef {
                doc /* The system shall support incident reporting workflows with placeholders for
                     * jurisdiction-specific reporting rules.
                     */
                attribute incidentRecordRequired : Boolean;
                attribute reportingRulePlaceholderUsed : Boolean;
                attribute reportGenerationSupported : Boolean;

                require constraint { reportingRulePlaceholderUsed == true }
            }

            requirement NFR_09_IncidentReporting_Workflow_Placeholders : IncidentReportingReadinessReqDef {
                doc /* Safety events (e.g., interlock trip, emergency stop, near miss) shall be captured
                     * in incident records and support report generation using assumption-driven reporting rules.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute supportsPartnerReporting : Boolean;

                require constraint { incidentRecordRequired == true }
            }

            // --- 10) Data privacy & minimization for logs (partner/site alignment) ---
            requirement def LogPrivacyMinimizationReqDef {
                doc /* Audit and access logs shall minimize collected personal data while preserving
                     * accountability and traceability.
                     */
                attribute actorIdPseudonymizationSupported : Boolean;
                attribute accessControlledLogViews : Boolean;
                attribute retentionPolicyDefined : Boolean;

                require constraint { retentionPolicyDefined == true }
            }

            requirement NFR_10_LogPrivacy_MinimizeButAuditable : LogPrivacyMinimizationReqDef {
                doc /* Logging shall support privacy controls (e.g., pseudonymization where appropriate),
                     * role-based access to sensitive logs, and defined retention consistent with partner/site policy.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute supportsLegalHoldPlaceholder : Boolean;

                require constraint { accessControlledLogViews == true }
            }

            // --- 11) Time synchronization for auditability ---
            requirement def TimeSyncForAuditabilityReqDef {
                doc /* All safety-relevant logs and event records shall use synchronized time sources
                     * to enable forensic reconstruction.
                     */
                attribute timeSyncRequired : Boolean;
                attribute maxClockSkewSeconds : Integer;

                require constraint { maxClockSkewSeconds >= 0 }
            }

            requirement NFR_11_TimeSync_Auditability_ClockSkewBound : TimeSyncForAuditabilityReqDef {
                doc /* Safety events and audit logs shall be time-stamped using a common time base
                     * with bounded clock skew across subsystems.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute timeServiceDefined : Boolean;

                require constraint { timeSyncRequired == true }
            }

            // --- 12) Periodic compliance review & internal audits ---
            requirement def ComplianceReviewReqDef {
                doc /* Compliance controls shall be subject to periodic review and internal audit, with
                     * tracked findings and closure.
                     */
                attribute reviewIntervalDays : Integer;
                attribute findingsTracked : Boolean;
                attribute closureEvidenceRequired : Boolean;

                require constraint { reviewIntervalDays > 0 }
            }

            requirement NFR_12_ComplianceReview_InternalAudit_FindingsClosure : ComplianceReviewReqDef {
                doc /* Perform periodic reviews of training status, access control lists, audit log integrity,
                     * and change control records; track findings to closure with evidence artifacts.
                     */
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute supportsAuditTrailOfAudits : Boolean;

                require constraint { findingsTracked == true }
            }
        }

        package Assumptions_Placeholders {
            requirement ASSUMP_01_RegulatoryFramework_UnknownPlaceholder {
                doc /* Assumption placeholder: Applicable local regulatory frameworks, standards, and
                     * permitting obligations for the site are not enumerated in this view. Specific obligations
                     * shall be captured in RD-ASM-001 and referenced here via identifiers.
                     */
                attribute assumptionId : String;
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute impactArea : String;

                require constraint { assumptionId != "" }
            }

            requirement ASSUMP_02_PartnerPolicies_Vary {
                doc /* Assumption placeholder: Partner organizations may have additional safety/compliance
                     * policies. The model supports attaching multiple policy/certification schemes without
                     * hardcoding their names.
                     */
                attribute assumptionId : String;
                attribute status : CommonComplianceModel::ComplianceStatusEnum;
                attribute impactArea : String;

                require constraint { assumptionId != "" }
            }
        }

        package ComplianceArtifacts_Examples {
            doc /* Example artifacts to support auditability/traceability NFRs (not exhaustive). */

            part TrainingEvidence : CommonComplianceModel::ComplianceArtifact {
                artifactId = "ART-TRN-PLACEHOLDER";
                evidenceType = CommonComplianceModel::EvidenceTypeEnum::TrainingRecord;
                ownerOrg = "TMT-OPS-PLACEHOLDER";
                retentionDays = 3650;
                tamperEvident = false;
            }

            part AccessEvidence : CommonComplianceModel::ComplianceArtifact {
                artifactId = "ART-ACC-PLACEHOLDER";
                evidenceType = CommonComplianceModel::EvidenceTypeEnum::AccessLog;
                ownerOrg = "TMT-SITE-PLACEHOLDER";
                retentionDays = 3650;
                tamperEvident = true;
            }

            part AuditEvidence : CommonComplianceModel::ComplianceArtifact {
                artifactId = "ART-AUD-PLACEHOLDER";
                evidenceType = CommonComplianceModel::EvidenceTypeEnum::AuditLog;
                ownerOrg = "TMT-OCS-PLACEHOLDER";
                retentionDays = 3650;
                tamperEvident = true;
            }

            part SafetyEventExample : CommonComplianceModel::SafetyEventRecord {
                eventId = "SEV-PLACEHOLDER-0001";
                eventType = "interlockTrip";
                eventTimestampUtc = "YYYY-MM-DDThh:mm:ssZ";
                severityLevel = 3;
                impactedSubsystem = "SafetyInterlockSubsystem-PLACEHOLDER";
                traceCorrelationId = "CORR-PLACEHOLDER-0001";
                requiresReporting = false;
            }
        }
    }
}
