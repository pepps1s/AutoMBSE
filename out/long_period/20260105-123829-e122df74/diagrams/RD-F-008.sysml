package TMT_RD_F_008_Safety_Interlocks_EmergencyResponse {
    import ScalarValues::*;
    import Quantities::*;
    
    doc /* 
     * diagram_id: RD-F-008
     * type: RD
     * title: Safety, Interlocks & Emergency Response Functional Requirements Set
     * goal: Capture functional requirements for personnel safety, interlocks, emergency stop, hazard detection, and controlled shutdown.
     * dependencies: RD-CTX-002, RD-ASM-001
     */

    /**********************************************************************
     * Context: hazards + mitigations catalog (functional obligations only)
     **********************************************************************/
    enum def HazardType {
        MotionHazard,
        HighVoltageHazard,
        LaserRadiationHazard,
        CryogenicHazard,
        OxygenDeficiencyHazard,
        FireSmokeHazard,
        SeismicHazard,
        WindStormHazard
    }

    requirement def HazardMitigationReqDef {
        doc /* Hazard shall be detected and mitigation shall be initiated to prevent harm.
             * This is a functional safety obligation; no implementation specifics. */
        attribute hazard : HazardType;
        attribute detectionTrigger : Boolean;
        attribute mitigationRequired : Boolean;
        attribute actionOutcomeAchieved : Boolean;

        require constraint { mitigationRequired == true }
    }

    requirement def SafeStateObligationReqDef {
        doc /* Upon safety trigger, the subsystem shall transition to a defined safe-state
             * and hold until an authorized reset/clear occurs. */
        attribute triggerDetected : Boolean;
        attribute safeStateEntered : Boolean;
        attribute safeStateMaintained : Boolean;

        require constraint { triggerDetected -> (safeStateEntered and safeStateMaintained) }
    }

    /**********************************************************************
     * RD: Safety, Interlocks & Emergency Response (functional requirements)
     **********************************************************************/
    package RD_F_008_SafetyInterlocksEmergencyResponse {

        /*******************************
         * Hazards + Required Mitigations
         *******************************/
        requirement Hazard_01_MotionMitigation : HazardMitigationReqDef {
            doc /* Motion hazard (moving telescope structure, rotating dome, segment handling) shall be detected,
                 * and motion shall be prevented or halted to protect personnel. */
            attribute hazard = HazardType::MotionHazard;
            attribute detectionTrigger : Boolean;
            attribute mitigationRequired = true;
            attribute actionOutcomeAchieved : Boolean;
            require constraint { detectionTrigger -> actionOutcomeAchieved }
        }

        requirement Hazard_02_HighVoltageMitigation : HazardMitigationReqDef {
            doc /* High-voltage hazard shall be detected and electrical hazard exposure shall be mitigated
                 * by transitioning affected domains to a non-energized safe condition. */
            attribute hazard = HazardType::HighVoltageHazard;
            attribute detectionTrigger : Boolean;
            attribute mitigationRequired = true;
            attribute actionOutcomeAchieved : Boolean;
            require constraint { detectionTrigger -> actionOutcomeAchieved }
        }

        requirement Hazard_03_LaserMitigation : HazardMitigationReqDef {
            doc /* Laser radiation hazard (if laser guide star operations are present) shall be detected,
                 * and emission shall be inhibited or terminated when safety conditions are violated. */
            attribute hazard = HazardType::LaserRadiationHazard;
            attribute detectionTrigger : Boolean;
            attribute mitigationRequired = true;
            attribute actionOutcomeAchieved : Boolean;
            require constraint { detectionTrigger -> actionOutcomeAchieved }
        }

        requirement Hazard_04_OxygenDeficiencyMitigation : HazardMitigationReqDef {
            doc /* Oxygen deficiency hazard shall be detected and personnel protection mitigation shall be initiated,
                 * including forced transition to safe operational constraints and emergency notification. */
            attribute hazard = HazardType::OxygenDeficiencyHazard;
            attribute detectionTrigger : Boolean;
            attribute mitigationRequired = true;
            attribute actionOutcomeAchieved : Boolean;
            require constraint { detectionTrigger -> actionOutcomeAchieved }
        }

        requirement Hazard_05_FireSmokeMitigation : HazardMitigationReqDef {
            doc /* Fire/smoke hazard shall be detected and emergency response actions shall be initiated
                 * to protect personnel and limit hazard escalation. */
            attribute hazard = HazardType::FireSmokeHazard;
            attribute detectionTrigger : Boolean;
            attribute mitigationRequired = true;
            attribute actionOutcomeAchieved : Boolean;
            require constraint { detectionTrigger -> actionOutcomeAchieved }
        }

        /********************************************
         * Safe-state behavior obligations by subsystem
         ********************************************/
        requirement SS_01_MountAndDrivesSafeState : SafeStateObligationReqDef {
            doc /* On any emergency stop or personnel safety trigger, mount/drives shall enter a safe-state
                 * that prevents hazardous motion and maintains that condition until authorized reset. */
            attribute triggerDetected : Boolean;
            attribute safeStateEntered : Boolean;
            attribute safeStateMaintained : Boolean;
        }

        requirement SS_02_EnclosureDomeSafeState : SafeStateObligationReqDef {
            doc /* On safety trigger, enclosure/dome mechanisms shall enter a safe-state that prevents hazardous motion
                 * and prevents creating new personnel hazards (e.g., moving shutters/rotation). */
            attribute triggerDetected : Boolean;
            attribute safeStateEntered : Boolean;
            attribute safeStateMaintained : Boolean;
        }

        requirement SS_03_OpticsHandlingSafeState : SafeStateObligationReqDef {
            doc /* On safety trigger, optics handling/segment handling operations shall enter safe-state
                 * to prevent unintended motion or release of loads and shall remain latched until cleared. */
            attribute triggerDetected : Boolean;
            attribute safeStateEntered : Boolean;
            attribute safeStateMaintained : Boolean;
        }

        requirement SS_04_HVPowerDomainsSafeState : SafeStateObligationReqDef {
            doc /* On electrical hazard trigger or emergency stop, high-voltage power domains shall transition
                 * to a safe-state that eliminates hazardous energy exposure and remains latched until cleared. */
            attribute triggerDetected : Boolean;
            attribute safeStateEntered : Boolean;
            attribute safeStateMaintained : Boolean;
        }

        requirement SS_05_LaserSubsystemSafeState : SafeStateObligationReqDef {
            doc /* On laser safety trigger or interlock violation, any laser emission capability shall enter safe-state
                 * with emission inhibited/terminated and remain inhibited until authorized reset. */
            attribute triggerDetected : Boolean;
            attribute safeStateEntered : Boolean;
            attribute safeStateMaintained : Boolean;
        }

        /********************************************
         * Functional safety requirements (>= 15)
         * Each includes explicit trigger/action outcomes.
         ********************************************/
        requirement FSR_01_EStop_Global : SafetyTriggerActionReqDef {
            doc /* If any emergency stop is actuated, the system shall initiate a global emergency response
                 * that transitions all motion-capable subsystems to safe-state and prevents restart until reset/clear. */
        }

        requirement def SafetyTriggerActionReqDef {
            doc /* A functional safety requirement expressed as Trigger -> Action Outcome(s). */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;

            require constraint { triggerOccurred -> (actionInitiated and safeOutcomeAchieved and restartPreventedUntilClear) }
        }

        requirement FSR_02_GuardDoorInterlock_MotionInhibit : SafetyTriggerActionReqDef {
            doc /* If a guarded access door/area interlock is open/violated, hazardous motion shall be inhibited
                 * or an active motion shall be brought to safe-state, and restart shall be prevented until cleared. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_03_PresenceDetected_MotionRestrict : SafetyTriggerActionReqDef {
            doc /* If personnel presence is detected within a defined hazard zone while hazardous motion is possible,
                 * the system shall enforce a safe operational restriction that prevents hazardous motion in that zone. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_04_SlewCommand_InterlockCheck : SafetyTriggerActionReqDef {
            doc /* If a slew/drive motion command is requested, the system shall verify interlocks/safety conditions;
                 * if any required condition is not satisfied, motion shall not start and an operator-visible safety alert shall be raised. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
            require constraint { triggerOccurred -> safeOutcomeAchieved } /* outcome includes "no hazardous motion starts" */
        }

        requirement FSR_05_UnexpectedMotionDetected_Halt : SafetyTriggerActionReqDef {
            doc /* If unexpected or unauthorized motion is detected, the system shall command transition to safe-state
                 * for affected motion axes and shall raise an emergency alarm. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_06_HighVoltageAccessInterlock_Deenergize : SafetyTriggerActionReqDef {
            doc /* If high-voltage access interlock is violated (e.g., panel open during energized state),
                 * the system shall initiate safe-state for the affected electrical domain such that hazardous energy is not exposed. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_07_GroundFaultOrOvercurrent_ShutdownDomain : SafetyTriggerActionReqDef {
            doc /* If a ground-fault or overcurrent safety trigger occurs, the system shall transition the affected power domain
                 * to a safe-state and prevent re-energization until the fault is cleared. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_08_FireOrSmokeDetected_EmergencyResponse : SafetyTriggerActionReqDef {
            doc /* If fire or smoke is detected, the system shall initiate emergency response actions:
                 * (a) transition hazardous operations to safe-state, (b) notify operators/emergency channels, and
                 * (c) maintain safe-state until all-clear. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_09_OxygenDeficiencyDetected_RestrictAccess : SafetyTriggerActionReqDef {
            doc /* If oxygen deficiency is detected, the system shall initiate emergency response actions:
                 * restrict hazardous operations, notify personnel, and prevent entry/continued operation in affected zones until cleared. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_10_CryogenicLeakOrOverpressure_ProtectPersonnel : SafetyTriggerActionReqDef {
            doc /* If cryogenic leak or overpressure trigger occurs, the system shall transition affected cryogenic operations
                 * to a safe-state and initiate alarms/notifications to protect personnel. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_11_LaserInterlockViolation_InhibitEmission : SafetyTriggerActionReqDef {
            doc /* If any laser safety interlock is violated (e.g., access/beam path safety condition not met),
                 * the system shall inhibit/terminate laser emission and prevent re-enable until authorized reset. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_12_SeismicEventDetected_ControlledSafeing : SafetyTriggerActionReqDef {
            doc /* If seismic event trigger exceeds defined threshold, the system shall initiate controlled safeing:
                 * suspend hazardous motion, transition to safe-state, and maintain until cleared. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_13_HighWindOrStormDetected_SecureOperations : SafetyTriggerActionReqDef {
            doc /* If high wind/storm trigger conditions are detected, the system shall initiate protective actions:
                 * restrict hazardous operations and transition relevant subsystems to safe-state until conditions clear. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_14_LossOfControlSystemHeartbeat_FailSafe : SafetyTriggerActionReqDef {
            doc /* If loss of safety monitoring/control heartbeat is detected, the system shall transition to safe-state
                 * for safety-critical operations and prevent restart until monitoring is restored and reset occurs. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_15_SafetyAlarm_LatchAndLog : SafetyTriggerActionReqDef {
            doc /* If any safety alarm is raised, the system shall latch the alarm condition, log the trigger and actions taken,
                 * and prevent clearing until an authorized acknowledge/clear is performed. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
            require constraint { triggerOccurred -> restartPreventedUntilClear }
        }

        requirement FSR_16_ControlledShutdown_OnCriticalHazard : SafetyTriggerActionReqDef {
            doc /* If a critical hazard trigger is detected while operations are active, the system shall perform a controlled shutdown sequence
                 * that transitions subsystems to safe-state without introducing new hazards, and holds until authorized recovery. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
        }

        requirement FSR_17_ResetAuthorization_GatedRecovery : SafetyTriggerActionReqDef {
            doc /* If a safety trip occurred, recovery shall be gated:
                 * only upon authorized reset/clear shall restart inhibition be removed; otherwise safe-state shall persist. */
            attribute triggerOccurred : Boolean;
            attribute actionInitiated : Boolean;
            attribute safeOutcomeAchieved : Boolean;
            attribute restartPreventedUntilClear : Boolean;
            require constraint { triggerOccurred -> restartPreventedUntilClear }
        }

        /********************************************
         * Trace/organization stubs (no implementation)
         ********************************************/
        requirement RD_F_008_TopLevel {
            doc /* This package aggregates functional requirements for safety, interlocks, emergency response,
                 * and safe-state obligations for TMT major subsystems. */
        }

        /* Optional simple derivations/relationships (kept lightweight) */
        derive reqt RD_F_008_TopLevel from Hazard_01_MotionMitigation;
        derive reqt RD_F_008_TopLevel from Hazard_02_HighVoltageMitigation;
        derive reqt RD_F_008_TopLevel from Hazard_03_LaserMitigation;
        derive reqt RD_F_008_TopLevel from Hazard_04_OxygenDeficiencyMitigation;
        derive reqt RD_F_008_TopLevel from Hazard_05_FireSmokeMitigation;

        derive reqt RD_F_008_TopLevel from FSR_01_EStop_Global;
        derive reqt RD_F_008_TopLevel from FSR_02_GuardDoorInterlock_MotionInhibit;
        derive reqt RD_F_008_TopLevel from FSR_03_PresenceDetected_MotionRestrict;
        derive reqt RD_F_008_TopLevel from FSR_04_SlewCommand_InterlockCheck;
        derive reqt RD_F_008_TopLevel from FSR_05_UnexpectedMotionDetected_Halt;
        derive reqt RD_F_008_TopLevel from FSR_06_HighVoltageAccessInterlock_Deenergize;
        derive reqt RD_F_008_TopLevel from FSR_07_GroundFaultOrOvercurrent_ShutdownDomain;
        derive reqt RD_F_008_TopLevel from FSR_08_FireOrSmokeDetected_EmergencyResponse;
        derive reqt RD_F_008_TopLevel from FSR_09_OxygenDeficiencyDetected_RestrictAccess;
        derive reqt RD_F_008_TopLevel from FSR_10_CryogenicLeakOrOverpressure_ProtectPersonnel;
        derive reqt RD_F_008_TopLevel from FSR_11_LaserInterlockViolation_InhibitEmission;
        derive reqt RD_F_008_TopLevel from FSR_12_SeismicEventDetected_ControlledSafeing;
        derive reqt RD_F_008_TopLevel from FSR_13_HighWindOrStormDetected_SecureOperations;
        derive reqt RD_F_008_TopLevel from FSR_14_LossOfControlSystemHeartbeat_FailSafe;
        derive reqt RD_F_008_TopLevel from FSR_15_SafetyAlarm_LatchAndLog;
        derive reqt RD_F_008_TopLevel from FSR_16_ControlledShutdown_OnCriticalHazard;
        derive reqt RD_F_008_TopLevel from FSR_17_ResetAuthorization_GatedRecovery;

        derive reqt SS_01_MountAndDrivesSafeState from FSR_01_EStop_Global;
        derive reqt SS_02_EnclosureDomeSafeState from FSR_01_EStop_Global;
        derive reqt SS_03_OpticsHandlingSafeState from FSR_01_EStop_Global;
        derive reqt SS_04_HVPowerDomainsSafeState from FSR_01_EStop_Global;
        derive reqt SS_05_LaserSubsystemSafeState from FSR_01_EStop_Global;
    }
}
