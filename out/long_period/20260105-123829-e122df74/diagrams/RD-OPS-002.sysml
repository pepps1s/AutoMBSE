package RD_OPS_002_RepresentativeOperationalScenariosSet {
    import ScalarValues::*;
    import Quantities::*;

    // Diagram metadata
    metadata def DiagramMeta {
        attribute diagramId : String;
        attribute diagramType : String;
        attribute title : String;
        attribute goal : String;
        attribute dependencies : String;
    }

    metadata diagram : DiagramMeta {
        diagramId = "RD-OPS-002";
        diagramType = "RD";
        title = "Representative Operational Scenarios Set (Primary + Alternates)";
        goal = "Capture scenario narratives that drive completeness: nominal observing and key exception paths.";
        dependencies = "RD-OPS-001, RD-F-002";
    }

    // Keywords (align to requirement set IDs in dependencies)
    enum ReqSetId {
        RD_OPS_001,
        RD_F_002
    }

    // Common scenario definition for operational narratives + trace keywords
    requirement def OperationalScenarioDef {
        doc /* Scenario narrative + trace keywords aligned to requirement set IDs. */

        attribute scenarioId : String;
        attribute scenarioName : String;

        // Alignment keywords: MUST include the relevant requirement set IDs by keyword match.
        attribute referencesRD_OPS_001 : Boolean;
        attribute referencesRD_F_002 : Boolean;

        // Optional: classify for acceptance checks
        attribute isPrimary : Boolean;
        attribute isAlternate : Boolean;
        attribute isException : Boolean;

        // Narrative blocks (kept as Strings so the view stays robust without custom types)
        attribute preconditions : String;
        attribute triggers : String;
        attribute mainFlow : String;
        attribute postconditions : String;
        attribute failureModes : String;

        require constraint {
            // Each scenario must reference at least one relevant requirement set by keyword alignment
            referencesRD_OPS_001 or referencesRD_F_002
        }
    }

    // ----------------------------
    // Primary (Nominal) Scenarios
    // ----------------------------

    requirement StartupScenario : OperationalScenarioDef {
        doc /* Nominal: Observatory startup sequence for TMT (三十米望远镜).
             * Keyword alignment: RD-OPS-001 (modes), RD-F-002 (StartupReq). */
        scenarioId = "OPS-SCN-001";
        scenarioName = "Startup (Power-up → Subsystem Self-check → Ready for Observing)";

        referencesRD_OPS_001 = true;   // modes/state catalog alignment
        referencesRD_F_002 = true;     // StartupReq alignment

        isPrimary = true;
        isAlternate = false;
        isException = false;

        preconditions = "Facility power available; safety interlocks armed; enclosure accessible; operator authorization granted.";
        triggers = "Night operations start; schedule initiates StartupReq; transition from StandbyMode/maintenance as applicable (RD-OPS-001).";
        mainFlow = "Enable core services (power/HVAC/network) → boot OCS/telemetry/logging → bring up subsystems in allowed order → run built-in tests → confirm subsystemStatus nominal → transition to ObservingMode readiness (RD-OPS-001).";
        postconditions = "Subsystem readiness confirmed; telescope in safe idle pointing; ready to slew to first target.";
        failureModes = "Any subsystem fails self-check → transition to DegradedMode or SafeParkMode per RD-OPS-001; create alert and require operator action.";
    }

    requirement SlewScenario : OperationalScenarioDef {
        doc /* Nominal: Slew to scheduled target.
             * Keyword alignment: RD-F-002 (SlewReq), RD-OPS-001 (ObservingMode). */
        scenarioId = "OPS-SCN-002";
        scenarioName = "Slew to Target (Pointing Model → Coarse Pointing → Settle)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = true;
        isAlternate = false;
        isException = false;

        preconditions = "Startup complete; telescope released for motion; dome shutters open or opening; target coordinates available.";
        triggers = "Observation sequencer issues SlewReq for next target; ObservingMode active (RD-OPS-001).";
        mainFlow = "Compute slew path using pointing model → rotate enclosure as needed → drive alt-az axes to target → settle and damp vibrations → verify accuracy meets SlewReq → hand off to acquisition.";
        postconditions = "Target field within acquisition window; mount tracking stabilized.";
        failureModes = "Encoder/drive anomaly or vibration exceed limits → pause motion → either retry with reduced rates (DegradedModeLimitedTrackingReq in RD-OPS-001) or safe park.";
    }

    requirement AcquisitionScenario : OperationalScenarioDef {
        doc /* Nominal: Acquisition for science target and guide stars.
             * Keyword alignment: RD-F-002 (AcquisitionReq). */
        scenarioId = "OPS-SCN-003";
        scenarioName = "Acquisition (Target Acquisition → Guide Star Acquisition → Focus/Alignment Checks)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = true;
        isAlternate = false;
        isException = false;

        preconditions = "Slew complete; tracking stable; instrument selected and available; WFS/cameras operational.";
        triggers = "Sequencer transitions from SlewReq to AcquisitionReq; guide star selection computed.";
        mainFlow = "Acquire target field image → centroid/plate-solve → acquire guide star(s) NGS/LGS as applicable → verify guideStarAccuracy meets AcquisitionReq → run quick focus check → proceed to phasing/alignment.";
        postconditions = "Target and guide references locked; system ready for alignment/phasing and optional AO.";
        failureModes = "Guide star not found or accuracy out of bounds → attempt alternate guide selection; if unavailable → go to alternate path instrument unavailable or weather abort handling as appropriate.";
    }

    requirement PhasingScenario : OperationalScenarioDef {
        doc /* Nominal: Segment alignment & phasing (APS loop) for 30m segmented M1.
             * Keyword alignment: RD-F-002 (AlignmentPhasingReq). */
        scenarioId = "OPS-SCN-004";
        scenarioName = "Alignment & Phasing (APS/WFS Loop for Segmented Primary)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = true;
        isAlternate = false;
        isException = false;

        preconditions = "Acquisition complete; wavefront sensing enabled; segment actuators/metrology ready.";
        triggers = "Sequencer issues AlignmentPhasingReq; phasing state requires update for current elevation/thermal state.";
        mainFlow = "Run WFS measurement → compute segment piston/tip/tilt errors → command segment actuators → iterate until alignmentAccuracy meets AlignmentPhasingReq → confirm stability window for exposures.";
        postconditions = "Optical train aligned; phasing within tolerance; ready for AO optional or exposure loop.";
        failureModes = "Actuator limit reached or metrology invalid → enter DegradedModeLimitedAlignmentReq (RD-OPS-001) and restrict observing or abort to safe-state.";
    }

    requirement AOOptionalScenario : OperationalScenarioDef {
        doc /* Optional: Adaptive Optics acquisition and loop closure.
             * Keyword alignment: RD-F-002 (AcquisitionReq / AlignmentPhasingReq integration), RD-OPS-001 (mode transitions). */
        scenarioId = "OPS-SCN-005";
        scenarioName = "AO Acquisition & Loop Closure (Optional)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = true;
        isAlternate = true;
        isException = false;

        preconditions = "Phasing complete; AO subsystem available; laser safety/interlocks satisfied if LGS used.";
        triggers = "Science program requests AO; sequencer branches to AO optional path; AO readiness confirmed.";
        mainFlow = "Select NGS/LGS → acquire WFS signals → calibrate reference slopes → close AO loops (RTC/DM) → verify performance meets observing constraints → hand off to exposure loop.";
        postconditions = "AO-corrected PSF achieved; exposure loop configured for AO telemetry and metadata.";
        failureModes = "AO loop unstable or interlock inhibits lasers → fall back to non-AO exposure if allowed (DegradedModePartialInstrumentCapabilityReq) or execute abort scenario.";
    }

    requirement ExposureLoopScenario : OperationalScenarioDef {
        doc /* Nominal: Science exposure loop (expose → readout → metadata → quick-look QA).
             * Keyword alignment: RD-F-002 (ExposureLoopReq). */
        scenarioId = "OPS-SCN-006";
        scenarioName = "Science Exposure Loop (Expose → Readout → Metadata → Quick-Look QA)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = true;
        isAlternate = false;
        isException = false;

        preconditions = "Acquisition and phasing complete; optional AO status set; instrument configured; tracking stable.";
        triggers = "Sequencer issues ExposureLoopReq for one or more exposures in observing block.";
        mainFlow = "Start exposure timer → maintain tracking/stability → collect telemetry and environmental data → readout detector → persist raw data + metadata → run quick-look QA → decide continue/adjust/repeat.";
        postconditions = "Science frames recorded with required metadata and quality assessment; next step is calibration or next target.";
        failureModes = "Quality below threshold or telemetry indicates drift → trigger re-phasing or re-acquisition; on hard fault transition to fault abort.";
    }

    requirement CalibrationScenario : OperationalScenarioDef {
        doc /* Nominal: Calibration (flat/dark/arc, WFS calibration, pointing model updates).
             * Keyword alignment: RD-F-002 (CalibrationReq), RD-OPS-001 (CalibrationMode). */
        scenarioId = "OPS-SCN-007";
        scenarioName = "Calibration (Instrument + WFS + Pointing Model Updates)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = true;
        isAlternate = false;
        isException = false;

        preconditions = "Observing block complete or calibration scheduled; calibration unit available; dome state supports calibrations.";
        triggers = "Sequencer transitions to CalibrationMode (RD-OPS-001) and issues CalibrationReq.";
        mainFlow = "Select calibrationType (flat/dark/arc/WFS) → configure sources and shutters → acquire calibration frames → tag metadata → update pointing model if required → validate calibration completeness.";
        postconditions = "Calibration products stored and linked to science data; system ready to resume observing or shutdown.";
        failureModes = "Calibration source failure or time overrun → defer optional calibrations; if critical calibration fails, mark data as provisional and notify operator.";
    }

    requirement ShutdownScenario : OperationalScenarioDef {
        doc /* Nominal: Observatory shutdown / safe park.
             * Keyword alignment: RD-F-002 (ShutdownReq), RD-OPS-001 (SafeParkMode). */
        scenarioId = "OPS-SCN-008";
        scenarioName = "Shutdown (Stop Observing → Safe Park → Power Down)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = true;
        isAlternate = false;
        isException = false;

        preconditions = "Observations complete or stop requested; no critical maintenance in progress.";
        triggers = "End of night; schedule complete; or operator requests ShutdownReq.";
        mainFlow = "Stop exposure activities → stow instruments as needed → park telescope to safe orientation → close dome shutters → transition to SafeParkMode/StandbyMode per RD-OPS-001 → power down nonessential subsystems → finalize logs and data transfers.";
        postconditions = "Telescope secured; subsystems in safe/standby state; logs archived; readiness for next startup.";
        failureModes = "If interlock prevents motion or closure, enter safe-state with alarms; maintain minimum power for safety monitoring.";
    }

    // ----------------------------
    // Exception / Alternate Scenarios
    // ----------------------------

    requirement WeatherAbortScenario : OperationalScenarioDef {
        doc /* Exception: Weather abort (high wind, humidity, clouds, etc.).
             * Keyword alignment: RD-F-002 (AlternatePathWeatherAbortReq), RD-OPS-001 (SafeParkMode/Degraded). */
        scenarioId = "OPS-EXC-001";
        scenarioName = "Exception: Weather Abort (Protect Hardware & Data)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = false;
        isAlternate = true;
        isException = true;

        preconditions = "Observing or startup in progress; environmental monitoring enabled.";
        triggers = "weatherCondition exceeds operational limits (RD-F-002 AlternatePathWeatherAbortReq).";
        mainFlow = "Pause exposures safely → secure instrument mechanisms → command dome closure/shutters → park telescope to safe pose → transition to SafeParkMode (RD-OPS-001) → record abortStatus + weatherCondition in logs and metadata.";
        postconditions = "Hardware protected; observation block halted with explicit abort status; recovery requires operator re-enable and conditions within limits.";
        failureModes = "If dome closure fails, escalate to fault abort; keep maximum protective posture and alarms active.";
    }

    requirement FaultAbortScenario : OperationalScenarioDef {
        doc /* Exception: Fault abort (critical subsystem anomaly).
             * Keyword alignment: RD-F-002 (AlternatePathFaultAbortReq), RD-OPS-001 (Degraded/SafePark modes). */
        scenarioId = "OPS-EXC-002";
        scenarioName = "Exception: Fault Abort (Detect → Safe-State → Notify → Diagnose)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = false;
        isAlternate = true;
        isException = true;

        preconditions = "Any operational mode; health monitoring active; fault detection rules enabled.";
        triggers = "faultCondition detected (RD-F-002 AlternatePathFaultAbortReq) such as drive fault, actuator fault, controller crash, or unsafe telemetry.";
        mainFlow = "Freeze or abort current action → move to safe-state (stop motion, stop lasers, safe instrument) → transition to DegradedMode or SafeParkMode (RD-OPS-001) depending on severity → set abortStatus and raise alarm → initiate diagnostic workflow and recovery steps.";
        postconditions = "System stable in safe configuration; fault logged with context; operator notified.";
        failureModes = "If fault blocks safe motion (e.g., brake failure), execute emergency stop procedures and lock out observing.";
    }

    requirement InstrumentSwapFailureScenario : OperationalScenarioDef {
        doc /* Exception: Instrument swap failure / instrument unavailable.
             * Keyword alignment: RD-F-002 (AlternatePathInstrumentUnavailableReq), RD-OPS-001 (Degraded partial capability). */
        scenarioId = "OPS-EXC-003";
        scenarioName = "Exception: Instrument Swap Failure (Instrument Unavailable / Partial Capability)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = false;
        isAlternate = true;
        isException = true;

        preconditions = "Instrument change requested; instrument ports and mechanisms enabled; safety clear.";
        triggers = "instrumentStatus indicates unavailable, swap mechanism error, or initialization failure (RD-F-002 AlternatePathInstrumentUnavailableReq).";
        mainFlow = "Abort swap motion safely → return to last known-safe configuration → mark requested instrument unavailable → decide: switch to alternate instrument (if science allows) or enter DegradedModePartialInstrumentCapabilityReq (RD-OPS-001) → update schedule and notify operator.";
        postconditions = "Instrument state consistent and safe; observation plan updated with explicit abort/alternate instrument selection.";
        failureModes = "If mechanism jam threatens damage, escalate to fault abort and require maintenance mode.";
    }

    requirement InterlockTripScenario : OperationalScenarioDef {
        doc /* Exception: Safety interlock trip (personnel, laser, access, motion limits).
             * Keyword alignment: RD-OPS-001 (SafeParkMode / safety-driven state), RD-F-002 (ShutdownReq interplay). */
        scenarioId = "OPS-EXC-004";
        scenarioName = "Exception: Safety Interlock Trip (Immediate Inhibit & Safe Response)";

        referencesRD_OPS_001 = true;
        referencesRD_F_002 = true;

        isPrimary = false;
        isAlternate = true;
        isException = true;

        preconditions = "Operations active; interlocks armed (doors, e-stop, laser safety, limit switches).";
        triggers = "Interlock trip (e-stop pressed, access breach, laser safety violation, limit switch).";
        mainFlow = "Immediately inhibit hazardous actions (stop motion/lasers) → hold or park as allowed → transition to SafeParkMode or safe-hold state (RD-OPS-001) → require explicit reset and authorization to resume → log event with timestamp and cause.";
        postconditions = "Hazard mitigated; system safe; resumption gated by procedures.";
        failureModes = "If interlock prevents park motion, remain in safe-hold with alarms; dispatch on-site response per safety procedures.";
    }

    // ----------------------------
    // Acceptance checks as constraints (view-level)
    // ----------------------------

    requirement def RD_OPS_002_AcceptanceDef {
        doc /* Acceptance checks for RD-OPS-002. */

        attribute nominalScenarioCount : Integer;
        attribute exceptionScenarioCount : Integer;

        require constraint {
            nominalScenarioCount >= 8
        }
        require constraint {
            exceptionScenarioCount >= 4
        }
    }

    requirement RD_OPS_002_Acceptance : RD_OPS_002_AcceptanceDef {
        doc /* Computed counts represented explicitly for parsing robustness. */
        nominalScenarioCount = 8;
        exceptionScenarioCount = 4;
    }
}
