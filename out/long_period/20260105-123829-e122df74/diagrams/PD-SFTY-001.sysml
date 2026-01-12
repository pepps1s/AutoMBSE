package PD_SFTY_001_SafetyIntegrityBudget {

  // ===========================================
  // Metadata
  // ===========================================
  doc /* diagram_id: PD-SFTY-001
         type: PD
         title: PD SFTY 001: Safety Integrity Budget (Interlock Coverage + Response Time)
         goal: Quantify safety interlock response times, coverage allocations, and logging/audit retention parameters
         dependencies: RD-F-008, RD-NF-006, RD-OPS-003, BDD-SUB09-001, PD-TIM-001
      */;

  // ===========================================
  // Imports (trace/dependency anchors)
  // ===========================================
  import RD_F_008_SafetyInterlocksEmergencyResponse::*;
  import RD_NF_006_SafetyCompliance_StandardsPoliciesAlignment::*;
  import FaultResponseAndSafeStateRequirements::*;
  import BDD_SUB09_001_Safety_TMT::*;
  import PD_Timing_Synchronization_Budget::*;

  // ===========================================
  // Value Types (Acceptance: response time, coverage %, retention)
  // ===========================================
  value type Seconds : Real {
    unit = "s";
  }

  value type Milliseconds : Real {
    unit = "ms";
  }

  value type CoveragePercent : Real {
    unit = "%";
    constraint { 0.0 <= self and self <= 100.0 }
  }

  value type Days : Integer {
    unit = "day";
    constraint { self >= 0 }
  }

  value type TimestampUtc : String; // ISO-8601 UTC timestamp representation

  // ===========================================
  // Core Budget Definitions
  // ===========================================
  part def SafetyInterlockBudget {

    // --- Response time allocations (detection → trip → safe state) ---
    attribute detectionToTripMax : Milliseconds;
    attribute tripToSafeStateMax : Milliseconds;
    attribute endToEndSafeStateMax : Milliseconds;

    // --- Coverage allocations ---
    attribute interlockCoverageTarget : CoveragePercent;
    attribute hazardCoverageMin : CoveragePercent;

    // --- Logging / audit parameters (RD-NF-006 aligned) ---
    attribute immutableStorageEnabled : Boolean;
    attribute searchable : Boolean;
    attribute includesActorId : Boolean;
    attribute includesChangeReason : Boolean;

    attribute eventRetentionDays : Days;
    attribute auditRetentionDays : Days;

    attribute timeSource : String;
    attribute timestampResolutionMs : Milliseconds;
    attribute timestampAccuracyMs : Milliseconds;

    // --- Timing accuracy reference (PD-TIM-001) ---
    attribute clockSyncAccuracyMs : Milliseconds;
    attribute jitterMs : Milliseconds;
    attribute offsetMs : Milliseconds;

    // --- Safe state bounding requirement (RD-OPS-003 style) ---
    attribute safeStateEntryBoundMax : Milliseconds;

    // --- Budget constraints (Acceptance: verify allocated maxima + bounded safe state entry) ---
    constraint ResponseTimeBudgetConstraint {
      // end-to-end must bound the sum of segments and remain within allocated maximum
      endToEndSafeStateMax <= (detectionToTripMax + tripToSafeStateMax);
    }

    constraint SafeStateEntryBoundConstraint {
      // safe state entry time is bounded by explicit safe state entry bound
      endToEndSafeStateMax <= safeStateEntryBoundMax;
    }

    constraint CoverageBudgetConstraint {
      // enforce target coverage and minimum hazard coverage
      interlockCoverageTarget >= hazardCoverageMin;
      interlockCoverageTarget >= 95.0;
      hazardCoverageMin >= 90.0;
    }

    constraint LoggingNFRConstraint {
      // RD-NF-006: immutability + retention + searchability expectations
      immutableStorageEnabled == true;
      searchable == true;
      includesActorId == true;
      includesChangeReason == true;
      eventRetentionDays >= 365;
      auditRetentionDays >= 3650; // 10 years typical compliance target (project allocatable)
    }

    constraint TimingAccuracyConstraintRef {
      // PD-TIM-001 reference: timestamp accuracy must be at least as good as overall clock sync accuracy
      timestampAccuracyMs <= clockSyncAccuracyMs;
      timestampResolutionMs <= 10.0 [ms];
    }

    constraint TimestampConsistencyCheck {
      // Timestamp consistency check: worst-case drift budget must not exceed accuracy allocation
      // driftBudget = abs(offset) + jitter + clockSyncAccuracy
      (abs(offsetMs) + jitterMs + clockSyncAccuracyMs) <= (timestampAccuracyMs * 2.0);
    }
  }

  // ===========================================
  // Hazard / Mitigation Lines (Acceptance: >= 8 tied to RD-F-008 hazard entries)
  // ===========================================
  part def HazardMitigationLine {
    attribute hazardId : String;
    attribute hazardName : String;

    attribute detectionTrigger : String;
    attribute mitigationRequired : String;
    attribute actionOutcomeAchieved : Boolean;

    attribute allocatedCoverage : CoveragePercent;

    attribute detectionToTripAllocated : Milliseconds;
    attribute tripToSafeStateAllocated : Milliseconds;
    attribute endToEndSafeStateAllocated : Milliseconds;

    // Constraint: each hazard line must meet the global maxima
    constraint HazardResponseWithinAllocation(maxDetTrip: Milliseconds,
                                             maxTripSafe: Milliseconds,
                                             maxEndToEnd: Milliseconds) {
      detectionToTripAllocated <= maxDetTrip;
      tripToSafeStateAllocated <= maxTripSafe;
      endToEndSafeStateAllocated <= maxEndToEnd;
    }

    // Constraint: safe state must be reached within end-to-end bound (non-negative and bounded)
    constraint SafeStateBounded(boundMax: Milliseconds) {
      0.0 [ms] <= endToEndSafeStateAllocated;
      endToEndSafeStateAllocated <= boundMax;
    }

    // Constraint: coverage must be reasonable
    constraint CoverageReasonable(minCov: CoveragePercent) {
      allocatedCoverage >= minCov;
      allocatedCoverage <= 100.0;
    }
  }

  // ===========================================
  // PD View: Safety Integrity Budget Instance
  // ===========================================
  part PD_SFTY_001_View : SafetyInterlockBudget {

    // --- Response time maxima (system allocation) ---
    :>> detectionToTripMax = 75.0 [ms];
    :>> tripToSafeStateMax = 250.0 [ms];
    :>> endToEndSafeStateMax = 325.0 [ms];

    :>> safeStateEntryBoundMax = 400.0 [ms];

    // --- Coverage targets ---
    :>> interlockCoverageTarget = 97.5;
    :>> hazardCoverageMin = 92.0;

    // --- Logging / audit parameters (RD-NF-006 aligned) ---
    :>> immutableStorageEnabled = true;
    :>> searchable = true;
    :>> includesActorId = true;
    :>> includesChangeReason = true;

    :>> eventRetentionDays = 730;    // 2 years safety event records
    :>> auditRetentionDays = 3650;   // 10 years audit logs

    :>> timeSource = "UniversalTime/UTC (PD-TIM-001 TimeStandard)";
    :>> timestampResolutionMs = 1.0 [ms];
    :>> timestampAccuracyMs = 5.0 [ms];

    // --- Timing accuracy reference from PD-TIM-001 allocations (example allocation) ---
    :>> clockSyncAccuracyMs = 2.0 [ms];
    :>> jitterMs = 1.0 [ms];
    :>> offsetMs = 1.0 [ms];

    // ===========================================
    // Hazard lines (>= 8) — tied to RD-F-008 hazard entries by ID/name
    // ===========================================

    part HZ_01_Motion : HazardMitigationLine {
      :>> hazardId = "RD-F-008/Hazard_01_MotionMitigation";
      :>> hazardName = "Hazard_01_MotionMitigation";
      :>> detectionTrigger = "Unexpected motion / drive runaway / slew limit exceeded";
      :>> mitigationRequired = "Motion inhibit + immediate stop + brake engage + safe park";
      :>> actionOutcomeAchieved = true;
      :>> allocatedCoverage = 98.0;
      :>> detectionToTripAllocated = 50.0 [ms];
      :>> tripToSafeStateAllocated = 200.0 [ms];
      :>> endToEndSafeStateAllocated = 250.0 [ms];

      assert HazardResponseWithinAllocation(PD_SFTY_001_View.detectionToTripMax,
                                           PD_SFTY_001_View.tripToSafeStateMax,
                                           PD_SFTY_001_View.endToEndSafeStateMax);

      assert SafeStateBounded(PD_SFTY_001_View.safeStateEntryBoundMax);
      assert CoverageReasonable(PD_SFTY_001_View.hazardCoverageMin);
    }

    part HZ_02_HighVoltage : HazardMitigationLine {
      :>> hazardId = "RD-F-008/Hazard_02_HighVoltageMitigation";
      :>> hazardName = "Hazard_02_HighVoltageMitigation";
      :>> detectionTrigger = "HV cabinet access or ground fault / overcurrent";
      :>> mitigationRequired = "HV shutdown + discharge + access interlock lockout";
      :>> actionOutcomeAchieved = true;
      :>> allocatedCoverage = 97.0;
      :>> detectionToTripAllocated = 60.0 [ms];
      :>> tripToSafeStateAllocated = 225.0 [ms];
      :>> endToEndSafeStateAllocated = 285.0 [ms];

      assert HazardResponseWithinAllocation(PD_SFTY_001_View.detectionToTripMax,
                                           PD_SFTY_001_View.tripToSafeStateMax,
                                           PD_SFTY_001_View.endToEndSafeStateMax);

      assert SafeStateBounded(PD_SFTY_001_View.safeStateEntryBoundMax);
      assert CoverageReasonable(PD_SFTY_001_View.hazardCoverageMin);
    }

    part HZ_03_Laser : HazardMitigationLine {
      :>> hazardId = "RD-F-008/Hazard_03_LaserMitigation";
      :>> hazardName = "Hazard_03_LaserMitigation";
      :>> detectionTrigger = "Laser propagation without clearance / beam path breach";
      :>> mitigationRequired = "Laser shutter close + beam inhibit + safety annunciation";
      :>> actionOutcomeAchieved = true;
      :>> allocatedCoverage = 99.0;
      :>> detectionToTripAllocated = 40.0 [ms];
      :>> tripToSafeStateAllocated = 150.0 [ms];
      :>> endToEndSafeStateAllocated = 190.0 [ms];

      assert HazardResponseWithinAllocation(PD_SFTY_001_View.detectionToTripMax,
                                           PD_SFTY_001_View.tripToSafeStateMax,
                                           PD_SFTY_001_View.endToEndSafeStateMax);

      assert SafeStateBounded(PD_SFTY_001_View.safeStateEntryBoundMax);
      assert CoverageReasonable(PD_SFTY_001_View.hazardCoverageMin);
    }

    part HZ_04_OxygenDeficiency : HazardMitigationLine {
      :>> hazardId = "RD-F-008/Hazard_04_OxygenDeficiencyMitigation";
      :>> hazardName = "Hazard_04_OxygenDeficiencyMitigation";
      :>> detectionTrigger = "O2 sensor below threshold / confined space event";
      :>> mitigationRequired = "Ventilation enable + access control + alarm + evacuation";
      :>> actionOutcomeAchieved = true;
      :>> allocatedCoverage = 95.0;
      :>> detectionToTripAllocated = 70.0 [ms];
      :>> tripToSafeStateAllocated = 240.0 [ms];
      :>> endToEndSafeStateAllocated = 310.0 [ms];

      assert HazardResponseWithinAllocation(PD_SFTY_001_View.detectionToTripMax,
                                           PD_SFTY_001_View.tripToSafeStateMax,
                                           PD_SFTY_001_View.endToEndSafeStateMax);

      assert SafeStateBounded(PD_SFTY_001_View.safeStateEntryBoundMax);
      assert CoverageReasonable(PD_SFTY_001_View.hazardCoverageMin);
    }

    part HZ_05_FireSmoke : HazardMitigationLine {
      :>> hazardId = "RD-F-008/Hazard_05_FireSmokeMitigation";
      :>> hazardName = "Hazard_05_FireSmokeMitigation";
      :>> detectionTrigger = "Smoke detector trip / thermal rise / flame sensor";
      :>> mitigationRequired = "Power isolate + suppression enable + evacuation alarm";
      :>> actionOutcomeAchieved = true;
      :>> allocatedCoverage = 96.0;
      :>> detectionToTripAllocated = 65.0 [ms];
      :>> tripToSafeStateAllocated = 230.0 [ms];
      :>> endToEndSafeStateAllocated = 295.0 [ms];

      assert HazardResponseWithinAllocation(PD_SFTY_001_View.detectionToTripMax,
                                           PD_SFTY_001_View.tripToSafeStateMax,
                                           PD_SFTY_001_View.endToEndSafeStateMax);

      assert SafeStateBounded(PD_SFTY_001_View.safeStateEntryBoundMax);
      assert CoverageReasonable(PD_SFTY_001_View.hazardCoverageMin);
    }

    part HZ_06_AccessBreach : HazardMitigationLine {
      :>> hazardId = "RD-F-008/Hazard_06_AccessInterlockBreach";
      :>> hazardName = "Hazard_06_AccessInterlockBreach";
      :>> detectionTrigger = "Door/gate opened during hazardous operation";
      :>> mitigationRequired = "Immediate inhibit of motion/laser/HV + alarm annunciation";
      :>> actionOutcomeAchieved = true;
      :>> allocatedCoverage = 97.0;
      :>> detectionToTripAllocated = 55.0 [ms];
      :>> tripToSafeStateAllocated = 210.0 [ms];
      :>> endToEndSafeStateAllocated = 265.0 [ms];

      assert HazardResponseWithinAllocation(PD_SFTY_001_View.detectionToTripMax,
                                           PD_SFTY_001_View.tripToSafeStateMax,
                                           PD_SFTY_001_View.endToEndSafeStateMax);

      assert SafeStateBounded(PD_SFTY_001_View.safeStateEntryBoundMax);
      assert CoverageReasonable(PD_SFTY_001_View.hazardCoverageMin);
    }

    part HZ_07_EStopLoop : HazardMitigationLine {
      :>> hazardId = "RD-F-008/Hazard_07_EStopLoopFailure";
      :>> hazardName = "Hazard_07_EStopLoopFailure";
      :>> detectionTrigger = "E-Stop loop open / contactor fault / PLC trip";
      :>> mitigationRequired = "Hardwired shutdown + safe torque off + alarm + lockout";
      :>> actionOutcomeAchieved = true;
      :>> allocatedCoverage = 99.5;
      :>> detectionToTripAllocated = 25.0 [ms];
      :>> tripToSafeStateAllocated = 175.0 [ms];
      :>> endToEndSafeStateAllocated = 200.0 [ms];

      assert HazardResponseWithinAllocation(PD_SFTY_001_View.detectionToTripMax,
                                           PD_SFTY_001_View.tripToSafeStateMax,
                                           PD_SFTY_001_View.endToEndSafeStateMax);

      assert SafeStateBounded(PD_SFTY_001_View.safeStateEntryBoundMax);
      assert CoverageReasonable(PD_SFTY_001_View.hazardCoverageMin);
    }

    part HZ_08_CryoThermal : HazardMitigationLine {
      :>> hazardId = "RD-F-008/Hazard_08_CryoThermalOvertemp";
      :>> hazardName = "Hazard_08_CryoThermalOvertemp";
      :>> detectionTrigger = "Cryo/thermal control fault, overtemp/undertemp threshold violation";
      :>> mitigationRequired = "Controlled shutdown + isolate cooling + safe configuration + notify operator";
      :>> actionOutcomeAchieved = true;
      :>> allocatedCoverage = 93.0;
      :>> detectionToTripAllocated = 75.0 [ms];
      :>> tripToSafeStateAllocated = 240.0 [ms];
      :>> endToEndSafeStateAllocated = 315.0 [ms];

      assert HazardResponseWithinAllocation(PD_SFTY_001_View.detectionToTripMax,
                                           PD_SFTY_001_View.tripToSafeStateMax,
                                           PD_SFTY_001_View.endToEndSafeStateMax);

      assert SafeStateBounded(PD_SFTY_001_View.safeStateEntryBoundMax);
      assert CoverageReasonable(PD_SFTY_001_View.hazardCoverageMin);
    }

    // ===========================================
    // Model-level constraint: all hazards must meet coverage and response bounds
    // (explicit aggregate constraints help acceptance checks)
    // ===========================================
    constraint AllHazardsBoundedAndCovered {
      HZ_01_Motion.endToEndSafeStateAllocated <= endToEndSafeStateMax;
      HZ_02_HighVoltage.endToEndSafeStateAllocated <= endToEndSafeStateMax;
      HZ_03_Laser.endToEndSafeStateAllocated <= endToEndSafeStateMax;
      HZ_04_OxygenDeficiency.endToEndSafeStateAllocated <= endToEndSafeStateMax;
      HZ_05_FireSmoke.endToEndSafeStateAllocated <= endToEndSafeStateMax;
      HZ_06_AccessBreach.endToEndSafeStateAllocated <= endToEndSafeStateMax;
      HZ_07_EStopLoop.endToEndSafeStateAllocated <= endToEndSafeStateMax;
      HZ_08_CryoThermal.endToEndSafeStateAllocated <= endToEndSafeStateMax;

      HZ_01_Motion.allocatedCoverage >= hazardCoverageMin;
      HZ_02_HighVoltage.allocatedCoverage >= hazardCoverageMin;
      HZ_03_Laser.allocatedCoverage >= hazardCoverageMin;
      HZ_04_OxygenDeficiency.allocatedCoverage >= hazardCoverageMin;
      HZ_05_FireSmoke.allocatedCoverage >= hazardCoverageMin;
      HZ_06_AccessBreach.allocatedCoverage >= hazardCoverageMin;
      HZ_07_EStopLoop.allocatedCoverage >= hazardCoverageMin;
      HZ_08_CryoThermal.allocatedCoverage >= hazardCoverageMin;
    }

  }

}
