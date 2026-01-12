package PD_VIB_001_StructuralVibrationAndWindshakeAllocation {

  //
  // Diagram: PD-VIB-001
  // Title: PD VIB 001: Structural Vibration & Windshake Allocation (Mount/Enclosure Coupling)
  // Goal: Quantify and allocate vibration/windshake contributions due to enclosure wind environment and mount/structure dynamics,
  //       providing parameters used by pointing/jitter budgets.
  // Dependencies: PD-PT-002, RD-NF-004, BDD-SUB02-001, BDD-SUB03-001
  //

  //========================
  // Units & Value Types
  //========================

  value type MeterPerSecond {
    unit = "m/s";
  }

  value type Radian {
    unit = "rad";
  }

  value type MicroRadian {
    unit = "urad";
  }

  value type Newton {
    unit = "N";
  }

  value type Pascal {
    unit = "Pa";
  }

  value type Hertz {
    unit = "Hz";
  }

  value type Dimensionless {
    unit = "1";
  }

  value type WindSpeed : MeterPerSecond;

  value type TurbulenceProxy : Dimensionless;

  value type VibrationAmplitude : MicroRadian;

  value type JitterRMS : MicroRadian;

  value type SensitivitySlope : MicroRadian {
    // Interpreted as (urad / (m/s)) for slope parameters
    unit = "urad/(m/s)";
  }

  //========================
  // Imported Requirement Anchor (RD-NF-004)
  //========================

  requirement def RD_NF_004_WindOperationalLimit {
    attribute windSpeedThreshold : WindSpeed;
  }

  //========================
  // Environment + Dynamics Parameter Model
  //========================

  part def PD_VIB_001_StructuralWindshakeAllocation {

    //---- Operating environment inputs (enclosure wind)
    attribute windSpeedAtEnclosure : WindSpeed;
    attribute turbulenceProxy : TurbulenceProxy;

    //---- Structural / mount dynamic state proxies
    attribute mountModeHz : Hertz;
    attribute structureDampingRatio : Dimensionless;
    attribute enclosurePressureFluct : Pascal;

    //---- Sensitivity parameter (explicitly named)
    attribute windToJitterSlope_per_mps : SensitivitySlope;

    //---- Coupling terms (>= 6) between enclosure airflow and structural response
    //     These terms represent allocations or transfer contributions from wind environment to structural vibration/jitter.
    attribute coupling_term_01_shearLayerBuffet : Dimensionless;
    attribute coupling_term_02_vortexShedding : Dimensionless;
    attribute coupling_term_03_pressureFluctCoupling : Dimensionless;
    attribute coupling_term_04_openingJetImpinge : Dimensionless;
    attribute coupling_term_05_boundaryLayerSeparation : Dimensionless;
    attribute coupling_term_06_doorShutterLeakage : Dimensionless;
    attribute coupling_term_07_turbulenceIngressToStructure : Dimensionless;

    //---- Intermediate vibration allocations (structural response)
    attribute mountVibrationRMS : VibrationAmplitude;
    attribute enclosureInducedVibrationRMS : VibrationAmplitude;
    attribute couplingInducedVibrationRMS : VibrationAmplitude;

    //---- Exports: named jitter contribution attributes referenced by PD-PT-002
    //     These names align to PD-PT-002 contributor buckets (wind/structure/etc.), enabling direct binding.
    attribute windContrib : JitterRMS;
    attribute structureContrib : JitterRMS;

    //---- Optional banded exports (useful when mapped into PD-PT-002 low/mid/high bins)
    attribute windLowBandJitter : JitterRMS;
    attribute windMidBandJitter : JitterRMS;
    attribute windHighBandJitter : JitterRMS;

    //========================
    // Constraints
    //========================

    // Anchor requirement instance (from RD-NF-004)
    requirement windOperationalLimit : RD_NF_004_WindOperationalLimit;

    // Constraint: Maximum allowable wind speed operation consistent with RD-NF-004
    constraint maxAllowableWindSpeedOperation {
      windSpeedAtEnclosure <= windOperationalLimit.windSpeedThreshold
    }

    // Constraint: Coupling / sensitivity-based jitter estimate (robust, explicit)
    // Uses at least 6 coupling terms and turbulence proxy to scale effective forcing.
    //
    // EffectiveCoupling = sum(coupling_terms) * (1 + turbulenceProxy)
    // windContrib = windToJitterSlope_per_mps * windSpeedAtEnclosure * EffectiveCoupling
    //
    constraint windshakeJitterAllocation {

      let EffectiveCoupling : Dimensionless =
        (coupling_term_01_shearLayerBuffet
       + coupling_term_02_vortexShedding
       + coupling_term_03_pressureFluctCoupling
       + coupling_term_04_openingJetImpinge
       + coupling_term_05_boundaryLayerSeparation
       + coupling_term_06_doorShutterLeakage
       + coupling_term_07_turbulenceIngressToStructure) * (1 + turbulenceProxy);

      windContrib == windToJitterSlope_per_mps * windSpeedAtEnclosure * EffectiveCoupling;

      // Structural contribution allocated from vibration amplitudes (simple mapping proxy)
      structureContrib == mountVibrationRMS + couplingInducedVibrationRMS;

      // Partition wind contributor into bands (proportional allocation; placeholders for later refinement)
      windLowBandJitter  == windContrib * 0.30;
      windMidBandJitter  == windContrib * 0.50;
      windHighBandJitter == windContrib * 0.20;

      // Coupling-induced vibration allocation as function of pressure fluctuation and coupling
      couplingInducedVibrationRMS == enclosureInducedVibrationRMS * EffectiveCoupling;

      // Enclosure-induced vibration proxy driven by pressure fluctuations and mount resonance amplification
      enclosureInducedVibrationRMS == (enclosurePressureFluct * 0.001) * (1 + (mountModeHz * 0.01)) * (1 - structureDampingRatio);
    }

  }

  //========================
  // Default Instance (for binding into budgets)
  //========================

  part PD_VIB_001_Model : PD_VIB_001_StructuralWindshakeAllocation {
    // Provide conservative defaults (can be overridden by analysis data)
    :>> windSpeedAtEnclosure = 8 [m/s];
    :>> turbulenceProxy = 0.6;
    :>> mountModeHz = 1.2 [Hz];
    :>> structureDampingRatio = 0.08;
    :>> enclosurePressureFluct = 15 [Pa];

    // Explicit sensitivity parameter per acceptance check
    :>> windToJitterSlope_per_mps = 0.5 [urad/(m/s)];

    // Coupling term defaults (dimensionless scaling factors)
    :>> coupling_term_01_shearLayerBuffet = 0.20;
    :>> coupling_term_02_vortexShedding = 0.15;
    :>> coupling_term_03_pressureFluctCoupling = 0.18;
    :>> coupling_term_04_openingJetImpinge = 0.10;
    :>> coupling_term_05_boundaryLayerSeparation = 0.12;
    :>> coupling_term_06_doorShutterLeakage = 0.08;
    :>> coupling_term_07_turbulenceIngressToStructure = 0.17;

    // Requirement default: consistent with RD-NF-004 operational wind threshold (placeholder; bind to authoritative value)
    :>> windOperationalLimit.windSpeedThreshold = 15 [m/s];

    // Seed a baseline mount vibration (placeholder; replace with measured/FEA results)
    :>> mountVibrationRMS = 1.0 [urad];
  }

}
