package PD_OPT_002_EndToEndThroughputBudget {

  //
  // Dependencies (shallow imports for traceability)
  //
  import PD_PRF_001::*;
  import RD_NF_001::*;
  import BDD_SUB01_001::*;
  import BDD_SUB06_001::*;

  //
  // View metadata
  //
  metadata def DiagramMeta {
    attribute diagramId : String;
    attribute diagramType : String;
    attribute title : String;
    attribute goal : String;
    attribute dependencies : String[0..*];
  }

  metadata PD_OPT_002_Meta : DiagramMeta {
    :>> diagramId = "PD-OPT-002";
    :>> diagramType = "PD";
    :>> title = "PD OPT 002: End-to-End Throughput Budget (Optics + Instruments)";
    :>> goal = "Define an end-to-end throughput budget across optical surfaces and instrument interfaces, including allocation per surface group and an overall minimum throughput target tied to sensitivity NFR.";
    :>> dependencies = {"PD-PRF-001", "RD-NF-001", "BDD-SUB01-001", "BDD-SUB06-001"};
  }

  //
  // Value types
  //
  value type ThroughputFraction : Real {
    // dimensionless fraction in [0,1]
  }

  value type ThroughputPercent : Real {
    // dimensionless percent in [0,100]
  }

  value type MarginFraction : Real {
    // dimensionless fraction in [0,1]
  }

  //
  // Wavelength band indexing
  //
  enum def WavelengthBand {
    item U;
    item B;
    item V;
    item R;
    item I;
    item J;
    item H;
    item K;
    item L;
    item M;
  }

  //
  // Helper functions
  //
  function def asPercent(x : ThroughputFraction) : ThroughputPercent {
    return 100.0 * x;
  }

  function def clamp01(x : Real) : ThroughputFraction {
    return if (x < 0.0) then 0.0
           else if (x > 1.0) then 1.0
           else x;
  }

  //
  // Contributor definition (each contributor provides a per-band throughput factor)
  //
  part def ThroughputContributor {
    attribute name : String;
    attribute tau : ThroughputFraction[WavelengthBand]; // per-band multiplicative factor
  }

  //
  // Budget model for TMT end-to-end throughput (Optics + Instruments)
  //
  part def TMT_EndToEndThroughputBudget {

    //
    // Trace anchors (conceptual)
    //
    attribute targetSystem : String = "Thirty Meter Telescope (TMT, 三十米望远镜)";

    //
    // Representative bands
    //
    attribute bands : WavelengthBand[0..*] = {
      WavelengthBand::J,
      WavelengthBand::H,
      WavelengthBand::K
    };

    //
    // Throughput contributors (>= 8) with allocation by surface group / interface
    //
    part m1GroupReflectivity : ThroughputContributor {
      :>> name = "M1 segment group reflectivity (effective)";
    }
    part m2Reflectivity : ThroughputContributor {
      :>> name = "M2 reflectivity";
    }
    part m3Reflectivity : ThroughputContributor {
      :>> name = "M3 reflectivity";
    }
    part relayOpticsTransmission : ThroughputContributor {
      :>> name = "Relay optics transmission (folds/lenses/windows)";
    }
    part scatteringLossFactor : ThroughputContributor {
      :>> name = "Scattering loss factor (1 - scatter)";
    }
    part obscurationFactor : ThroughputContributor {
      :>> name = "Obscuration/vignetting factor";
    }
    part instrumentInterfaceLoss : ThroughputContributor {
      :>> name = "Instrument interface coupling (port, AO feed, alignment)";
    }
    part instrumentInternalTransmission : ThroughputContributor {
      :>> name = "Instrument internal optics transmission";
    }
    part detectorQuantumEfficiency : ThroughputContributor {
      :>> name = "Detector QE factor";
    }

    //
    // Margin / contingency line items (dimensionless fractions)
    //
    attribute marginEngineering : MarginFraction[WavelengthBand];     // e.g., model uncertainty
    attribute contingencyProgrammatic : MarginFraction[WavelengthBand]; // e.g., growth allowance

    //
    // Computed end-to-end throughput (fraction and percent) per band
    //
    attribute throughputEndToEnd : ThroughputFraction[WavelengthBand];
    attribute throughputEndToEndPct : ThroughputPercent[WavelengthBand];

    //
    // Minimum throughput target tied to Sensitivity NFR (RD-NF-001)
    // NOTE: If RD-NF-001 provides SensitivityPerformanceNFRDef.throughputEndToEnd (per band),
    //       we reference it as the minimum threshold. Otherwise this attribute can be bound during integration.
    //
    attribute minThroughputTarget : ThroughputFraction[WavelengthBand];

    //
    // Representative band for acceptance check
    //
    attribute representativeBand : WavelengthBand = WavelengthBand::K;

    //
    // Roll-up constraint: multiply contributors and apply margin/contingency to compute end-to-end throughput
    //
    constraint def RollupConstraint {

      // Helper aliases
      let tau_m1 : ThroughputFraction[WavelengthBand] = m1GroupReflectivity.tau;
      let tau_m2 : ThroughputFraction[WavelengthBand] = m2Reflectivity.tau;
      let tau_m3 : ThroughputFraction[WavelengthBand] = m3Reflectivity.tau;
      let tau_relay : ThroughputFraction[WavelengthBand] = relayOpticsTransmission.tau;
      let tau_scatter : ThroughputFraction[WavelengthBand] = scatteringLossFactor.tau;
      let tau_obsc : ThroughputFraction[WavelengthBand] = obscurationFactor.tau;
      let tau_if : ThroughputFraction[WavelengthBand] = instrumentInterfaceLoss.tau;
      let tau_inst : ThroughputFraction[WavelengthBand] = instrumentInternalTransmission.tau;
      let tau_qe : ThroughputFraction[WavelengthBand] = detectorQuantumEfficiency.tau;

      // End-to-end (pre-margin) product
      let tau_product : ThroughputFraction[WavelengthBand] =
        tau_m1 * tau_m2 * tau_m3 * tau_relay * tau_scatter * tau_obsc * tau_if * tau_inst * tau_qe;

      // Apply margin + contingency as multiplicative de-rating factors (1 - x)
      let tau_derated : ThroughputFraction[WavelengthBand] =
        tau_product * (1.0 - marginEngineering) * (1.0 - contingencyProgrammatic);

      // Assign computed outputs (clamped)
      throughputEndToEnd = clamp01(tau_derated);
      throughputEndToEndPct = asPercent(throughputEndToEnd);

      // Basic sanity bounds for contributors and margins/contingency
      assert all(b in WavelengthBand) {
        (0.0 <= tau_m1[b] & tau_m1[b] <= 1.0) &
        (0.0 <= tau_m2[b] & tau_m2[b] <= 1.0) &
        (0.0 <= tau_m3[b] & tau_m3[b] <= 1.0) &
        (0.0 <= tau_relay[b] & tau_relay[b] <= 1.0) &
        (0.0 <= tau_scatter[b] & tau_scatter[b] <= 1.0) &
        (0.0 <= tau_obsc[b] & tau_obsc[b] <= 1.0) &
        (0.0 <= tau_if[b] & tau_if[b] <= 1.0) &
        (0.0 <= tau_inst[b] & tau_inst[b] <= 1.0) &
        (0.0 <= tau_qe[b] & tau_qe[b] <= 1.0) &
        (0.0 <= marginEngineering[b] & marginEngineering[b] <= 1.0) &
        (0.0 <= contingencyProgrammatic[b] & contingencyProgrammatic[b] <= 1.0);
      }
    }

    //
    // RD-NF-001 linkage (binding placeholder)
    //
    // If RD-NF-001 exposes SensitivityPerformanceNFRDef.throughputEndToEnd (per band),
    // integrate by binding minThroughputTarget to that NFR attribute in the integrated model.
    //

    //
    // Acceptance check: computed throughput meets minimum threshold for at least one representative band
    //
    constraint def MeetsMinimumThresholdCheck {
      assert throughputEndToEnd[representativeBand] >= minThroughputTarget[representativeBand];
    }

    //
    // Apply constraints
    //
    constraint rollup : RollupConstraint;
    constraint meetsMin : MeetsMinimumThresholdCheck;
  }

  //
  // Concrete instance for PD view (example values per representative bands; others can be populated in integration)
  //
  part TMT_PD_OPT_002 : TMT_EndToEndThroughputBudget {

    // Example minimum target (to be bound to RD-NF-001 Sensitivity terms during integration)
    // Provide at least one representative band threshold for validation
    :>> minThroughputTarget[WavelengthBand::K] = 0.25;

    // Contributor allocations (illustrative placeholders)
    // M1 effective reflectivity over segment group and number of bounces in group
    :>> m1GroupReflectivity.tau[WavelengthBand::K] = 0.90;
    :>> m2Reflectivity.tau[WavelengthBand::K] = 0.92;
    :>> m3Reflectivity.tau[WavelengthBand::K] = 0.92;
    :>> relayOpticsTransmission.tau[WavelengthBand::K] = 0.88;
    :>> scatteringLossFactor.tau[WavelengthBand::K] = 0.97;   // (1 - scatter)
    :>> obscurationFactor.tau[WavelengthBand::K] = 0.93;      // central obscuration/vignetting
    :>> instrumentInterfaceLoss.tau[WavelengthBand::K] = 0.90; // port + coupling
    :>> instrumentInternalTransmission.tau[WavelengthBand::K] = 0.85;
    :>> detectorQuantumEfficiency.tau[WavelengthBand::K] = 0.90;

    // Margin/contingency line items (illustrative)
    :>> marginEngineering[WavelengthBand::K] = 0.05;
    :>> contingencyProgrammatic[WavelengthBand::K] = 0.05;

    // Representative band for acceptance
    :>> representativeBand = WavelengthBand::K;
  }

}
