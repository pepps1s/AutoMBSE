package PD_DATA_001_DataRateAndVolumeBudget_TMT {

  // ------------------------------------------------------------
  // Diagram/View Metadata (PD-DATA-001)
  // ------------------------------------------------------------
  doc /* diagram_id: PD-DATA-001 */ {
    title = "PD DATA 001: Data Rate & Volume Budget (Acquisition → Archive)";
    goal = "Budget raw data rates, metadata overhead, quick-look products, pipeline outputs, and archive ingest volumes to ensure feasibility against interface and data requirements.";
    dependsOn = { "RD-F-007", "RD-NF-005", "RD-IF-001", "BDD-SUB08-001", "BDD-SUB07-001", "BDD-SUB06-001" };
    targetSystem = "Thirty Meter Telescope (TMT, 三十米望远镜)";
  }

  // ------------------------------------------------------------
  // Value Types + Units (Acceptance: rate/volume/latency with units)
  // ------------------------------------------------------------

  // Minimal unit stubs (so models remain self-contained and parsable)
  unit def bit;
  unit def s;
  unit def B;
  unit def day;

  unit def bps = bit / s;
  unit def Gbps = 1_000_000_000 * bps;

  unit def Byte = 8 * bit;
  unit def TB = 1_000_000_000_000 * Byte;     // decimal TB for budgeting

  unit def ms = 0.001 * s;
  unit def min = 60 * s;
  unit def hr  = 3_600 * s;

  valueType def DataRate {
    unit = Gbps;
  }

  valueType def DataVolume {
    unit = TB;
  }

  valueType def Latency {
    unit = ms;
  }

  valueType def Duration {
    unit = s;
  }

  valueType def Ratio {
    unit = 1;
  }

  // ------------------------------------------------------------
  // Reusable Interface Capacity Parameters (Acceptance)
  // ------------------------------------------------------------
  part def InterfaceCapacityDef {
    attribute name : String;
    attribute maxThroughput : DataRate;
    attribute maxNightlyIngest : DataVolume;
  }

  package InterfaceCapacities {
    // Named capacity parameters (reusable by other PD diagrams)
    part SiteBackboneLink : InterfaceCapacityDef {
      :>> name = "IF_SiteNetwork_Backbone";
      :>> maxThroughput = 80 [Gbps];          // placeholder budget value
      :>> maxNightlyIngest = 2_000 [TB];
    }

    part PartnerNetworkLink : InterfaceCapacityDef {
      :>> name = "IF_PartnerNetwork_Bidir";
      :>> maxThroughput = 20 [Gbps];
      :>> maxNightlyIngest = 800 [TB];
    }

    part OnsiteArchiveIngest : InterfaceCapacityDef {
      :>> name = "IF_DataArchive_Ingest";
      :>> maxThroughput = 40 [Gbps];
      :>> maxNightlyIngest = 1_500 [TB];
    }

    part PipelineComputeFabric : InterfaceCapacityDef {
      :>> name = "IF_Pipeline_Compute_Fabric";
      :>> maxThroughput = 100 [Gbps];
      :>> maxNightlyIngest = 3_000 [TB];
    }

    part StorageArrayWrite : InterfaceCapacityDef {
      :>> name = "IF_Storage_Array_Write";
      :>> maxThroughput = 60 [Gbps];
      :>> maxNightlyIngest = 2_500 [TB];
    }
  }

  // ------------------------------------------------------------
  // NFR Timing/Latency Terms (Referenced from RD-NF-005)
  // ------------------------------------------------------------
  // This package provides anchors mirroring RD-NF-005 timing/latency/sync terms.
  // Other views can "import PD_DATA_001_DataRateAndVolumeBudget_TMT::RD_NF_005_Ref::*"
  package RD_NF_005_Ref {
    // Names aligned to sample terms from RD-NF-005
    attribute clockSyncAccuracy : Latency = 1 [ms];
    attribute timestampAccuracy : Latency = 1 [ms];
    attribute commandLatency : Latency = 250 [ms];
    attribute telemetryRate : DataRate = 2 [Gbps];

    // End-to-end latency requirement placeholder (timing/latency check will reference it)
    attribute endToEndDataAvailabilityMax : Latency = 5_000 [ms]; // 5 s
  }

  // ------------------------------------------------------------
  // Data Contributors (Acceptance: >=10 contributors)
  // ------------------------------------------------------------
  part def DataContributorDef {
    attribute name : String;

    // Throughput and volume budgeted at contributor boundary
    attribute avgRate : DataRate;          // sustained/average
    attribute peakRate : DataRate;         // burst/peak
    attribute nightlyVolume : DataVolume;  // may be derived via constraints
    attribute enabled : Boolean = true;
  }

  // ------------------------------------------------------------
  // Budget Context (Acquisition → Archive)
  // ------------------------------------------------------------
  part def DataRateAndVolumeBudgetDef {

    // ---- Inputs / knobs ----
    attribute observingHoursPerNight : Duration = 10 [hr];
    attribute overheadFactor : Ratio = 1.0;          // additional multiplicative overhead
    attribute archiveReplicationFactor : Ratio = 2.0; // e.g., primary + replica

    // Latency knobs (for end-to-end check)
    attribute acquisitionToQuicklookLatency : Latency = 1_000 [ms];
    attribute quicklookToPipelineLatency : Latency = 1_500 [ms];
    attribute pipelineToArchiveIngestLatency : Latency = 2_000 [ms];

    // ---- Reusable interface capacities (referenced parts) ----
    part siteBackbone : InterfaceCapacityDef = InterfaceCapacities::SiteBackboneLink;
    part archiveIngestIF : InterfaceCapacityDef = InterfaceCapacities::OnsiteArchiveIngest;
    part pipelineFabricIF : InterfaceCapacityDef = InterfaceCapacities::PipelineComputeFabric;
    part storageWriteIF : InterfaceCapacityDef = InterfaceCapacities::StorageArrayWrite;

    // ---- Contributors (>=10; spanning raw, metadata, telemetry, QA, pipeline, replication) ----
    part instrumentRaw : DataContributorDef {
      :>> name = "Instrument Raw Frames (science detectors)";
      :>> avgRate = 12 [Gbps];
      :>> peakRate = 18 [Gbps];
    }

    part instrumentHousekeeping : DataContributorDef {
      :>> name = "Instrument Housekeeping (temps, voltages, configs)";
      :>> avgRate = 0.2 [Gbps];
      :>> peakRate = 0.5 [Gbps];
    }

    part metadataOverhead : DataContributorDef {
      :>> name = "Metadata Overhead (RD-F-007 provenance/config/time refs)";
      :>> avgRate = 0.6 [Gbps];
      :>> peakRate = 1.0 [Gbps];
    }

    part observatoryTelemetry : DataContributorDef {
      :>> name = "Observatory Telemetry Stream (OCS + subsystems)";
      :>> avgRate = 1.5 [Gbps];
      :>> peakRate = 3.0 [Gbps];
    }

    part engineeringLogs : DataContributorDef {
      :>> name = "Engineering Logs (events, alarms, trace logs)";
      :>> avgRate = 0.3 [Gbps];
      :>> peakRate = 0.8 [Gbps];
    }

    part quicklookQAProducts : DataContributorDef {
      :>> name = "Quick-Look / QA Products (RD-F-007 QuickLookQADef)";
      :>> avgRate = 2.0 [Gbps];
      :>> peakRate = 4.0 [Gbps];
    }

    part calibrationFrames : DataContributorDef {
      :>> name = "Calibration Frames (darks/flats/arcs)";
      :>> avgRate = 1.0 [Gbps];
      :>> peakRate = 2.0 [Gbps];
    }

    part pipelineIntermediates : DataContributorDef {
      :>> name = "Pipeline Intermediates (staging, chunking, scratch)";
      :>> avgRate = 6.0 [Gbps];
      :>> peakRate = 10.0 [Gbps];
    }

    part pipelineFinalProducts : DataContributorDef {
      :>> name = "Pipeline Final Products (L1/L2 science products)";
      :>> avgRate = 3.0 [Gbps];
      :>> peakRate = 5.0 [Gbps];
    }

    part archivePackaging : DataContributorDef {
      :>> name = "Archive Packaging/Manifests (indexes, checksums)";
      :>> avgRate = 0.4 [Gbps];
      :>> peakRate = 0.9 [Gbps];
    }

    part archiveReplication : DataContributorDef {
      :>> name = "Archive Replication (primary->replica)";
      :>> avgRate = 0.0 [Gbps];   // derived via constraint from ingest volume
      :>> peakRate = 8.0 [Gbps];
    }

    part partnerDistribution : DataContributorDef {
      :>> name = "Partner Distribution / Export (selected products)";
      :>> avgRate = 1.0 [Gbps];
      :>> peakRate = 3.0 [Gbps];
    }

    // ---- Aggregates ----
    attribute totalAvgRate : DataRate;
    attribute totalPeakRate : DataRate;
    attribute totalNightlyVolume : DataVolume;

    attribute acquisitionToArchiveEndToEndLatency : Latency;

    // ----------------------------------------------------------
    // Constraints: compute nightly volume and peak throughput
    // ----------------------------------------------------------

    constraint def NightlyVolumeFromRate {
      in rate : DataRate;
      in dur : Duration;
      out vol : DataVolume;

      // Convert: vol(TB) = rate(Gbps) * dur(s) / 8 (Gb->GB) / 1000 (GB->TB)  [using decimal TB]
      // Note: This is budgeting math; units are handled conceptually in this stub.
      vol == (rate * dur) / (8 * 1_000); 
    }

    constraint def SumRatesAndVolumes {
      in overhead : Ratio;
      in dur : Duration;

      in r1 : DataContributorDef;
      in r2 : DataContributorDef;
      in r3 : DataContributorDef;
      in r4 : DataContributorDef;
      in r5 : DataContributorDef;
      in r6 : DataContributorDef;
      in r7 : DataContributorDef;
      in r8 : DataContributorDef;
      in r9 : DataContributorDef;
      in r10 : DataContributorDef;
      in r11 : DataContributorDef;
      in r12 : DataContributorDef;

      out totalAvg : DataRate;
      out totalPeak : DataRate;
      out totalVol : DataVolume;

      // Sum sustained/average and peak rates; apply overhead factor
      totalAvg == overhead * (r1.avgRate + r2.avgRate + r3.avgRate + r4.avgRate + r5.avgRate + r6.avgRate + r7.avgRate + r8.avgRate + r9.avgRate + r10.avgRate + r11.avgRate + r12.avgRate);
      totalPeak == overhead * (r1.peakRate + r2.peakRate + r3.peakRate + r4.peakRate + r5.peakRate + r6.peakRate + r7.peakRate + r8.peakRate + r9.peakRate + r10.peakRate + r11.peakRate + r12.peakRate);

      // Total nightly volume derived from total average rate
      totalVol == (totalAvg * dur) / (8 * 1_000);
    }

    constraint def ArchiveReplicationVolume {
      in ingestVol : DataVolume;
      in replFactor : Ratio;
      out replVol : DataVolume;

      replVol == ingestVol * (replFactor - 1.0);
    }

    constraint def InterfaceCapacityCheck {
      in requiredPeak : DataRate;
      in requiredNightly : DataVolume;
      in cap : InterfaceCapacityDef;
      out ok : Boolean;

      ok == (requiredPeak <= cap.maxThroughput) & (requiredNightly <= cap.maxNightlyIngest);
    }

    // End-to-end latency check (Acceptance: references RD-NF-005; at least one check)
    constraint def EndToEndLatencyCheck {
      in t1 : Latency;
      in t2 : Latency;
      in t3 : Latency;
      in maxAllowed : Latency;
      out ok : Boolean;

      (t1 + t2 + t3) <= maxAllowed;
      ok == ((t1 + t2 + t3) <= maxAllowed);
    }

    // ----------------------------------------------------------
    // Constraint bindings (compute contributor volumes + totals)
    // ----------------------------------------------------------

    // Derive nightly volumes for each contributor from its avgRate
    constraint instrumentRawVol : NightlyVolumeFromRate {
      rate = instrumentRaw.avgRate;
      dur = observingHoursPerNight;
      vol = instrumentRaw.nightlyVolume;
    }

    constraint instrumentHKVol : NightlyVolumeFromRate {
      rate = instrumentHousekeeping.avgRate;
      dur = observingHoursPerNight;
      vol = instrumentHousekeeping.nightlyVolume;
    }

    constraint metadataVol : NightlyVolumeFromRate {
      rate = metadataOverhead.avgRate;
      dur = observingHoursPerNight;
      vol = metadataOverhead.nightlyVolume;
    }

    constraint telemetryVol : NightlyVolumeFromRate {
      rate = observatoryTelemetry.avgRate;
      dur = observingHoursPerNight;
      vol = observatoryTelemetry.nightlyVolume;
    }

    constraint logsVol : NightlyVolumeFromRate {
      rate = engineeringLogs.avgRate;
      dur = observingHoursPerNight;
      vol = engineeringLogs.nightlyVolume;
    }

    constraint quicklookVol : NightlyVolumeFromRate {
      rate = quicklookQAProducts.avgRate;
      dur = observingHoursPerNight;
      vol = quicklookQAProducts.nightlyVolume;
    }

    constraint calVol : NightlyVolumeFromRate {
      rate = calibrationFrames.avgRate;
      dur = observingHoursPerNight;
      vol = calibrationFrames.nightlyVolume;
    }

    constraint pipeIntVol : NightlyVolumeFromRate {
      rate = pipelineIntermediates.avgRate;
      dur = observingHoursPerNight;
      vol = pipelineIntermediates.nightlyVolume;
    }

    constraint pipeFinalVol : NightlyVolumeFromRate {
      rate = pipelineFinalProducts.avgRate;
      dur = observingHoursPerNight;
      vol = pipelineFinalProducts.nightlyVolume;
    }

    constraint archPackVol : NightlyVolumeFromRate {
      rate = archivePackaging.avgRate;
      dur = observingHoursPerNight;
      vol = archivePackaging.nightlyVolume;
    }

    // Replication contributor volume derived from total ingest (computed later); rate may be budgeted separately
    attribute archiveReplicationNightlyVolume : DataVolume;

    constraint replVol : ArchiveReplicationVolume {
      ingestVol = totalNightlyVolume;
      replFactor = archiveReplicationFactor;
      replVol = archiveReplicationNightlyVolume;
    }

    // Bind replication contributor nightly volume (rate remains independently peak-budgeted)
    constraint replContributorVolBind {
      archiveReplication.nightlyVolume == archiveReplicationNightlyVolume;
    }

    constraint partnerVol : NightlyVolumeFromRate {
      rate = partnerDistribution.avgRate;
      dur = observingHoursPerNight;
      vol = partnerDistribution.nightlyVolume;
    }

    // Compute totals (12 contributors listed above)
    constraint totals : SumRatesAndVolumes {
      overhead = overheadFactor;
      dur = observingHoursPerNight;

      r1 = instrumentRaw;
      r2 = instrumentHousekeeping;
      r3 = metadataOverhead;
      r4 = observatoryTelemetry;
      r5 = engineeringLogs;
      r6 = quicklookQAProducts;
      r7 = calibrationFrames;
      r8 = pipelineIntermediates;
      r9 = pipelineFinalProducts;
      r10 = archivePackaging;
      r11 = archiveReplication;
      r12 = partnerDistribution;

      totalAvg = totalAvgRate;
      totalPeak = totalPeakRate;
      totalVol = totalNightlyVolume;
    }

    // ----------------------------------------------------------
    // Interface feasibility checks (named, reusable parameters)
    // ----------------------------------------------------------
    attribute okSiteBackbone : Boolean;
    attribute okArchiveIngest : Boolean;
    attribute okPipelineFabric : Boolean;
    attribute okStorageWrite : Boolean;

    constraint checkSiteBackbone : InterfaceCapacityCheck {
      requiredPeak = totalPeakRate;
      requiredNightly = totalNightlyVolume;
      cap = siteBackbone;
      ok = okSiteBackbone;
    }

    constraint checkArchiveIngest : InterfaceCapacityCheck {
      requiredPeak = totalPeakRate;
      requiredNightly = totalNightlyVolume;
      cap = archiveIngestIF;
      ok = okArchiveIngest;
    }

    constraint checkPipelineFabric : InterfaceCapacityCheck {
      requiredPeak = totalPeakRate;
      requiredNightly = totalNightlyVolume;
      cap = pipelineFabricIF;
      ok = okPipelineFabric;
    }

    constraint checkStorageWrite : InterfaceCapacityCheck {
      requiredPeak = totalPeakRate;
      requiredNightly = totalNightlyVolume;
      cap = storageWriteIF;
      ok = okStorageWrite;
    }

    // ----------------------------------------------------------
    // End-to-end latency check referencing RD-NF-005 terms
    // ----------------------------------------------------------
    attribute okEndToEndLatency : Boolean;

    constraint e2eLatency : EndToEndLatencyCheck {
      t1 = acquisitionToQuicklookLatency;
      t2 = quicklookToPipelineLatency;
      t3 = pipelineToArchiveIngestLatency;
      maxAllowed = RD_NF_005_Ref::endToEndDataAvailabilityMax;
      ok = okEndToEndLatency;
    }

    // Also publish computed end-to-end latency (sum)
    constraint e2eLatencySum {
      acquisitionToArchiveEndToEndLatency == (acquisitionToQuicklookLatency + quicklookToPipelineLatency + pipelineToArchiveIngestLatency);
    }

  }

  // ------------------------------------------------------------
  // Instance for TMT Data System budget context (Acquisition → Archive)
  // ------------------------------------------------------------
  part TMT_DataRateAndVolumeBudget : DataRateAndVolumeBudgetDef {
    // Example overhead: 8% framing/headers + buffering overhead
    :>> overheadFactor = 1.08;

    // Example latencies (ms) aligned to RD-NF-005 timing notion
    :>> acquisitionToQuicklookLatency = 1_200 [ms];
    :>> quicklookToPipelineLatency = 1_400 [ms];
    :>> pipelineToArchiveIngestLatency = 1_800 [ms];

    // Example observing duration
    :>> observingHoursPerNight = 10 [hr];

    // Example: dual-write to two archive copies
    :>> archiveReplicationFactor = 2.0;
  }

}
