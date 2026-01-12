package IBD_IF_INT_008_SiteInfrastructure_AllSubsystems_UtilityServicesBundle {

  // diagram_id: IBD-IF-INT-008
  // type: IBD
  // title: IBD Internal IF: Site Infrastructure↔All Subsystems (Utility Services Bundle)
  // goal: Define the internal contract for facility services (power, cooling, network, environment telemetry) delivered to each subsystem.
  // dependencies: RD-IF-003, BDD-SUB03-003, IBD-ROOT-010, IBD-ROOT-014

  // Use only service port definitions that exist in BDD-SUB03-003
  import BDD_SUB03_003_EnclosureDome_SiteInfrastructure::*;

  // =========
  // Part defs
  // =========

  part def FacilityServicesBundle {
    // Facility-side instantiated service capacity/quality attributes
    attribute powerAvailable : Real;
    attribute thermalPower   : Real;
    attribute bandwidth      : Real;
    attribute latency        : Real;

    // Distinct service ports (power vs cooling vs network vs environment telemetry)
    port powerSvc   : PowerPort;
    port coolingSvc : CoolingPort;
    port netSvc     : DataPort;
    port envTelSvc  : EnvPort;
  }

  part def SubsystemUtilityClient {
    // Client-facing ports (conjugated to reflect "consumes services / receives telemetry")
    port powerIn    : ~PowerPort;
    port coolingIn  : ~CoolingPort;
    port netIn      : ~DataPort;
    port envTelIn   : ~EnvPort;
  }

  // Subsystems (specializations for clarity in the IBD)
  part def TelescopeOpticalSystem        : SubsystemUtilityClient;
  part def TelescopeStructureAndMount    : SubsystemUtilityClient;
  part def AdaptiveOpticsSubsystem       : SubsystemUtilityClient;
  part def InstrumentsSubsystem          : SubsystemUtilityClient;
  part def ObservatoryControlSystem      : SubsystemUtilityClient;
  part def DataSystem                    : SubsystemUtilityClient;

  // ======================
  // IBD: internal interface
  // ======================

  part def IBD_IF_INT_008_InternalIF {

    // Facility provider
    part facility : FacilityServicesBundle {
      powerAvailable = 1.0;
      thermalPower   = 1.0;
      bandwidth      = 1.0;
      latency        = 1.0;
    }

    // At least five subsystems (we provide six)
    part telescopeOptics : TelescopeOpticalSystem;
    part structureMount  : TelescopeStructureAndMount;
    part ao              : AdaptiveOpticsSubsystem;
    part instruments     : InstrumentsSubsystem;
    part ocs             : ObservatoryControlSystem;
    part data            : DataSystem;

    // ---------------------------------------
    // Power service distribution (distinct)
    // ---------------------------------------
    connector pwr_to_telescopeOptics (facility.powerSvc, telescopeOptics.powerIn);
    connector pwr_to_structureMount  (facility.powerSvc, structureMount.powerIn);
    connector pwr_to_ao              (facility.powerSvc, ao.powerIn);
    connector pwr_to_instruments     (facility.powerSvc, instruments.powerIn);
    connector pwr_to_ocs             (facility.powerSvc, ocs.powerIn);
    connector pwr_to_data            (facility.powerSvc, data.powerIn);

    // ---------------------------------------
    // Cooling service distribution (distinct)
    // ---------------------------------------
    connector cool_to_telescopeOptics (facility.coolingSvc, telescopeOptics.coolingIn);
    connector cool_to_structureMount  (facility.coolingSvc, structureMount.coolingIn);
    connector cool_to_ao              (facility.coolingSvc, ao.coolingIn);
    connector cool_to_instruments     (facility.coolingSvc, instruments.coolingIn);
    connector cool_to_ocs             (facility.coolingSvc, ocs.coolingIn);
    connector cool_to_data            (facility.coolingSvc, data.coolingIn);

    // ---------------------------------------
    // Network service distribution (distinct)
    // ---------------------------------------
    connector net_to_telescopeOptics (facility.netSvc, telescopeOptics.netIn);
    connector net_to_structureMount  (facility.netSvc, structureMount.netIn);
    connector net_to_ao              (facility.netSvc, ao.netIn);
    connector net_to_instruments     (facility.netSvc, instruments.netIn);
    connector net_to_ocs             (facility.netSvc, ocs.netIn);
    connector net_to_data            (facility.netSvc, data.netIn);

    // -------------------------------------------------
    // Environment telemetry distribution (distinct)
    // -------------------------------------------------
    connector env_to_telescopeOptics (facility.envTelSvc, telescopeOptics.envTelIn);
    connector env_to_structureMount  (facility.envTelSvc, structureMount.envTelIn);
    connector env_to_ao              (facility.envTelSvc, ao.envTelIn);
    connector env_to_instruments     (facility.envTelSvc, instruments.envTelIn);
    connector env_to_ocs             (facility.envTelSvc, ocs.envTelIn);
    connector env_to_data            (facility.envTelSvc, data.envTelIn);
  }

}
