package PD_IF_001_InstrumentServicesCapacityBudget {
    doc /* diagram_id: PD-IF-001
           type: PD
           title: PD IF 001: Instrument Services Capacity Budget (Power/Data/Thermal/Control)
           goal: Translate black-box instrument port requirements into allocated service capacities and margins per port/service category for PD baseline.
           dependencies: RD-IF-002, RD-F-005, BDD-SUB06-001, BDD-SUB03-001, PD-PWR-001, PD-NET-001, PD-TH-001
        */;

    // -----------------------------
    // Units (minimal, for readability)
    // -----------------------------
    unit W;
    unit Gbps;
    unit K;
    unit percent;

    // -----------------------------
    // Service categories (>= 6)
    // -----------------------------
    enum ServiceCategory {
        Power_HVDC,
        Power_LVDC,
        Data_Control,
        Data_Telemetry,
        Thermal_Cooling,
        Thermal_HeatRejection,
        Control_Commands,
        Control_SafetyInterlock
    }

    // -----------------------------
    // Upstream budget sources (explicit references)
    // -----------------------------
    part def PD_PWR_001_ElectricalPowerBudget_Source {
        attribute steadyStatePowerAvailable : W;
        attribute peakPowerAvailable       : W;
        attribute backupPowerAvailable     : W;
    }

    part def PD_NET_001_NetworkBandwidthAvailabilityBudget_Source {
        attribute controlNetworkBandwidthAvailable   : Gbps;
        attribute telemetryNetworkBandwidthAvailable : Gbps;
        attribute partnerNetworkBandwidthAvailable   : Gbps;
        attribute timeSyncNetworkBandwidthAvailable  : Gbps;
        attribute endToEndAvailabilityAvailable      : percent;
    }

    part def PD_TH_001_ObservatoryThermalStabilityBudget_Source {
        attribute instrumentBayCoolingAvailable : W;
        attribute hvacHeatRejectionAvailable   : W;
        attribute allowableInstrumentBayDeltaT : K;
    }

    // -----------------------------
    // RD-IF-002 requirement "views" (black-box per-port requirements)
    // -----------------------------
    part def RD_IF_002_InstrumentPortInterfaceRequirements {
        // Required capacities per port (black-box)
        attribute powerCapacityRequired                : W;
        attribute dataBandwidthRequired                : Gbps;
        attribute thermalCapacityRequired              : W;        // removal / cooling required
        attribute controlProtocolReliabilityRequired   : percent;  // e.g., 99.9%
    }

    // -----------------------------
    // BDD-SUB06-001 alignment: InstrumentPort with portId
    // (Naming consistency: portId like "PORT-01", etc.)
    // -----------------------------
    part def InstrumentPort {
        attribute portId       : String;
        attribute isOccupied   : Boolean;
    }

    // Representative port classes (>= 3)
    part def InstrumentPortClass {
        attribute portClassName : String;
        attribute portIdPrefix  : String;
    }

    // -----------------------------
    // Per-port capacity allocations and margins (actual capacities)
    // -----------------------------
    part def PortServiceCapacities {
        // Identification / alignment
        attribute portId : String;

        // "Actual"/allocated capacities for the port baseline
        attribute powerCapacityActual              : W;
        attribute dataBandwidthActual              : Gbps;
        attribute thermalCapacityActual            : W;
        attribute controlProtocolReliabilityActual : percent;

        // Optional: service-category bookkeeping (at least 6 categories covered)
        attribute usesPowerHVDC          : Boolean;
        attribute usesPowerLVDC          : Boolean;
        attribute usesDataControl        : Boolean;
        attribute usesDataTelemetry      : Boolean;
        attribute usesThermalCooling     : Boolean;
        attribute usesThermalHeatReject  : Boolean;
        attribute usesControlCommands    : Boolean;
        attribute usesControlSafetyILK   : Boolean;

        // Margins (computed)
        attribute powerMargin       : W;
        attribute dataMargin        : Gbps;
        attribute thermalMargin     : W;
        attribute reliabilityMargin : percent;
    }

    // -----------------------------
    // Constraint library: required vs actual with margin checks
    // -----------------------------
    constraint def CapacityMeetsOrExceedsRequired {
        in  reqPower       : W;
        in  actPower       : W;
        in  reqData        : Gbps;
        in  actData        : Gbps;
        in  reqThermal     : W;
        in  actThermal     : W;
        in  reqReliability : percent;
        in  actReliability : percent;

        out powerMargin       : W;
        out dataMargin        : Gbps;
        out thermalMargin     : W;
        out reliabilityMargin : percent;

        constraint {
            powerMargin       = actPower - reqPower;
            dataMargin        = actData  - reqData;
            thermalMargin     = actThermal - reqThermal;
            reliabilityMargin = actReliability - reqReliability;

            actPower       >= reqPower;
            actData        >= reqData;
            actThermal     >= reqThermal;
            actReliability >= reqReliability;
        }
    }

    constraint def PortAllocationWithinUpstreamBudgets {
        // Compare allocations against upstream available sources (explicitly PD-PWR-001/PD-NET-001/PD-TH-001)
        in portPowerAlloc       : W;
        in portDataControlAlloc : Gbps;
        in portDataTelemAlloc   : Gbps;
        in portCoolingAlloc     : W;

        in pwrSteadyAvail : W;
        in netCtrlAvail   : Gbps;
        in netTelemAvail  : Gbps;
        in thCoolAvail    : W;

        constraint {
            portPowerAlloc       <= pwrSteadyAvail;
            portDataControlAlloc <= netCtrlAvail;
            portDataTelemAlloc   <= netTelemAvail;
            portCoolingAlloc     <= thCoolAvail;
        }
    }

    // -----------------------------
    // Main PD baseline model
    // -----------------------------
    part def PD_IF_001_Baseline {
        // Upstream sources (explicit references)
        part pwrBudget  : PD_PWR_001_ElectricalPowerBudget_Source;
        part netBudget  : PD_NET_001_NetworkBandwidthAvailabilityBudget_Source;
        part thBudget   : PD_TH_001_ObservatoryThermalStabilityBudget_Source;

        // Port classes (>= 3 representative)
        part portClass_NASMYTH : InstrumentPortClass {
            :>> portClassName = "NASMYTH";
            :>> portIdPrefix  = "PORT-N";
        }
        part portClass_CASSEGRAIN : InstrumentPortClass {
            :>> portClassName = "CASSEGRAIN";
            :>> portIdPrefix  = "PORT-C";
        }
        part portClass_AO_BENCH : InstrumentPortClass {
            :>> portClassName = "AO_BENCH";
            :>> portIdPrefix  = "PORT-AO";
        }

        // Representative ports aligned with BDD-SUB06-001 naming consistency (portId)
        part port_01 : InstrumentPort { :>> portId = "PORT-01"; :>> isOccupied = false; }
        part port_02 : InstrumentPort { :>> portId = "PORT-02"; :>> isOccupied = true;  }
        part port_03 : InstrumentPort { :>> portId = "PORT-03"; :>> isOccupied = true;  }

        // RD-IF-002 requirements per port (black-box)
        part rd_if_002_port_01 : RD_IF_002_InstrumentPortInterfaceRequirements;
        part rd_if_002_port_02 : RD_IF_002_InstrumentPortInterfaceRequirements;
        part rd_if_002_port_03 : RD_IF_002_InstrumentPortInterfaceRequirements;

        // Allocations / actual capacities per port
        part alloc_port_01 : PortServiceCapacities { :>> portId = "PORT-01"; }
        part alloc_port_02 : PortServiceCapacities { :>> portId = "PORT-02"; }
        part alloc_port_03 : PortServiceCapacities { :>> portId = "PORT-03"; }

        // -----------------------------
        // Example baseline values (can be redefined by downstream instantiations)
        // -----------------------------
        // Upstream available capacities (from PD-PWR-001 / PD-NET-001 / PD-TH-001)
        :>> pwrBudget.steadyStatePowerAvailable = 25000[W];
        :>> pwrBudget.peakPowerAvailable       = 40000[W];
        :>> pwrBudget.backupPowerAvailable     = 5000[W];

        :>> netBudget.controlNetworkBandwidthAvailable   = 40[Gbps];
        :>> netBudget.telemetryNetworkBandwidthAvailable = 100[Gbps];
        :>> netBudget.partnerNetworkBandwidthAvailable   = 20[Gbps];
        :>> netBudget.timeSyncNetworkBandwidthAvailable  = 1[Gbps];
        :>> netBudget.endToEndAvailabilityAvailable      = 99.95[percent];

        :>> thBudget.instrumentBayCoolingAvailable = 15000[W];
        :>> thBudget.hvacHeatRejectionAvailable   = 30000[W];
        :>> thBudget.allowableInstrumentBayDeltaT = 1.0[K];

        // RD-IF-002 per-port requirements (power/data/thermal/control reliability)
        :>> rd_if_002_port_01.powerCapacityRequired              = 1200[W];
        :>> rd_if_002_port_01.dataBandwidthRequired              = 5[Gbps];
        :>> rd_if_002_port_01.thermalCapacityRequired            = 900[W];
        :>> rd_if_002_port_01.controlProtocolReliabilityRequired = 99.90[percent];

        :>> rd_if_002_port_02.powerCapacityRequired              = 2500[W];
        :>> rd_if_002_port_02.dataBandwidthRequired              = 12[Gbps];
        :>> rd_if_002_port_02.thermalCapacityRequired            = 2000[W];
        :>> rd_if_002_port_02.controlProtocolReliabilityRequired = 99.95[percent];

        :>> rd_if_002_port_03.powerCapacityRequired              = 1800[W];
        :>> rd_if_002_port_03.dataBandwidthRequired              = 8[Gbps];
        :>> rd_if_002_port_03.thermalCapacityRequired            = 1400[W];
        :>> rd_if_002_port_03.controlProtocolReliabilityRequired = 99.90[percent];

        // Allocated/actual capacities (PD baseline allocations)
        :>> alloc_port_01.powerCapacityActual              = 1500[W];
        :>> alloc_port_01.dataBandwidthActual              = 8[Gbps];
        :>> alloc_port_01.thermalCapacityActual            = 1200[W];
        :>> alloc_port_01.controlProtocolReliabilityActual = 99.95[percent];

        :>> alloc_port_02.powerCapacityActual              = 3200[W];
        :>> alloc_port_02.dataBandwidthActual              = 15[Gbps];
        :>> alloc_port_02.thermalCapacityActual            = 2600[W];
        :>> alloc_port_02.controlProtocolReliabilityActual = 99.97[percent];

        :>> alloc_port_03.powerCapacityActual              = 2200[W];
        :>> alloc_port_03.dataBandwidthActual              = 10[Gbps];
        :>> alloc_port_03.thermalCapacityActual            = 1700[W];
        :>> alloc_port_03.controlProtocolReliabilityActual = 99.93[percent];

        // Service category coverage per port (>= 6 categories across the set)
        :>> alloc_port_01.usesPowerLVDC         = true;
        :>> alloc_port_01.usesDataControl       = true;
        :>> alloc_port_01.usesDataTelemetry     = true;
        :>> alloc_port_01.usesThermalCooling    = true;
        :>> alloc_port_01.usesControlCommands   = true;
        :>> alloc_port_01.usesControlSafetyILK  = false;

        :>> alloc_port_02.usesPowerHVDC         = true;
        :>> alloc_port_02.usesPowerLVDC         = true;
        :>> alloc_port_02.usesDataControl       = true;
        :>> alloc_port_02.usesDataTelemetry     = true;
        :>> alloc_port_02.usesThermalCooling    = true;
        :>> alloc_port_02.usesThermalHeatReject = true;
        :>> alloc_port_02.usesControlCommands   = true;
        :>> alloc_port_02.usesControlSafetyILK  = true;

        :>> alloc_port_03.usesPowerLVDC         = true;
        :>> alloc_port_03.usesDataControl       = true;
        :>> alloc_port_03.usesThermalCooling    = true;
        :>> alloc_port_03.usesControlCommands   = true;
        :>> alloc_port_03.usesControlSafetyILK  = true;

        // -----------------------------
        // RD-IF-002 compliance constraints per port (required vs actual + margins)
        // -----------------------------
        constraint chk_port_01 : CapacityMeetsOrExceedsRequired(
            reqPower       = rd_if_002_port_01.powerCapacityRequired,
            actPower       = alloc_port_01.powerCapacityActual,
            reqData        = rd_if_002_port_01.dataBandwidthRequired,
            actData        = alloc_port_01.dataBandwidthActual,
            reqThermal     = rd_if_002_port_01.thermalCapacityRequired,
            actThermal     = alloc_port_01.thermalCapacityActual,
            reqReliability = rd_if_002_port_01.controlProtocolReliabilityRequired,
            actReliability = alloc_port_01.controlProtocolReliabilityActual,
            powerMargin       = alloc_port_01.powerMargin,
            dataMargin        = alloc_port_01.dataMargin,
            thermalMargin     = alloc_port_01.thermalMargin,
            reliabilityMargin = alloc_port_01.reliabilityMargin
        );

        constraint chk_port_02 : CapacityMeetsOrExceedsRequired(
            reqPower       = rd_if_002_port_02.powerCapacityRequired,
            actPower       = alloc_port_02.powerCapacityActual,
            reqData        = rd_if_002_port_02.dataBandwidthRequired,
            actData        = alloc_port_02.dataBandwidthActual,
            reqThermal     = rd_if_002_port_02.thermalCapacityRequired,
            actThermal     = alloc_port_02.thermalCapacityActual,
            reqReliability = rd_if_002_port_02.controlProtocolReliabilityRequired,
            actReliability = alloc_port_02.controlProtocolReliabilityActual,
            powerMargin       = alloc_port_02.powerMargin,
            dataMargin        = alloc_port_02.dataMargin,
            thermalMargin     = alloc_port_02.thermalMargin,
            reliabilityMargin = alloc_port_02.reliabilityMargin
        );

        constraint chk_port_03 : CapacityMeetsOrExceedsRequired(
            reqPower       = rd_if_002_port_03.powerCapacityRequired,
            actPower       = alloc_port_03.powerCapacityActual,
            reqData        = rd_if_002_port_03.dataBandwidthRequired,
            actData        = alloc_port_03.dataBandwidthActual,
            reqThermal     = rd_if_002_port_03.thermalCapacityRequired,
            actThermal     = alloc_port_03.thermalCapacityActual,
            reqReliability = rd_if_002_port_03.controlProtocolReliabilityRequired,
            actReliability = alloc_port_03.controlProtocolReliabilityActual,
            powerMargin       = alloc_port_03.powerMargin,
            dataMargin        = alloc_port_03.dataMargin,
            thermalMargin     = alloc_port_03.thermalMargin,
            reliabilityMargin = alloc_port_03.reliabilityMargin
        );

        // -----------------------------
        // Explicit upstream-budget referencing constraints (available sources)
        // Split port data into control vs telemetry allocations (illustrative)
        // -----------------------------
        constraint upstream_chk_port_01 : PortAllocationWithinUpstreamBudgets(
            portPowerAlloc       = alloc_port_01.powerCapacityActual,
            portDataControlAlloc = 2[Gbps],
            portDataTelemAlloc   = 6[Gbps],
            portCoolingAlloc     = alloc_port_01.thermalCapacityActual,
            pwrSteadyAvail = pwrBudget.steadyStatePowerAvailable,
            netCtrlAvail   = netBudget.controlNetworkBandwidthAvailable,
            netTelemAvail  = netBudget.telemetryNetworkBandwidthAvailable,
            thCoolAvail    = thBudget.instrumentBayCoolingAvailable
        );

        constraint upstream_chk_port_02 : PortAllocationWithinUpstreamBudgets(
            portPowerAlloc       = alloc_port_02.powerCapacityActual,
            portDataControlAlloc = 5[Gbps],
            portDataTelemAlloc   = 10[Gbps],
            portCoolingAlloc     = alloc_port_02.thermalCapacityActual,
            pwrSteadyAvail = pwrBudget.steadyStatePowerAvailable,
            netCtrlAvail   = netBudget.controlNetworkBandwidthAvailable,
            netTelemAvail  = netBudget.telemetryNetworkBandwidthAvailable,
            thCoolAvail    = thBudget.instrumentBayCoolingAvailable
        );

        constraint upstream_chk_port_03 : PortAllocationWithinUpstreamBudgets(
            portPowerAlloc       = alloc_port_03.powerCapacityActual,
            portDataControlAlloc = 3[Gbps],
            portDataTelemAlloc   = 7[Gbps],
            portCoolingAlloc     = alloc_port_03.thermalCapacityActual,
            pwrSteadyAvail = pwrBudget.steadyStatePowerAvailable,
            netCtrlAvail   = netBudget.controlNetworkBandwidthAvailable,
            netTelemAvail  = netBudget.telemetryNetworkBandwidthAvailable,
            thCoolAvail    = thBudget.instrumentBayCoolingAvailable
        );
    }

    // Root instance for the view/baseline
    part PD_IF_001 : PD_IF_001_Baseline;
}
