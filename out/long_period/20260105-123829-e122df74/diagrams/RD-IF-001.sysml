package RD_IF_001_ExternalInterfacesInventory {
    // diagram_id: RD-IF-001
    // type: RD
    // title: External Interfaces Inventory (System Context I/O)
    // goal: List and characterize external interfaces to seed interface requirements.
    // dependencies: RD-CTX-001

    import ScalarValues::*;
    import Quantities::*;

    // -------------------------
    // Common typing
    // -------------------------

    enum def DirectionKind {
        in;
        out;
        bidirectional;
    }

    enum def FlowKind {
        data;
        energy;
        material;
    }

    enum def CriticalityKind {
        low;
        medium;
        high;
        missionCritical;
    }

    // Inventory item describing one external interface at the system boundary
    part def ExternalInterfaceItem {
        doc /* Inventory record for a single external interface at the TMT system boundary. */

        attribute interfaceName : String;
        attribute direction : DirectionKind;
        attribute flowType : FlowKind;

        // Flags to support later requirement and hazard analysis alignment
        attribute criticality : CriticalityKind;
        attribute safetyRelevant : Boolean;

        // Brief characterization to help align names with later requirement statements
        attribute characterization : String;
    }

    // Minimal structural placeholders for context (kept intentionally lightweight for RD view)
    part def TargetSystem {
        attribute systemName : String;
    }

    part def ExternalEntity {
        attribute entityName : String;
    }

    // -------------------------
    // Target system
    // -------------------------

    part TMT : TargetSystem {
        systemName = "Thirty Meter Telescope (TMT, 三十米望远镜)";
    }

    // -------------------------
    // External entities (context actors/sources/sinks)
    // -------------------------

    part PowerGrid : ExternalEntity { entityName = "Site Electrical Power Grid / Utility Feed"; }
    part UPS_Battery : ExternalEntity { entityName = "UPS / Battery Backup System"; }
    part Onsite_Generator : ExternalEntity { entityName = "On-site Backup Generator"; }

    part Site_Network_Backbone : ExternalEntity { entityName = "Site Network Backbone (LAN/WAN Edge)"; }
    part Time_Frequency_Reference : ExternalEntity { entityName = "Time & Frequency Reference (e.g., GPSDO / PTP Grandmaster)"; }
    part External_Observatory_Network : ExternalEntity { entityName = "External Observatory/Partner Network"; }

    part Operators : ExternalEntity { entityName = "Operators / Observers (Human Users)"; }
    part Maintainers : ExternalEntity { entityName = "Maintenance Personnel"; }
    part Safety_Authority : ExternalEntity { entityName = "Site Safety Authority / Compliance Body"; }

    part Science_Instruments : ExternalEntity { entityName = "Science Instruments (Hosted/Connected)"; }
    part Calibration_Equipment : ExternalEntity { entityName = "Calibration Sources / Calibration Equipment"; }
    part Metrology_Equipment : ExternalEntity { entityName = "External Metrology / Alignment Equipment"; }

    part Data_Archive : ExternalEntity { entityName = "Science Data Archive / Data Center"; }
    part Pipeline_Compute : ExternalEntity { entityName = "External Compute / Pipeline Cluster"; }

    part Environment : ExternalEntity { entityName = "Natural Environment (wind/temperature/humidity/dust)"; }
    part Cooling_Water_Supply : ExternalEntity { entityName = "Cooling Water / Glycol Supply (Facility)"; }
    part HVAC_Exhaust : ExternalEntity { entityName = "Facility HVAC Exhaust / Venting"; }

    part Physical_Access : ExternalEntity { entityName = "Physical Access / Handling Systems (doors, lifts, cranes)"; }
    part Emergency_Services : ExternalEntity { entityName = "Emergency Services / Response Interface"; }

    // -------------------------
    // External interface inventory (>= 15 items with direction + type + flags)
    // Names are written to align with later requirement statements (e.g., "TMT shall accept/provide ... via <interfaceName>").
    // -------------------------

    // Power / energy interfaces
    part IF_UtilityPower_In : ExternalInterfaceItem {
        interfaceName = "UtilityPowerInput";
        direction = DirectionKind::in;
        flowType = FlowKind::energy;
        criticality = CriticalityKind::missionCritical;
        safetyRelevant = true;
        characterization = "Primary AC power feed from site utility/grid into TMT facility distribution.";
    }

    part IF_UPS_Power_In : ExternalInterfaceItem {
        interfaceName = "UPSPowerInput";
        direction = DirectionKind::in;
        flowType = FlowKind::energy;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Conditioned/battery-backed power feed to maintain safe operation during brief outages.";
    }

    part IF_Generator_Power_In : ExternalInterfaceItem {
        interfaceName = "GeneratorPowerInput";
        direction = DirectionKind::in;
        flowType = FlowKind::energy;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Backup generator power input supporting safe-state and critical loads during extended outages.";
    }

    // Network / data interfaces
    part IF_SiteNetwork_Bidir : ExternalInterfaceItem {
        interfaceName = "SiteNetworkLink";
        direction = DirectionKind::bidirectional;
        flowType = FlowKind::data;
        criticality = CriticalityKind::missionCritical;
        safetyRelevant = false;
        characterization = "Primary LAN/WAN connectivity for command, telemetry, logging, services, and integration.";
    }

    part IF_TimeSync_In : ExternalInterfaceItem {
        interfaceName = "TimeSynchronizationInput";
        direction = DirectionKind::in;
        flowType = FlowKind::data;
        criticality = CriticalityKind::high;
        safetyRelevant = false;
        characterization = "Time/frequency reference input for timestamping, correlation, sequencing, and observatory timing.";
    }

    part IF_PartnerNetwork_Bidir : ExternalInterfaceItem {
        interfaceName = "PartnerNetworkExchange";
        direction = DirectionKind::bidirectional;
        flowType = FlowKind::data;
        criticality = CriticalityKind::medium;
        safetyRelevant = false;
        characterization = "External partner/observatory data exchange link (remote ops coordination, status, transfer).";
    }

    // Operator / human interfaces
    part IF_OperatorCommands_In : ExternalInterfaceItem {
        interfaceName = "OperatorCommandInput";
        direction = DirectionKind::in;
        flowType = FlowKind::data;
        criticality = CriticalityKind::missionCritical;
        safetyRelevant = true;
        characterization = "Operator/observer command inputs (observation sequences, overrides, acknowledgements).";
    }

    part IF_OperatorDisplays_Out : ExternalInterfaceItem {
        interfaceName = "OperatorStatusDisplayOutput";
        direction = DirectionKind::out;
        flowType = FlowKind::data;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Status/alarms/telemetry presentation to operators (HMIs, consoles, dashboards).";
    }

    part IF_MaintenanceAccess_Bidir : ExternalInterfaceItem {
        interfaceName = "MaintenanceServiceAccess";
        direction = DirectionKind::bidirectional;
        flowType = FlowKind::data;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Engineering/maintenance access for diagnostics, patching, calibration procedures, and service tools.";
    }

    // Instrumentation interfaces
    part IF_InstrumentControl_Bidir : ExternalInterfaceItem {
        interfaceName = "InstrumentControlAndTelemetry";
        direction = DirectionKind::bidirectional;
        flowType = FlowKind::data;
        criticality = CriticalityKind::missionCritical;
        safetyRelevant = false;
        characterization = "Command/control + telemetry interface between TMT control system and hosted science instruments.";
    }

    part IF_InstrumentPower_Out : ExternalInterfaceItem {
        interfaceName = "InstrumentPowerProvision";
        direction = DirectionKind::out;
        flowType = FlowKind::energy;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Provision of power to connected/hosted instruments through defined facility power services.";
    }

    part IF_InstrumentCoolant_Out : ExternalInterfaceItem {
        interfaceName = "InstrumentCoolingService";
        direction = DirectionKind::out;
        flowType = FlowKind::material;
        criticality = CriticalityKind::medium;
        safetyRelevant = true;
        characterization = "Cooling service delivery (air/liquid) to instrument ports to meet thermal constraints.";
    }

    // Calibration / metrology interfaces
    part IF_CalibrationInputs_In : ExternalInterfaceItem {
        interfaceName = "CalibrationSourceInput";
        direction = DirectionKind::in;
        flowType = FlowKind::material;
        criticality = CriticalityKind::medium;
        safetyRelevant = false;
        characterization = "External calibration stimuli (light sources/fixtures) introduced for calibration procedures.";
    }

    part IF_MetrologyData_In : ExternalInterfaceItem {
        interfaceName = "ExternalMetrologyDataInput";
        direction = DirectionKind::in;
        flowType = FlowKind::data;
        criticality = CriticalityKind::high;
        safetyRelevant = false;
        characterization = "Metrology/alignment measurement data ingested from external test equipment during integration/maintenance.";
    }

    // Data system / archive interfaces
    part IF_DataProducts_Out : ExternalInterfaceItem {
        interfaceName = "ScienceDataProductsOutput";
        direction = DirectionKind::out;
        flowType = FlowKind::data;
        criticality = CriticalityKind::missionCritical;
        safetyRelevant = false;
        characterization = "Raw and processed science data products output for storage, distribution, and science use.";
    }

    part IF_Metadata_Out : ExternalInterfaceItem {
        interfaceName = "ObservationMetadataOutput";
        direction = DirectionKind::out;
        flowType = FlowKind::data;
        criticality = CriticalityKind::high;
        safetyRelevant = false;
        characterization = "Observation/configuration metadata output required for provenance, pipelines, and archive indexing.";
    }

    part IF_ArchiveIngest_Bidir : ExternalInterfaceItem {
        interfaceName = "ArchiveIngestAndQuery";
        direction = DirectionKind::bidirectional;
        flowType = FlowKind::data;
        criticality = CriticalityKind::high;
        safetyRelevant = false;
        characterization = "Archive ingest, retrieval, and query interface for long-term storage and access services.";
    }

    part IF_ExternalCompute_Bidir : ExternalInterfaceItem {
        interfaceName = "ExternalPipelineComputeInterface";
        direction = DirectionKind::bidirectional;
        flowType = FlowKind::data;
        criticality = CriticalityKind::medium;
        safetyRelevant = false;
        characterization = "Data transfer and job control interface with external compute for pipelines/QA/processing.";
    }

    // Environment / facility interfaces
    part IF_EnvironmentalLoads_In : ExternalInterfaceItem {
        interfaceName = "EnvironmentalLoadsInput";
        direction = DirectionKind::in;
        flowType = FlowKind::material;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Environmental inputs (wind, dust, humidity, temperature gradients) affecting safety/performance envelopes.";
    }

    part IF_CoolingSupply_In : ExternalInterfaceItem {
        interfaceName = "FacilityCoolingSupplyInput";
        direction = DirectionKind::in;
        flowType = FlowKind::material;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Facility cooling supply (water/glycol/air) feeding thermal management for enclosure and equipment.";
    }

    part IF_HVAC_Exhaust_Out : ExternalInterfaceItem {
        interfaceName = "FacilityHVACExhaustOutput";
        direction = DirectionKind::out;
        flowType = FlowKind::material;
        criticality = CriticalityKind::medium;
        safetyRelevant = true;
        characterization = "Exhaust/venting output to maintain safe environmental conditions and heat rejection.";
    }

    // Safety / emergency interfaces
    part IF_EmergencyStop_In : ExternalInterfaceItem {
        interfaceName = "EmergencyStopInput";
        direction = DirectionKind::in;
        flowType = FlowKind::data;
        criticality = CriticalityKind::missionCritical;
        safetyRelevant = true;
        characterization = "Emergency stop / trip input initiating safe-state transitions and motion/power inhibit.";
    }

    part IF_SafetyInterlock_Bidir : ExternalInterfaceItem {
        interfaceName = "SafetyInterlockInterface";
        direction = DirectionKind::bidirectional;
        flowType = FlowKind::data;
        criticality = CriticalityKind::missionCritical;
        safetyRelevant = true;
        characterization = "Interlock status/control exchange for personnel safety, access control, and hazard mitigation.";
    }

    part IF_EmergencyNotification_Out : ExternalInterfaceItem {
        interfaceName = "EmergencyNotificationOutput";
        direction = DirectionKind::out;
        flowType = FlowKind::data;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Outbound emergency notifications (alarms, incident status) to site emergency response/services.";
    }

    // Physical access / handling interfaces
    part IF_PhysicalAccess_Bidir : ExternalInterfaceItem {
        interfaceName = "PhysicalAccessAndHandling";
        direction = DirectionKind::bidirectional;
        flowType = FlowKind::material;
        criticality = CriticalityKind::high;
        safetyRelevant = true;
        characterization = "Physical access and handling interactions (doors, lifts, cranes) for operations and maintenance.";
    }
}
