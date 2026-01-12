package TMTSystemModel {

    package IBDROOT019 {

        // Part Definitions for Internal Interface Bundles (as inventory hubs)
        part def TimeInterfaceBundle {
            attribute latency: Real;
            attribute bandwidth: Real;
            part timeSyncPort: TimeSyncPort;
            part telescopeOpticalSystemPort: TimeSyncPort;
            part mountStructurePort: TimeSyncPort;
            part enclosureSiteInfrastructurePort: TimeSyncPort;
            part alignmentPhasingSystemPort: TimeSyncPort;
            part adaptiveOpticsSubsystemPort: TimeSyncPort;
            part instrumentsPort: TimeSyncPort;
        }

        part def CmdInterfaceBundle {
            attribute latency: Real;
            attribute bandwidth: Real;
            part commandPort: CommandPort;
            part telescopeOpticalSystemPort: CommandPort;
            part mountStructurePort: CommandPort;
            part enclosureSiteInfrastructurePort: CommandPort;
            part alignmentPhasingSystemPort: CommandPort;
            part adaptiveOpticsSubsystemPort: CommandPort;
            part instrumentsPort: CommandPort;
        }

        part def TelemetryInterfaceBundle {
            attribute latency: Real;
            attribute bandwidth: Real;
            part telemetryPort: TelemetryPort;
            part telescopeOpticalSystemPort: TelemetryPort;
            part mountStructurePort: TelemetryPort;
            part enclosureSiteInfrastructurePort: TelemetryPort;
            part alignmentPhasingSystemPort: TelemetryPort;
            part adaptiveOpticsSubsystemPort: TelemetryPort;
            part instrumentsPort: TelemetryPort;
        }

        part def PowerInterfaceBundle {
            attribute voltage: Real;
            attribute frequency: Real;
            part powerPort: PowerPort;
            part telescopeOpticalSystemPort: PowerPort;
            part mountStructurePort: PowerPort;
            part enclosureSiteInfrastructurePort: PowerPort;
            part alignmentPhasingSystemPort: PowerPort;
            part adaptiveOpticsSubsystemPort: PowerPort;
            part instrumentsPort: PowerPort;
        }

        part def NetworkInterfaceBundle {
            attribute latency: Real;
            attribute bandwidth: Real;
            part networkPort: NetworkPort;
            part telescopeOpticalSystemPort: NetworkPort;
            part mountStructurePort: NetworkPort;
            part enclosureSiteInfrastructurePort: NetworkPort;
            part alignmentPhasingSystemPort: NetworkPort;
            part adaptiveOpticsSubsystemPort: NetworkPort;
            part instrumentsPort: NetworkPort;
        }

        part def SafetyInhibitInterfaceBundle {
            attribute latency: Real;
            attribute bandwidth: Real;
            part safetyPort: SafetyPort;
            part telescopeOpticalSystemPort: SafetyPort;
            part mountStructurePort: SafetyPort;
            part enclosureSiteInfrastructurePort: SafetyPort;
            part alignmentPhasingSystemPort: SafetyPort;
            part adaptiveOpticsSubsystemPort: SafetyPort;
            part instrumentsPort: SafetyPort;
        }

        part def DataInterfaceBundle {
            attribute latency: Real;
            attribute bandwidth: Real;
            part dataPort: DataPort;
            part telescopeOpticalSystemPort: DataPort;
            part mountStructurePort: DataPort;
            part enclosureSiteInfrastructurePort: DataPort;
            part alignmentPhasingSystemPort: DataPort;
            part adaptiveOpticsSubsystemPort: DataPort;
            part instrumentsPort: DataPort;
        }

        // Connector Definitions for Internal Interface Bundles
        connector def TimeInterfaceConnectors { part source: TimeSyncPort; part target: TimeSyncPort; }
        connector def CmdInterfaceConnectors { part source: CommandPort; part target: CommandPort; }
        connector def TelemetryInterfaceConnectors { part source: TelemetryPort; part target: TelemetryPort; }
        connector def PowerInterfaceConnectors { part source: PowerPort; part target: PowerPort; }
        connector def NetworkInterfaceConnectors { part source: NetworkPort; part target: NetworkPort; }
        connector def SafetyInhibitInterfaceConnectors { part source: SafetyPort; part target: SafetyPort; }
        connector def DataInterfaceConnectors { part source: DataPort; part target: DataPort; }

        // Subsystem Participation (reused names/types)
        part def TelescopeOpticalSystem {
            part timeSyncPort: TimeSyncPort;
            part commandPort: CommandPort;
            part telemetryPort: TelemetryPort;
            part powerPort: PowerPort;
            part networkPort: NetworkPort;
            part safetyPort: SafetyPort;
            part dataPort: DataPort;
        }

        part def MountStructure {
            part timeSyncPort: TimeSyncPort;
            part commandPort: CommandPort;
            part telemetryPort: TelemetryPort;
            part powerPort: PowerPort;
            part networkPort: NetworkPort;
            part safetyPort: SafetyPort;
            part dataPort: DataPort;
        }

        part def EnclosureSiteInfrastructure {
            part timeSyncPort: TimeSyncPort;
            part commandPort: CommandPort;
            part telemetryPort: TelemetryPort;
            part powerPort: PowerPort;
            part networkPort: NetworkPort;
            part safetyPort: SafetyPort;
            part dataPort: DataPort;
        }

        // IBD inventory root (instances + connections)
        part IBD_ROOT_019_InternalInterfaceInventory {

            // Subsystems (every subsystem participates in at least one bundle)
            part telescopeOpticalSystem : TelescopeOpticalSystem;
            part mountStructure : MountStructure;
            part enclosureSiteInfrastructure : EnclosureSiteInfrastructure;
            part alignmentPhasingSystem : AlignmentPhasingSystem;
            part adaptiveOpticsSubsystem : AdaptiveOpticsSubsystem;
            part instruments : Instruments;

            // Interface bundles (BBD-1: instantiate types so BDD-defined attributes exist in instance context)
            part timeBundle : TimeInterfaceBundle {
                attribute latency = 0.001;
                attribute bandwidth = 1000.0;
            }

            part cmdBundle : CmdInterfaceBundle {
                attribute latency = 0.010;
                attribute bandwidth = 10.0;
            }

            part telemetryBundle : TelemetryInterfaceBundle {
                attribute latency = 0.050;
                attribute bandwidth = 200.0;
            }

            part powerBundle : PowerInterfaceBundle {
                attribute voltage = 480.0;
                attribute frequency = 60.0;
            }

            part networkBundle : NetworkInterfaceBundle {
                attribute latency = 0.002;
                attribute bandwidth = 10000.0;
            }

            part safetyInhibitBundle : SafetyInhibitInterfaceBundle {
                attribute latency = 0.005;
                attribute bandwidth = 1.0;
            }

            part dataBundle : DataInterfaceBundle {
                attribute latency = 0.020;
                attribute bandwidth = 5000.0;
            }

            // Connections (each bundle connects to >= 3 subsystems)

            // Time
            connect tmt_time_tos : TimeInterfaceConnectors between timeBundle.telescopeOpticalSystemPort and telescopeOpticalSystem.timeSyncPort;
            connect tmt_time_mount : TimeInterfaceConnectors between timeBundle.mountStructurePort and mountStructure.timeSyncPort;
            connect tmt_time_esi : TimeInterfaceConnectors between timeBundle.enclosureSiteInfrastructurePort and enclosureSiteInfrastructure.timeSyncPort;
            connect tmt_time_aps : TimeInterfaceConnectors between timeBundle.alignmentPhasingSystemPort and alignmentPhasingSystem;
            connect tmt_time_ao : TimeInterfaceConnectors between timeBundle.adaptiveOpticsSubsystemPort and adaptiveOpticsSubsystem;
            connect tmt_time_inst : TimeInterfaceConnectors between timeBundle.instrumentsPort and instruments;

            // Command
            connect tmt_cmd_tos : CmdInterfaceConnectors between cmdBundle.telescopeOpticalSystemPort and telescopeOpticalSystem.commandPort;
            connect tmt_cmd_mount : CmdInterfaceConnectors between cmdBundle.mountStructurePort and mountStructure.commandPort;
            connect tmt_cmd_aps : CmdInterfaceConnectors between cmdBundle.alignmentPhasingSystemPort and alignmentPhasingSystem;
            connect tmt_cmd_ao : CmdInterfaceConnectors between cmdBundle.adaptiveOpticsSubsystemPort and adaptiveOpticsSubsystem;

            // Telemetry
            connect tmt_tlm_tos : TelemetryInterfaceConnectors between telemetryBundle.telescopeOpticalSystemPort and telescopeOpticalSystem.telemetryPort;
            connect tmt_tlm_mount : TelemetryInterfaceConnectors between telemetryBundle.mountStructurePort and mountStructure.telemetryPort;
            connect tmt_tlm_esi : TelemetryInterfaceConnectors between telemetryBundle.enclosureSiteInfrastructurePort and enclosureSiteInfrastructure.telemetryPort;
            connect tmt_tlm_aps : TelemetryInterfaceConnectors between telemetryBundle.alignmentPhasingSystemPort and alignmentPhasingSystem;
            connect tmt_tlm_ao : TelemetryInterfaceConnectors between telemetryBundle.adaptiveOpticsSubsystemPort and adaptiveOpticsSubsystem;
            connect tmt_tlm_inst : TelemetryInterfaceConnectors between telemetryBundle.instrumentsPort and instruments;

            // Power
            connect tmt_pwr_tos : PowerInterfaceConnectors between powerBundle.telescopeOpticalSystemPort and telescopeOpticalSystem.powerPort;
            connect tmt_pwr_mount : PowerInterfaceConnectors between powerBundle.mountStructurePort and mountStructure.powerPort;
            connect tmt_pwr_esi : PowerInterfaceConnectors between powerBundle.enclosureSiteInfrastructurePort and enclosureSiteInfrastructure.powerPort;
            connect tmt_pwr_inst : PowerInterfaceConnectors between powerBundle.instrumentsPort and instruments;

            // Network
            connect tmt_net_tos : NetworkInterfaceConnectors between networkBundle.telescopeOpticalSystemPort and telescopeOpticalSystem.networkPort;
            connect tmt_net_mount : NetworkInterfaceConnectors between networkBundle.mountStructurePort and mountStructure.networkPort;
            connect tmt_net_esi : NetworkInterfaceConnectors between networkBundle.enclosureSiteInfrastructurePort and enclosureSiteInfrastructure.networkPort;
            connect tmt_net_aps : NetworkInterfaceConnectors between networkBundle.alignmentPhasingSystemPort and alignmentPhasingSystem;
            connect tmt_net_ao : NetworkInterfaceConnectors between networkBundle.adaptiveOpticsSubsystemPort and adaptiveOpticsSubsystem;
            connect tmt_net_inst : NetworkInterfaceConnectors between networkBundle.instrumentsPort and instruments;

            // Safety Inhibit
            connect tmt_safe_tos : SafetyInhibitInterfaceConnectors between safetyInhibitBundle.telescopeOpticalSystemPort and telescopeOpticalSystem.safetyPort;
            connect tmt_safe_mount : SafetyInhibitInterfaceConnectors between safetyInhibitBundle.mountStructurePort and mountStructure.safetyPort;
            connect tmt_safe_esi : SafetyInhibitInterfaceConnectors between safetyInhibitBundle.enclosureSiteInfrastructurePort and enclosureSiteInfrastructure.safetyPort;

            // Data
            connect tmt_data_tos : DataInterfaceConnectors between dataBundle.telescopeOpticalSystemPort and telescopeOpticalSystem.dataPort;
            connect tmt_data_esi : DataInterfaceConnectors between dataBundle.enclosureSiteInfrastructurePort and enclosureSiteInfrastructure.dataPort;
            connect tmt_data_ao : DataInterfaceConnectors between dataBundle.adaptiveOpticsSubsystemPort and adaptiveOpticsSubsystem;
            connect tmt_data_inst : DataInterfaceConnectors between dataBundle.instrumentsPort and instruments;
        }
    }
}
