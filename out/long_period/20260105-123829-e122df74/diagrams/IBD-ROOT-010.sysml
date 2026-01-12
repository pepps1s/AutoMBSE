package IBD_ROOT_InternalNetworksSegmentation {

    // Part Definitions
    part def mgmtNet;
    part def controlBus;
    part def dataNet;

    // Network Attributes
    attribute def bandwidth;
    attribute def latency;

    // Subsystem Parts Connections
    part def TelescopeOpticalSystem {
        port mgmtNetPort: mgmtNet;
        port controlBusPort: controlBus;
        port dataNetPort: dataNet;
    }

    part def EnclosureSiteInfrastructure {
        port mgmtNetPort: mgmtNet;
        port controlBusPort: controlBus;
        port dataNetPort: dataNet;
    }

    part def AdaptiveOpticsSubsystem {
        port mgmtNetPort: mgmtNet;
        port controlBusPort: controlBus;
        port dataNetPort: dataNet;
    }

    part def Instruments {
        port mgmtNetPort: mgmtNet;
        port controlBusPort: controlBus;
        port dataNetPort: dataNet;
    }

    // Internal Network Segments and Connections
    connector mgmtNetToTelescope {
        source TelescopeOpticalSystem.mgmtNetPort;
        target mgmtNet;
        direction out;
    }

    connector controlBusToTelescope {
        source TelescopeOpticalSystem.controlBusPort;
        target controlBus;
        direction out;
    }

    connector dataNetToTelescope {
        source TelescopeOpticalSystem.dataNetPort;
        target dataNet;
        direction out;
    }

    connector mgmtNetToEnclosure {
        source EnclosureSiteInfrastructure.mgmtNetPort;
        target mgmtNet;
        direction out;
    }

    connector controlBusToEnclosure {
        source EnclosureSiteInfrastructure.controlBusPort;
        target controlBus;
        direction out;
    }

    connector dataNetToEnclosure {
        source EnclosureSiteInfrastructure.dataNetPort;
        target dataNet;
        direction out;
    }

    connector mgmtNetToAO {
        source AdaptiveOpticsSubsystem.mgmtNetPort;
        target mgmtNet;
        direction out;
    }

    connector controlBusToAO {
        source AdaptiveOpticsSubsystem.controlBusPort;
        target controlBus;
        direction out;
    }

    connector dataNetToAO {
        source AdaptiveOpticsSubsystem.dataNetPort;
        target dataNet;
        direction out;
    }

    connector mgmtNetToInstruments {
        source Instruments.mgmtNetPort;
        target mgmtNet;
        direction out;
    }

    connector controlBusToInstruments {
        source Instruments.controlBusPort;
        target controlBus;
        direction out;
    }

    connector dataNetToInstruments {
        source Instruments.dataNetPort;
        target dataNet;
        direction out;
    }

    // Network Segment Details (Attributes: bandwidth, latency)
    part mgmtNet {
        attribute bandwidth: Bandwidth;
        attribute latency: Latency;
    }

    part controlBus {
        attribute bandwidth: Bandwidth;
        attribute latency: Latency;
    }

    part dataNet {
        attribute bandwidth: Bandwidth;
        attribute latency: Latency;
    }

    // Define data types for bandwidth and latency
    scalar def Bandwidth {
        unit: Mbps;
    }

    scalar def Latency {
        unit: ms;
    }
}
