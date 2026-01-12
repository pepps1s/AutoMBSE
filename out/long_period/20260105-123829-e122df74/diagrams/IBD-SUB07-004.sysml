package IBD_SUB07_OCSCommandRouterInternalWiring {

    // Part Definitions
    part def OCS_CommandRouter {
        port cmdPort: CommandPort;
        port configPort: ConfigPort;
        port timeSyncPort: TimeSyncPort;
        port authorizationHook: AuthorizationHook;
        port transportLayer: TransportLayer;
    }

    part def Adapter {
        port cmdIn: CommandInPort;
        port cmdOut: CommandOutPort;
        port configIn: ConfigInPort;
        port configOut: ConfigOutPort;
        port timeSyncIn: TimeSyncInPort;
        port timeSyncOut: TimeSyncOutPort;
    }

    part def AuthorizationHook {
        port authIn: AuthInPort;
        port authOut: AuthOutPort;
    }

    part def TransportLayer {
        port dataIn: DataInPort;
        port dataOut: DataOutPort;
    }

    // Port Definitions
    port def CommandPort {
        out cmdIn: CommandInPort;
        in cmdOut: CommandOutPort;
    }

    port def ConfigPort {
        out configIn: ConfigInPort;
        in configOut: ConfigOutPort;
    }

    port def TimeSyncPort {
        out timeSyncIn: TimeSyncInPort;
        in timeSyncOut: TimeSyncOutPort;
    }

    // Subsystem Adapters
    part def MotionAdapter {
        port cmdIn: CommandInPort;
        port cmdOut: CommandOutPort;
        port configIn: ConfigInPort;
        port configOut: ConfigOutPort;
        port timeSyncIn: TimeSyncInPort;
        port timeSyncOut: TimeSyncOutPort;
    }

    part def DataAdapter {
        port cmdIn: CommandInPort;
        port cmdOut: CommandOutPort;
        port configIn: ConfigInPort;
        port configOut: ConfigOutPort;
        port timeSyncIn: TimeSyncInPort;
        port timeSyncOut: TimeSyncOutPort;
    }

    part def SafetyAdapter {
        port cmdIn: CommandInPort;
        port cmdOut: CommandOutPort;
        port configIn: ConfigInPort;
        port configOut: ConfigOutPort;
        port timeSyncIn: TimeSyncInPort;
        port timeSyncOut: TimeSyncOutPort;
    }

    // Internal Connections
    connector def CommandRouterToAdapter {
        source: OCS_CommandRouter.cmdPort;
        target: MotionAdapter.cmdIn;
    }

    connector def ConfigRouterToAdapter {
        source: OCS_CommandRouter.configPort;
        target: MotionAdapter.configIn;
    }

    connector def TimeSyncRouterToAdapter {
        source: OCS_CommandRouter.timeSyncPort;
        target: MotionAdapter.timeSyncIn;
    }

    connector def AdapterToCommandRouter {
        source: MotionAdapter.cmdOut;
        target: OCS_CommandRouter.cmdPort;
    }

    connector def AdapterToConfigRouter {
        source: MotionAdapter.configOut;
        target: OCS_CommandRouter.configPort;
    }

    connector def AdapterToTimeSyncRouter {
        source: MotionAdapter.timeSyncOut;
        target: OCS_CommandRouter.timeSyncPort;
    }

}
