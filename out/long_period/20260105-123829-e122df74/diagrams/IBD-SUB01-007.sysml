package IBD_SUB01_M2M3_Control_Telemetry_Thermal {

    // --- Item (flow) Definitions ---
    item def Command;
    item def ControlStatus;
    item def TelemetrySignal;
    item def CoolantFlow;
    item def HeatFlow;

    // --- Port Definitions (reuse BDD-consistent names) ---
    port def ControlPort {
        in item command : Command;
        out item controlStatus : ControlStatus;
    }

    port def TelemetryPort {
        out item telemetry : TelemetrySignal;
    }

    port def ThermalPort {
        in item coolant : CoolantFlow;
        out item heat : HeatFlow;
    }

    // --- Attribute Definitions (names aligned to dependency views) ---
    attribute def temperatureGradient;
    attribute def Temperature;
    attribute def Pressure;

    // --- Part Definitions ---
    part def M2_Mirror {
        port controlPort : ControlPort;
        port telemetryPort : TelemetryPort;
        port thermalPort : ThermalPort;
    }

    part def M3_Mirror {
        port controlPort : ControlPort;
        port telemetryPort : TelemetryPort;
        port thermalPort : ThermalPort;
    }

    part def M2_Mirror_Controller {
        port controlPort : ~ControlPort;
        port telemetryPort : ~TelemetryPort;
        port thermalPort : ~ThermalPort;
    }

    part def M3_Mirror_Controller {
        port controlPort : ~ControlPort;
        port telemetryPort : ~TelemetryPort;
        port thermalPort : ~ThermalPort;
    }

    // --- Internal Connectivity (IBD structure) ---
    part def InternalConnectivity {
        part M2 : M2_Mirror;
        part M3 : M3_Mirror;
        part M2Controller : M2_Mirror_Controller;
        part M3Controller : M3_Mirror_Controller;

        // Command / control loop
        connector M2ControlConnection : M2Controller.controlPort -> M2.controlPort;
        connector M3ControlConnection : M3Controller.controlPort -> M3.controlPort;

        // Telemetry (from mirrors to controllers)
        connector M2TelemetryConnection : M2.telemetryPort -> M2Controller.telemetryPort;
        connector M3TelemetryConnection : M3.telemetryPort -> M3Controller.telemetryPort;

        // Thermal / cooling (coolant to mirrors, heat return from mirrors)
        connector M2ThermalConnection : M2Controller.thermalPort -> M2.thermalPort;
        connector M3ThermalConnection : M3Controller.thermalPort -> M3.thermalPort;

        // Environmental/thermal attribute consistent with RD-NF-004
        attribute temperatureGradient : temperatureGradient;

        // BDD-SUB01-003 required attributes instantiated in this context
        attribute temperature : Temperature;
        attribute pressure : Pressure;
    }

    // --- Required instantiation for BBD-1 ---
    part internalConnectivity : InternalConnectivity;

}
