package IBD-ROOT-009_TMT_InterfaceBaselineReconstitution {

    // ---- Minimal port type stubs (for port-block matching) ----
    port def ControlPort;
    port def SignalPort;
    port def DataPort;
    port def TelemetryPort;
    port def PowerPort;
    port def TimeSyncPort;
    port def SafetyPort;
    port def BeamPort;
    port def NetworkPort;

    // ---- Part definitions (types) ----
    part def TelescopeOpticalSystem {
        port out opticalControlPort: ControlPort;
        port in  opticalSignalPort: SignalPort;
        port out telemetryOut: TelemetryPort;
    }

    part def MountStructure {
        port in  mountControlIn: ControlPort;
        port out mountSignalOut: SignalPort;
    }

    part def EnclosureSiteInfrastructure {
        port in  enclosureControlIn: ControlPort;
        port out environmentalSignalOut: SignalPort;
    }

    part def AlignmentPhasingSystem {
        port in  alignmentControlIn: ControlPort;
        port out phasingSignalOut: SignalPort;
        port in  timingIn: TimeSyncPort;
        port in  safetyIn: SafetyPort;
        port out telemOut: TelemetryPort;
    }

    // [BBD-1] Must instantiate AdaptiveOpticsSubsystem (type name must match BDD exactly)
    part def AdaptiveOpticsSubsystem {
        // Attributes required by BDD-SUB05-002
        attribute aoEnabled: Boolean;
        attribute aoLoopRate_Hz: Real;
        attribute residualWFE_nm: Real;
        attribute strehlRatio: Real;

        // Representative ports consistent with BDD-SUB05-002 sample names
        port in  ocsCmdIn: ControlPort;
        port out statusOut: SignalPort;
        port out telemetryOut: TelemetryPort;
        port in  timeIn: TimeSyncPort;
        port in  pwrIn: PowerPort;
        port inout net: NetworkPort;
        port in  scienceBeamIn: BeamPort;
        port out scienceBeamOut: BeamPort;
    }

    part def Instruments {
        port in  instrumentControlIn: ControlPort;
        port in  beamIn: BeamPort;
        port out beamOut: BeamPort;
        port out dataOut: DataPort;
    }

    part def DataSystem {
        port in  rawIn: DataPort;
        port out dataProductsOut: DataPort;
        port out telemetryOut: TelemetryPort;
    }

    part def Safety {
        port in  cmdIn: SafetyPort;
        port out tlmOut: SafetyPort;
        port in  pwr: PowerPort;
        port out inhibitOut: SafetyPort;
    }

    part def OperationsMaintenance {
        port in  maintenanceCmdIn: ControlPort;
        port out maintenanceStatusOut: SignalPort;
        port in  safetyInterlockIn: SafetyPort;
        port inout dataIfc: DataPort;
    }

    // ---- Top-level SoS IBD instance with all subsystem parts ----
    part tmtInternalInterfaceBaseline {
        part tos   : TelescopeOpticalSystem;
        part mount : MountStructure;
        part enc   : EnclosureSiteInfrastructure;
        part aps   : AlignmentPhasingSystem;
        part ao    : AdaptiveOpticsSubsystem;
        part inst  : Instruments;
        part data  : DataSystem;
        part saf   : Safety;
        part ops   : OperationsMaintenance;

        // ---- Connectors (ensure no isolated blocks) ----
        connect tos.opticalControlPort to mount.mountControlIn;
        connect mount.mountSignalOut to enc.enclosureControlIn;
        connect enc.environmentalSignalOut to aps.alignmentControlIn;

        connect aps.phasingSignalOut to ao.ocsCmdIn;
        connect ao.scienceBeamOut to inst.beamIn;
        connect inst.dataOut to data.rawIn;

        connect data.telemetryOut to saf.cmdIn;
        connect saf.inhibitOut to ops.safetyInterlockIn;

        // Additional connectivity to ensure broad coupling
        connect ao.telemetryOut to aps.telemOut;
        connect ao.timeIn to aps.timingIn;
        connect ao.pwrIn to saf.pwr;
        connect ops.dataIfc to data.dataProductsOut;
        connect inst.instrumentControlIn to ao.statusOut;
        connect tos.telemetryOut to data.telemetryOut;
    }

}
