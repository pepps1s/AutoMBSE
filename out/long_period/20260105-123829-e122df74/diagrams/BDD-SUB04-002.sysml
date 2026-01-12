package BDD_SUB04_002_AlignmentAndPhasingSystem {
    import ScalarValues::*;
    import SI::*;
    import Quantities::*;

    //========================
    // Abstract base concepts
    //========================
    abstract part def Subsystem;
    abstract part def HardwareUnit;
    abstract part def Sensor :> HardwareUnit;
    abstract part def Controller :> HardwareUnit;
    abstract part def Actuator :> HardwareUnit;
    abstract part def ElectronicsUnit :> HardwareUnit;

    //========================
    // Value / attribute defs
    //========================
    attribute def SegmentId { id : Integer; }
    attribute def SegmentCount { n : Integer; }

    attribute def WavelengthBand {
        name : String;
        wavelengthMin : LengthValue;
        wavelengthMax : LengthValue;
    }

    attribute def WFEValue { rms : LengthValue; }          // Wavefront Error (RMS)
    attribute def TipTiltValue { tip : Real; tilt : Real; } // dimensionless angles (model-level)
    attribute def PistonValue { piston : LengthValue; }
    attribute def UpdateCadence { period : TimeValue; }
    attribute def SampleRate { rate : FrequencyValue; }

    //========================
    // Interface / port defs
    //========================
    interface def OpticalWavefrontIf {
        attribute band : WavelengthBand;
        attribute wfe : WFEValue;
    }

    interface def WavefrontMeasurementIf {
        attribute measuredWfe : WFEValue;
        attribute tipTilt : TipTiltValue;
        attribute timestamp : TimeValue;
    }

    interface def SegmentStateIf {
        attribute segmentId : SegmentId;
        attribute piston : PistonValue;
        attribute tipTilt : TipTiltValue;
        attribute wfeResidual : WFEValue;
    }

    interface def SegmentCommandIf {
        attribute segmentId : SegmentId;
        attribute pistonCmd : PistonValue;
        attribute tipCmd : Real;
        attribute tiltCmd : Real;
        attribute timestamp : TimeValue;
    }

    interface def ControlStatusIf {
        attribute enabled : Boolean;
        attribute loopCadence : UpdateCadence;
        attribute residualWfe : WFEValue;
    }

    interface def TelemetryIf {
        attribute timestamp : TimeValue;
        attribute text : String;
    }

    interface def PowerIf {
        attribute voltage : ElectricPotentialValue;
        attribute current : ElectricCurrentValue;
        attribute enabled : Boolean;
    }

    interface def TimeSyncIf {
        attribute timestamp : TimeValue;
    }

    //========================
    // Key units (parts)
    //========================
    part def AlignmentAndPhasingSystem :> Subsystem {
        attribute name : String;
        attribute segmentCount : SegmentCount;

        // Ports / external interfaces
        port opticalIn  : in OpticalWavefrontIf;
        port statusOut  : out ControlStatusIf;
        port telemetryOut : out TelemetryIf;

        port powerIn    : in PowerIf;
        port timeIn     : in TimeSyncIf;

        // Internal decomposition (APS, WFS, segment actuators)
        part apsController : APSController;
        part wfs : WavefrontSensorSuite;
        part segmentActuation : SegmentActuatorSystem;
        part metrology : MetrologySubsystem;
        part signalChain : SignalConditioningAndIO;
    }

    // APS Controller (computes commands from measurements, closes loop)
    part def APSController :> Controller {
        attribute loopCadence : UpdateCadence;
        attribute phasingResidualTarget : WFEValue;

        port measIn     : in WavefrontMeasurementIf;
        port segStateIn : in SegmentStateIf;
        port segCmdOut  : out SegmentCommandIf;

        port statusOut  : out ControlStatusIf;
        port telemetryOut : out TelemetryIf;

        port timeIn     : in TimeSyncIf;
        port powerIn    : in PowerIf;
    }

    // Wavefront sensor suite (can include multiple sensors)
    part def WavefrontSensorSuite :> Sensor {
        attribute sampleRate : SampleRate;
        attribute band : WavelengthBand;

        port opticalIn  : in OpticalWavefrontIf;
        port measOut    : out WavefrontMeasurementIf;

        port telemetryOut : out TelemetryIf;
        port timeIn     : in TimeSyncIf;
        port powerIn    : in PowerIf;

        // Representative internal sensor (e.g., Shack-Hartmann WFS)
        part shWfs : ShackHartmannWFS;
    }

    part def ShackHartmannWFS :> Sensor {
        attribute lensletCount : Integer;
        attribute sampleRate : SampleRate;

        port opticalIn  : in OpticalWavefrontIf;
        port measOut    : out WavefrontMeasurementIf;

        port timeIn     : in TimeSyncIf;
        port powerIn    : in PowerIf;
    }

    // Segment actuator system (commands -> physical segment motions)
    part def SegmentActuatorSystem :> Actuator {
        attribute supportedSegments : SegmentCount;

        port cmdIn      : in SegmentCommandIf;
        port stateOut   : out SegmentStateIf;

        port telemetryOut : out TelemetryIf;
        port timeIn     : in TimeSyncIf;
        port powerIn    : in PowerIf;

        // Representative composition: per-segment actuation unit (arrayed)
        part segmentUnit[*] : SegmentActuationUnit;
        part actuatorElectronics : ActuatorDriveElectronics;
    }

    part def SegmentActuationUnit :> Actuator {
        attribute segmentId : SegmentId;

        port cmdIn    : in SegmentCommandIf;
        port stateOut : out SegmentStateIf;

        // Example local actuators (piston + tip/tilt)
        part pistonAct : PistonActuator;
        part tipTiltAct : TipTiltActuator;
    }

    part def PistonActuator :> Actuator {
        attribute strokeMax : LengthValue;
        port cmdIn : in SegmentCommandIf;
        port stateOut : out SegmentStateIf;
    }

    part def TipTiltActuator :> Actuator {
        attribute angleMax : Real;
        port cmdIn : in SegmentCommandIf;
        port stateOut : out SegmentStateIf;
    }

    part def ActuatorDriveElectronics :> ElectronicsUnit {
        port cmdIn : in SegmentCommandIf;
        port powerIn : in PowerIf;
        port telemetryOut : out TelemetryIf;
    }

    // Metrology subsystem (independent alignment/phasing sensing)
    part def MetrologySubsystem :> Sensor {
        attribute sampleRate : SampleRate;

        port segStateOut : out SegmentStateIf;
        port telemetryOut : out TelemetryIf;

        port timeIn : in TimeSyncIf;
        port powerIn : in PowerIf;

        part edgeSensors : SegmentEdgeSensorNetwork;
    }

    part def SegmentEdgeSensorNetwork :> Sensor {
        attribute channels : Integer;
        attribute sampleRate : SampleRate;

        port segStateOut : out SegmentStateIf;
        port timeIn : in TimeSyncIf;
        port powerIn : in PowerIf;
    }

    // Signal chain / I/O (interfaces to observatory control, data/telemetry)
    part def SignalConditioningAndIO :> ElectronicsUnit {
        port wfsMeasIn : in WavefrontMeasurementIf;
        port segStateIn : in SegmentStateIf;
        port segCmdIn : in SegmentCommandIf;

        port telemetryOut : out TelemetryIf;

        port timeIn : in TimeSyncIf;
        port powerIn : in PowerIf;
    }

    //========================
    // Top-level system context
    //========================
    part def ThirtyMeterTelescope {
        attribute name : String;

        // Subsystem of interest
        part aps : AlignmentAndPhasingSystem;

        // External boundary ports (representative)
        port power : out PowerIf;
        port time  : out TimeSyncIf;

        // Optical wavefront presented to APS/WFS (from telescope optics)
        port opticalWavefront : out OpticalWavefrontIf;

        // Observatory telemetry sink (abstract)
        port telemetrySink : in TelemetryIf;
    }

    //========================
    // Example usage instance (optional but parsable)
    //========================
    part tmt : ThirtyMeterTelescope {
        attribute name = "TMT (三十米望远镜)";

        // Wire key boundary connections (BDD-level via binding connectors)
        bind power to aps.powerIn;
        bind time to aps.timeIn;
        bind opticalWavefront to aps.opticalIn;
        bind aps.telemetryOut to telemetrySink;

        // Internal bindings inside APS (structural intent)
        bind aps.opticalIn to aps.wfs.opticalIn;
        bind aps.wfs.measOut to aps.apsController.measIn;

        bind aps.metrology.segStateOut to aps.apsController.segStateIn;
        bind aps.apsController.segCmdOut to aps.segmentActuation.cmdIn;

        bind aps.segmentActuation.stateOut to aps.signalChain.segStateIn;
        bind aps.wfs.measOut to aps.signalChain.wfsMeasIn;
        bind aps.apsController.segCmdOut to aps.signalChain.segCmdIn;

        bind aps.signalChain.telemetryOut to aps.telemetryOut;
        bind aps.apsController.statusOut to aps.statusOut;
    }
}
