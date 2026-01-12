package BDD_INS_001_ScienceInstrumentsSubsystemDecomposition {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Define abstract components and subsystems %
    abstract part def Instrument;
    abstract part def InstrumentPort;
    abstract part def InstrumentType;
    abstract part def InstrumentControl;
    abstract part def CalibrationUnitInterface;

    % Define instrument port %
    part def InstrumentPort {
        direction : Direction;
        portType : String;
    }

    % Define instrument types %
    part def InstrumentType {
        name : String;
        supportedFunctions : List<String>;
    }

    % Define instrument control interface %
    part def InstrumentControl {
        controlType : String;
        controlProtocol : String;
    }

    % Define calibration unit interface %
    part def CalibrationUnitInterface {
        calibrationMethod : String;
        calibrationFrequency : Integer;
    }

    % Define the Instrument framework subsystem %
    part def ScienceInstrumentSubsystem {
        part ports : InstrumentPort;
        part types : InstrumentType;
        part control : InstrumentControl;
        part calibrationUnit : CalibrationUnitInterface;
    }

    % Define the connections and relationships between components %
    ScienceInstrumentSubsystem {
        part ports : InstrumentPort {
            direction = "out";
            portType = "data";
        }
        part types : InstrumentType {
            name = "Optical";
            supportedFunctions = ["Imaging", "Spectroscopy"];
        }
        part control : InstrumentControl {
            controlType = "Digital";
            controlProtocol = "TCP/IP";
        }
        part calibrationUnit : CalibrationUnitInterface {
            calibrationMethod = "Flat-field";
            calibrationFrequency = 100; % in hours
        }
    }

    % Acceptance Checks %
    acceptance_check InstrumentPortsDefined {
        condition : "ports != null";
        description : "Instrument ports are defined and connected.";
    }

    acceptance_check InstrumentTypesDefined {
        condition : "types != null";
        description : "Instrument types are specified.";
    }

    acceptance_check ControlCalibrationDefined {
        condition : "control != null && calibrationUnit != null";
        description : "Control and calibration unit interfaces are defined.";
    }

}
