package BDD_SUB07_003_ObservatoryControlSystem {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    abstract part def ObservatoryControlSystem;
    
    part def Sequencer :> ObservatoryControlSystem {
        part definition : ControlSequence;
        part interface : TelemetryInterface;
        part interface : AlarmInterface;
    }
    
    part def Configurator :> ObservatoryControlSystem {
        part definition : ConfigurationParameters;
        part interface : TelemetryInterface;
    }
    
    part def TelemetrySystem :> ObservatoryControlSystem {
        part interface : TelemetryData;
        part interface : AlarmData;
    }
    
    part def AlarmSystem :> ObservatoryControlSystem {
        part interface : AlarmData;
    }
    
    part ObservatoryControlSystemStructure {
        part sequencer : Sequencer;
        part configurator : Configurator;
        part telemetry : TelemetrySystem;
        part alarms : AlarmSystem;
    }
    
    part def TelemetryInterface {
        port telemetryPort : TelemetryData inout;
    }
    
    part def AlarmInterface {
        port alarmPort : AlarmData inout;
    }

    part def ControlSequence {
        attribute sequenceType : String;
        attribute sequenceDuration : Real;
    }
    
    part def ConfigurationParameters {
        attribute configType : String;
        attribute configValue : Real;
    }
    
    part def TelemetryData {
        attribute telemetryValue : Real;
        attribute timestamp : Time;
    }
    
    part def AlarmData {
        attribute alarmStatus : String;
        attribute alarmType : String;
        attribute severityLevel : Integer;
    }

}
