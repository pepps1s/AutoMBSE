package BDD_SUB07_ObservatoryControlSystem {

    import ScalarValues::*;
    import SI::*;
    import Quantities::*;
    
    part def Sequencer {
        part controlUnit : ControlUnit {
            attribute state : String = "Idle";
        }
        part timingModule : TimingModule {
            attribute sequenceDuration : Time = 0.5 [s];
        }
    }
    
    part def Configuration {
        part configModule : ConfigModule {
            attribute configType : String = "Standard";
            attribute configurationID : String;
        }
    }
    
    part def Telemetry {
        part telemetryModule : TelemetryModule {
            attribute telemetryID : String;
            attribute dataRate : Real = 10.0 [Mbps];
        }
        part telemetryInterface : Interface {
            direction : out;
            type : TelemetryData;
        }
    }
    
    part def Alarms {
        part alarmModule : AlarmModule {
            attribute alarmType : String = "Critical";
            attribute triggerThreshold : Real = 0.75 [unit];
        }
    }
    
    part def ObservatoryControlSystem {
        part sequencer : Sequencer;
        part configuration : Configuration;
        part telemetry : Telemetry;
        part alarms : Alarms;
        
        part healthMonitor : HealthMonitor {
            attribute status : String = "Operational";
        }
        
        part operatorInterface : OperatorInterface {
            attribute userLevel : String = "Admin";
        }
    }

    interface def TelemetryData {
        attribute dataContent : String;
        attribute timestamp : Time;
        attribute signalStrength : Real = 100 [dBm];
    }
    
    part def ControlUnit {
        attribute controlID : String;
        attribute operationalMode : String = "Automatic";
    }
    
    part def TimingModule {
        attribute cycleTime : Time = 0.1 [s];
        attribute sequencePhase : String = "Start";
    }
    
    part def ConfigModule {
        attribute configVersion : String = "v1.0";
        attribute configStatus : String = "Active";
    }
    
    part def AlarmModule {
        attribute alarmStatus : String = "Inactive";
    }
    
    part def HealthMonitor {
        attribute operationalStatus : String = "OK";
        attribute faultCode : String = "None";
    }
    
    part def OperatorInterface {
        attribute interfaceType : String = "Graphical";
        attribute screenResolution : String = "1920x1080";
    }

}
