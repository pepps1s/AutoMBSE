package BDD_STR_002_PointingTrackingControlHardware {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Define the main system for the Pointing and Tracking Control Hardware Breakdown
    block def PointingTrackingControlSystem {
        part def Controller;
        part def Sensor;
        part def Actuator;

        part def PointingTrackingControlHardware {
            part controller : Controller;
            part sensor : Sensor;
            part actuator : Actuator;
        }
        
        % Define hardware elements related to Pointing and Tracking loops
        block def Controller {
            attribute controlType : String;
            attribute powerRequirement : Power;
            attribute dataInput : Data;
            attribute dataOutput : Data;
        }

        block def Sensor {
            attribute sensorType : String;
            attribute accuracy : Real;
            attribute powerConsumption : Power;
            attribute outputSignal : Data;
        }

        block def Actuator {
            attribute actuatorType : String;
            attribute responseTime : Time;
            attribute powerRequirement : Power;
            attribute controlSignal : Data;
        }

        % Define the relationship between hardware components
        PointingTrackingControlHardware::controller --> PointingTrackingControlSystem::Controller;
        PointingTrackingControlHardware::sensor --> PointingTrackingControlSystem::Sensor;
        PointingTrackingControlHardware::actuator --> PointingTrackingControlSystem::Actuator;
    }

    % Define attributes for the TMT-specific elements for targeting the system
    block def TMTSystem {
        part def Telescope;
        part def Mount;
        part def Enclosure;
        
        % Defining elements under Telescope
        block def Telescope {
            part primaryMirror : Mirror;
            part secondaryMirror : Mirror;
            part tertiaryMirror : Mirror;
        }

        % Define Mirror structure for each telescope mirror
        block def Mirror {
            attribute size : Length;
            attribute material : String;
        }

        % Mounting system of the telescope
        block def Mount {
            part altAzMount : AltAzMount;
            part bearing : Bearing;
            part drive : Drive;
            part encoder : Encoder;
        }

        block def AltAzMount {
            attribute capacity : Real;
            attribute rotationSpeed : Real;
        }

        block def Bearing {
            attribute type : String;
            attribute material : String;
        }

        block def Drive {
            attribute motorType : String;
            attribute powerConsumption : Power;
        }

        block def Encoder {
            attribute precision : Real;
            attribute type : String;
        }

        % Define Enclosure system for telescope infrastructure
        block def Enclosure {
            attribute rotationSpeed : Real;
            attribute material : String;
            attribute size : Length;
        }
    }

    % Define dependency relationships between elements within the system
    PointingTrackingControlHardware --> TMTSystem::Telescope;
    TMTSystem::Telescope --> TMTSystem::Mount;
    TMTSystem::Telescope --> TMTSystem::Enclosure;
}
