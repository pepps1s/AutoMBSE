package BDD_SUB02_Mount_Structure {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    abstract part def MountStructure;
    abstract part def Drive;
    abstract part def Encoder;
    abstract part def VibrationControl;
    abstract part def AltAzMount;
    
    % Define attributes for the parts %
    attribute def DriveCapacity {
        capacity : PowerValue; % in watts
    }

    attribute def EncoderResolution {
        resolution : Integer; % in arc-seconds
    }

    attribute def VibrationThreshold {
        threshold : FrequencyValue; % in Hz
    }

    % Define parts with specific attributes %
    part def AltAzMount {
        part baseStructure : MountStructure;
        part driveSystem : Drive {
            attribute capacity = 1500 [watts];
        }
        part encoderSystem : Encoder {
            attribute resolution = 0.0001 [arc-seconds];
        }
        part vibrationControlSystem : VibrationControl {
            attribute threshold = 0.01 [Hz];
        }
    }

    % Define system with associated parts %
    part TMT_MountStructure {
        part altAzMount : AltAzMount;
    }

    % Define connections via ports %
    port def mountControlPort {
        direction = inout;
        type = ControlSignal; % defines the control communication type
    }

    % Define interfaces between parts %
    part altAzControlInterface {
        port controlPort : mountControlPort;
        part controlSystem : TMT_MountStructure;
    }
    
    % Define the subsystem relationships %
    part MountStructure_Sys {
        part mountStructure : TMT_MountStructure;
        part altAzControl : altAzControlInterface;
    }

}
