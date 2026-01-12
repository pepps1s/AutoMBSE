package BDD_SUB02_Mount_Structure {
    
    import ScalarValues::*;
    import SI::*;
    
    % Define Parts %
    part def MountStructure;
    part def AltAzMount;
    part def Drives;
    part def Encoders;
    part def VibrationControl;
    
    % Define Attributes for Mount Structure %
    attribute def MountType {
        type : String;
        value = "Alt-Az";
    }
    
    attribute def DriveType {
        type : String;
        value = "Electric";
    }
    
    % Define Subsystem Structure for Mount/Structure %
    part def MountStructureSystem {
        part altAzMount : AltAzMount;
        part drives : Drives;
        part encoders : Encoders;
        part vibrationControl : VibrationControl;
        
        attribute systemType : MountType {
            type = String;
            value = "Alt-Az";
        }
    }
    
    % Define Ports %
    part mountPort {
        part : Real;   % Real-type port for motion control %
    }
    
    part encoderPort {
        part : Integer;   % Integer-type port for encoder data %
    }
    
    % Connections between Parts and Interfaces %
    part def MountControl {
        part connectionPort : mountPort;
        part encoderInterface : encoderPort;
    }
    
    % System Behavior and Connections %
    MountStructureSystem {
        part body : MountStructure {
            part drives : Drives;
            part encoders : Encoders;
            part vibrationControl : VibrationControl;
        }
        
        % Connections %
        connection mountPort -> mountPort;
        connection encoderPort -> encoderPort;
    }
}
