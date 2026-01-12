package BDD_SUB05_003_AdaptiveOptics {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    % Definitions for Adaptive Optics Subsystem %
    
    abstract part def AdaptiveOpticsSystem;
    abstract part def WavefrontSensor;
    abstract part def DeformableMirror;
    abstract part def AORealTimeController;
    abstract part def AOAcquisition;
    abstract part def AOLoopClosure;
    abstract part def AOResolution;
    
    % Attributes for Adaptive Optics Components %
    
    attribute def AOResolutionAccuracy {
        accuracy : LengthValue;
    }
    
    % Part Definitions for Adaptive Optics Subsystem %
    part AdaptiveOptics : AdaptiveOpticsSystem {
        part acquisition : AOAcquisition;
        part loopClosure : AOLoopClosure;
        part resolution : AOResolution {
            accuracy = 0.1 [micron];
        }
        part wavefrontSensor : WavefrontSensor;
        part deformableMirror : DeformableMirror;
        part controller : AORealTimeController;
    }
    
    % Interface Definitions for Adaptive Optics System %
    
    part inputPort :> ControlPorts::InputPort {
        attribute signalType : String;
    }
    
    part outputPort :> ControlPorts::OutputPort {
        attribute signalType : String;
    }
    
    % Connections and Relationships %
    
    inputPort :> AdaptiveOptics::acquisition;
    outputPort :> AdaptiveOptics::loopClosure;
    
    part telescopeControl :> TelescopeControlSystem {
        part adaptiveOpticsControl : AdaptiveOptics;
    }
    
    % Defining Key Interfaces %
    interface AdaptiveOpticsInterface {
        part wavefrontInput :> inputPort;
        part loopClosureOutput :> outputPort;
    }
    
    % Dependencies with Telescope System %
    part telescopeInterface :> TelescopeInterfaces::ControlInterface {
        part adaptiveOptics : AdaptiveOpticsInterface;
    }
    
}
