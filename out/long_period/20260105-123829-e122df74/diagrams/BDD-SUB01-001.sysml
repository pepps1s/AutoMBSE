package BDD_SUB01_TelescopeOpticalSystem {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    part def M1Segment :> Structure::MirrorSegment {
        attribute size : LengthValue {
            size = 0.5 [m];  % Approximate segment size
        }
        part alignmentSensor : Structure::WavefrontSensor;
    }

    part def M2 :> Structure::Mirror {
        attribute diameter : LengthValue {
            diameter = 6.0 [m];  % Secondary mirror diameter
        }
        part alignmentSystem : Structure::AlignmentSystem;
    }

    part def M3 :> Structure::Mirror {
        attribute diameter : LengthValue {
            diameter = 4.0 [m];  % Tertiary mirror diameter
        }
        part steeringSystem : Structure::SteeringSystem;
    }

    part def OpticalPath :> Structure::OpticalPath {
        part path : OpticalPathElement[3] {
            part : M1Segment;
            part : M2;
            part : M3;
        }
    }

    part def PhasingSystem :> Structure::PhasingSystem {
        part wavefrontSensor : Structure::WavefrontSensor;
        part actuator : Structure::SegmentActuator[492];  % For segmented primary mirror (M1)
    }

    part TelescopeOpticalSystem {
        part primaryMirror : M1Segment[492];
        part secondaryMirror : M2;
        part tertiaryMirror : M3;
        part opticalPath : OpticalPath;
        part phasingSystem : PhasingSystem;
    }

    % Define key interfaces and ports

    part def OpticalInterface :> Interface::OpticalPort {
        direction = inout;
        type = Optical;
    }

    part TelescopeOpticalSystem_Interface {
        part opticalPort : OpticalInterface;
        part alignmentPort : Interface::AlignmentPort;
    }

    % Dependencies

    % The following are abstracted dependency definitions
    part def RD_F_001 :> Requirement::FunctionalRequirement {
        description = "Defines key functional requirements for telescope optical system.";
    }

    part def RD_NF_001 :> Requirement::NonFunctionalRequirement {
        description = "Defines performance and operational constraints for telescope optical system.";
    }

}
