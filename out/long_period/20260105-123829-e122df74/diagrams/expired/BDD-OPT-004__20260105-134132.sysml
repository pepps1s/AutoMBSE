package BDD-OPT-004_M3_Structure {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Define the Tertiary Mirror and Articulation Mechanisms %
    part def TertiaryMirror :> TelescopeOpticalSystem::Mirror {
        attribute mirrorDiameter : Length {
            value = 12 [m]; % Diameter of the tertiary mirror %
        }
        part articulationMechanism : ArticulationSystem {
            part : SteeringMechanism;
        }
    }

    % Define the Steering Mechanism for articulation %
    part def SteeringMechanism :> MechanicalSystems::ActuatorSystem {
        attribute actuatorType : String {
            value = "Rotary Actuator"; % Type of actuator used for articulation %
        }
        part rotationAxis : Direction {
            value = "Yaw"; % Steering axis for mirror movement %
        }
    }

    % Define the interfaces towards instrument feed paths %
    part def InstrumentFeedInterface {
        part : OpticalPath;
        part : PowerSupply;
        part : ThermalManagementSystem;
    }

    % Define the connection between parts %
    part M3_Structure {
        part body : TertiaryMirror;
        part articulation : SteeringMechanism;
        part interface : InstrumentFeedInterface;
    }

    % Define the structural dependencies and related parts for the system %
    dependency def BDD-OPT-001 :> BDD-OPT-001_TelescopeOpticalSystemDecomposition {
        part : TertiaryMirror;
        part : SteeringMechanism;
        part : InstrumentFeedInterface;
    }

    % Acceptance Checks %
    acceptance_checks {
        - articulation mechanisms for M3 defined.
        - interfaces toward instrument feeds defined.
    }
}
