package IBD_Internal_IF_Optics_AO_Beam_State_Interface_Bundle {
    import ScalarValues::*;
    import Quantities::*;

    // Local interface items (non-conflicting with dependency libraries)
    item def TelescopeState {
        attribute timestamp_s : Real;
        attribute frameId : String;
    }

    port def TelescopeStateOutPort {
        out item telescopeState : TelescopeState;
    }

    port def TelescopeStateInPort {
        in item telescopeState : TelescopeState;
    }

    // IBD: Internal IF between Telescope Optics (SUB01) and Adaptive Optics (SUB05)
    part def IBD_IF_INT_005_Optics_AO_InterfaceBundle {

        // Instantiate dependency-defined part types (BBD-1: instantiates beamWavelength/beamIntensity context)
        part optics : TelescopeOptics;
        part ao     : AdaptiveOpticsSubsystem;

        // Typed interface ports for robust connector typing
        port opticsBeamOut : OpticalBeamPort;
        port aoBeamIn      : OpticalBeamPort;

        port opticsStateOut : TelescopeStateOutPort;
        port aoStateIn      : TelescopeStateInPort;

        // At least one constraint/attribute for alignment/timing reference
        attribute frameAlignmentTol_arcsec : Real = 0.05;
        attribute timeSkewTol_ms           : Real = 1.0;

        constraint FrameAlignmentConstraint {
            condition: opticsStateOut.telescopeState.frameId == aoStateIn.telescopeState.frameId;
        }

        constraint TelescopeStateTimingConstraint {
            condition: (opticsStateOut.telescopeState.timestamp_s - aoStateIn.telescopeState.timestamp_s) <= (timeSkewTol_ms / 1000.0);
        }

        // Bind bundle ports to subsystem ports (reuse existing port names where available)
        bind opticsBeamOut to optics.opticalOutPort;
        bind aoBeamIn      to ao.opticalInPort;

        // Provide explicit telescopeState ports on the part usages to avoid name mismatches
        port optics.telescopeStateOutPort : TelescopeStateOutPort;
        port ao.telescopeStateInPort      : TelescopeStateInPort;

        bind opticsStateOut to optics.telescopeStateOutPort;
        bind aoStateIn      to ao.telescopeStateInPort;

        // Interface flows
        connect opticsBeamOut  to aoBeamIn;
        connect opticsStateOut to aoStateIn;
    }

    // IBD root usage for the diagram/view
    part ibd_if_int_005 : IBD_IF_INT_005_Optics_AO_InterfaceBundle;
}
