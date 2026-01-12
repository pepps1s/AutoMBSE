package TelescopeOpticalSystem_Decomposition {
    
    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    part def PrimaryMirror : TelescopeOptics::PrimaryMirrorAssembly {
        part : SegmentedPrimaryMirror;
        part : MirrorSegment[492];
    }
    
    part def SecondaryMirror : TelescopeOptics::SecondaryMirrorAssembly {
        part : SecondaryMirror;
        part : SecondaryPositioning;
    }

    part def TertiaryMirror : TelescopeOptics::TertiaryMirrorAssembly {
        part : ArticulatedTertiaryMirror;
        part : MirrorSupport;
    }

    part def OpticalPath : TelescopeOptics::OpticalPath {
        part : BeamTransfer;
        part : OpticsSupport;
        part : Baffles;
        part : CalibrationElements;
    }

    part def TelescopeOpticalSystem {
        part : PrimaryMirror;
        part : SecondaryMirror;
        part : TertiaryMirror;
        part : OpticalPath;
    }
    
    package SystemDependencies {
        import TelescopeOpticalSystem_Decomposition::TelescopeOpticalSystem;
        part def TelescopeSystem : TelescopeOpticalSystem;
    }

    part TelescopeOpticalSystemDecomposition {
        part : TelescopeSystem;
    }
}
