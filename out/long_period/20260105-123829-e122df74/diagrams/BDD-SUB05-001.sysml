package BDD_SUB05_001_AdaptiveOptics {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    // Dependencies (expected to exist in the model repository)
    // RD-F-001, RD-NF-001
    // (Imports kept generic to remain parsable even if namespaces differ.)
    import RD_F_001_BlackBoxFunctionsOverview::*;
    import RD_NF_001_PerformanceNFR_SensitivityAndAngularResolution::*;

    // --------------------------
    // Common interface/port types
    // --------------------------

    interface def OpticalBeamIF;
    interface def WavefrontSensorIF;
    interface def DeformableMirrorIF;
    interface def TelescopeStateIF;
    interface def TimeSyncIF;
    interface def CommandIF;
    interface def TelemetryIF;
    interface def SafetyInterlockIF;
    interface def LaserGuideStarIF;

    port def OpticalBeamPort {
        inout beam : OpticalBeamIF;
    }

    port def WavefrontSensePort {
        inout wfs : WavefrontSensorIF;
    }

    port def DMControlPort {
        inout dm : DeformableMirrorIF;
    }

    port def TelescopeStatePort {
        inout state : TelescopeStateIF;
    }

    port def TimeSyncPort {
        inout time : TimeSyncIF;
    }

    port def CommandPort {
        inout cmd : CommandIF;
    }

    port def TelemetryPort {
        inout tlm : TelemetryIF;
    }

    port def SafetyPort {
        inout safety : SafetyInterlockIF;
    }

    port def LaserGuideStarPort {
        inout lgs : LaserGuideStarIF;
    }

    // --------------------------
    // Structural decomposition
    // --------------------------

    abstract part def Subsystem;

    // External/context subsystems (stubs for traceability + interfaces)
    part def TelescopeOpticalSystem :> Subsystem {
        port opticalOut : OpticalBeamPort;
        port telescopeState : TelescopeStatePort;
        port timeSync : TimeSyncPort;
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port safety : SafetyPort;
    }

    part def PointingTrackingAndStability :> Subsystem {
        port telescopeState : TelescopeStatePort;
        port cmd : CommandPort;
        port tlm : TelemetryPort;
    }

    part def ObservationPhasingAndExposureControl :> Subsystem {
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port timeSync : TimeSyncPort;
    }

    part def AlignmentAndPhasingSystem :> Subsystem {
        port telescopeState : TelescopeStatePort;
        port cmd : CommandPort;
        port tlm : TelemetryPort;
    }

    part def ScienceInstrumentInterface :> Subsystem {
        port opticalIn : OpticalBeamPort;
        port cmd : CommandPort;
        port tlm : TelemetryPort;
    }

    part def ObservatoryControlSystem :> Subsystem {
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port safety : SafetyPort;
        port timeSync : TimeSyncPort;
    }

    // AO internal elements
    part def AO_WavefrontSensorAssembly :> Subsystem {
        port opticalIn : OpticalBeamPort;
        port wfsOut : WavefrontSensePort;
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port timeSync : TimeSyncPort;
    }

    part def AO_DeformableMirrorAssembly :> Subsystem {
        port opticalIn : OpticalBeamPort;
        port opticalOut : OpticalBeamPort;
        port dmIn : DMControlPort;
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port safety : SafetyPort;
        port timeSync : TimeSyncPort;
    }

    part def AO_RealTimeController :> Subsystem {
        port wfsIn : WavefrontSensePort;
        port dmOut : DMControlPort;
        port telescopeState : TelescopeStatePort;
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port timeSync : TimeSyncPort;

        attribute aoEnabled : Boolean;
        attribute aoLoopRate : Real;
        attribute strehlRatio : Real;
        attribute residualWFE_AO : Real;
    }

    // Optional acquisition / loop closure / RTC support elements
    part def AO_AcquisitionAndGuiding :> Subsystem {
        port telescopeState : TelescopeStatePort;
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port timeSync : TimeSyncPort;
        port lgs : LaserGuideStarPort;
        port wfs : WavefrontSensePort;
    }

    part def AO_LoopClosureManager :> Subsystem {
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port timeSync : TimeSyncPort;
        port wfs : WavefrontSensePort;
        port dm : DMControlPort;
    }

    part def AO_LaserGuideStarFacility :> Subsystem {
        port cmd : CommandPort;
        port tlm : TelemetryPort;
        port safety : SafetyPort;
        port lgs : LaserGuideStarPort;
        port timeSync : TimeSyncPort;
    }

    // Subsystem of interest
    part def AdaptiveOpticsSubsystem :> Subsystem {
        // Interfaces to/from TMT context
        port opticalIn : OpticalBeamPort;
        port opticalOut : OpticalBeamPort;

        port ocsCmd : CommandPort;
        port ocsTlm : TelemetryPort;
        port safety : SafetyPort;
        port telescopeState : TelescopeStatePort;
        port timeSync : TimeSyncPort;

        // Internal AO decomposition
        part wfsAssembly : AO_WavefrontSensorAssembly;
        part dmAssembly : AO_DeformableMirrorAssembly;
        part rtc : AO_RealTimeController;

        // Optional elements (present but may be disabled by configuration/requirements)
        part acquisition : AO_AcquisitionAndGuiding;
        part loopClosure : AO_LoopClosureManager;
        part lgsFacility : AO_LaserGuideStarFacility;
    }

    // Top-level TMT structural context (minimal for this BDD)
    part def TMT_System {
        part optical : TelescopeOpticalSystem;
        part pointing : PointingTrackingAndStability;
        part exposure : ObservationPhasingAndExposureControl;
        part aps : AlignmentAndPhasingSystem;
        part ocs : ObservatoryControlSystem;
        part instruments : ScienceInstrumentInterface;

        part ao : AdaptiveOpticsSubsystem;
    }

    // --------------------------
    // Requirement-to-implementation trace (fix RD-2)
    // NOTE: These requirement names are referenced exactly as reported by validation.
    // --------------------------

    // Science/Mission goal packages -> implemented at system/observatory + AO capability level
    satisfy MissionGoalsAndScienceDrivers by TMT_System;
    satisfy EarlyUniverseGoalMapping by TMT_System;
    satisfy GalaxyEvolutionGoalMapping by TMT_System;
    satisfy ExoplanetsGoalMapping by TMT_System;
    satisfy BHgrowthGoalMapping by TMT_System;
    satisfy StakeholderGroupConcerns by ObservatoryControlSystem;

    // Observing assumptions -> handled by OCS/config + exposure control + AO configuration
    satisfy ObservingAssumptionsAndConditionsDef by ObservationPhasingAndExposureControl;
    satisfy ObservingAssumptionsAndConditions by ObservationPhasingAndExposureControl;

    // Functional requirement defs/instances -> mapped to structural modules that realize them
    satisfy FR_PhasedSegmentedApertureDef by AlignmentAndPhasingSystem;
    satisfy FR_PhasedSegmentedAperture by AlignmentAndPhasingSystem;

    satisfy FR_PointingTrackingAndStabilityDef by PointingTrackingAndStability;
    satisfy FR_PointingTrackingAndStability by PointingTrackingAndStability;

    satisfy FR_ObservationPhasingAndExposureControlDef by ObservationPhasingAndExposureControl;
    satisfy FR_ObservationPhasingAndExposureControl by ObservationPhasingAndExposureControl;

    // AO optional requirement def/instance -> implemented by the AO subsystem (incl. acquisition/loop closure/RTC)
    satisfy FR_AdaptiveOpticsOptionalDef by AdaptiveOpticsSubsystem;
    satisfy FR_AdaptiveOpticsOptional by AdaptiveOpticsSubsystem;

    // Performance NFR defs -> realized by AO RTC + overall AO subsystem performance allocations
    satisfy SensitivityPerformanceNFRDef by AdaptiveOpticsSubsystem;
    satisfy AngularResolutionPerformanceNFRDef by AdaptiveOpticsSubsystem;

}
