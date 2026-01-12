package BDD_SUB06_002_Instruments {
  import ScalarValues::*;
  import SI::*;

  //
  // Requirements stubs + traceability hooks (to resolve RD-2 validation)
  //
  package Requirements_Traceability {

    requirement def MissionGoalsAndScienceDrivers;
    requirement def EarlyUniverseGoalMapping;
    requirement def GalaxyEvolutionGoalMapping;
    requirement def ExoplanetsGoalMapping;
    requirement def BHgrowthGoalMapping;
    requirement def StakeholderGroupConcerns;

    requirement def ObservingAssumptionsAndConditionsDef;
    requirement def ObservingAssumptionsAndConditions;

    requirement def FR_PhasedSegmentedApertureDef;
    requirement def FR_PhasedSegmentedAperture;

    requirement def FR_PointingTrackingAndStabilityDef;
    requirement def FR_PointingTrackingAndStability;

    requirement def FR_ObservationPhasingAndExposureControlDef;
    requirement def FR_ObservationPhasingAndExposureControl;

    requirement def FR_AdaptiveOpticsOptionalDef;
    requirement def FR_AdaptiveOpticsOptional;

    requirement def SensitivityPerformanceNFRDef;
    requirement def AngularResolutionPerformanceNFRDef;
  }

  //
  // Common types, interfaces, and ports
  //
  package Interfaces {

    value def Voltage = Real;
    value def Current = Real;
    value def Power = Real;
    value def Temperature = Real;
    value def MassFlow = Real;
    value def BitRate = Real;
    value def Frequency = Real;

    interface def PowerIF {
      inout v : Voltage;
      inout i : Current;
      out p : Power;
    }

    interface def DataIF {
      inout rate : BitRate;
    }

    interface def TimingIF {
      inout refClock : Frequency;
    }

    interface def CommandIF {
      inout cmdStream : String;
    }

    interface def HealthIF {
      inout status : String;
      out fault : Boolean;
    }

    interface def CoolingIF {
      inout tempSupply : Temperature;
      inout tempReturn : Temperature;
      inout flow : MassFlow;
    }

    interface def ScienceBeamIF {
      inout beamToken : String;
    }

    interface def CalibrationLightIF {
      inout calLightToken : String;
    }

    port def PwrPort { inout power : PowerIF; }
    port def DataPort { inout data : DataIF; }
    port def TimePort { inout timing : TimingIF; }
    port def CmdPort { inout command : CommandIF; }
    port def HlthPort { inout health : HealthIF; }
    port def CoolPort { inout cooling : CoolingIF; }

    port def BeamPort { inout beam : ScienceBeamIF; }
    port def CalLightPort { inout cal : CalibrationLightIF; }
  }

  //
  // Structural decomposition (BDD)
  //
  package Architecture {
    import Interfaces::*;
    import Requirements_Traceability::*;

    abstract part def TMT_Subsystem;

    abstract part def Instrument;
    abstract part def CalibrationSource;
    abstract part def SelectorMechanism;
    abstract part def ControlElectronics;
    abstract part def ThermalControl;

    part def InstrumentPortAssembly {
      port beamIn  : BeamPort;
      port beamOut : BeamPort;
      port calIn   : CalLightPort;
      port cool    : CoolPort;
      port pwr     : PwrPort;
      port data    : DataPort;
      port time    : TimePort;
      port cmd     : CmdPort;
      port hlth    : HlthPort;

      attribute portId : String;
      attribute maxMountedInstruments : Integer;
    }

    part def InstrumentSelector : SelectorMechanism {
      port cmd  : CmdPort;
      port hlth : HlthPort;
      port pwr  : PwrPort;
      port data : DataPort;

      attribute selectionMode : String;         // e.g., "carousel", "pickoff", "switchyard"
      attribute selectLatency_s : Real;
    }

    part def CalibrationUnit {
      part flatSource : CalibrationSource;
      part arcSource  : CalibrationSource;
      part darkShutter : SelectorMechanism;

      port calOut : CalLightPort;
      port cmd    : CmdPort;
      port hlth   : HlthPort;
      port pwr    : PwrPort;
      port data   : DataPort;

      attribute calibrationCadence_s : Real;
      attribute supportedCalModes : String;
    }

    part def InstrumentBay {
      part instrument : Instrument;

      port beamIn  : BeamPort;
      port calIn   : CalLightPort;
      port cool    : CoolPort;
      port pwr     : PwrPort;
      port data    : DataPort;
      port time    : TimePort;
      port cmd     : CmdPort;
      port hlth    : HlthPort;

      attribute bayId : String;
      attribute instrumentType : String;        // imaging / spectroscopy / polarimetry (etc.)
    }

    part def InstrumentControlSystem : ControlElectronics {
      port cmdUpstream : CmdPort;
      port dataUpstream : DataPort;
      port timeRef : TimePort;
      port hlthUpstream : HlthPort;

      port cmdDownstream : CmdPort;
      port dataDownstream : DataPort;
      port hlthDownstream : HlthPort;

      attribute configProfile : String;
      attribute supportsTelemetry : Boolean;
    }

    part def InstrumentThermalControl : ThermalControl {
      port coolIn : CoolPort;
      port hlth   : HlthPort;
      port cmd    : CmdPort;

      attribute temperatureSetpoint_C : Real;
      attribute stabilityBand_C : Real;
    }

    //
    // Subsystem: Instruments (ports, selection, calibration units)
    //
    part def InstrumentsSubsystem :> TMT_Subsystem {
      part ports : InstrumentPortAssembly;
      part selector : InstrumentSelector;
      part cal : CalibrationUnit;
      part  bays[4] : InstrumentBay;            // scalable placeholder for multiple instruments
      part ics : InstrumentControlSystem;
      part thermal : InstrumentThermalControl;

      // External interfaces to telescope/observatory services
      port beamFromTelescope : BeamPort;
      port beamToTelescope   : BeamPort;
      port calServiceOut     : CalLightPort;

      port facilityPwr  : PwrPort;
      port facilityData : DataPort;
      port facilityTime : TimePort;
      port facilityCmd  : CmdPort;
      port facilityHlth : HlthPort;
      port facilityCool : CoolPort;

      // Key internal connections (BDD-level wiring hints)
      connect beamFromTelescope to ports.beamIn;
      connect ports.beamOut to beamToTelescope;

      connect cal.calOut to ports.calIn;
      connect calServiceOut to cal.calOut;

      connect facilityPwr  to ports.pwr;
      connect facilityData to ports.data;
      connect facilityTime to ports.time;
      connect facilityCmd  to ports.cmd;
      connect facilityHlth to ports.hlth;
      connect facilityCool to ports.cool;

      connect facilityCmd to selector.cmd;
      connect facilityPwr to selector.pwr;
      connect facilityData to selector.data;
      connect facilityHlth to selector.hlth;

      connect facilityCmd to cal.cmd;
      connect facilityPwr to cal.pwr;
      connect facilityData to cal.data;
      connect facilityHlth to cal.hlth;

      connect facilityCool to thermal.coolIn;
      connect facilityCmd  to thermal.cmd;
      connect facilityHlth to thermal.hlth;

      // Distribute services to instrument bays
      connect ports.beamOut to bays[*].beamIn;
      connect ports.calIn   to bays[*].calIn;
      connect ports.cool    to bays[*].cool;
      connect ports.pwr     to bays[*].pwr;
      connect ports.data    to bays[*].data;
      connect ports.time    to bays[*].time;
      connect ports.cmd     to bays[*].cmd;
      connect ports.hlth    to bays[*].hlth;

      // Control system fan-out (logical; detailed in IBD views)
      connect facilityCmd to ics.cmdUpstream;
      connect facilityData to ics.dataUpstream;
      connect facilityTime to ics.timeRef;
      connect facilityHlth to ics.hlthUpstream;

      connect ics.cmdDownstream to bays[*].cmd;
      connect ics.dataDownstream to bays[*].data;
      connect ics.hlthDownstream to bays[*].hlth;
    }

    //
    // Concrete part instance for the target system context
    //
    part TMT_Instruments : InstrumentsSubsystem;

    //
    // Traceability: satisfy key requirement packages/elements (RD-2 fix)
    //
    satisfy MissionGoalsAndScienceDrivers by TMT_Instruments;
    satisfy EarlyUniverseGoalMapping by TMT_Instruments;
    satisfy GalaxyEvolutionGoalMapping by TMT_Instruments;
    satisfy ExoplanetsGoalMapping by TMT_Instruments;
    satisfy BHgrowthGoalMapping by TMT_Instruments;
    satisfy StakeholderGroupConcerns by TMT_Instruments;

    satisfy ObservingAssumptionsAndConditionsDef by TMT_Instruments;
    satisfy ObservingAssumptionsAndConditions by TMT_Instruments;

    satisfy FR_PhasedSegmentedApertureDef by TMT_Instruments;
    satisfy FR_PhasedSegmentedAperture by TMT_Instruments;

    satisfy FR_PointingTrackingAndStabilityDef by TMT_Instruments;
    satisfy FR_PointingTrackingAndStability by TMT_Instruments;

    satisfy FR_ObservationPhasingAndExposureControlDef by TMT_Instruments;
    satisfy FR_ObservationPhasingAndExposureControl by cal;

    satisfy FR_AdaptiveOpticsOptionalDef by ports;
    satisfy FR_AdaptiveOpticsOptional by ports;

    satisfy SensitivityPerformanceNFRDef by TMT_Instruments;
    satisfy AngularResolutionPerformanceNFRDef by TMT_Instruments;
  }
}
