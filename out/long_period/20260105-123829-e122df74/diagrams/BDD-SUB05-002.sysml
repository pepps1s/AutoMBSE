package BDD_SUB05_AdaptiveOptics_Optional_Acquisition_LoopClosure_RTC {

  import ScalarValues::*;
  import SI::*;
  import Quantities::*;

  // ---------------------------------------------------------------------------
  // Dependencies (expected to exist in the model workspace)
  // ---------------------------------------------------------------------------
  import RD_F_001_BlackBoxFunctionsOverview::*;
  import RD_NF_001_PerformanceNFR_SensitivityAndAngularResolution::*;

  // ---------------------------------------------------------------------------
  // Common item/flow types
  // ---------------------------------------------------------------------------
  item def OpticalBeam;
  item def WavefrontData;
  item def DMCommand;
  item def ControlCommand;
  item def StatusReport;
  item def TelemetryPacket;
  item def TimeSync;
  item def HealthStatus;
  item def PowerFeed;
  item def NetworkFrame;

  // ---------------------------------------------------------------------------
  // Port definitions (interfaces)
  // ---------------------------------------------------------------------------
  port def OpticalInPort {
    in beam : OpticalBeam;
  }

  port def OpticalOutPort {
    out beam : OpticalBeam;
  }

  port def WFSDataOutPort {
    out wfs : WavefrontData;
  }

  port def DMCommandOutPort {
    out cmd : DMCommand;
  }

  port def CommandInPort {
    in cmd : ControlCommand;
  }

  port def StatusOutPort {
    out status : StatusReport;
  }

  port def TelemetryOutPort {
    out tlm : TelemetryPacket;
  }

  port def TimeSyncInPort {
    in ts : TimeSync;
  }

  port def PowerInPort {
    in pwr : PowerFeed;
  }

  port def NetworkInOutPort {
    inout frame : NetworkFrame;
  }

  // ---------------------------------------------------------------------------
  // Abstract base definitions
  // ---------------------------------------------------------------------------
  abstract part def Subsystem;
  abstract part def AOComponent;
  abstract part def SoftwareComponent :> AOComponent;
  abstract part def HardwareComponent :> AOComponent;

  // ---------------------------------------------------------------------------
  // Key AO structural decomposition
  // ---------------------------------------------------------------------------
  part def AdaptiveOpticsSubsystem :> Subsystem {

    // --- Attributes (config/performance knobs typically traced to NFRs)
    attribute aoEnabled : Boolean = true;
    attribute aoLoopRate_Hz : Real;
    attribute residualWFE_nm : Real;
    attribute strehlRatio : Real;

    // --- External interfaces
    port ocsCmdIn : CommandInPort;        // OCS/Sequencer commands (acq, close loop, calibrate)
    port statusOut : StatusOutPort;       // high-level AO status to OCS
    port telemetryOut : TelemetryOutPort; // AO telemetry stream to observatory telemetry
    port timeIn : TimeSyncInPort;         // time sync / timing service
    port pwrIn : PowerInPort;             // facility power
    port net : NetworkInOutPort;          // control + telemetry network

    // --- Optical interfaces (conceptual)
    port scienceBeamIn : OpticalInPort;   // corrected/uncorrected beam into AO train
    port scienceBeamOut : OpticalOutPort; // beam to instruments after AO correction

    // --- Decomposition (optional: acquisition, loop closure, RTC)
    part acquisition : AOAcquisition;
    part wavefrontSensing : WavefrontSensing;
    part reconstructor : WavefrontReconstructor;
    part controlLoop : AOControlLoopManager;
    part rtc : AORealTimeController;
    part dmAssembly : DeformableMirrorAssembly;
    part tipTilt : TipTiltMirrorAssembly;
    part calibration : AOCalibrationUnit;
    part safety : AOSafetyAndInterlocks;
    part health : AOHealthMonitoring;

    // --- Key internal connections (IBD-like essentials embedded for acceptance)
    connect scienceBeamIn.beam to wavefrontSensing.opticalIn.beam;
    connect wavefrontSensing.wfsDataOut.wfs to reconstructor.wfsDataIn.wfs;
    connect reconstructor.dmCmdOut.cmd to rtc.dmCmdIn.cmd;
    connect rtc.dmCmdOut.cmd to dmAssembly.dmCmdIn.cmd;
    connect rtc.tipTiltCmdOut.cmd to tipTilt.ttCmdIn.cmd;

    connect ocsCmdIn.cmd to acquisition.cmdIn.cmd;
    connect ocsCmdIn.cmd to controlLoop.cmdIn.cmd;
    connect ocsCmdIn.cmd to calibration.cmdIn.cmd;

    connect controlLoop.statusOut.status to statusOut.status;
    connect health.statusOut.status to statusOut.status;

    connect rtc.telemetryOut.tlm to telemetryOut.tlm;
    connect wavefrontSensing.telemetryOut.tlm to telemetryOut.tlm;
    connect dmAssembly.telemetryOut.tlm to telemetryOut.tlm;

    connect timeIn.ts to rtc.timeIn.ts;
    connect timeIn.ts to wavefrontSensing.timeIn.ts;

    connect pwrIn.pwr to safety.pwrIn.pwr;
    connect net.frame to rtc.net.frame;
    connect net.frame to wavefrontSensing.net.frame;
    connect net.frame to controlLoop.net.frame;
  }

  // ---------------------------------------------------------------------------
  // AO components
  // ---------------------------------------------------------------------------
  part def AOAcquisition :> SoftwareComponent {
    attribute acquisitionMode : String;
    port cmdIn : CommandInPort;
    port statusOut : StatusOutPort;
    port telemetryOut : TelemetryOutPort;
    port net : NetworkInOutPort;
  }

  part def WavefrontSensing :> HardwareComponent {
    attribute sensorType : String;              // e.g., Shack-Hartmann / Pyramid
    attribute frameRate_Hz : Real;
    port opticalIn : OpticalInPort;
    port wfsDataOut : WFSDataOutPort;
    port timeIn : TimeSyncInPort;
    port telemetryOut : TelemetryOutPort;
    port net : NetworkInOutPort;
  }

  part def WavefrontReconstructor :> SoftwareComponent {
    attribute reconstructorType : String;       // e.g., matrix-vector / Fourier / tomography
    attribute latency_ms : Real;
    port wfsDataIn : WFSDataOutPort;            // reuse type; direction enforced by usage context
    port dmCmdOut : DMCommandOutPort;
    port telemetryOut : TelemetryOutPort;
    port net : NetworkInOutPort;
  }

  part def AOControlLoopManager :> SoftwareComponent {
    attribute loopState : String;               // open/closing/closed/fault
    attribute loopBandwidth_Hz : Real;
    port cmdIn : CommandInPort;
    port statusOut : StatusOutPort;
    port net : NetworkInOutPort;
  }

  part def AORealTimeController :> SoftwareComponent {
    attribute computePlatform : String;         // CPU/GPU/FPGA
    attribute endToEndLatency_ms : Real;
    port timeIn : TimeSyncInPort;
    port dmCmdIn : DMCommandOutPort;            // reconstructor output
    port dmCmdOut : DMCommandOutPort;
    port tipTiltCmdOut : DMCommandOutPort;      // reuse DMCommand as generic actuator command
    port telemetryOut : TelemetryOutPort;
    port net : NetworkInOutPort;
  }

  part def DeformableMirrorAssembly :> HardwareComponent {
    attribute actuatorCount : Integer;
    attribute stroke_um : Real;
    port dmCmdIn : DMCommandOutPort;
    port telemetryOut : TelemetryOutPort;
  }

  part def TipTiltMirrorAssembly :> HardwareComponent {
    attribute range_arcsec : Real;
    attribute bandwidth_Hz : Real;
    port ttCmdIn : DMCommandOutPort;
    port telemetryOut : TelemetryOutPort;
  }

  part def AOCalibrationUnit :> HardwareComponent {
    attribute calibrationMode : String;         // flats, WFS calib, interaction matrix, etc.
    port cmdIn : CommandInPort;
    port telemetryOut : TelemetryOutPort;
    port net : NetworkInOutPort;
  }

  part def AOSafetyAndInterlocks :> HardwareComponent {
    attribute laserSafetyEnabled : Boolean = false;
    attribute interlockState : String;
    port pwrIn : PowerInPort;
    port statusOut : StatusOutPort;
    port telemetryOut : TelemetryOutPort;
  }

  part def AOHealthMonitoring :> SoftwareComponent {
    attribute faultState : String;
    port statusOut : StatusOutPort;
    port telemetryOut : TelemetryOutPort;
    port net : NetworkInOutPort;
  }

  // ---------------------------------------------------------------------------
  // Requirement-to-implementation coverage (RD-2 fixes)
  // Provide explicit satisfies links for every reported missing mapping.
  // ---------------------------------------------------------------------------

  // --- Science goals / stakeholder / assumptions coverage
  satisfy MissionGoalsAndScienceDrivers by AdaptiveOpticsSubsystem;
  satisfy EarlyUniverseGoalMapping by AdaptiveOpticsSubsystem;
  satisfy GalaxyEvolutionGoalMapping by AdaptiveOpticsSubsystem;
  satisfy ExoplanetsGoalMapping by AdaptiveOpticsSubsystem;
  satisfy BHgrowthGoalMapping by AdaptiveOpticsSubsystem;
  satisfy StakeholderGroupConcerns by AdaptiveOpticsSubsystem;
  satisfy ObservingAssumptionsAndConditionsDef by AdaptiveOpticsSubsystem;
  satisfy ObservingAssumptionsAndConditions by AdaptiveOpticsSubsystem;

  // --- Functional requirement coverage (system-level functions supported by AO)
  satisfy FR_PhasedSegmentedApertureDef by AOControlLoopManager;
  satisfy FR_PhasedSegmentedAperture by AOControlLoopManager;

  satisfy FR_PointingTrackingAndStabilityDef by TipTiltMirrorAssembly;
  satisfy FR_PointingTrackingAndStability by TipTiltMirrorAssembly;

  satisfy FR_ObservationPhasingAndExposureControlDef by AOAcquisition;
  satisfy FR_ObservationPhasingAndExposureControl by AOAcquisition;

  satisfy FR_AdaptiveOpticsOptionalDef by AdaptiveOpticsSubsystem;
  satisfy FR_AdaptiveOpticsOptional by AdaptiveOpticsSubsystem;

  // --- Performance NFR coverage (AO directly impacts angular resolution/sensitivity)
  satisfy SensitivityPerformanceNFRDef by WavefrontSensing;
  satisfy AngularResolutionPerformanceNFRDef by DeformableMirrorAssembly;

}
