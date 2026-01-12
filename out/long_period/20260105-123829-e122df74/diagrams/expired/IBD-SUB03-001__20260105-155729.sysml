package IBD_SUB03_001_EnclosureDomeAndSiteInfrastructure_InternalConnectivity {

  // Dependency: BDD-SUB03-001
  import ScalarValues::*;
  import BDD_SUB03_EnclosureDomeAndSiteInfrastructure::*;
  
  // IBD-SUB03-001
  part IBD_SUB03_EnclosureDomeAndSiteInfrastructure {

    // --- Instantiate key parts from BDD (consistent part names) ---
    part facilitySubsystem : FacilitySubsystem;
    part enclosureDomeAndSiteInfrastructure : BDD_SUB03_EnclosureDomeAndSiteInfrastructure;
    part controlUnit : ControlUnit;

    // --- Instantiate BDD-defined properties (to satisfy BBD-1 validation) ---
    // (Typed with ScalarValues where exact domain types are not required for connectivity modeling)
    attribute Availability : Boolean;

    attribute earlyUniverse : Boolean;
    attribute galaxyEvolution : Boolean;
    attribute exoplanets : Boolean;
    attribute BHgrowth : Boolean;

    attribute highAngularResolution : Boolean;
    attribute longExposures : Boolean;

    attribute highAngularResolutionCapability : Boolean;
    attribute longExposureCapability : Boolean;

    attribute mappedToCapability : String;

    attribute scienceUsersConcerns : String;
    attribute operatorsConcerns : String;
    attribute maintainersConcerns : String;
    attribute safetyConcerns : String;
    attribute dataUsersConcerns : String;
    attribute siteConcerns : String;
    attribute partnersConcerns : String;

    attribute title : String;
    attribute theme : String;
    attribute traceableToRequirement : String;

    attribute observingBand : String;
    attribute wavelengthEffective : Real;
    attribute zenithAngleMax : Real;
    attribute seeingFWHM : Real;
    attribute skyBackground : Real;
    attribute airmassMax : Real;
    attribute pwv : Real;
    attribute moonIlluminationMax : Real;
    attribute windSpeedMax : Real;
    attribute temperatureRange : String;

    attribute phasingResidualWFE : Real;
    attribute phasingUpdateCadence : Real;

    attribute trackingJitterRMS : Real;
    attribute guidingErrorRMS : Real;
    attribute vibrationLineOfSightRMS : Real;

    attribute maxAllowableImageSmear : Real;
    attribute maxExposureTime : Real;
    attribute dutyCycleMin : Real;
    attribute calibrationCadence : Real;

    attribute aoEnabled : Boolean;
    attribute residualWFE_AO : Real;
    attribute strehlRatio : Real;
    attribute aoLoopRate : Real;

    attribute limitingMagnitudePointSource : Real;
    attribute snrTarget : Real;
    attribute exposureTimeForSNR : Real;

    attribute spectralResolution : Real;
    attribute throughputEndToEnd : Real;
    attribute photometricStability : Real;

    attribute encircledEnergyRadius : Real;
    attribute diffractionLimited : Boolean;
    attribute wavelengthForResolution : Real;

    attribute angularResolutionFWHM : Real;
    attribute psfCoreFWHM : Real;

    attribute totalWFERMS : Real;
    attribute imageMotionRMS : Real;
    attribute strehlRatioAchieved : Real;

    // --- Internal connectivity (no new connector/port defs; avoid IBD-1 / IBD-3 violations) ---
    // Power distribution/support path
    connect facilitySubsystem to enclosureDomeAndSiteInfrastructure;
    connect facilitySubsystem to controlUnit;

    // HVAC/thermal management support path
    connect enclosureDomeAndSiteInfrastructure to facilitySubsystem;

    // Networking/telemetry/control path
    connect controlUnit to enclosureDomeAndSiteInfrastructure;
    connect controlUnit to facilitySubsystem;
  }

}
