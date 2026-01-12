package TMT_BDD_AO_001 {
  import ScalarValues::*;
  import SI::*;
  import Quantities::*;

  // Diagram: BDD-AO-001
  // Title : Adaptive Optics (AO) Subsystem Decomposition (Optional Scope)
  // Goal  : Define AO structure including wavefront sensors, deformable mirrors,
  //         real-time controller, guide star acquisition placeholders.
  // Depends on: BDD-SYS-003

  // --- Common abstractions (lightweight) ---
  abstract part def TMTComponent;
  abstract part def AOComponent :> TMTComponent;

  // --- AO elements required by acceptance checks ---
  part def WavefrontSensor :> AOComponent;
  part def DeformableMirror :> AOComponent;
  part def AORealTimeController :> AOComponent;

  // Placeholders requested (explicitly modeled but not decomposed here)
  part def GuideStarAcquisition :> AOComponent;
  part def NaturalGuideStarAcquisition :> GuideStarAcquisition;
  part def LaserGuideStarAcquisition :> GuideStarAcquisition;

  // Optional additional (kept minimal)
  part def AOCorrectionOptics :> AOComponent;
  part def AOCalibrationUnit :> AOComponent;

  // --- AO subsystem decomposition ---
  part def AdaptiveOpticsSubsystem :> AOComponent {
    // Wavefront sensors
    part wfs : WavefrontSensor;
    part wfsArray[0..*] : WavefrontSensor;

    // Deformable mirrors
    part dm : DeformableMirror;
    part dmArray[0..*] : DeformableMirror;

    // AO Real-Time Controller (RTC)
    part rtc : AORealTimeController;

    // Guide star acquisition placeholders (NGS/LGS)
    part gsa : GuideStarAcquisition;
    part ngsAcq : NaturalGuideStarAcquisition;
    part lgsAcq : LaserGuideStarAcquisition;

    // Optional supporting components
    part correctionOptics : AOCorrectionOptics;
    part calUnit : AOCalibrationUnit;
  }

  // --- View anchor for this diagram ---
  part ao : AdaptiveOpticsSubsystem;
}
