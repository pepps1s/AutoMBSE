package PD_CAL_001 {

    import PD-AVL-001::*;
    import RD-F-002::*;
    import RD-F-007::*;
    import PD-DATA-001::*;
    import BDD-SUB06-001::*;
    import BDD-SUB07-001::*;
    import BDD-SUB08-001::*;
    
    part CalibrationOverheadBudget {
        description "Defines the calibration overhead budget, cadences, and metadata impacts."
        goal "Budget calibration time overheads, cadences, and resulting data products/metadata impacts to ensure observing efficiency and data integrity targets."
        dependency {
            RD-F-002;
            RD-F-007;
            PD-DATA-001;
            PD-AVL-001;
            BDD-SUB06-001;
            BDD-SUB07-001;
            BDD-SUB08-001;
        }
    }

    part CalibrationActivities {
        part flats : CalibrationActivity {
            description "Flat field calibration."
            duration 15; 
            overheadFraction 0.1;
            metadataFields ["flatFieldData", "flatFieldTimestamp"];
        }
        part darks : CalibrationActivity {
            description "Dark field calibration."
            duration 30;
            overheadFraction 0.12;
            metadataFields ["darkFieldData", "darkFieldTimestamp"];
        }
        part arcs : CalibrationActivity {
            description "Arc calibration for spectral lines."
            duration 20;
            overheadFraction 0.15;
            metadataFields ["arcData", "arcTimestamp"];
        }
        part wfsCalib : CalibrationActivity {
            description "Wavefront sensing calibration."
            duration 40;
            overheadFraction 0.18;
            metadataFields ["wfsData", "wfsTimestamp"];
        }
        part pointingModel : CalibrationActivity {
            description "Pointing model update."
            duration 25;
            overheadFraction 0.2;
            metadataFields ["pointingModelData", "pointingModelTimestamp"];
        }
        part pointingCorrection : CalibrationActivity {
            description "Pointing correction using updated model."
            duration 30;
            overheadFraction 0.18;
            metadataFields ["pointingCorrectionData", "pointingCorrectionTimestamp"];
        }
        part alignmentCalibration : CalibrationActivity {
            description "Segment alignment calibration."
            duration 35;
            overheadFraction 0.16;
            metadataFields ["alignmentData", "alignmentTimestamp"];
        }
        part systemCalibration : CalibrationActivity {
            description "Overall system calibration for all subsystems."
            duration 50;
            overheadFraction 0.2;
            metadataFields ["systemCalibData", "systemCalibTimestamp"];
        }
    }

    part CalibrationConstraints {
        constraint { 
            sum(overheadFraction) == 1.0; 
            "Total calibration overhead fraction must be 1.0."
        }
        constraint { 
            quickLookQAAvailableBeforeExposureDecision == true; 
            "Quick-look QA must be available before the next exposure decision, in line with RD-F-007 requirements."
        }
        constraint { 
            calibrationActivityMetadataCompleteness == true; 
            "Each calibration activity must produce the necessary metadata or reference a complete metadata set."
        }
    }

    part CalibrationOverheadRollup {
        description "Roll-up constraint computes the overall calibration overhead fraction and feeds into PD-AVL-001 planned loss category."
        dependency PD-AVL-001;
        constraint { 
            calibrationOverheadFraction == sum(CalibrationActivities.overheadFraction); 
            "Calibration overhead fraction is rolled up from individual activities."
        }
    }

    acceptance_check {
        check "Defines calibration types, cadence, per-activity duration, and overhead fraction value types.";
        check "Includes at least 8 calibration activities (flats/darks/arcs/WFS calib/pointing model updates, etc.).";
        check "Roll-up constraint computes calibration overhead fraction and feeds PD-AVL-001 planned loss category.";
        check "Includes constraints for quick-look QA availability timing tied to RD-F-007 requirements.";
        check "Ensures each calibration activity produces defined metadata fields or references a metadata completeness parameter set.";
    }
}
