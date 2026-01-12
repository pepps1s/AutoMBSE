package BDD_APS_WFS_Subsystem_Decomposition {

    import ScalarValues::*;
    import SI::*;
    
    % Block Definitions for APS/WFS Subsystem %
    
    abstract part def AlignmentPhasingWavefrontSensingSubsystem;
    abstract part def WFS_Sensors;
    abstract part def Metrology;
    abstract part def SegmentPhasingControlElements;
    abstract part def CalibrationUnits;
    
    % Part Definitions for APS/WFS Subsystem %
    
    part AlignmentPhasingWavefrontSensingSubsystem {
        part wfsSensors : WFS_Sensors;
        part metrologySystem : Metrology;
        part segmentPhasingControl : SegmentPhasingControlElements;
        part calibrationUnits : CalibrationUnits;
    }

    % Part Definitions for WFS Sensors %
    
    part def WFS_Sensors {
        attribute sensorType : String;
        attribute resolution : Real;
        attribute sensitivity : Real;
        attribute fieldOfView : AngleValue;
    }

    % Part Definitions for Metrology System %
    
    part def Metrology {
        attribute systemType : String;
        attribute accuracy : Real;
        attribute calibrationFrequency : Real;
    }
    
    % Part Definitions for Segment Phasing Control Elements %
    
    part def SegmentPhasingControlElements {
        attribute controlType : String;
        attribute actuationRange : LengthValue;
        attribute responseTime : TimeValue;
    }

    % Part Definitions for Calibration Units %
    
    part def CalibrationUnits {
        attribute calibrationType : String;
        attribute calibrationPeriod : TimeValue;
        attribute precision : Real;
    }

    % System Description for Alignment, Phasing, and Wavefront Sensing Subsystem %
    
    part APS_WFS_System : AlignmentPhasingWavefrontSensingSubsystem {
        part wfsSensors : WFS_Sensors {
            sensorType = "Shack-Hartmann";
            resolution = 0.1 [micron];
            sensitivity = 95 [percent];
            fieldOfView = 2 [degree];
        }
        part metrologySystem : Metrology {
            systemType = "Segmented Primary Mirror Metrology";
            accuracy = 0.5 [micron];
            calibrationFrequency = 30 [days];
        }
        part segmentPhasingControl : SegmentPhasingControlElements {
            controlType = "Piezoelectric";
            actuationRange = 5 [micron];
            responseTime = 10 [ms];
        }
        part calibrationUnits : CalibrationUnits {
            calibrationType = "Flat Field";
            calibrationPeriod = 7 [days];
            precision = 0.01 [micron];
        }
    }

    % Relationships & Dependencies %
    
    AlignmentPhasingWavefrontSensingSubsystem *- wfsSensors : "provides" -> WFS_Sensors;
    AlignmentPhasingWavefrontSensingSubsystem *- metrologySystem : "integrates" -> Metrology;
    AlignmentPhasingWavefrontSensingSubsystem *- segmentPhasingControl : "includes" -> SegmentPhasingControlElements;
    AlignmentPhasingWavefrontSensingSubsystem *- calibrationUnits : "utilizes" -> CalibrationUnits;
    
}
