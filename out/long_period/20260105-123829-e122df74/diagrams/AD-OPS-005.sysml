package AD_OPS_CalibrationSequence {

    // Define inputs
    item def CalibrationType;
    item def InstrumentPort;

    // Define actions
    action CalibrationSequence {
        in item calibrationType : CalibrationType;
        in item instrumentPort : InstrumentPort;

        // Start of Calibration Sequence
        first start;
        
        // Step 1: Select Calibration Type and Instrument Port
        then action SelectCalibrationType {
            in item calibrationType : CalibrationType;
            first start;
            then action 'Choose Calibration Type' {
                in item calibrationType : CalibrationType;
            }
            then done;
        }
        
        then action SelectInstrumentPort {
            in item instrumentPort : InstrumentPort;
            first start;
            then action 'Select Instrument Port' {
                in item instrumentPort : InstrumentPort;
            }
            then done;
        }

        // Step 2: Perform Calibration (Flat/Dark)
        then action CalibrationExecution {
            in item calibrationType : CalibrationType;
            first start;
            then action 'Execute Calibration' {
                in item calibrationType : CalibrationType;
            }
            then done;
        }

        // Step 3: Optional WFS Calibration
        then action WFSCalibration {
            in item calibrationType : CalibrationType;
            first start;
            then action 'Perform WFS Calibration' {
                in item calibrationType : CalibrationType;
            }
            then done;
        }

        // Step 4: Validation of Calibration Quality and Completeness
        then action Validation {
            first start;
            then action 'Validate Calibration' {
                in item calibrationType : CalibrationType;
            }
            then action RetryOrFlag {
                first start;
                then action 'Retry Calibration';
                then action 'Flag Calibration Issue';
                then done;
            }
            then done;
        }

        // Step 5: Data/Metadata Tagging for Data System
        then action Tagging {
            first start;
            then action 'Tag Data/Metadata' {
                in item calibrationType : CalibrationType;
            }
            then done;
        }

        // End of Calibration Sequence
        then done;

        // Postcondition: Calibration Ready
        postcondition CalibrationReady;
    }
}
