package behavior {

  item def ExposurePlan;
  item def InstrumentConfig;
  
  action scienceExposureLoop {
    in item exposurePlan : ExposurePlan;
    in item instrumentConfig : InstrumentConfig;

    first start;
    
    then action startExposure {
      in item exposurePlan : ExposurePlan;
      in item instrumentConfig : InstrumentConfig;
      // Initialize and configure the exposure process
      first start;
      then action 'Configure instrument';
      then action 'Start exposure';
      then done;
    }
    
    then action readout {
      // Readout the data from the exposure
      in item exposurePlan : ExposurePlan;
      first start;
      then action 'Readout data';
      then done;
    }

    then action captureMetadata {
      // Capture metadata and provenance of the exposure
      in item exposurePlan : ExposurePlan;
      first start;
      then action 'Capture exposure metadata';
      then action 'Capture provenance';
      then done;
    }

    then action quickLookQA {
      // Perform quick-look QA on the data
      in item exposurePlan : ExposurePlan;
      first start;
      then action 'Perform quick-look QA';
      then done;
    }

    then action decisionQA {
      // Decision point: is the QA ok? Continue or adjust/abort the loop
      in item exposurePlan : ExposurePlan;
      first start;
      then action 'QA ok?';
      then decision {
        choice continueLoop : 'Continue to next exposure';
        choice abortLoop : 'Abort and adjust exposure parameters';
      }
    }

    then action dataHandoff {
      // Hand off data to the data system
      in item exposurePlan : ExposurePlan;
      first start;
      then action 'Handoff data to Data System';
      then done;
    }

    then done;
  }

}
