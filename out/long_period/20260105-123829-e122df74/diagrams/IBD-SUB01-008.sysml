package IBD_SUB01_008 {

    import ScalarValues::*;
    import Quantities::*;
    
    // Defining the parts for the Optical Beam Path Handover
    part def OpticalInputPort {
        port opticalInPort: OpticalPort;
    }

    part def BeamHandlingElement {
        port opticalInPort: OpticalPort;
        port opticalOutPort: OpticalPort;
        attribute beamPropagationStatus: BeamStatus;
    }

    part def OpticalOutputPort {
        port opticalOutPort: OpticalPort;
    }

    // Optical Port definition
    port def OpticalPort {
        in opticalBeam: OpticalBeam;
        out opticalBeam: OpticalBeam;
    }

    // Beam propagation status attribute
    attribute def BeamStatus {
        value: String;
        description: "Indicates the status of beam propagation";
    }

    // Optical Beam definition
    part def OpticalBeam {
        attribute beamIntensity: Real;
        attribute beamWavelength: Frequency;
    }

    // Internal routing connections (connectors)
    connector def BeamPropagation {
        part OpticalInputPort to BeamHandlingElement: opticalInPort;
        part BeamHandlingElement to OpticalOutputPort: opticalOutPort;
    }

    // Define attributes related to adaptive optics subsystem
    part def AdaptiveOpticsSubsystem {
        attribute aoEnabled: Boolean;
        attribute aoLoopRate_Hz: Real;
        attribute residualWFE_nm: Real;
        attribute strehlRatio: Real;
    }

    // Port connectors for adaptive optics
    connector def AOControl {
        part AdaptiveOpticsSubsystem to BeamHandlingElement: opticalInPort;
        part BeamHandlingElement to AdaptiveOpticsSubsystem: opticalOutPort;
    }
    
    // Instance of the optical system and configuration
    part OpticalSystem {
        part opticalInput: OpticalInputPort;
        part beamHandling: BeamHandlingElement;
        part opticalOutput: OpticalOutputPort;
        part adaptiveOptics: AdaptiveOpticsSubsystem;
        connector beamProp: BeamPropagation;
        connector aoControl: AOControl;
    }
}
