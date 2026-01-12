package IBD_SUB06_CalibrationUnitWiring {

    package CalibrationSystem {

        part def CalibrationSource {
            port out CalLightPort: CalLightPortType;
            port in MetadataHook: MetadataPortType;
            attribute calibrationType: CalibrationType;
            attribute calibrationCadence: CadenceType;
        }

        part def SelectorMechanism {
            port out BeamPort: BeamPortType;
            port in MetadataHook: MetadataPortType;
            attribute selectorType: SelectorType;
        }

        part def CalibrationUnit {
            part CalibrationSource: CalibrationSource;
            part SelectorMechanism: SelectorMechanism;
            port out CalLightPort: CalLightPortType;
            port in BeamPort: BeamPortType;
            port in MetadataHook: MetadataPortType;
        }

        port def CalLightPortType {
            out lightSignal: LightSignal;
        }

        port def BeamPortType {
            in beamSignal: BeamSignal;
        }

        port def MetadataPortType {
            in calibrationMetadata: MetadataType;
        }

        // Dependencies
        dependency IBD_SysML_Connection {
            connects CalibrationSource.CalLightPort to SelectorMechanism.BeamPort;
            connects CalibrationSource.MetadataHook to MetadataPort;
            connects SelectorMechanism.MetadataHook to MetadataPort;
            connects CalibrationUnit.CalLightPort to CalibrationSource.CalLightPort;
            connects CalibrationUnit.BeamPort to SelectorMechanism.BeamPort;
            connects CalibrationUnit.MetadataHook to MetadataPort;
        }

        // Attributes and Types
        attribute def CalibrationType {
            value: String;
        }

        attribute def CadenceType {
            value: String;
        }

        attribute def LightSignal {
            value: String;
        }

        attribute def BeamSignal {
            value: String;
        }

        attribute def MetadataType {
            value: String;
        }

        attribute def SelectorType {
            value: String;
        }

    }
}
