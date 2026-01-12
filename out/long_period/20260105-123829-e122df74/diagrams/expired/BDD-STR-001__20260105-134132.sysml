package TelescopeStructureAndMountDecomposition {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    abstract part def AltAzMount;
    abstract part def Bearing;
    abstract part def Drive;
    abstract part def Encoder;
    abstract part def VibrationElement;
    abstract part def ThermalElement;
    
    part def TelescopeStructureMount {
        part altAzMount : AltAzMount {
            description = "Alt-Az Mount for telescope structure."
        }
        part bearing : Bearing {
            description = "Bearings for rotating the telescope structure."
        }
        part drive : Drive {
            description = "Drives for moving the telescope along the alt-az axes."
        }
        part encoder : Encoder {
            description = "Encoders for measuring the position of the mount."
        }
        part vibrationControl : VibrationElement {
            description = "Vibration control elements to minimize vibrations affecting observations."
        }
        part thermalControl : ThermalElement {
            description = "Thermal control elements to stabilize temperature variations in the structure."
        }
    }

    attribute def AltAzMountCapacity {
        maxLoad : ForceValue;
        deviation : LengthValue;
    }

    part def AltAzMountPart :> TelescopeStructureAndMountDecomposition::AltAzMount {
        attribute loadCapacity : AltAzMountCapacity {
            maxLoad = 5000 [N];
            deviation = 2 [mm];
        }
    }
    
    part def BearingPart :> TelescopeStructureAndMountDecomposition::Bearing {
        attribute bearingType : String {
            bearingType = "Ball Bearing";
        }
    }

    part def DrivePart :> TelescopeStructureAndMountDecomposition::Drive {
        attribute motorType : String {
            motorType = "Stepper Motor";
        }
    }

    part def EncoderPart :> TelescopeStructureAndMountDecomposition::Encoder {
        attribute encoderPrecision : Integer {
            encoderPrecision = 1024;
        }
    }

    part def VibrationElementPart :> TelescopeStructureAndMountDecomposition::VibrationElement {
        attribute vibrationDamping : Real {
            vibrationDamping = 0.95;
        }
    }

    part def ThermalElementPart :> TelescopeStructureAndMountDecomposition::ThermalElement {
        attribute thermalCapacity : Real {
            thermalCapacity = 1000 [J/K];
        }
    }
}
