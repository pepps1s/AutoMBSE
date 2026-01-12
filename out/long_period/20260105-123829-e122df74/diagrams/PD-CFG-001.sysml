package TMT_ConfigurationVariant {

    import TMT_Operations::*;
    import TMT_Requirements::*;
    
    package TMT_ConfigurationParameters {
        
        part AOEnabled :> TelescopeOpticalSystem {
            part redefines adaptiveOptics = AOEnabledFlag;
        }

        part AODisabled :> TelescopeOpticalSystem {
            part redefines adaptiveOptics = AODisabledFlag;
        }

        part InstrumentModeClass :> TelescopeOpticalSystem {
            part redefines instrumentType = selectedInstrument;
        }
        
        part InvalidConfiguration :> TelescopeOpticalSystem {
            part redefines adaptiveOptics = AOEnabledFlag;
            part redefines instrumentType = unsupportedInstrument;
        }
    }
    
    package TMT_Variants {

        part AOEnabledFlag : AdaptiveOptics {
            :>> isActive = true;
        }

        part AODisabledFlag : AdaptiveOptics {
            :>> isActive = false;
        }

        part selectedInstrument : Instrument {
            :>> type = 'Imaging';
        }

        part unsupportedInstrument : Instrument {
            :>> type = 'Unsupported';
        }

        abstract part def AdaptiveOptics;
        part def Instrument {
            attribute type;
        }
    }

    package TMT_VariationDefinitions {

        abstract part TelescopeOpticalSystem :> Telescope {
        
            variation part redefines adaptiveOptics : AdaptiveOptics {
                variant part AOEnabledFlag;
                variant part AODisabledFlag;
            }
            variation part redefines instrumentType : Instrument {
                variant part selectedInstrument;
                variant part unsupportedInstrument;
            }

            abstract constraint { 
                (adaptiveOptics == AOEnabledFlag & instrumentType == selectedInstrument) ^ 
                (adaptiveOptics == AODisabledFlag & instrumentType == unsupportedInstrument) 
            }
        }
    }
}
