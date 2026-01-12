package InstrumentHostingAndServicesRequirements {

    import ScalarValues::*;
    import Quantities::*;
    
    requirement def FunctionalRequirements {
        doc /* The functional requirements for instrument ports, selection, service provisioning (power/data/thermal/control),
              and calibration unit interaction in the TMT system are captured in this set. */

        attribute portSelection : Boolean;
        attribute serviceEnableDisable : Boolean;
        attribute healthCheck : Boolean;
        attribute safeIsolation : Boolean;
        attribute metadataIntegration : Boolean;
        attribute configurationManagement : Boolean;
        attribute serviceCategories : String;

        require constraint{ 
            portSelection == true or portSelection == false;
            serviceEnableDisable == true or serviceEnableDisable == false;
            healthCheck == true or healthCheck == false;
            safeIsolation == true or safeIsolation == false;
            metadataIntegration == true or metadataIntegration == false;
            configurationManagement == true or configurationManagement == false;
        }

        doc /* The system shall include at least 10 functional requirements covering port selection, service enable/disable, 
              health checks, safe isolation, and the integration of metadata and configuration management. */

        require check {
            count(FunctionalRequirements) >= 10;
        }

        doc /* The system must define the required service categories without prescribing the exact protocols. */

        require constraint {
            serviceCategories != "";
        }
    }

    requirement def InstrumentMetadataIntegration {
        doc /* The system shall ensure hooks for instrument metadata and configuration management are included in the functional requirements. */

        attribute metadataHook : Boolean;
        attribute configManagementHook : Boolean;

        require constraint{
            metadataHook == true or metadataHook == false;
            configManagementHook == true or configManagementHook == false;
        }
    }
}
