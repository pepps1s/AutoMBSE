package RD_IF_002_InstrumentPortInterfaceRequirements {

    import ScalarValues::*;
    import Quantities::*;
    
    requirement def InterfacePowerRequirement {
        doc /* The instrument port shall provide a stable power supply with a capacity of at least 5kW. */
        
        attribute powerCapacityRequired : Real;
        attribute powerCapacityActual : Real;

        require constraint { powerCapacityActual >= powerCapacityRequired }
    }

    requirement def InterfaceDataRequirement {
        doc /* The instrument port shall support data transmission with a minimum bandwidth of 10Gbps. */
        
        attribute dataBandwidthRequired : Real;
        attribute dataBandwidthActual : Real;

        require constraint { dataBandwidthActual >= dataBandwidthRequired }
    }

    requirement def InterfaceThermalRequirement {
        doc /* The instrument port shall manage thermal output with a cooling capacity of at least 500W. */
        
        attribute thermalCapacityRequired : Real;
        attribute thermalCapacityActual : Real;

        require constraint { thermalCapacityActual >= thermalCapacityRequired }
    }

    requirement def InterfaceControlRequirement {
        doc /* The instrument port shall support a control protocol with 99.9% uptime for reliability. */
        
        attribute controlProtocolReliabilityRequired : Real;
        attribute controlProtocolReliabilityActual : Real;

        require constraint { controlProtocolReliabilityActual >= controlProtocolReliabilityRequired }
    }

    requirement def ServiceEnableDisableRequirement {
        doc /* The instrument port shall allow for service enable and disable functionality with safety interlocks. */
        
        attribute serviceEnable : Boolean;
        attribute serviceDisable : Boolean;
        
        require constraint { serviceEnable == true or serviceDisable == true }
    }

    requirement def IsolationRequirement {
        doc /* The instrument port shall be isolated when disabled to prevent power leakage. */
        
        attribute isolationStatus : Boolean;
        
        require constraint { isolationStatus == true }
    }

    requirement def SafetyInterlockRequirement {
        doc /* The instrument port shall have a safety interlock to prevent operation under unsafe conditions. */
        
        attribute safetyInterlockStatus : Boolean;
        
        require constraint { safetyInterlockStatus == true }
    }

    requirement def InterfaceMechanicalRequirement {
        doc /* The instrument port shall support mechanical connections with a standard connector class. */
        
        attribute connectorClassRequired : String;
        attribute connectorClassActual : String;

        require constraint { connectorClassActual == connectorClassRequired }
    }

    requirement def ServiceCategoryRequirement {
        doc /* The instrument port shall be able to categorize services (power, data, thermal, control). */
        
        attribute serviceCategory : String;
        
        require constraint { serviceCategory in ["power", "data", "thermal", "control"] }
    }

    requirement def CapacityRequirement {
        doc /* The instrument port shall have a maximum capacity of 10kW for power, 10Gbps for data, and 500W for thermal. */
        
        attribute powerCapacityMax : Real;
        attribute dataBandwidthMax : Real;
        attribute thermalCapacityMax : Real;

        require constraint { powerCapacityMax == 10000 and dataBandwidthMax == 10000 and thermalCapacityMax == 500 }
    }

    requirement def ConnectorClassificationRequirement {
        doc /* The instrument port shall use a connector with an industry-standard classification for reliability. */
        
        attribute connectorClass : String;
        
        require constraint { connectorClass in ["Class A", "Class B", "Class C"] }
    }

    requirement def CoolantFlowRequirement {
        doc /* The instrument port shall require a coolant flow rate of at least 5 liters per minute. */
        
        attribute coolantFlowRequired : Real;
        attribute coolantFlowActual : Real;

        require constraint { coolantFlowActual >= coolantFlowRequired }
    }
    
}
