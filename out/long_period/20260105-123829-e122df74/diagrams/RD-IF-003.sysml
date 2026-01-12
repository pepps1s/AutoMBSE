package RD_IF_003_SiteInfrastructureInterfaceRequirements {
    
    import ScalarValues::*;
    import Quantities::*;
    
    requirement def PowerQualityReqDef {
        doc /* The site infrastructure shall provide power with stable voltage 
              and frequency to ensure continuous operation of the TMT system. 
              Power quality shall meet the specified limits for all operational conditions. */
        
        attribute voltage : Real;
        attribute frequency : Real;
        
        require constraint{ voltage >= 210.0 && voltage <= 250.0 }
        require constraint{ frequency >= 59.5 && frequency <= 60.5 }
    }
    
    requirement def BackupBehaviorReqDef {
        doc /* In the event of power failure, backup systems (UPS, Generator) shall 
              take over the power supply within 2 seconds, ensuring continuous operations. */
        
        attribute responseTime : Real;
        attribute backupPowerCapacity : Real;
        
        require constraint{ responseTime <= 2.0 }
        require constraint{ backupPowerCapacity >= 100.0 }
    }
    
    requirement def NetworkConnectivityReqDef {
        doc /* The site network shall provide reliable bidirectional connectivity with the TMT for real-time data exchange. 
              Connectivity should ensure data rates of at least 1 Gbps under all conditions. */
        
        attribute dataRate : Real;
        attribute uptimePercentage : Real;
        
        require constraint{ dataRate >= 1000.0 }
        require constraint{ uptimePercentage >= 99.9 }
    }
    
    requirement def HVACExchangesReqDef {
        doc /* The HVAC system shall ensure proper ventilation and cooling for the TMT system. 
              This includes maintaining a temperature range of 18°C to 22°C and relative humidity between 40% and 60%. */
        
        attribute temperature : Real;
        attribute humidity : Real;
        
        require constraint{ temperature >= 18.0 && temperature <= 22.0 }
        require constraint{ humidity >= 40.0 && humidity <= 60.0 }
    }
    
    requirement def MonitoringTelemetryReqDef {
        doc /* The infrastructure systems (Power, HVAC, Network) shall provide continuous monitoring data to the TMT for health checks and predictive maintenance. 
              Monitoring data shall be transmitted to the observatory control system in real-time. */
        
        attribute telemetryData : Boolean;
        attribute transmissionFrequency : Real;
        
        require constraint{ telemetryData == true }
        require constraint{ transmissionFrequency >= 1.0 }
    }
    
    requirement def FailoverBehaviorReqDef {
        doc /* In the event of failure of a primary infrastructure system (Power, Network, HVAC), 
              failover to a secondary system shall occur automatically without operational interruption. */
        
        attribute failoverTime : Real;
        
        require constraint{ failoverTime <= 5.0 }
    }
    
    requirement def ControlledShutdownReqDef {
        doc /* In case of critical failure, the system shall perform a controlled shutdown sequence 
              to preserve equipment integrity and data. This shall be completed within 15 minutes of failure detection. */
        
        attribute shutdownTime : Real;
        
        require constraint{ shutdownTime <= 15.0 }
    }
    
    requirement def InterfaceDocumentationReqDef {
        doc /* All interface requirements between TMT and site infrastructure shall be documented and tracked, ensuring traceability and consistency across all systems. */
        
        attribute documentationStatus : Boolean;
        
        require constraint{ documentationStatus == true }
    }

    requirement def InterfaceConsistencyReqDef {
        doc /* The interface specifications between TMT and site infrastructure shall be consistent with previously defined dependencies. */
        
        attribute dependenciesConsistent : Boolean;
        
        require constraint{ dependenciesConsistent == true }
    }

}
