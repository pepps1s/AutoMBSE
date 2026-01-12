package BDD-SITE-001 {
    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Define the abstract parts for the subsystems in the site infrastructure %
    abstract part def PowerSubsystem;
    abstract part def HVACSubsystem;
    abstract part def NetworkSubsystem;
    abstract part def FacilityMonitoringSubsystem;
    
    % Define the specific parts for the site infrastructure components %
    part def UPSDistribution : PowerSubsystem {
        part batteryBank : PowerSubsystem;
        part powerControlUnit : PowerSubsystem;
    }

    part def InternalNetwork : NetworkSubsystem {
        part dataSwitch : NetworkSubsystem;
        part router : NetworkSubsystem;
        part firewall : NetworkSubsystem;
    }

    part def FacilityMonitoring : FacilityMonitoringSubsystem {
        part temperatureSensor : FacilityMonitoringSubsystem;
        part humiditySensor : FacilityMonitoringSubsystem;
        part alarmSystem : FacilityMonitoringSubsystem;
    }

    part SiteInfrastructure {
        part upsDistribution : UPSDistribution;
        part internalNetwork : InternalNetwork;
        part facilityMonitoring : FacilityMonitoring;
    }

    % Define the dependencies of this subsystem model to another BDD %
    dependency BDD-SYS-003 {
        part TMT_BDD_SYS_003 :> PowerSubsystem;
        part TMT_BDD_SYS_003 :> NetworkSubsystem;
        part TMT_BDD_SYS_003 :> FacilityMonitoringSubsystem;
    }

    % Define the acceptance checks %
    acceptance_checks {
        verify UPSDistributionModel : UPSDistribution;
        verify InternalNetworkModel : InternalNetwork;
        verify FacilityMonitoringPlaceholders : FacilityMonitoring;
    }
}
