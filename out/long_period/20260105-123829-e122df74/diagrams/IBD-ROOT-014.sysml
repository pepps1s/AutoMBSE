package IBD_ROOT_014_PowerDistributionInternalLinks {
    
    import ScalarValues::*;
    import Quantities::*;
    
    % Power Distribution Model for TMT %
    package PowerDistributionSystem {

        % Part Definitions %
        part def FacilityPowerSystem {
            port powerFeedPort: PowerFeedPort;
            port backupPowerPort: BackupPowerPort;
            attribute voltage: Voltage;
            attribute frequency: Frequency;
        }

        part def SubsystemPower {
            port powerInputPort: PowerInputPort;
            attribute powerDemand: PowerDemand;
        }

        % Power Port Definitions %
        port def PowerFeedPort {
            out power: PowerLevel;
            in safetySignal: SafetySignal;
        }

        port def BackupPowerPort {
            out backupPower: PowerLevel;
            in shutdownSignal: ShutdownSignal;
        }

        port def PowerInputPort {
            in power: PowerLevel;
            out controlSignal: ControlSignal;
        }

        % Attributes Definitions %
        attribute def PowerLevel;
        attribute def Voltage;
        attribute def Frequency;
        attribute def PowerDemand;
        
        % Control & Shutdown Signals %
        part def SafetySignal;
        part def ShutdownSignal;
        part def ControlSignal;

        % Subsystem Part Definitions %
        part def PowerSubsystem {
            part powerDistribution: FacilityPowerSystem;
            part powerConsumer: SubsystemPower;
            port powerInput: PowerInputPort;
        }

        % Internal Wiring between Power Sources and Subsystems %
        part def PowerDistributionSystem {
            part facility: FacilityPowerSystem;
            part subsystem: PowerSubsystem;
            port powerFeed: PowerFeedPort;
            port backupFeed: BackupPowerPort;
        }

        % Internal connections %
        connect PowerDistributionSystem.facility.powerFeedPort to PowerSubsystem.powerInputPort;
        connect PowerSubsystem.powerInputPort to PowerFeedPort;

        % Constraints to manage backup and controlled shutdown behavior %
        constraint def BackupPowerBehavior {
            when PowerSubsystem.powerDemand > FacilityPowerSystem.powerFeedPort.power then
                FacilityPowerSystem.backupPowerPort.backupPower := PowerLevel.high;
        }

        constraint def ShutdownTrigger {
            when SafetySignal.estopTriggered then
                PowerSubsystem.powerInputPort.controlSignal := ControlSignal.off;
                PowerSubsystem.powerInputPort.power := PowerLevel.zero;
        }
    }

}
