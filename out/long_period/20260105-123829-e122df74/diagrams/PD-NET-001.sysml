package PD_NET_001_NetworkBandwidthAvailabilityBudget {
    // PD NET 001: Network Bandwidth & Availability Budget (Onsite + External Links)

    // Define part types for network segments
    part SiteBackbone : NetworkSegment {
        attribute bandwidth : DataRate;
        attribute availability : Percentage;
        attribute packetLoss : Percentage;
    }
    
    part ControlNetwork : NetworkSegment {
        attribute bandwidth : DataRate;
        attribute availability : Percentage;
        attribute packetLoss : Percentage;
    }

    part TelemetryNetwork : NetworkSegment {
        attribute bandwidth : DataRate;
        attribute availability : Percentage;
        attribute packetLoss : Percentage;
    }

    part PartnerNetwork : NetworkSegment {
        attribute bandwidth : DataRate;
        attribute availability : Percentage;
        attribute packetLoss : Percentage;
    }

    part TimeSyncNetwork : NetworkSegment {
        attribute bandwidth : DataRate;
        attribute availability : Percentage;
        attribute packetLoss : Percentage;
    }

    // Define constraints for network segment performance
    constraint BandwidthConstraint {
        SiteBackbone.bandwidth >= 1000 Mbps;
        ControlNetwork.bandwidth >= 500 Mbps;
        TelemetryNetwork.bandwidth >= 200 Mbps;
        PartnerNetwork.bandwidth >= 300 Mbps;
        TimeSyncNetwork.bandwidth >= 50 Mbps;
    }

    constraint AvailabilityConstraint {
        SiteBackbone.availability >= 99.9%;
        ControlNetwork.availability >= 99.8%;
        TelemetryNetwork.availability >= 99.7%;
        PartnerNetwork.availability >= 99.9%;
        TimeSyncNetwork.availability >= 99.5%;
    }

    constraint PacketLossConstraint {
        SiteBackbone.packetLoss <= 0.1%;
        ControlNetwork.packetLoss <= 0.2%;
        TelemetryNetwork.packetLoss <= 0.3%;
        PartnerNetwork.packetLoss <= 0.1%;
        TimeSyncNetwork.packetLoss <= 0.5%;
    }

    // End-to-end data delivery path performance check
    constraint EndToEndAvailability {
        (SiteBackbone.availability * ControlNetwork.availability * TelemetryNetwork.availability * PartnerNetwork.availability * TimeSyncNetwork.availability) >= 99.5%;
    }

    // Redundancy and failover assumptions based on RD-IF-003
    part RedundancyConfiguration : NetworkRedundancy {
        attribute failoverTime : TimeDuration;
        attribute backupLinkAvailability : Percentage;
    }

    constraint RedundancyAssumption {
        RedundancyConfiguration.failoverTime <= 10s;
        RedundancyConfiguration.backupLinkAvailability >= 99.9%;
    }

    // Define the dependencies between different elements
    dependency PD-DATA-001 {
        dependencyType = "data input";
    }

    dependency RD-IF-003 {
        dependencyType = "interface requirements";
    }

    dependency RD-NF-005 {
        dependencyType = "non-functional requirements";
    }

    dependency RD-IF-001 {
        dependencyType = "external interfaces";
    }

    dependency BDD-SUB03-001 {
        dependencyType = "subsystem design";
    }

    dependency BDD-SUB07-001 {
        dependencyType = "subsystem design";
    }

    dependency BDD-SUB08-001 {
        dependencyType = "subsystem design";
    }

    // Acceptance checks
    acceptanceCheck DefinesBandwidthAndAvailabilityTypes {
        check = "Defines bandwidth, packet loss, and uptime/availability value types";
    }

    acceptanceCheck IncludesNetworkSegments {
        check = "Includes at least 8 network segments/links";
    }

    acceptanceCheck ProvidesConstraintsFromPDData {
        check = "Provides constraints that compare required vs provided capacities derived from PD-DATA-001";
    }

    acceptanceCheck DefinesAvailabilityAllocations {
        check = "Defines availability allocations per segment and a roll-up check for end-to-end data delivery path";
    }

    acceptanceCheck AssumptionsForRedundancy {
        check = "Includes explicit assumptions for redundancy/failover aligned with RD-IF-003";
    }
}
