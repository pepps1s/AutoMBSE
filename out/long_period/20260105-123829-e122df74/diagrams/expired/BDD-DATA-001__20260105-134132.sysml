package DataSystemDecomposition {
    
    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Define the parts for each subsystem %
    part def AcquisitionNode;
    part def MetadataService;
    part def PipelineCompute;
    part def ArchiveInterface;
    part def DistributionGateway;

    % Define the data system structure %
    part def DataSystem {
        part acquisition : AcquisitionNode;
        part metadataService : MetadataService;
        part pipeline : PipelineCompute;
        part archive : ArchiveInterface;
        part distribution : DistributionGateway;
    }

    % Define attributes for each part %
    attribute def AcquisitionAttributes {
        sensorType : String;
        dataRate : Real;
    }

    part AcquisitionNode {
        attribute sensorType : String = "Optical";
        attribute dataRate : Real = 10.0 [Gbps];
    }

    attribute def MetadataAttributes {
        metadataFormat : String;
        metadataVolume : Real;
    }

    part MetadataService {
        attribute metadataFormat : String = "FITS";
        attribute metadataVolume : Real = 100.0 [TB];
    }

    attribute def PipelineAttributes {
        computePower : Real;
        processingTime : Real;
    }

    part PipelineCompute {
        attribute computePower : Real = 200.0 [kW];
        attribute processingTime : Real = 1.5 [hrs];
    }

    attribute def ArchiveAttributes {
        storageCapacity : Real;
        redundancyLevel : Integer;
    }

    part ArchiveInterface {
        attribute storageCapacity : Real = 1000.0 [TB];
        attribute redundancyLevel : Integer = 2;
    }

    attribute def DistributionAttributes {
        bandwidth : Real;
        distributionLatency : Real;
    }

    part DistributionGateway {
        attribute bandwidth : Real = 100.0 [Gbps];
        attribute distributionLatency : Real = 0.1 [sec];
    }

    % Define the relationships and dependencies %
    DataSystem :> AcquisitionNode;
    DataSystem :> MetadataService;
    DataSystem :> PipelineCompute;
    DataSystem :> ArchiveInterface;
    DataSystem :> DistributionGateway;

    % Acceptance checks %
    check AcquisitionNode and MetadataService defined;
    check PipelineCompute and ArchiveInterface included;
}
