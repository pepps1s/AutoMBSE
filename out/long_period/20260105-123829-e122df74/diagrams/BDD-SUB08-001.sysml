package BDD_SUB08_DataSystem {
    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Part Definitions %
    part def DataSystem {
        part acquisition : AcquisitionSubsystem;
        part metadata : MetadataSubsystem;
        part pipeline : PipelineSubsystem;
        part archive : ArchiveSubsystem;
    }

    part def AcquisitionSubsystem {
        part sensorArray : SensorArray;
        part dataCapture : DataCapture;
    }

    part def MetadataSubsystem {
        part metadataHandler : MetadataHandler;
    }

    part def PipelineSubsystem {
        part dataProcessing : DataProcessing;
        part qualityControl : QualityControl;
    }

    part def ArchiveSubsystem {
        part dataStorage : DataStorage;
        part dataRetrieval : DataRetrieval;
    }

    % Port Definitions %
    part acquisitionPort {
        port acquisitionIn : DataCapture;
        port acquisitionOut : DataStream;
    }

    part metadataPort {
        port metadataIn : DataStream;
        port metadataOut : MetadataStream;
    }

    part pipelinePort {
        port pipelineIn : MetadataStream;
        port pipelineOut : ProcessedDataStream;
    }

    part archivePort {
        port archiveIn : ProcessedDataStream;
        port archiveOut : ArchivedData;
    }

    % Interfaces and Connections %
    acquisitionPort.acquisitionOut -> metadataPort.metadataIn;
    metadataPort.metadataOut -> pipelinePort.pipelineIn;
    pipelinePort.pipelineOut -> archivePort.archiveIn;
}
