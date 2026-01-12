package IBD_SUB08_007 {
    package DataStagingModel {
        // Part Definitions for Archive/Distribution Staging to External Interfaces (Internal Side v2)
        
        part def StagingBuffer {
            attribute storageCapacityTB: Integer;
            attribute latencySeconds: Integer;
            port inBufferPort: StagingPort;
            port outBufferPort: StagingPort;
        }

        part def ExternalEndpoint {
            port inEndpointPort: ExternalInterfacePort;
            port outEndpointPort: ExternalInterfacePort;
        }

        // Port Definitions for Staging Buffer
        port def StagingPort {
            out item dataFlow: DataProduct;
            out item metadataFlow: MetadataRecord;
            in item dataFlowIn: DataProduct;
            in item metadataFlowIn: MetadataRecord;
        }

        // Port Definitions for External Interfaces
        port def ExternalInterfacePort {
            in item dataFlow: DataProduct;
            in item metadataFlow: MetadataRecord;
            out item dataFlowOut: DataProduct;
            out item metadataFlowOut: MetadataRecord;
        }

        // Flow Definitions for Data Product and Metadata Flow
        part def DataProduct;
        part def MetadataRecord;

        // Model Instances
        part StagingBufferInstance: StagingBuffer {
            storageCapacityTB = 100; // Example of instantiating an attribute
            latencySeconds = 5;      // Example latency
        }

        part ExternalEndpointInstance: ExternalEndpoint;

        // Connections to represent data and metadata flow
        connector DataFlowConnection {
            from StagingBufferInstance.outBufferPort.dataFlow;
            to ExternalEndpointInstance.inEndpointPort.dataFlow;
        }

        connector MetadataFlowConnection {
            from StagingBufferInstance.outBufferPort.metadataFlow;
            to ExternalEndpointInstance.inEndpointPort.metadataFlow;
        }
    }
}
