package AO_RTC_Structural_Breakdown {

    import ScalarValues::*;
    import SI::*;
    
    part def ComputeNode;
    part def TimingSyncDistribution;
    part def Telemetry;
    part def ControlIO;

    part def AO_RealTime_Controller {
        part computeNode : ComputeNode;
        part timingSync : TimingSyncDistribution;
        part telemetry : Telemetry;
        part controlIO : ControlIO;
    }

    part def ComputeNode {
        attribute processorType : String;
        attribute processorPower : PowerValue;
        attribute coreCount : Integer;
    }

    part def TimingSyncDistribution {
        attribute syncMethod : String;
        attribute distributionRate : FrequencyValue;
    }

    part def Telemetry {
        attribute telemetryType : String;
        attribute telemetryFrequency : FrequencyValue;
        attribute dataRate : DataRateValue;
    }

    part def ControlIO {
        attribute controlType : String;
        attribute ioCount : Integer;
        attribute interfaceProtocol : String;
    }

    // Define AO Real-Time Controller System Breakdown
    part AO_RTC_Structure {
        part rtcController : AO_RealTime_Controller {
            part computeNode[4]; // Define four compute nodes
            part timingSync[2];  // Define two timing/sync modules
            part telemetry[2];   // Define two telemetry modules
            part controlIO[3];   // Define three control I/O modules
        }
    }
}
