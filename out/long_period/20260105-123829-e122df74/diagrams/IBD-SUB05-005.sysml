package IBD_SUB05_AO_Internal_Loop_Closure {

    import ScalarValues::*;
    import Quantities::*;

    // Reuse ports/types defined in dependency views (BDD-SUB05-002, IBD-ROOT-013)
    import BDD_SUB05_AdaptiveOptics_Optional_Acquisition_LoopClosure_RTC::*;
    import IBD_ROOT_013::*;

    package IBD_SUB05_005 {

        // Local internal parts for this IBD (ports are reused from BDD-SUB05-002)
        part def WavefrontSensor {
            port wfsDataOut : WFSDataOutPort;
        }

        part def ReconstructorRTC {
            port wfsDataIn    : ~WFSDataOutPort;
            port dmCommandOut : DMCommandOutPort;
            port timeIn       : TimeSyncInPort;
        }

        part def DMAndTipTiltActuator {
            port dmCommandIn      : ~DMCommandOutPort;
            port tipTiltCommandIn : CommandInPort;
        }

        // IBD instance for: AO_Internal_Loop_Closure (type must match BDD exactly)
        part ibd_sub05_005 : AO_Internal_Loop_Closure {

            // Instantiate BDD-defined attributes in the IBD instance context
            attribute aoLoopRate_Hz  = 1000[Hz];
            attribute loopLatency_ms = 1.5[ms];

            // Parts (WFS acquisition, RTC/reconstructor, DM/tip-tilt actuator)
            part wfs        : WavefrontSensor;
            part rtc        : ReconstructorRTC;
            part dmTipTilt  : DMAndTipTiltActuator;

            // Time distribution dependency (from IBD-ROOT-013)
            part tsProc     : TimeSyncProcessor;

            // Connectors (required)
            wfs.wfsDataOut       -> rtc.wfsDataIn;        // WFS data out
            rtc.dmCommandOut     -> dmTipTilt.dmCommandIn; // DM command out
            tsProc.timeOut       -> rtc.timeIn;           // timeSync in
        }
    }
}
