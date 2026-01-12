package IBD_SUB06_005_InstrumentPortServicesInternalWiring {

  // Dependencies (referenced by name; definitions assumed in the dependent packages):
  // - BDD-SUB06-002 provides: InstrumentPortAssembly, Instrument, and service port defs PwrPort/DataPort/TimePort/CoolPort/BeamPort
  // - RD-IF-002 provides: capacity attributes such as powerCapacityMax, dataBandwidthMax, thermalCapacityMax
  // - IBD-ROOT-013, IBD-ROOT-014 provide upstream backbone contexts (not expanded here)

  import ScalarValues::*;
  import Quantities::*;

  // View/Diagram metadata
  metadata diagramId = "IBD-SUB06-005";
  metadata diagramType = "IBD";
  metadata title = "IBD SUB06: Instrument Port Services Internal Wiring (Power/Data/Time/Cool/Beam)";
  metadata goal = "Detail internal connectivity within instrument hosting for service distribution to an instrument port assembly and attached instruments.";

  // Internal wiring context for SUB06 instrument hosting
  part def SUB06_InstrumentPortServicesInternalWiring {

    // Parts (types defined in BDD-SUB06-002)
    part ipa : InstrumentPortAssembly;
    part inst1 : Instrument;

    // Service ports on the InstrumentPortAssembly side (port defs from BDD-SUB06-002)
    port ipaPwr  : PwrPort;
    port ipaData : DataPort;
    port ipaTime : TimePort;
    port ipaCool : CoolPort;
    port ipaBeam : BeamPort;

    // Service ports on the Instrument side (port defs from BDD-SUB06-002)
    port instPwr  : PwrPort;
    port instData : DataPort;
    port instTime : TimePort;
    port instCool : CoolPort;
    port instBeam : BeamPort;

    // Capacity attributes (names aligned to RD-IF-002 CapacityRequirement vocabulary)
    // Instantiated here to capture interface budget limits for this hosting wiring context.
    attribute powerCapacityMax   : Real = 8.0;     // (e.g., kW)
    attribute dataBandwidthMax   : Real = 100.0;   // (e.g., Gbps)
    attribute thermalCapacityMax : Real = 5.0;     // (e.g., kW heat lift)

    // Separate connectors for each service
    connect pwrLink  : ipaPwr  to instPwr;
    connect dataLink : ipaData to instData;
    connect timeLink : ipaTime to instTime;
    connect coolLink : ipaCool to instCool;
    connect beamLink : ipaBeam to instBeam;

    // Bind exposed ports to the typed parts (so the IBD ports represent the parts' service interfaces)
    // (These bindings keep the view explicit without assuming the internal port names of ipa/inst1.)
    connect ipaPwrBind  : ipaPwr  to ipa;
    connect ipaDataBind : ipaData to ipa;
    connect ipaTimeBind : ipaTime to ipa;
    connect ipaCoolBind : ipaCool to ipa;
    connect ipaBeamBind : ipaBeam to ipa;

    connect instPwrBind  : instPwr  to inst1;
    connect instDataBind : instData to inst1;
    connect instTimeBind : instTime to inst1;
    connect instCoolBind : instCool to inst1;
    connect instBeamBind : instBeam to inst1;
  }

  // IBD root instance (what the view would typically show)
  part ibd_sub06_005 : SUB06_InstrumentPortServicesInternalWiring;

}
