package IBD_QA_001_IBD_QA_CrossView_Consistency_Gate_SUB03 {

  // Diagram metadata
  metadata diagram_id = "IBD-QA-001";
  metadata diagram_type = "IBD";
  metadata title = "IBD QA: Cross-View Consistency Gate (Ports & Attributes) — SUB03 Focus";
  metadata goal =
    "Explicit QA IBD checklist to validate SUB03 BDD↔IBD port+attribute consistency and prevent regression.";
  metadata dependencies = {"BDD-SUB03-003","IBD-SUB03-011","IBD-SUB03-012","IBD-SUB03-013"};
  metadata target_system = "Thirty Meter Telescope (TMT, 三十米望远镜)";

  // Shallow imports (expected to exist per dependencies)
  // NOTE: Names follow dependency sample package names; adjust to your repo naming conventions if needed.
  import BDD_SUB03_003_EnclosureDome_SiteInfrastructure::*;
  import IBD_SUB03_011_EnclosureSiteAttributeInstantiationCompletion::*;
  import IBD_SUB03_HVAC_Loops_Environmental_Sensing_Wiring_Internal::*;
  import IBD_SUB03_013_EnclosureMechanisms_InternalWiring::*;

  // ---------------------------
  // Named acceptance element (reusable by other QA views)
  // ---------------------------
  verification def SUB03_CrossViewConsistencyGate_PortsAndAttributes {
    doc /*
      Acceptance element for SUB03 cross-view consistency (BDD ↔ IBD).
      References IBD-QA-001 checklist structure and rules:
      - No orphan ports (every port has an owning part typed from BDD-SUB03-003)
      - No missing attributes (key BDD attributes appear on IBD instances)
    */;
  }

  // ---------------------------
  // QA rule capture (constraint + note)
  // ---------------------------
  constraint def SUB03_NoOrphanPorts_NoMissingAttributes {

    doc /*
      Rules captured by this constraint (checklist intent):
      1) No orphan ports:
         - Every port shown/used in SUB03 IBDs must be owned by a part instance.
         - That owning part instance must be typed by a part definition from BDD-SUB03-003.
      2) No missing attributes:
         - Key SUB03 attributes defined in BDD-SUB03-003 must be present on the IBD instances
           (at least as attribute usages with matching names/types).
    */;

    // Parameters (kept explicit for reuse in other QA views)
    in owningParts : Part[0..*];
    in requiredAttributeNames : String[0..*];
    in requiredPortNames : String[0..*];

    // Assertions (expression style is intentionally simple and tool-friendly)
    // If your tool supports richer query/collection operators, you can strengthen these.
    assert noOrphanPorts :
      owningParts->forAll(p |
        p.ownedFeature->select(f | f.isKindOf(Port))->forAll(prt |
          prt.owner == p
        )
      );

    assert noMissingAttributes :
      owningParts->forAll(p |
        requiredAttributeNames->forAll(n |
          p.ownedFeature->exists(f | f.isKindOf(Attribute) and f.name == n)
        )
      );

    assert allUsedPortsHaveOwners :
      requiredPortNames->forAll(n |
        owningParts->exists(p |
          p.ownedFeature->exists(f | f.isKindOf(Port) and f.name == n)
        )
      );
  }

  note def SUB03_QA_Note_CrossViewGate {
    doc /*
      Checklist note:
      - "No orphan ports": every port in SUB03 IBDs must appear as an owned port of a typed part instance.
      - "No missing attributes": ensure BDD key attributes (e.g., voltage, frequency, bandwidth, latency,
        temperature, humidity, pressure, windSpeed, commandRate, powerAvailable, thermalPower) are present
        in IBD instance context.
      - Regression guard: if a port/attribute is added/renamed in BDD-SUB03-003, update this QA IBD to match.
    */;
  }

  // ---------------------------
  // QA IBD focus structure (SUB03 cross-view anchor)
  // ---------------------------
  part def SUB03_QA_CrossViewGate_IBD {

    doc /*
      This is an explicit QA IBD "view" that instantiates/anchors key SUB03 parts, attributes, and ports.
      It is intended as a checklist for BDD-SUB03-003 ↔ IBD-SUB03-011/012/013 consistency.
    */;

    // --- Key BDD attributes (explicitly listed as required checklist items)
    // (Modeled as local QA attribute usages so the view is self-contained and diff-friendly.)
    attribute voltage            : voltage;
    attribute frequency          : frequency;
    attribute powerAvailable     : powerAvailable;
    attribute thermalPower       : thermalPower;
    attribute bandwidth          : bandwidth;
    attribute latency            : latency;
    attribute commandRate        : commandRate;

    attribute temperature        : temperature;
    attribute humidity           : humidity;
    attribute pressure           : pressure;
    attribute windSpeed          : windSpeed;

    // --- Key SUB03 parts (typed from BDD-SUB03-003 and/or dependency IBD definitions)
    // Enclosure/Site facility anchor (from IBD-SUB03-011)
    part enclosureSiteFacility : EnclosureSiteFacility;

    // Typical infrastructure elements (typed from BDD-SUB03-003)
    part powerSupply   : InfrastructureElement;
    part hvacPlant     : InfrastructureElement;
    part networkSwitch : InfrastructureElement;
    part envMonitor    : InfrastructureElement;

    // Facility controller (typed from BDD-SUB03-003)
    part facilityCtrl  : FacilityController;

    // HVAC/control and sensing blocks (from IBD-SUB03-012 dependency)
    part hvacControlSystem          : HVACControlSystem;
    part environmentalSensingSystem : EnvironmentalSensingSystem;
    part telemetrySystem            : Telemetry_System;

    // Enclosure mechanisms (from IBD-SUB03-013 dependency)
    part enclosureMechanismsController : EnclosureMechanismController;
    part enclosureMechanismAssembly    : EnclosureMechanismAssembly;

    // --- Required ports used across SUB03 IBDs (explicit ownership on typed parts)
    // Site-level service ports (BDD-SUB03-003)
    port sitePower   : PowerPort;
    port siteCooling : CoolingPort;
    port siteData    : DataPort;
    port siteControl : ControlPort;
    port siteEnv     : EnvPort;

    // Infrastructure owned ports (from IBD-SUB03-011)
    port powerSupply_pwr     : PowerPort;
    port hvacPlant_pwr       : PowerPort;
    port hvacPlant_cool      : CoolingPort;
    port networkSwitch_pwr   : PowerPort;
    port networkSwitch_net   : DataPort;
    port envMonitor_env      : EnvPort;
    port facilityCtrl_ctl    : ControlPort;
    port facilityCtrl_net    : DataPort;
    port facilityCtrl_env    : EnvPort;

    // Enclosure mechanisms ports (from IBD-SUB03-013)
    port cmdIn      : ControlPort;
    port tlmOut     : DataPort;
    port status     : DataPort;
    port inhibitIn  : ControlPort;
    port alarmOut   : DataPort;

    // --- QA bindings showing attribute presence on IBD instances (checklist instantiation)
    // (These usages deliberately mirror dependency IBD attribute names.)
    attribute enclosure_voltage     : voltage  = voltage;
    attribute enclosure_frequency   : frequency = frequency;

    attribute env_temperature       : temperature = temperature;
    attribute env_humidity          : humidity    = humidity;
    attribute env_pressure          : pressure    = pressure;
    attribute env_windSpeed         : windSpeed   = windSpeed;

    attribute net_bandwidth         : bandwidth = bandwidth;
    attribute net_latency           : latency   = latency;

    // --- Minimal connective context (optional, to make ownership explicit)
    // Connections are illustrative: the checklist value is "who owns what port" + "attribute existence".
    connect enclosureSiteFacility.sitePower   to sitePower;
    connect enclosureSiteFacility.siteCooling to siteCooling;
    connect enclosureSiteFacility.siteData    to siteData;
    connect enclosureSiteFacility.siteControl to siteControl;
    connect enclosureSiteFacility.siteEnv     to siteEnv;

    connect powerSupply.pwr         to powerSupply_pwr;
    connect hvacPlant.pwr           to hvacPlant_pwr;
    connect hvacPlant.cool          to hvacPlant_cool;
    connect networkSwitch.pwr       to networkSwitch_pwr;
    connect networkSwitch.net       to networkSwitch_net;
    connect envMonitor.env          to envMonitor_env;

    connect facilityCtrl.controlBus to facilityCtrl_ctl;
    connect facilityCtrl.mgmtNet    to facilityCtrl_net;
    connect facilityCtrl.envSense   to facilityCtrl_env;

    connect enclosureMechanismsController.cmdIn     to cmdIn;
    connect enclosureMechanismsController.tlmOut    to tlmOut;
    connect enclosureMechanismsController.inhibitIn to inhibitIn;
    connect enclosureMechanismsController.alarmOut  to alarmOut;

    // --- Attach QA rule constraint + note (explicitly referenced)
    constraint qaRules : SUB03_NoOrphanPorts_NoMissingAttributes {
      owningParts = {
        enclosureSiteFacility,
        powerSupply,
        hvacPlant,
        networkSwitch,
        envMonitor,
        facilityCtrl,
        hvacControlSystem,
        environmentalSensingSystem,
        telemetrySystem,
        enclosureMechanismsController,
        enclosureMechanismAssembly
      };

      requiredAttributeNames = {
        "voltage","frequency","powerAvailable","thermalPower","bandwidth","latency","commandRate",
        "temperature","humidity","pressure","windSpeed"
      };

      requiredPortNames = {
        "sitePower","siteCooling","siteData","siteControl","siteEnv",
        "powerSupply_pwr","hvacPlant_pwr","hvacPlant_cool","networkSwitch_pwr","networkSwitch_net",
        "envMonitor_env","facilityCtrl_ctl","facilityCtrl_net","facilityCtrl_env",
        "cmdIn","tlmOut","status","inhibitIn","alarmOut"
      };
    }

    note qaNote : SUB03_QA_Note_CrossViewGate;
  }

  // ---------------------------
  // View instance (the "IBD" entry point)
  // ---------------------------
  part IBD_QA_001_View : SUB03_QA_CrossViewGate_IBD;

  // Link the named acceptance element to this view (so other QA views can reference it)
  verification SUB03_CrossViewConsistencyGate_PortsAndAttributes_appliesTo_IBD_QA_001 :
    SUB03_CrossViewConsistencyGate_PortsAndAttributes {
      doc "Applies to IBD-QA-001 checklist view for SUB03 ports+attributes cross-view consistency.";
    }

}
