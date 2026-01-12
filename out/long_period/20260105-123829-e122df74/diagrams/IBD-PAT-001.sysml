package IBD_PAT_001_PortAttributeInstantiationChecklistHarness {

  // ---------------------------------------------------------------------------
  // Purpose:
  // Reusable IBD harness pattern that (1) forces instantiation of BDD-defined
  // attributes on IBD parts, and (2) prohibits orphan ports in IBD connections
  // by requiring that ports exist on the referenced BDD part types.
  // Designed to be imported without renaming collisions via unique namespace.
  // ---------------------------------------------------------------------------

  package DependencyRefs {
    // Shallow dependency marker (no direct import of external model required here)
    // dependsOn: BDD-SUB03-003
  }

  // ---------------------------------------------------------------------------
  // Library Types (lightweight) to keep this pattern self-contained and import-safe
  // ---------------------------------------------------------------------------

  package PatternLib {

    // ---- Minimal value/type stubs (can be replaced by project quantity libs) ----
    attribute def RealValue;
    attribute def IntValue;
    attribute def BoolValue;

    // ---- Generic item defs for ports (placeholders) ----
    item def PowerFlow;
    item def CoolingFlow;
    item def DataFlow;
    item def ControlFlow;
    item def EnvFlow;

    // -------------------------------------------------------------------------
    // BDD-side "Type Definitions" (Representative subset aligned to BDD-SUB03-003)
    // These are not intended to replace the authoritative BDD model. They exist
    // so constraints/examples can be expressed in a self-contained pattern.
    // -------------------------------------------------------------------------

    part def InfrastructureElement_BDD {
      // Representative BDD attributes (sample list mentions: voltage, frequency, etc.)
      attribute voltage : RealValue;
      attribute frequency : RealValue;
      attribute powerAvailable : RealValue;
      attribute thermalPower : RealValue;
      attribute bandwidth : RealValue;
      attribute latency : RealValue;
      attribute commandRate : RealValue;
      attribute temperature : RealValue;
      attribute humidity : RealValue;
      attribute pressure : RealValue;
      attribute windSpeed : RealValue;

      // Representative BDD ports (sample names mention: PowerPort, CoolingPort, DataPort, ControlPort, EnvPort)
      port PowerPort : PowerPort_BDD;
      port CoolingPort : CoolingPort_BDD;
      port DataPort : DataPort_BDD;
      port ControlPort : ControlPort_BDD;
      port EnvPort : EnvPort_BDD;
    }

    // ---- Port defs (BDD-side types) ----
    port def PowerPort_BDD {
      inout item power : PowerFlow;
    }

    port def CoolingPort_BDD {
      inout item coolant : CoolingFlow;
    }

    port def DataPort_BDD {
      inout item data : DataFlow;
    }

    port def ControlPort_BDD {
      inout item ctrl : ControlFlow;
    }

    port def EnvPort_BDD {
      inout item env : EnvFlow;
    }

    // -------------------------------------------------------------------------
    // Harness Contracts / Rules
    // -------------------------------------------------------------------------

    // Rule 1: "Attribute Instantiation Checklist"
    //
    // Use by specializing this abstract part in an IBD package and binding
    // subjectPart to the concrete IBD part. The checklist requires that a set
    // of "required attributes" are instantiated on the IBD part.
    //
    // Note: SysML v2 does not mandate a single canonical reflection API here;
    // this pattern encodes the checklist as explicit required attributes that
    // must be bound/assigned for the instantiation to be considered complete.
    abstract part def AttributeInstantiationChecklistHarness {

      // The concrete IBD part under test (instance usage)
      ref part subjectPart;

      // Checklist booleans to be satisfied by providing explicit bindings in the IBD.
      // Pattern users should set these via constraints/assignments in their IBDs
      // or replace with project-specific validation automation.
      attribute has_voltage : BoolValue;
      attribute has_frequency : BoolValue;
      attribute has_powerAvailable : BoolValue;
      attribute has_thermalPower : BoolValue;
      attribute has_bandwidth : BoolValue;
      attribute has_latency : BoolValue;
      attribute has_commandRate : BoolValue;

      // Constraint: all required flags must be true for "complete attribute instantiation"
      constraint attributes_instantiated {
        // Treat BoolValue as boolean-like; projects may refine typing.
        // If your tool requires explicit Boolean, map BoolValue accordingly.
        has_voltage
        and has_frequency
        and has_powerAvailable
        and has_thermalPower
        and has_bandwidth
        and has_latency
        and has_commandRate;
      }
    }

    // Rule 2: "No Orphan Ports"
    //
    // Use by binding the endpoint ports used in an IBD connection and the
    // referenced BDD part types for the owning parts. The constraint requires
    // that the endpoint port names/types exist on those BDD types.
    //
    // This is written generically but demonstrated with InfrastructureElement_BDD.
    abstract part def NoOrphanPortsHarness {

      // Endpoint parts (IBD parts) and their BDD types (the referenced "part def" types)
      ref part endA_part;
      ref part endB_part;

      // The BDD types that endA_part / endB_part are intended to conform to
      ref part def endA_bddType;
      ref part def endB_bddType;

      // The ports used in the IBD connection (as port usages on the IBD parts)
      ref port endA_port;
      ref port endB_port;

      // Constraint: ports used in IBD must exist in BDD for referenced types.
      //
      // This pattern expresses the rule by requiring conformance of endpoint ports
      // to some port defined on the BDD type. In the example below, we bind the
      // BDD type to InfrastructureElement_BDD and require the IBD ports to match
      // one of its known ports.
      //
      // Projects may extend this with full metamodel-based reflection or custom validation.
      constraint ports_must_exist_on_bdd_types {

        // For illustration, enforce that when BDD type is InfrastructureElement_BDD,
        // the used ports must be among {PowerPort, CoolingPort, DataPort, ControlPort, EnvPort}
        // and must have the matching port def type.
        (
          (endA_bddType == InfrastructureElement_BDD)
          implies
          (
            (endA_port == endA_part.PowerPort and endA_part.PowerPort :> PowerPort_BDD)
            or (endA_port == endA_part.CoolingPort and endA_part.CoolingPort :> CoolingPort_BDD)
            or (endA_port == endA_part.DataPort and endA_part.DataPort :> DataPort_BDD)
            or (endA_port == endA_part.ControlPort and endA_part.ControlPort :> ControlPort_BDD)
            or (endA_port == endA_part.EnvPort and endA_part.EnvPort :> EnvPort_BDD)
          )
        )
        and
        (
          (endB_bddType == InfrastructureElement_BDD)
          implies
          (
            (endB_port == endB_part.PowerPort and endB_part.PowerPort :> PowerPort_BDD)
            or (endB_port == endB_part.CoolingPort and endB_part.CoolingPort :> CoolingPort_BDD)
            or (endB_port == endB_part.DataPort and endB_part.DataPort :> DataPort_BDD)
            or (endB_port == endB_part.ControlPort and endB_part.ControlPort :> ControlPort_BDD)
            or (endB_port == endB_part.EnvPort and endB_part.EnvPort :> EnvPort_BDD)
          )
        );
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Example Mapping: BDD attribute -> required IBD instantiation
  // Demonstrates "voltage" from BDD-SUB03-003 sample_names and the harness rule.
  // ---------------------------------------------------------------------------

  package ExampleMapping_BDD_to_IBD {

    import PatternLib::*;

    // Example BDD type (would normally live in BDD-SUB03-003)
    part def FacilityController_BDD : InfrastructureElement_BDD {
      // Inherits voltage, frequency, etc. and ports
    }

    // Example IBD part usage (instantiation of a BDD type)
    part facilityController : FacilityController_BDD {
      // IBD instantiation: explicitly bind/instantiate BDD attributes as needed
      // (Tools may require explicit value assignment to count as "instantiated")
      attribute voltage : RealValue;
      attribute frequency : RealValue;
      attribute powerAvailable : RealValue;
      attribute thermalPower : RealValue;
      attribute bandwidth : RealValue;
      attribute latency : RealValue;
      attribute commandRate : RealValue;

      // Ports used in IBD (must not be orphan)
      port PowerPort : PowerPort_BDD;
      port DataPort : DataPort_BDD;
      port ControlPort : ControlPort_BDD;
    }

    // Apply Attribute Instantiation Checklist Harness
    part attrChecklist : AttributeInstantiationChecklistHarness {
      ref part subjectPart = facilityController;

      // Map BDD attribute "voltage" -> required IBD instantiation flag
      // (In practice, set these based on model queries or explicit assignments)
      attribute has_voltage = true;
      attribute has_frequency = true;
      attribute has_powerAvailable = true;
      attribute has_thermalPower = true;
      attribute has_bandwidth = true;
      attribute has_latency = true;
      attribute has_commandRate = true;
    }

    // Example peer element (also conforms to InfrastructureElement_BDD)
    part siteInfrastructure : InfrastructureElement_BDD {
      attribute voltage : RealValue;
      attribute frequency : RealValue;
      attribute powerAvailable : RealValue;
      attribute thermalPower : RealValue;
      attribute bandwidth : RealValue;
      attribute latency : RealValue;
      attribute commandRate : RealValue;

      port PowerPort : PowerPort_BDD;
      port DataPort : DataPort_BDD;
      port ControlPort : ControlPort_BDD;
    }

    // Example IBD connections (Power + Control) using non-orphan ports rule
    connect powerLink
      from facilityController.PowerPort.power
      to siteInfrastructure.PowerPort.power;

    connect controlLink
      from facilityController.ControlPort.ctrl
      to siteInfrastructure.ControlPort.ctrl;

    // Apply No-Orphan-Ports Harness to the power link endpoints
    part noOrphanPorts_power : NoOrphanPortsHarness {
      ref part endA_part = facilityController;
      ref part def endA_bddType = InfrastructureElement_BDD;
      ref port endA_port = facilityController.PowerPort;

      ref part endB_part = siteInfrastructure;
      ref part def endB_bddType = InfrastructureElement_BDD;
      ref port endB_port = siteInfrastructure.PowerPort;
    }

    // Apply No-Orphan-Ports Harness to the control link endpoints
    part noOrphanPorts_ctrl : NoOrphanPortsHarness {
      ref part endA_part = facilityController;
      ref part def endA_bddType = InfrastructureElement_BDD;
      ref port endA_port = facilityController.ControlPort;

      ref part endB_part = siteInfrastructure;
      ref part def endB_bddType = InfrastructureElement_BDD;
      ref port endB_port = siteInfrastructure.ControlPort;
    }
  }

}
