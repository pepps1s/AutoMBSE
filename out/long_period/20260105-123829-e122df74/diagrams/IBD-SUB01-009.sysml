package IBD_SUB01_009_OpticalConfigurationStateReportingWiring {

  import ScalarValues::*;
  import Quantities::*;

  //
  // Item (signal) definitions
  //
  item def ConfigStateTelemetry {
    attribute configId : String;
    attribute mechanismName : String;
    attribute stateLabel : String;
    attribute stateCode : Integer;
    attribute timestamp_utc : String;
  }

  item def OpsConfigStateEvent {
    attribute configId : String;
    attribute mechanismName : String;
    attribute stateLabel : String;
  }

  //
  // Port definitions (typed per BDD-SUB01-003 category: ControlPort / TelemetryPort)
  //
  port def TelemetryPort {
    out item telemetry : ConfigStateTelemetry;
  }

  port def TelemetryInPort {
    in item telemetry : ConfigStateTelemetry;
  }

  port def ControlPort {
    in item cmd : String;
    out item ack : String;
  }

  port def OpsSequencingPort {
    out item cfgStateEvent : OpsConfigStateEvent;
  }

  //
  // Mechanisms (configuration mechanisms) and reporting
  //
  part def OpticalPathConfigMechanism {
    attribute mechanismName : String;
    attribute currentConfigId : String;
    attribute currentStateLabel : String;

    port ctrl : ControlPort;
    port statePub : TelemetryPort;
  }

  part def OpticalConfigurationIdentifier {
    attribute configId : String;
  }

  part def OpticalStateReporterTelemetryPublisher {
    attribute publisherName : String;
    attribute lastPublishedConfigId : String;

    // aggregate multiple mechanism telemetry into a single publish stream
    port inMech1 : TelemetryInPort;
    port inMech2 : TelemetryInPort;

    // publish to telemetry backbone + provide an ops-facing event feed
    port telemetryOut : TelemetryPort;
    port opsOut : OpsSequencingPort;
  }

  //
  // External sinks (typed to match root IBD dependencies)
  //
  part def UnifiedTelemetryBackboneGateway {
    attribute telemetryCadence_Hz : Real;
    port telemetryIn : TelemetryInPort;
  }

  part def OCS_SequencerConfigStateIngress {
    attribute subscriberName : String;
    port cfgStateIn : OpsSequencingPort;
  }

  //
  // IBD: internal wiring for optical configuration state reporting
  //
  part def TelescopeOpticalSystem_OpticalConfigurationStateReportingWiring {

    // Configuration mechanisms (examples)
    part m3Steering : OpticalPathConfigMechanism {
      mechanismName = "M3_Steering";
      currentConfigId = "CFG-OCS-OPT-0007";
      currentStateLabel = "IN_POSITION";
    }

    part instrumentSelector : OpticalPathConfigMechanism {
      mechanismName = "Instrument_Selector";
      currentConfigId = "CFG-OCS-OPT-0007";
      currentStateLabel = "LOCKED";
    }

    // Explicit configuration identifier instance captured by telemetry
    part opticalConfig : OpticalConfigurationIdentifier {
      configId = "CFG-OCS-OPT-0007";
    }

    // State reporter / telemetry publisher
    part stateReporter : OpticalStateReporterTelemetryPublisher {
      publisherName = "SUB01_OpticalConfigStateReporter";
      lastPublishedConfigId = "CFG-OCS-OPT-0007";
    }

    // Ops/Telemetry paths (external gateways)
    part telemetryBackbone : UnifiedTelemetryBackboneGateway {
      telemetryCadence_Hz = 1.0;
    }

    part ocsSequencer : OCS_SequencerConfigStateIngress {
      subscriberName = "OCS_Sequencer_ConfigState";
    }

    //
    // Wiring: mechanism state telemetry -> reporter -> telemetry backbone + OCS/Ops sequencing
    //
    connection m3SteeringStateToReporter
      connect m3Steering.statePub::telemetry to stateReporter.inMech1::telemetry;

    connection instrumentSelectorStateToReporter
      connect instrumentSelector.statePub::telemetry to stateReporter.inMech2::telemetry;

    connection reporterToTelemetryBackbone
      connect stateReporter.telemetryOut::telemetry to telemetryBackbone.telemetryIn::telemetry;

    connection reporterToOCSSequencerOpsPath
      connect stateReporter.opsOut::cfgStateEvent to ocsSequencer.cfgStateIn::cfgStateEvent;
  }

}
