package BDD-SFT-002_CommandTelemetryBackbone {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Define the parts for the command and telemetry transport backbone system %
    part def OCS : ObservatoryControlSystem {
        part commandProcessing : CommandProcessingUnit;
        part telemetryProcessing : TelemetryProcessingUnit;
    }

    part def CommandProcessingUnit {
        part commandGenerator : CommandGenerator;
        part commandParser : CommandParser;
        part commandEncoder : CommandEncoder;
        part commandSender : CommandSender;
    }

    part def TelemetryProcessingUnit {
        part telemetryReceiver : TelemetryReceiver;
        part telemetryDecoder : TelemetryDecoder;
        part telemetryLogger : TelemetryLogger;
    }

    % Define interfaces between OCS and subsystems %
    interface def CommandInterface {
        direction : inout;
        type : CommandSignal;
    }

    interface def TelemetryInterface {
        direction : inout;
        type : TelemetrySignal;
    }

    % Define the connections between OCS and subsystems %
    part OCSCommandInterface : CommandInterface {
        part : CommandSignal;
    }

    part OCSCommandInterface : TelemetryInterface {
        part : TelemetrySignal;
    }

    % Define shared command and telemetry transport %
    part def SharedTransport : TelemetryCommandTransport {
        part commandLine : CommandLine;
        part telemetryLine : TelemetryLine;
    }

    % Final block with shared command and telemetry transport integration %
    block def CommandTelemetryBackbone {
        part ocs : OCS;
        part sharedTransport : SharedTransport;
        
        connection OCSCommandInterface {
            port : CommandSignal;
        }

        connection OCSCommandInterface {
            port : TelemetrySignal;
        }
    }
}
