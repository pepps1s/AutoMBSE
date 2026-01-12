package IBD_SUB03_HVAC_Loops_Environmental_Sensing_Wiring_Internal {

    // Import dependencies for parts, ports, and attributes.
    import ScalarValues::*;
    import Quantities::*;
    
    // Define HVAC System Blocks and Parts
    block def HVAC_Control_System {
        attribute power: PowerLevel;
        attribute temperatureSetpoint: Temperature;
        port def temperatureControlPort: TemperatureControl;
        port def humidityControlPort: HumidityControl;
        part fan : HVAC_Actuator;
        part valve : HVAC_Actuator;
    }

    block def HVAC_Actuator {
        attribute actuatorPower: PowerLevel;
        port def actuatorControlPort: ControlSignal;
    }
    
    // Define Sensor Blocks and Ports
    block def Environmental_Sensors {
        part temperatureSensor : TemperatureSensor;
        part humiditySensor : HumiditySensor;
        part pressureSensor : PressureSensor;
        port def temperatureOutputPort: TemperatureOutput;
        port def humidityOutputPort: HumidityOutput;
        port def pressureOutputPort: PressureOutput;
    }
    
    // Define Telemetry System
    block def Telemetry_System {
        port def telemetryInPort: TelemetryIn;
        port def telemetryOutPort: TelemetryOut;
        part telemetryCadence : Cadence;
    }
    
    // Define Environmental Sensing Sensor Types
    block def TemperatureSensor {
        attribute temperature: Temperature;
    }
    
    block def HumiditySensor {
        attribute humidity: Humidity;
    }
    
    block def PressureSensor {
        attribute pressure: Pressure;
    }

    // Define Telemetry Input/Output Ports and Attributes
    port def TelemetryIn {
        in signal: TelemetrySignal;
    }
    
    port def TelemetryOut {
        out signal: TelemetrySignal;
    }
    
    // Define Telemetry Signal Flow
    part def TelemetrySignal;
    
    // Define Control Signal Port for Fan/Valve Control
    port def ControlSignal {
        out signal: ControlCommand;
    }

    // Define the connections for wiring between parts
    part HVACControlSystem : HVAC_Control_System {
        part temperatureControl : HVAC_Actuator;
        part humidityControl : HVAC_Actuator;
        port outPort : temperatureControlPort;
        port inPort : temperatureControlPort;
    }
    
    part EnvironmentalSensingSystem : Environmental_Sensors {
        part temperatureSensor : TemperatureSensor;
        part humiditySensor : HumiditySensor;
        part pressureSensor : PressureSensor;
        port outPort : temperatureOutputPort;
        port inPort : temperatureOutputPort;
    }
    
    part TelemetrySystem : Telemetry_System {
        part telemetryCadence : Cadence;
        port outPort : telemetryInPort;
        port inPort : telemetryOutPort;
    }
}
