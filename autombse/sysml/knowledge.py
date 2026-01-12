SysMLKnowledge = {
    "RD": """Using the SysML v2 rendering engine, the standard code format for defining a requirements diagram is as follows:

% Define a package and requirements inside it %
package requirePackage {
    import ScalarValues::*;
    import Quantities::*;

    requirement def reqDef {
        % Define the requirement description %
        doc /* The description of the requirement */

        % Define attributes and types; the type should be one of: Integer, Real, Boolean %
        attribute require1 : ValueType;
        attribute require2 : ValueType;

        % Define a constraint, typically an inequality involving only attributes %
        require constraint{
            capacityActual <= capacityRequired
        }
    }
}

An example of modeling an eVehicle requirements diagram is shown below:

package eVehicleRequirementDefinitions {

    import ScalarValues::*;
    import Quantities::*;

    requirement def BatteryCapacityReqDef {
        doc /* The actual battery capacity shall be greater than or equal
             * to the required capacity. */

            attribute capacityActual : Real;
            attribute capacityRequired : Real;

            require constraint{ capacityActual <= capacityRequired }
    }

    requirement def MaxSpeedReqDef {
        doc /* The maximum speed of the vehicle shall be
             * not greater than the required maximum speed. */

            attribute maxSpeedVehicle : Real;
            attribute maxSpeedRequired : Real;

            require constraint{ maxSpeedVehicle <= maxSpeedRequired }
    }

}
""",
    "BDD": """Using the SysML v2 rendering engine, the standard code format for defining a Block Definition Diagram (BDD) is as follows:

In SysML v2, a component is defined via `part def` and used via `part`. With this define-and-use pattern, the relationship between (for example) smallVehicle and Vehicle is a “defined by” relationship, which is a form of generalization.

A *port definition* is used to define features that a component exposes through ports. Features with direction are called *directed features*. The direction can be `in`, `out`, or `inout`. Ports can have attributes and reference features. Directed features are reference features, so the keyword `ref` can be omitted.

```sysml
package blockDef {

    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    % Define abstract types. Any types used in Parts other than Integer/Real/Boolean should be pre-defined %
    abstract part def customComponent;
    abstract part def customPart;

    % Define a part %
    part customPart1 : customComponent{
        part body : customPart;
    }

    % Define a port %
    part part1Output{
        part : Real;
    }
}
```

A minimal example is shown below:

```sysml
package eVehicle_LogicalArchitecture {

    import ScalarValues::*;
    import SI::*;

    abstract part def LogicalComponent;
    abstract part def System;
    abstract part def SoI;

    attribute def WheelSize {
        size : LengthValue;
        deviation : LengthValue;
    }

    part def Wheel :> ShapeItems::CircularCylinder {
        attribute sizeOfWheel : WheelSize {
            size = 325 [mm];
            deviation = 1 [mm];
        }
    }

    part eVehicle{
        part body : LogicalComponent;
        part battery : LogicalComponent;
        part wheel : Wheel;
        part frontWheel[2] {
            attribute size : Integer;
        }
        part rearWheel[2] {
            attribute size : Integer;
        }
    }
}
```

Note: before assigning a type to a value using `attribute value : type`, define it using `abstract attribute def`.
When defining `value`, use `value = <value>` rather than `:>> value := <value>`, and the assigned value should not include a unit.
""",
    "IBD": """In SysML v2, a component is defined via `part def` and used via `part`. With this define-and-use pattern, the relationship between (for example) smallVehicle and Vehicle is a “defined by” relationship, which is a form of generalization.

A *port definition* is used to define features that a component exposes through ports. Features with direction are called *directed features*. The direction can be `in`, `out`, or `inout`. Ports can have attributes and reference features. Directed features are reference features, so the keyword `ref` can be omitted.

```sysml
package blockDef {

    import ScalarValues::*;
    import Quantities::*;

    % Define abstract types. Any types used in Parts other than Integer/Real/Boolean should be pre-defined %
    abstract part def customComponent;
    abstract part def customPart;
    abstract attribute def attPart;
    abstract item def itemPart;

    % Define a port type (exposes features via ports) %
    % A port definition should declare out item / in item inside; items inherit from an abstract item type %
    port def part1Output {
        attribute temperature : attPart;
        out item fuelSupply : itemPart;
        in item fuelReturn: itemPart;
    }

    port def part1InPort {
      attribute temperature : attPart;
      in item fuelSupply : itemPart;
      out item fuelReturn : itemPart;
    }

    % Define a part %
    part customPart1 : customComponent{
        part body : customPart;
        % Note: port properties must specify a type <portTypeName> and use ':' to inherit: `port <portName> : portTypeName` %
        port outPort : part1Output;
        port inPort : ~part1Output; // equivalent to part1Input
    }

}
```

A minimal example is shown below:

```sysml
package RadarSystemModel {
    package RadarDefinitionLibrary {
        // Part Definitions
        part def Radar {
            port radarControlPort: RadarControl;
            port radarSignalPort: RadarSignal;
            attribute radarPower: PowerLevel;
            attribute operatingFrequency: Frequency;
        }

        // Port Definitions
        port def RadarControl {
            out radarCommand: RadarCommand;
        }

        port def RadarSignal {
            in radarEcho: RadarEcho;
            out radarWave: RadarWave;
        }

        // Flow Definitions
        part def RadarCommand;
        part def RadarEcho;
        part def RadarWave;

        // Attribute Definitions
        attribute def PowerLevel;
        attribute def Frequency;
    }
}
```
""",
    "AD": """In SysML v2, an activity/behavior diagram is defined through the following key elements:

- `action`: the basic unit of behavior.
- `in item`: defines the input interface of an action.
- `first start`: the start node of a behavior.
- `then action`: defines the sequence of actions.
- `then done`: marks the end of a behavior.

An activity/behavior diagram is defined via `action` and uses `in item` to define input interfaces. Execution starts with `first start`. Each step is defined using `then action`. Within an `action`, use `in item` when an interface is needed. End the behavior with `then done;`.

```sysml
package behavior {

  // Define items
  item def inPort;

  // Define behavior
  action mainAction {
    in item input : inPort; // use input interface

    // Action sequence
    first start; // start
    then action step1; // step 1
    then action step2 { // step 2 (with sub-actions)
      in item input : inPort; // input interface for sub-action
      first start;
      then action subStep1;
      then action subStep2;
      then done;
    }
    then action step3; // step 3
    then done; // end
  }

}

```

A minimal example is shown below:

```sysml
package eVehicle_Behavior {

    action chargeBattery {
        in item power : ElectricalEnergy;

        succession flow chargeBattery.power to charge.power;

        first start;
        then action 'insert charging plug';
        then action 'secure charging plug';
        then action charge {
            in item power : ElectricalEnergy;
        }
        then action 'unsecure charging plug';
        then action removePlug : 'Remove charging plug';
        then done;
    }
    action def 'Remove charging plug';
    item def ElectricalEnergy;
}
```
""",
    "PD": """To implement a SysML v2 variant diagram, define an abstract base component such as `part def Component`. Create concrete parts for each variant, e.g. `part variant1 : Component` and `part variant2 : Component`.
Create an abstract part that represents all possible configurations, using inheritance and redefinition to specify a configuration, e.g. `abstract part Configurations :> Component`.
For each configuration, define a concrete part, for example:
- `part Config1 :> Configurations` where `part redefines partProperty = variant1`
- `part Config2 :> Configurations` where `part redefines partProperty = variant2`
Define constraints to ensure configuration correctness, e.g. using the `constraint` keyword.

A minimal example is shown below:

```sysml
package eVehicle_VariantModel {

    import eVehicle_LogicalArchitecture::*;

    package eVehicle_Configurations {

        import eVehicle_Variations::*;

        part eVehicleStandard :> eVehicleVariations {
            part redefines engine = standardEngine;
            part redefines battery = batteryLow;
        }
        part eVehiclePremium :> eVehicleVariations {
            part redefines engine = powerEngine;
            part redefines battery = batteryHigh;
        }
        part INVALIDeVehicle :> eVehicleVariations {
            part redefines engine = powerEngine;
            part redefines battery = batteryLow;
        }
    }

    package eVehicle_Variants {

        part batteryLow : Battery {
            :>> capacity = 40;
        }
        part batteryHigh : Battery {
            :>> capacity = 60;
        }

        part powerEngine : Engine;
        part standardEngine : Engine;
        abstract part def Engine;
        part def Battery {
            attribute capacity;
        }
    }

    package eVehicle_Variations {
        import eVehicle_VariantModel::*;
        import eVehicle_Variants::*;
        import eVehicle_Variants::*;

        abstract part eVehicleVariations :> eVehicle {

            variation part redefines battery : Battery {
                variant part batterLow;
                variant part batteryHigh;
            }
            variation part redefines engine : Engine {
                variant part standardEngine;
                variant part powerEngine;
            }

            abstract constraint { (battery == batteryLow & engine == standardEngine) ^ (battery == batteryHigh) }
        }
    }
}
```
""",
}
