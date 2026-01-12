package BDD_INS_002_InstrumentPortModel {
    
    import ScalarValues::*;
    import Quantities::*;
    import SI::*;
    
    part def InstrumentPort {
        part mechanicalInterface : MechanicalInterface;
        part serviceBundle : ServiceBundle;
    }

    part def MechanicalInterface {
        port def mechanicalPlaceholder {
            direction = out;
            type = MechanicalConnection;
        }
        part placeholder : MechanicalPart;
    }
    
    part def ServiceBundle {
        part def powerService : PowerService;
        part def dataService : DataService;
        part def thermalService : ThermalService;
    }
    
    part def PowerService {
        port def powerPort {
            direction = inout;
            type = PowerConnection;
        }
        part powerUnit : PowerComponent;
    }
    
    part def DataService {
        port def dataPort {
            direction = inout;
            type = DataConnection;
        }
        part dataUnit : DataComponent;
    }
    
    part def ThermalService {
        port def thermalPort {
            direction = inout;
            type = ThermalConnection;
        }
        part thermalUnit : ThermalComponent;
    }
    
    abstract part def MechanicalConnection;
    abstract part def PowerConnection;
    abstract part def DataConnection;
    abstract part def ThermalConnection;

    abstract part def MechanicalPart;
    abstract part def PowerComponent;
    abstract part def DataComponent;
    abstract part def ThermalComponent;
    
    package ServiceInterfaces {
        part def Power {
            port powerIn : PowerService;
            port powerOut : PowerService;
        }
        
        part def Data {
            port dataIn : DataService;
            port dataOut : DataService;
        }
        
        part def Thermal {
            port thermalIn : ThermalService;
            port thermalOut : ThermalService;
        }
    }

    instrumentPortStructure : InstrumentPort {
        part instrumentMechanicalInterface : MechanicalInterface;
        part instrumentServiceBundle : ServiceBundle;
    }
}
