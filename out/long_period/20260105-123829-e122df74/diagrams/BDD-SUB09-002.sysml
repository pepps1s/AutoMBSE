package BDD_SUB09_002_Safety {
    import ScalarValues::*;
    import Quantities::*;
    import SI::*;

    //============================================================
    // Context + Trace stubs (dependencies)
    //============================================================

    package Dependencies {
        // Minimal stubs to allow trace references from this package
        requirement def RD_F_001;
        requirement def RD_NF_001;
    }

    //============================================================
    // Value types, enums, and common abstractions
    //============================================================

    package SafetyTypes {

        enum def SafetySeverityKind { info; warning; critical; emergency; }

        enum def SafetyModeKind {
            NormalOps;
            Maintenance;
            Engineering;
            RestrictedOps;
            SafeState;
            EmergencyStop;
        }

        enum def InterlockStateKind { disarmed; armed; tripped; bypassed; fault; }

        enum def EStopStateKind { released; pressed; latched; resetRequired; fault; }

        enum def PermitKind {
            PersonnelAccess;
            MotionEnable;
            LaserEnable;
            HighVoltageEnable;
            CryogenicEnable;
            InstrumentEnable;
        }

        attribute def SafetyEventId {
            value : String;
        }

        attribute def SafetyTimestamp {
            value : String;
        }

        attribute def SafetyMessage {
            value : String;
        }

        attribute def BoolFlag {
            value : Boolean;
        }

        attribute def Severity {
            value : SafetySeverityKind;
        }

        attribute def InterlockState {
            value : InterlockStateKind;
        }

        attribute def EStopState {
            value : EStopStateKind;
        }

        attribute def SafetyMode {
            value : SafetyModeKind;
        }

        attribute def PermitType {
            value : PermitKind;
        }

        attribute def AccessZoneId {
            value : String;
        }

        attribute def SensorHealth {
            value : Boolean;
        }

        attribute def SubsystemId {
            value : String;
        }
    }

    //============================================================
    // Interfaces (ports)
    //============================================================

    package SafetyInterfaces {
        import SafetyTypes::*;

        interface def I_SafetyEvent {
            in eventId : SafetyEventId;
            in timestamp : SafetyTimestamp;
            in severity : Severity;
            in message : SafetyMessage;
        }

        interface def I_EStopCommand {
            in requestEStop : Boolean;
            in requestReset : Boolean;
        }

        interface def I_EStopStatus {
            out state : EStopState;
            out isLatched : Boolean;
        }

        interface def I_InterlockStatus {
            out state : InterlockState;
            out isTripped : Boolean;
            out isBypassed : Boolean;
        }

        interface def I_InterlockTrip {
            in trip : Boolean;
            in cause : SafetyMessage;
        }

        interface def I_PermitRequest {
            in permit : PermitType;
            in requester : SubsystemId;
            in requested : Boolean;
        }

        interface def I_PermitGrant {
            out permit : PermitType;
            out granted : Boolean;
            out reason : SafetyMessage;
        }

        interface def I_MotionInhibit {
            out inhibitAllMotion : Boolean;
            out inhibitAz : Boolean;
            out inhibitEl : Boolean;
            out inhibitRotator : Boolean;
        }

        interface def I_OperationalConstraints {
            out maxWindSpeedOps : Real;
            out temperatureMinOps : Real;
            out temperatureMaxOps : Real;
            out allowLaserOps : Boolean;
            out allowPersonnelAccess : Boolean;
        }

        interface def I_ZoneAccess {
            in zone : AccessZoneId;
            in requestEntry : Boolean;
            out entryGranted : Boolean;
            out interlockState : InterlockState;
        }

        interface def I_HealthStatus {
            out ok : Boolean;
            out detail : SafetyMessage;
        }
    }

    //============================================================
    // Structural decomposition: Safety subsystem for TMT
    //============================================================

    package SafetyStructure {
        import SafetyTypes::*;
        import SafetyInterfaces::*;
        import Dependencies::*;

        abstract part def SafetyComponent {
            attribute id : SubsystemId;
            attribute healthOk : Boolean;
            port health : I_HealthStatus;
        }

        //--------------- Sensors / Inputs ---------------

        part def DoorAccessSensor :> SafetyComponent {
            attribute zone : AccessZoneId;
            attribute doorClosed : Boolean;
            attribute lockEngaged : Boolean;
            attribute sensorOk : SensorHealth;
            port eventOut : I_SafetyEvent;
        }

        part def EmergencyStopButton :> SafetyComponent {
            attribute location : String;
            attribute state : EStopState;
            port estopStatus : I_EStopStatus;
            port eventOut : I_SafetyEvent;
        }

        part def EnvironmentalSafetyMonitor :> SafetyComponent {
            attribute windSpeed : Real;
            attribute temperature : Real;
            attribute lightningAlert : Boolean;
            attribute sensorOk : SensorHealth;
            port constraintsOut : I_OperationalConstraints;
            port eventOut : I_SafetyEvent;
        }

        part def LaserSafetyInput :> SafetyComponent {
            attribute lgsEnabled : Boolean;
            attribute interlockLoopOk : Boolean;
            attribute aircraftSpotterClear : Boolean;
            port eventOut : I_SafetyEvent;
        }

        //--------------- Core logic / controllers ---------------

        part def InterlockController :> SafetyComponent {
            attribute state : InterlockState;
            attribute bypassActive : Boolean;
            attribute armed : Boolean;

            port tripIn : I_InterlockTrip;
            port statusOut : I_InterlockStatus;

            port zoneAccess : I_ZoneAccess;
            port eventOut : I_SafetyEvent;
        }

        part def EmergencyStopController :> SafetyComponent {
            attribute state : EStopState;
            attribute latched : Boolean;

            port cmdIn : I_EStopCommand;
            port statusOut : I_EStopStatus;

            port motionInhibitOut : I_MotionInhibit;
            port eventOut : I_SafetyEvent;
        }

        part def PermitManager :> SafetyComponent {
            attribute currentMode : SafetyMode;
            attribute allowBypassForMaintenance : Boolean;

            port permitReqIn : I_PermitRequest;
            port permitGrantOut : I_PermitGrant;

            port constraintsIn : I_OperationalConstraints;
            port interlockStatusIn : I_InterlockStatus;
            port estopStatusIn : I_EStopStatus;

            port eventOut : I_SafetyEvent;
        }

        part def OperationalConstraintsController :> SafetyComponent {
            attribute maxWindSpeedOps : Real;
            attribute temperatureMinOps : Real;
            attribute temperatureMaxOps : Real;
            attribute allowLaserOps : Boolean;
            attribute allowPersonnelAccess : Boolean;

            port envIn : I_OperationalConstraints;
            port constraintsOut : I_OperationalConstraints;
            port eventOut : I_SafetyEvent;
        }

        //--------------- Actuation / outputs ---------------

        part def SafetyActuationInterface :> SafetyComponent {
            // Abstract output interface to other subsystems (mount, enclosure, laser, instruments)
            port motionInhibitIn : I_MotionInhibit;
            port permitGrantIn : I_PermitGrant;
            port interlockStatusIn : I_InterlockStatus;
            port estopStatusIn : I_EStopStatus;
            port eventOut : I_SafetyEvent;
        }

        part def SafetyHMI :> SafetyComponent {
            attribute operatorAckRequired : Boolean;
            port eventIn : I_SafetyEvent;
            port permitReqOut : I_PermitRequest;
            port estopCmdOut : I_EStopCommand;
        }

        part def SafetyLoggingAndAudit :> SafetyComponent {
            port eventIn : I_SafetyEvent;
            attribute retainDays : Integer;
        }

        //============================================================
        // Safety Subsystem root (TMT::Safety)
        //============================================================

        part def TMT_SafetySubsystem :> SafetyComponent {
            // Key state/constraints
            attribute currentMode : SafetyMode;
            attribute interlockState : InterlockState;
            attribute estopState : EStopState;

            // Decomposition
            part accessSensors[*] : DoorAccessSensor;
            part estopButtons[*] : EmergencyStopButton;
            part envMonitor : EnvironmentalSafetyMonitor;
            part laserInput : LaserSafetyInput;

            part interlocks : InterlockController;
            part estopCtrl : EmergencyStopController;
            part permits : PermitManager;
            part opsConstraints : OperationalConstraintsController;

            part safetyActuation : SafetyActuationInterface;
            part hmi : SafetyHMI;
            part audit : SafetyLoggingAndAudit;

            // External-facing ports (top-level)
            port permitReq : I_PermitRequest;
            port permitGrant : I_PermitGrant;

            port motionInhibit : I_MotionInhibit;
            port constraints : I_OperationalConstraints;

            port interlockStatus : I_InterlockStatus;
            port estopStatus : I_EStopStatus;

            port safetyEvents : I_SafetyEvent;

            //========================================================
            // Internal connections (IBD-style wiring within BDD package)
            //========================================================

            // Environmental constraints feed operational constraints controller and permits
            connect envMonitor.constraintsOut to opsConstraints.envIn;
            connect opsConstraints.constraintsOut to permits.constraintsIn;
            connect opsConstraints.constraintsOut to constraints;

            // Interlock controller status goes to permits and external status port
            connect interlocks.statusOut to permits.interlockStatusIn;
            connect interlocks.statusOut to interlockStatus;

            // E-stop controller status goes to permits and external status port
            connect estopCtrl.statusOut to permits.estopStatusIn;
            connect estopCtrl.statusOut to estopStatus;

            // Permits output to external and to safety actuation interface
            connect permits.permitGrantOut to permitGrant;
            connect permits.permitGrantOut to safetyActuation.permitGrantIn;

            // External permit requests routed into permits (and from HMI)
            connect permitReq to permits.permitReqIn;
            connect hmi.permitReqOut to permits.permitReqIn;

            // E-stop commands from HMI to controller
            connect hmi.estopCmdOut to estopCtrl.cmdIn;

            // Motion inhibit from e-stop controller to external + actuation interface
            connect estopCtrl.motionInhibitOut to motionInhibit;
            connect estopCtrl.motionInhibitOut to safetyActuation.motionInhibitIn;

            // Zone access requests/decisions through interlock controller (e.g., access control)
            connect interlocks.zoneAccess to interlocks.zoneAccess;

            // Safety events fan-in to logging + HMI + external event port
            connect interlocks.eventOut to safetyEvents;
            connect estopCtrl.eventOut to safetyEvents;
            connect permits.eventOut to safetyEvents;
            connect opsConstraints.eventOut to safetyEvents;
            connect envMonitor.eventOut to safetyEvents;
            connect laserInput.eventOut to safetyEvents;

            connect safetyEvents to audit.eventIn;
            connect safetyEvents to hmi.eventIn;

            // Status fan-in to safety actuation interface for downstream consumers
            connect interlocks.statusOut to safetyActuation.interlockStatusIn;
            connect estopCtrl.statusOut to safetyActuation.estopStatusIn;
        }

        //============================================================
        // Traceability hooks (lightweight)
        //============================================================

        satisfy SafetySubsystemSatisfies_F_SafetyAndInterlocks {
            // Stub satisfy to show trace structure; the referenced requirement defs are stubs above.
            // In a full model, bind to specific RD-F-001 / RD-NF-001 requirement instances.
            requirement = RD_F_001;
            satisfiedBy = TMT_SafetySubsystem;
        }

        satisfy SafetySubsystemSatisfies_NFR_SafetyConstraints {
            requirement = RD_NF_001;
            satisfiedBy = TMT_SafetySubsystem;
        }
    }

    //============================================================
    // Diagram package wrapper (for stable naming)
    //============================================================

    package BDD_SUB09_002 {
        import SafetyStructure::*;
        // View intent: Safety (interlocks, emergency stop, operational constraints)
        part safety : TMT_SafetySubsystem;
    }
}
