package TMT_BDD_ENC_001 {
    import ScalarValues::*;
    import SI::*;

    // Diagram: BDD-ENC-001
    // Title: Enclosure/Dome System Decomposition (Level 2)
    // Depends on: BDD-SYS-003 (TMT top-level structure incl. enclosure)

    abstract part def TMTComponent;
    abstract part def MechanicalSubsystem :> TMTComponent;
    abstract part def FacilitySubsystem :> TMTComponent;

    // ---------- Enclosure/Dome (Level 2) ----------
    part def EnclosureDomeSystem :> FacilitySubsystem {
        part domeRotation : DomeRotationSubsystem;
        part shuttersVents : ShuttersAndVentsSubsystem;
        part windScreens : WindScreenSubsystem;
        part thermalManagementStructures : ThermalManagementStructures;
        part servicePlatforms : ServicePlatformsSubsystem;
    }

    // ---------- Dome rotation ----------
    part def DomeRotationSubsystem :> MechanicalSubsystem {
        part azimuthStructure : DomeAzimuthStructure;
        part rotationDrive : RotationDriveTrain;
        part azimuthBearing : AzimuthBearingAssembly;
        part rotationEncoder : DomeAzimuthEncoder;
        part controlInterlocks : RotationInterlocks;
    }

    part def DomeAzimuthStructure :> MechanicalSubsystem;
    part def RotationDriveTrain :> MechanicalSubsystem {
        part motorGroup : DriveMotorAssembly;
        part gearbox : GearboxAssembly;
        part pinionRing : PinionToRingInterface;
        part brake : RotationBrakeAssembly;
    }
    part def DriveMotorAssembly :> MechanicalSubsystem;
    part def GearboxAssembly :> MechanicalSubsystem;
    part def PinionToRingInterface :> MechanicalSubsystem;
    part def RotationBrakeAssembly :> MechanicalSubsystem;

    part def AzimuthBearingAssembly :> MechanicalSubsystem;
    part def DomeAzimuthEncoder :> MechanicalSubsystem;

    part def RotationInterlocks :> MechanicalSubsystem {
        part limitSwitches : LimitSwitchSet;
        part overspeedProtection : OverspeedProtection;
        part emergencyStopInterface : EmergencyStopInterface;
    }
    part def LimitSwitchSet :> MechanicalSubsystem;
    part def OverspeedProtection :> MechanicalSubsystem;
    part def EmergencyStopInterface :> MechanicalSubsystem;

    // ---------- Shutters / vents ----------
    part def ShuttersAndVentsSubsystem :> MechanicalSubsystem {
        part mainApertureShutter : MainApertureShutterAssembly;
        part ventLouvers : VentLouverArray;
        part shutterDrives : ShutterDriveSubsystem;
        part shutterInterlocks : ShutterInterlocks;
    }

    part def MainApertureShutterAssembly :> MechanicalSubsystem {
        part shutterPanels : ShutterPanel[2];
        part shutterRails : ShutterRailSet;
        part weatherSeals : ShutterSealSet;
    }
    part def ShutterPanel :> MechanicalSubsystem;
    part def ShutterRailSet :> MechanicalSubsystem;
    part def ShutterSealSet :> MechanicalSubsystem;

    part def VentLouverArray :> MechanicalSubsystem {
        part louverBank[*] : VentLouverBank;
    }
    part def VentLouverBank :> MechanicalSubsystem {
        part louverBlades[*] : LouverBlade;
        part louverActuator : LouverActuator;
        part louverPositionSensor : LouverPositionSensor;
    }
    part def LouverBlade :> MechanicalSubsystem;
    part def LouverActuator :> MechanicalSubsystem;
    part def LouverPositionSensor :> MechanicalSubsystem;

    part def ShutterDriveSubsystem :> MechanicalSubsystem {
        part driveMotors : ShutterMotorAssembly;
        part transmission : ShutterTransmission;
        part brakes : ShutterBrakeAssembly;
    }
    part def ShutterMotorAssembly :> MechanicalSubsystem;
    part def ShutterTransmission :> MechanicalSubsystem;
    part def ShutterBrakeAssembly :> MechanicalSubsystem;

    part def ShutterInterlocks :> MechanicalSubsystem {
        part openCloseLimits : ShutterLimitSet;
        part windStowLogic : WindStowInterlock;
        part personnelSafetyLocks : PersonnelSafetyLockout;
    }
    part def ShutterLimitSet :> MechanicalSubsystem;
    part def WindStowInterlock :> MechanicalSubsystem;
    part def PersonnelSafetyLockout :> MechanicalSubsystem;

    // ---------- Wind screens ----------
    part def WindScreenSubsystem :> MechanicalSubsystem {
        part windScreenPanels[*] : WindScreenPanel;
        part windScreenActuation : WindScreenActuationSubsystem;
        part windScreenSensors : WindScreenSensorSuite;
    }

    part def WindScreenPanel :> MechanicalSubsystem;
    part def WindScreenActuationSubsystem :> MechanicalSubsystem {
        part actuators[*] : WindScreenActuator;
        part linkages[*] : WindScreenLinkage;
    }
    part def WindScreenActuator :> MechanicalSubsystem;
    part def WindScreenLinkage :> MechanicalSubsystem;

    part def WindScreenSensorSuite :> MechanicalSubsystem {
        part localAnemometer : Anemometer;
        part panelPositionSensors[*] : PanelPositionSensor;
    }
    part def Anemometer :> MechanicalSubsystem;
    part def PanelPositionSensor :> MechanicalSubsystem;

    // ---------- Thermal management structures ----------
    part def ThermalManagementStructures :> FacilitySubsystem {
        part insulation : EnclosureInsulationSystem;
        part thermalBaffles : ThermalBaffleSystem;
        part ventilationPlenum : VentilationPlenumSystem;
        part heatShielding : HeatShieldingSystem;
    }

    part def EnclosureInsulationSystem :> FacilitySubsystem {
        part insulationLayers[*] : InsulationLayer;
        part vaporBarrier : VaporBarrier;
    }
    part def InsulationLayer :> FacilitySubsystem;
    part def VaporBarrier :> FacilitySubsystem;

    part def ThermalBaffleSystem :> FacilitySubsystem {
        part baffles[*] : ThermalBaffle;
        part radiantSurfaces[*] : RadiantControlSurface;
    }
    part def ThermalBaffle :> FacilitySubsystem;
    part def RadiantControlSurface :> FacilitySubsystem;

    part def VentilationPlenumSystem :> FacilitySubsystem {
        part plenums[*] : VentPlenum;
        part flowGuides[*] : FlowGuide;
    }
    part def VentPlenum :> FacilitySubsystem;
    part def FlowGuide :> FacilitySubsystem;

    part def HeatShieldingSystem :> FacilitySubsystem {
        part shields[*] : HeatShieldPanel;
        part reflectiveCoatings[*] : ReflectiveCoating;
    }
    part def HeatShieldPanel :> FacilitySubsystem;
    part def ReflectiveCoating :> FacilitySubsystem;

    // ---------- Service platforms ----------
    part def ServicePlatformsSubsystem :> FacilitySubsystem {
        part accessPlatforms[*] : AccessPlatform;
        part maintenanceWalkways[*] : MaintenanceWalkway;
        part laddersStairs[*] : LadderOrStairAssembly;
        part fallProtection : FallProtectionSystem;
        part liftingHandlingPoints : LiftingAndHandlingPoints;
    }

    part def AccessPlatform :> FacilitySubsystem {
        part platformDeck : PlatformDeck;
        part guardRails : GuardRailSet;
        part toeBoards : ToeBoardSet;
    }
    part def PlatformDeck :> FacilitySubsystem;
    part def GuardRailSet :> FacilitySubsystem;
    part def ToeBoardSet :> FacilitySubsystem;

    part def MaintenanceWalkway :> FacilitySubsystem;
    part def LadderOrStairAssembly :> FacilitySubsystem;

    part def FallProtectionSystem :> FacilitySubsystem {
        part anchorPoints[*] : SafetyAnchorPoint;
        part lifelines[*] : SafetyLifeline;
    }
    part def SafetyAnchorPoint :> FacilitySubsystem;
    part def SafetyLifeline :> FacilitySubsystem;

    part def LiftingAndHandlingPoints :> FacilitySubsystem {
        part hoistPoints[*] : HoistPoint;
        part riggingInterfaces[*] : RiggingInterface;
    }
    part def HoistPoint :> FacilitySubsystem;
    part def RiggingInterface :> FacilitySubsystem;

    // ---------- Usage (instance) ----------
    part enclosure : EnclosureDomeSystem;
}
