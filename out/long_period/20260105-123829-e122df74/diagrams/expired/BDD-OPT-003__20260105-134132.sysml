package M2_Active_Optics_Structural_Model {

    import ScalarValues::*;
    import SI::*;
    
    part def TelescopeOpticalSystem {
        part primaryMirror : PrimaryMirror;
        part secondaryMirror : SecondaryMirror;
        part tertiaryMirror : TertiaryMirror;
        part opticalPath : OpticalPath;
        part systemDependencies : SystemDependencies;
    }

    part def PrimaryMirror {
        part segment[492] : Segment;
        part supportStructure : SupportStructure;
    }

    part def SecondaryMirror {
        part actuator : Actuator;
        part metrologySystem : MetrologySystem;
    }

    part def TertiaryMirror {
        part steeringMechanism : SteeringMechanism;
    }

    part def OpticalPath {
        part opticalElement : OpticalElement;
    }

    part def SystemDependencies {
        part telescopeStructure : TelescopeStructure;
        part alignmentSystem : AlignmentSystem;
    }

    part def Segment {
        part shape : OpticalComponent;
        part actuation : ActuationSystem;
        part metrology : MetrologySystem;
    }

    part def SupportStructure {
        part baseSupport : StructuralElement;
        part actuatorMount : ActuatorMount;
    }

    part def Actuator {
        part linearActuator : LinearActuator;
        part controlSystem : ActuatorControl;
    }

    part def MetrologySystem {
        part wavefrontSensor : WavefrontSensor;
        part positionSensor : PositionSensor;
    }

    part def SteeringMechanism {
        part motors : MotorSystem;
        part controller : SteeringController;
    }

    part def OpticalElement {
        part lens : OpticalLens;
        part mirror : OpticalMirror;
    }

    part def TelescopeStructure {
        part mount : Mount;
        part drives : DriveSystem;
        part encoders : EncoderSystem;
        part vibrationControl : VibrationControl;
        part thermalControl : ThermalControl;
    }

    part def AlignmentSystem {
        part alignmentController : AlignmentController;
        part segmentActuators : SegmentActuator[492];
    }

    part def LinearActuator {
        part motor : Motor;
        part feedbackSystem : FeedbackSystem;
    }

    part def ActuatorControl {
        part controllerUnit : ControllerUnit;
        part signalProcessor : SignalProcessor;
    }

    part def WavefrontSensor {
        part sensorElement : SensorElement;
        part dataProcessor : SensorProcessor;
    }

    part def PositionSensor {
        part positionElement : SensorElement;
        part dataProcessor : SensorProcessor;
    }

    part def MotorSystem {
        part motorUnit : MotorUnit;
        part powerSupply : PowerSupply;
    }

    part def SteeringController {
        part controllerUnit : ControllerUnit;
        part feedbackSystem : FeedbackSystem;
    }

    part def OpticalLens {
        part lensElement : LensElement;
        part coating : Coating;
    }

    part def OpticalMirror {
        part mirrorElement : MirrorElement;
        part coating : Coating;
    }

    part def Mount {
        part azimuth : Azimuth;
        part altitude : Altitude;
    }

    part def DriveSystem {
        part azimuthDrive : DriveUnit;
        part altitudeDrive : DriveUnit;
    }

    part def EncoderSystem {
        part azimuthEncoder : EncoderUnit;
        part altitudeEncoder : EncoderUnit;
    }

    part def VibrationControl {
        part dampers : VibrationDampers;
        part isolators : Isolators;
    }

    part def ThermalControl {
        part heatExchanger : HeatExchanger;
        part temperatureSensor : TemperatureSensor;
    }

    part def SegmentActuator {
        part linearActuator : LinearActuator;
        part positionFeedback : PositionSensor;
    }

    part def SensorElement {
        part opticalSensor : OpticalSensor;
        part electronicSensor : ElectronicSensor;
    }

    part def ControllerUnit {
        part processor : Processor;
        part interfaceUnit : InterfaceUnit;
    }

    part def PowerSupply {
        part energySource : EnergySource;
        part distributionUnit : PowerDistributor;
    }

    part def Motor {
        part motorType : MotorType;
        part powerInput : PowerInput;
    }

    part def FeedbackSystem {
        part feedbackSignal : Signal;
        part processingUnit : SignalProcessor;
    }

    part def SignalProcessor {
        part digitalProcessor : DigitalProcessor;
        part analogProcessor : AnalogProcessor;
    }

    part def LensElement {
        part curvature : Curvature;
        part material : Material;
    }

    part def MirrorElement {
        part curvature : Curvature;
        part material : Material;
    }

    part def Coating {
        part layer : CoatingLayer;
        part reflectiveMaterial : ReflectiveMaterial;
    }

    part def Azimuth {
        part angleSensor : AngleSensor;
        part driveMotor : Motor;
    }

    part def Altitude {
        part angleSensor : AngleSensor;
        part driveMotor : Motor;
    }

    part def DriveUnit {
        part motor : Motor;
        part gearbox : Gearbox;
    }

    part def EncoderUnit {
        part sensor : EncoderSensor;
        part signalProcessor : SignalProcessor;
    }

    part def VibrationDampers {
        part material : DampingMaterial;
        part design : DampingDesign;
    }

    part def Isolators {
        part material : IsolationMaterial;
        part design : IsolationDesign;
    }

    part def HeatExchanger {
        part fluidFlow : FluidFlow;
        part thermalConductivity : ThermalConductivity;
    }

    part def TemperatureSensor {
        part sensorType : SensorType;
        part calibration : Calibration;
    }

    part def EnergySource {
        part type : PowerType;
        part capacity : PowerCapacity;
    }

    part def PowerDistributor {
        part distributionMethod : PowerDistribution;
        part outputPorts : OutputPort[4];
    }

    part def MotorType {
        part type : MotorKind;
        part specification : MotorSpec;
    }

    part def PowerInput {
        part voltage : Voltage;
        part current : Current;
    }

    part def Signal {
        part type : SignalType;
        part frequency : SignalFrequency;
    }

    part def DigitalProcessor {
        part clockSpeed : ClockSpeed;
        part processingUnit : CPU;
    }

    part def AnalogProcessor {
        part gain : Gain;
        part frequencyResponse : FrequencyResponse;
    }

    part def Curvature {
        part radius : Radius;
        part surfaceFinish : SurfaceFinish;
    }

    part def Material {
        part type : MaterialType;
        part composition : MaterialComposition;
    }

    part def CoatingLayer {
        part thickness : Thickness;
        part material : CoatingMaterial;
    }

    part def ReflectiveMaterial {
        part reflectance : Reflectance;
        part absorption : Absorption;
    }

    part def AngleSensor {
        part resolution : Resolution;
        part range : AngleRange;
    }

    part def Motor {
        part motorType : MotorType;
        part powerRating : PowerRating;
    }

    part def EncoderSensor {
        part resolution : Resolution;
        part sensorType : EncoderType;
    }

    part def DampingMaterial {
        part absorptionCoefficient : AbsorptionCoefficient;
        part type : MaterialType;
    }

    part def IsolationMaterial {
        part materialType : MaterialType;
        part isolationCoefficient : IsolationCoefficient;
    }

    part def FluidFlow {
        part flowRate : FlowRate;
        part pressureDrop : PressureDrop;
    }

    part def ThermalConductivity {
        part value : ConductivityValue;
        part temperatureDependency : TemperatureDependency;
    }

    part def SensorType {
        part kind : SensorKind;
        part range : SensorRange;
    }

    part def Calibration {
        part accuracy : Accuracy;
        part traceability : Traceability;
    }

    part def PowerType {
        part kind : PowerKind;
        part voltage : Voltage;
    }

    part def PowerCapacity {
        part value : CapacityValue;
        part unit : CapacityUnit;
    }

    part def PowerDistribution {
        part method : DistributionMethod;
        part voltageLevels : VoltageLevel[3];
    }

    part def OutputPort {
        part type : PortType;
        part capacity : PortCapacity;
    }

    part def MotorKind {
        part type : MotorType;
        part application : MotorApplication;
    }

    part def MotorSpec {
        part torque : Torque;
        part efficiency : Efficiency;
    }

    part def Voltage {
        part level : VoltageLevel;
        part unit : VoltageUnit;
    }

    part def Current {
        part value : CurrentValue;
        part unit : CurrentUnit;
    }

    part def SignalType {
        part kind : SignalKind;
        part frequency : Frequency;
    }

    part def SignalFrequency {
        part value : FrequencyValue;
        part unit : FrequencyUnit;
    }

    part def ClockSpeed {
        part value : SpeedValue;
        part unit : SpeedUnit;
    }

    part def CPU {
        part type : CPUType;
        part frequency : ClockSpeed;
    }

    part def Gain {
        part value : GainValue;
        part unit : GainUnit;
    }

    part def FrequencyResponse {
        part range : FrequencyRange;
        part unit : FrequencyUnit;
    }

    part def Radius {
        part value : RadiusValue;
        part unit : RadiusUnit;
    }

    part def SurfaceFinish {
        part type : FinishType;
        part roughness : RoughnessValue;
    }

    part def MaterialType {
        part type : MaterialKind;
        part properties : MaterialProperties;
    }

    part def MaterialComposition {
        part elements : ElementType[5];
        part proportion : ProportionValue;
    }

    part def Thickness {
        part value : ThicknessValue;
        part unit : ThicknessUnit;
    }

    part def CoatingMaterial {
        part materialType : MaterialType;
        part coatingThickness : CoatingThickness;
    }

    part def Reflectance {
        part value : ReflectanceValue;
        part unit : ReflectanceUnit;
    }

    part def Absorption {
        part value : AbsorptionValue;
        part unit : AbsorptionUnit;
    }

    part def Resolution {
        part value : ResolutionValue;
        part unit : ResolutionUnit;
    }

    part def AngleRange {
        part min : AngleMin;
        part max : AngleMax;
    }

    part def PowerRating {
        part value : RatingValue;
        part unit : RatingUnit;
    }

    part def AbsorptionCoefficient {
        part value : AbsorptionValue;
        part unit : AbsorptionUnit;
    }

    part def IsolationCoefficient {
        part value : IsolationValue;
        part unit : IsolationUnit;
    }

    part def FlowRate {
        part value : FlowValue;
        part unit : FlowUnit;
    }

    part def PressureDrop {
        part value : PressureValue;
        part unit : PressureUnit;
    }

    part def ConductivityValue {
        part value : ConductivityValue;
        part unit : ConductivityUnit;
    }

    part def TemperatureDependency {
        part value : TemperatureDependencyValue;
    }

    part def SensorKind {
        part type : SensorType;
        part sensitivity : Sensitivity;
    }

    part def SensorRange {
        part min : RangeMin;
        part max : RangeMax;
    }

    part def Accuracy {
        part value : AccuracyValue;
        part unit : AccuracyUnit;
    }

    part def Traceability {
        part standard : Standard;
        part certification : Certification;
    }

    part def DistributionMethod {
        part type : MethodType;
        part redundancy : Redundancy;
    }

    part def VoltageLevel {
        part level : VoltageLevel;
        part tolerance : VoltageTolerance;
    }

    part def PortType {
        part type : ConnectionType;
        part rating : PortRating;
    }

    part def PortCapacity {
        part value : CapacityValue;
        part unit : CapacityUnit;
    }

    part def MotorApplication {
        part type : ApplicationType;
        part rating : ApplicationRating;
    }

    part def Torque {
        part value : TorqueValue;
        part unit : TorqueUnit;
    }

    part def Efficiency {
        part value : EfficiencyValue;
        part unit : EfficiencyUnit;
    }

    part def VoltageLevel {
        part level : VoltageValue;
        part tolerance : VoltageTolerance;
    }

    part def CurrentValue {
        part value : CurrentAmount;
        part unit : CurrentUnit;
    }

    part def VoltageUnit {
        part unit : UnitType;
    }

    part def CurrentUnit {
        part unit : UnitType;
    }

    part def SignalKind {
        part type : SignalType;
    }

    part def FrequencyValue {
        part value : FrequencyMagnitude;
        part unit : FrequencyUnit;
    }

    part def SpeedValue {
        part value : SpeedMagnitude;
        part unit : SpeedUnit;
    }

    part def SignalUnit {
        part unit : SignalUnitType;
    }

    part def ProportionValue {
        part value : Value;
        part unit : Unit;
    }

    part def ElementType {
        part element : Element;
    }

}
