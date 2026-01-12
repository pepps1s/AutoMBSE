package BDD_APS_003_Segment_Actuation_And_Edge_Sensing_Structural_Model {

    import ScalarValues::*;
    import SI::*;
    import Quantities::*;

    // Shallow dependencies (import packages defined in upstream BDDs)
    // NOTE: Package names are representative stable aliases for the dependency views.
    import BDD_OPT_002_Optical_PrimaryMirror_Segments::*;
    import BDD_APS_001_APS_WFS_Subsystem_Decomposition::*;

    //========================================
    // Common / reusable abstractions
    //========================================

    abstract part def SegmentAssembly;
    abstract part def SegmentElectroMechanicalDevice;
    abstract part def SegmentSensor;
    abstract part def SegmentController;
    abstract part def SegmentPowerInterface;
    abstract part def SegmentDataInterface;

    attribute def SegmentIndex {
        value : Integer;
    }

    attribute def LengthTolerance {
        nominal : LengthValue;
        plusMinus : LengthValue;
    }

    attribute def ForceRange {
        min : ForceValue;
        max : ForceValue;
    }

    attribute def StrokeRange {
        min : LengthValue;
        max : LengthValue;
    }

    attribute def SampleRate {
        hz : FrequencyValue;
    }

    attribute def LatencyBudget {
        max : TimeValue;
    }

    //========================================
    // Actuation (segment-level) reusable parts
    //========================================

    part def SegmentActuator : SegmentElectroMechanicalDevice {
        attribute stroke : StrokeRange;
        attribute force : ForceRange;
        attribute positionResolution : LengthTolerance;
        attribute bandwidth : FrequencyValue;
    }

    part def ActuatorDriveElectronics : SegmentController {
        attribute controlLoopRate : SampleRate;
        attribute commandLatency : LatencyBudget;

        // Directional interfaces (logical)
        in  commandIn  : SegmentDataInterface;
        out statusOut  : SegmentDataInterface;
        in  powerIn    : SegmentPowerInterface;
    }

    part def SegmentActuationAssembly : SegmentAssembly {
        // Typical segment supports are implemented with multiple actuators;
        // the exact quantity can be specialized by downstream views.
        part actuators[3] : SegmentActuator;
        part drives[3]    : ActuatorDriveElectronics;

        // Internal defined-by relationship: each drive corresponds to an actuator.
        // (Allocation/connection elaborated in IBD views)
    }

    //========================================
    // Edge sensing (segment-level) reusable parts
    //========================================

    part def SegmentEdgeSensor : SegmentSensor {
        attribute measurementResolution : LengthTolerance;
        attribute sampleRate : SampleRate;
        attribute dynamicRange : LengthValue;
        out edgeGapOut : SegmentDataInterface;
        in  powerIn    : SegmentPowerInterface;
    }

    part def EdgeSensingConditioner : SegmentController {
        attribute sampleRate : SampleRate;
        attribute latency : LatencyBudget;

        in  sensorIn   : SegmentDataInterface;
        out dataOut    : SegmentDataInterface;
        in  powerIn    : SegmentPowerInterface;
    }

    part def SegmentEdgeSensingAssembly : SegmentAssembly {
        // Edge sensors around the segment perimeter; count is a design choice
        // (e.g., per-edge or per-interface). Keep reusable and configurable.
        part edgeSensors[6]  : SegmentEdgeSensor;
        part conditioners[6] : EdgeSensingConditioner;
    }

    //========================================
    // Combined segment actuation + edge sensing
    //========================================

    part def SegmentActuationAndEdgeSensingAssembly : SegmentAssembly {
        part actuation   : SegmentActuationAssembly;
        part edgeSensing : SegmentEdgeSensingAssembly;

        // Segment-local controller for health/status aggregation
        part localController : SegmentController {
            attribute telemetryRate : SampleRate;
            attribute maxLatency    : LatencyBudget;

            in  powerIn   : SegmentPowerInterface;
            in  dataIn    : SegmentDataInterface;
            out dataOut   : SegmentDataInterface;
        }
    }

    //========================================
    // Reusable part definitions for each M1 segment
    //========================================

    // This defines the per-segment structural pattern that downstream views can instantiate
    // for PrimaryMirrorSegment (from BDD-OPT-002) and connect into APS (from BDD-APS-001).
    part def M1Segment_ActuationAndSensingUnit : SegmentAssembly {
        attribute segmentId : SegmentIndex;

        // Link to the optical segment (defined upstream); kept as a reusable reference slot.
        part segment : PrimaryMirrorSegment;

        // Segment-level assemblies (required by acceptance checks)
        part actuationAndEdgeSensing : SegmentActuationAndEdgeSensingAssembly;

        // Optional placeholders for integration with APS metrology / phasing control elements
        // defined upstream in BDD-APS-001 (kept as references to avoid deep coupling).
        part segmentPhasingControl : SegmentPhasingControlElements;
        part metrologyHook         : Metrology;
    }

    // Convenience specialization: a canonical M1 segment unit (can be used as the standard reusable type)
    part def M1SegmentReusableModule :> M1Segment_ActuationAndSensingUnit;

}
