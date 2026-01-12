package BDD_APS_002_WFS_Suite_Structural_Breakdown {
    import ScalarValues::*;
    import SI::*;
    import Quantities::*;

    // Diagram: BDD-APS-002
    // Title: Wavefront Sensor Suite Structural Breakdown
    // Dependency: BDD-APS-001

    // =========
    // Abstract bases / placeholders
    // =========
    abstract part def WFS_Element;
    abstract part def WFS_Sensor :> WFS_Element;
    abstract part def WFS_Optics :> WFS_Element;
    abstract part def WFS_Detector :> WFS_Element;
    abstract part def WFS_Electronics :> WFS_Element;
    abstract part def WFS_Software :> WFS_Element;
    abstract part def WFS_CalibrationPath :> WFS_Element;

    // =========
    // Attributes (lightweight, no units assigned per guidance)
    // =========
    abstract attribute def WFS_SensorPerf {
        resolution : Real;
        sensitivity : Real;
        fieldOfView : Real;
        bandwidth : Real;
    }

    abstract attribute def WFS_DetectorPerf {
        pixelFormat : String;
        frameRate : Real;
        readNoise : Real;
        quantumEfficiency : Real;
    }

    abstract attribute def WFS_OpticsPerf {
        aperture : Real;
        focalLength : Real;
        throughput : Real;
        pupilSampling : Real;
    }

    abstract attribute def WFS_ElectronicsPerf {
        adcBits : Integer;
        dataRate : Real;
        latency : Real;
        power : Real;
    }

    // =========
    // Sensor Types (breakdown)
    // =========
    part def ShackHartmannWFS :> WFS_Sensor {
        attribute perf : WFS_SensorPerf;
        part lensletArray : LensletArrayOptics;
        part detector : ImagingDetector;
        part readoutElectronics : DetectorReadoutElectronics;
    }

    part def PyramidWFS :> WFS_Sensor {
        attribute perf : WFS_SensorPerf;
        part pyramidOptic : PyramidPrismOptics;
        part modulationOptics : ModulationOptics;
        part detector : ImagingDetector;
        part readoutElectronics : DetectorReadoutElectronics;
    }

    part def CurvatureWFS :> WFS_Sensor {
        attribute perf : WFS_SensorPerf;
        part defocusOptics : CurvatureDefocusOptics;
        part detector : ImagingDetector;
        part readoutElectronics : DetectorReadoutElectronics;
    }

    part def ZernikeWFS :> WFS_Sensor {
        attribute perf : WFS_SensorPerf;
        part phaseMaskOptic : PhaseMaskOptics;
        part detector : ImagingDetector;
        part readoutElectronics : DetectorReadoutElectronics;
    }

    // =========
    // Optics breakdown (suite-level)
    // =========
    part def BeamPickoffOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    part def RelayOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    part def PupilImagingOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    part def FilterWheelOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    part def LensletArrayOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    part def PyramidPrismOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    part def ModulationOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    part def CurvatureDefocusOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    part def PhaseMaskOptics :> WFS_Optics {
        attribute perf : WFS_OpticsPerf;
    }

    // =========
    // Detector breakdown
    // =========
    part def ImagingDetector :> WFS_Detector {
        attribute perf : WFS_DetectorPerf;
    }

    part def DetectorFrontEnd :> WFS_Detector {
        attribute perf : WFS_DetectorPerf;
    }

    // =========
    // Readout / Control Electronics (placeholders)
    // =========
    part def DetectorReadoutElectronics :> WFS_Electronics {
        attribute perf : WFS_ElectronicsPerf;
        part adc : ADC_Module;
        part timing : TimingSyncModule;
        part dataLink : HighSpeedDataInterface;
    }

    part def SensorControlElectronics :> WFS_Electronics {
        attribute perf : WFS_ElectronicsPerf;
        part powerRegulation : PowerConditioningModule;
        part io : ControlIO_Module;
    }

    part def ThermalControlElectronics :> WFS_Electronics {
        attribute perf : WFS_ElectronicsPerf;
        part temperatureSense : TemperatureSenseModule;
        part heaterDriver : HeaterDriverModule;
    }

    // Electronics leaf placeholders
    part def ADC_Module :> WFS_Electronics;
    part def TimingSyncModule :> WFS_Electronics;
    part def HighSpeedDataInterface :> WFS_Electronics;
    part def PowerConditioningModule :> WFS_Electronics;
    part def ControlIO_Module :> WFS_Electronics;
    part def TemperatureSenseModule :> WFS_Electronics;
    part def HeaterDriverModule :> WFS_Electronics;

    // =========
    // Software / firmware placeholders
    // =========
    part def WFSSignalProcessing :> WFS_Software;
    part def WFSCalibrationProcessing :> WFS_Software;
    part def WFSControlInterface :> WFS_Software;

    // =========
    // Calibration path stubs (placeholders included by requirement)
    // =========
    part def InternalCalibrationSourcePath :> WFS_CalibrationPath {
        part calSource : CalibrationSource;
        part injectionOptics : CalibrationInjectionOptics;
        part shutter : CalibrationShutter;
    }

    part def ExternalCalibrationInterfacePath :> WFS_CalibrationPath {
        part externalPort : CalibrationExternalPort;
        part couplingOptics : CalibrationCouplingOptics;
        part shutter : CalibrationShutter;
    }

    // Calibration leaf placeholders
    part def CalibrationSource :> WFS_Element;
    part def CalibrationInjectionOptics :> WFS_Optics;
    part def CalibrationExternalPort :> WFS_Element;
    part def CalibrationCouplingOptics :> WFS_Optics;
    part def CalibrationShutter :> WFS_Element;

    // =========
    // Suite (the thing being decomposed)
    // =========
    part def WavefrontSensorSuite :> WFS_Element {

        // Sensor types (clear breakdown)
        part shackHartmann : ShackHartmannWFS;
        part pyramid : PyramidWFS;
        part curvature : CurvatureWFS;
        part zernike : ZernikeWFS;

        // Shared optics (clear breakdown)
        part beamPickoff : BeamPickoffOptics;
        part relay : RelayOptics;
        part pupilImager : PupilImagingOptics;
        part filters : FilterWheelOptics;

        // Detector and electronics placeholders (clear breakdown)
        part detector : ImagingDetector;
        part detectorFrontEnd : DetectorFrontEnd;

        part readoutElectronics : DetectorReadoutElectronics;
        part sensorControlElectronics : SensorControlElectronics;
        part thermalControlElectronics : ThermalControlElectronics;

        // Software placeholders
        part signalProcessing : WFSSignalProcessing;
        part calibrationProcessing : WFSCalibrationProcessing;
        part controlInterface : WFSControlInterface;

        // Calibration path placeholders (stubs)
        part internalCalibrationPath : InternalCalibrationSourcePath;
        part externalCalibrationPath : ExternalCalibrationInterfacePath;
    }

    // =========
    // Context package root for this BDD (stable naming)
    // =========
    part BDD_APS_WFS_Suite_Structural_Breakdown {
        part wfsSuite : WavefrontSensorSuite;
    }
}
