package TMT_PerformanceBudget {

    import TMT_SystemStructure::*;
    import TMT_Requirements::*;

    package TMT_PerformanceBudget_001 {

        part TMT_PerformanceBudget :> TMT_Budget {
            part redefines wavelength = 1.0;
            part redefines WFE = 0.03;
            part redefines throughput = 0.85;
            part redefines jitter = 0.002;
            part redefines drift = 0.001;
            part redefines exposureTime = 300;
            part redefines dutyCycle = 0.9;
        }

        part TMT_PerformanceBudget_Anchor :> TMT_PerformanceBudget {
            part redefines wavelength = 1.5;
            part redefines WFE = 0.02;
            part redefines throughput = 0.90;
            part redefines jitter = 0.0015;
            part redefines drift = 0.0008;
            part redefines exposureTime = 250;
            part redefines dutyCycle = 0.95;
        }

        constraint PerformanceBudgetConstraint {
            part redefines WFE = (sqrt(WFE^2 + jitter^2));
            part redefines jitter = (sqrt(jitter^2 + drift^2));
            part redefines throughput = (throughput * 0.9);
        }
    }

    package TMT_PerformanceRequirements {
        
        import TMT_PerformanceBudget_001::*;
        
        part TMT_RD_NF_001_Performance :> TMT_Requirements {
            requirement sensitivityAndAngularResolution :> TMT_NF_001 {
                attribute sensitivity = "High";
                attribute angularResolution = 0.01;
            }

            requirement jitterAndTrackingStability :> TMT_NF_002 {
                attribute jitterTolerance = 0.002;
                attribute trackingError = 0.005;
            }
        }

        part TMT_RD_NF_002_Performance :> TMT_Requirements {
            requirement jitterSensitivity :> TMT_NF_002 {
                attribute jitterTolerance = 0.002;
            }

            requirement throughput :> TMT_NF_001 {
                attribute targetThroughput = 0.85;
            }
        }
    }

    package TMT_Budget {
        part def TMT_Budget {
            attribute wavelength;
            attribute WFE;
            attribute throughput;
            attribute jitter;
            attribute drift;
            attribute exposureTime;
            attribute dutyCycle;
        }
    }

    package TMT_Requirements {
        part def TMT_NF_001 {
            attribute sensitivity;
            attribute angularResolution;
        }

        part def TMT_NF_002 {
            attribute jitterTolerance;
            attribute trackingError;
        }
    }

}
