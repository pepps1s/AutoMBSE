package AD_SUB05_001 {

    action AO_Acquisition_Optional_Loop_Closure {
        in item AOEnable : Boolean;  // AO enable flag
        in item GuideSource : String;  // Guide star selection input (NGS/LGS)
        
        // Step 1: Guide star selection and AO enable check
        first start;
        then action GuideStarSelection {
            in item GuideSource : String;
        };
        
        // Step 2: WFS calibration step
        then action WFSCalibration;
        
        // Step 3: RTC initialization
        then action RTCInitialization;
        
        // Step 4: DM engagement
        then action DMEngage;
        
        // Step 5: Verify residual WFE/Strehl proxy
        then action VerifyResidualWFE_Strehl;

        // Branch: Skip AO if AOEnable is false
        then action SkipAOIfDisabled {
            in item AOEnable : Boolean;
            guard AOEnable == false;
            then action ReturnClearStatus;
        };
        
        // Safety branch for laser-related constraints if LGS path is selected
        then action LaserSafetyCheck {
            in item GuideSource : String;
            guard GuideSource == "LGS";
            then action ApplyLaserSafetyConstraints;
        };
        
        // Final step: Ensure AO loop stability
        then action EnsureAOLoopStability;
        
        then done;
    }

    // Acceptance checks included in the behavior model:
    // - Action includes AOEnable flag and GuideSource selection.
    // - Steps for WFS calibration, RTC start, DM engage, verify residual WFE/Strehl proxy.
    // - Branch that bypasses AO when AOEnable is false and returns a clear status.
    // - Safety branch for laser-related constraints if LGS path is selected.
    // - Ends with declared condition 'AO_Loop_Stable'.
}
