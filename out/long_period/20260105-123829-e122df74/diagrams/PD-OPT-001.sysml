package PD_OPT_001_Optical_Wavefront_Error_Allocation_Budget {

    import TMT_System_Architecture::*;
    
    part WFE_Allocation_Budget {
        part M1_Segments_Allocation : WFE_Contributor {
            :>> value = 5.0;
            :>> units = "nm";
        }
        part M2_Allocation : WFE_Contributor {
            :>> value = 3.0;
            :>> units = "nm";
        }
        part M3_Allocation : WFE_Contributor {
            :>> value = 2.0;
            :>> units = "nm";
        }
        part Alignment_Phasing_Allocation : WFE_Contributor {
            :>> value = 1.5;
            :>> units = "nm";
        }
        part AO_Residual_Allocation : WFE_Contributor {
            :>> value = 0.5;
            :>> units = "nm";
        }
        
        constraint Total_WFE_Allocation {
            :>> WFE_Allocation_Sum <= PD-PRF-001.WFE_Limit;
        }
        
        part AO_Enabled_Flag : Boolean {
            :>> value = true;
        }
        
        constraint AO_Impact_on_Allocation {
            :>> if AO_Enabled_Flag == true then WFE_Allocation_Sum += AO_Residual_Allocation;
        }

        constraint RSS_Rollup {
            :>> WFE_Allocation_Sum = sqrt(M1_Segments_Allocation.value^2 + M2_Allocation.value^2 + M3_Allocation.value^2 + Alignment_Phasing_Allocation.value^2 + AO_Residual_Allocation.value^2);
        }
        
        // Allocations per Subsystem and Corresponding BDD Elements
        part TelescopeOpticalSystem :> BDD_SUB01_TelescopeOpticalSystem {
            part redefines WFE_Contributor = M1_Segments_Allocation;
            part redefines WFE_Contributor = M2_Allocation;
            part redefines WFE_Contributor = M3_Allocation;
        }
        
        part AlignmentAndPhasingSystem :> BDD_SUB04_001_AlignmentAndPhasingSystem {
            part redefines WFE_Contributor = Alignment_Phasing_Allocation;
        }
        
        part AdaptiveOpticsSystem :> BDD_SUB05_001_AdaptiveOptics {
            part redefines WFE_Contributor = AO_Residual_Allocation;
        }
        
    }
    
    // Define WFE Contributor type
    abstract part WFE_Contributor {
        attribute value : Real;
        attribute units : String;
    }

    // Roll-up calculation for RSS
    abstract constraint RSS_Rollup {
        value = sqrt(M1_Segments_Allocation.value^2 + M2_Allocation.value^2 + M3_Allocation.value^2 + Alignment_Phasing_Allocation.value^2 + AO_Residual_Allocation.value^2);
    }
}
